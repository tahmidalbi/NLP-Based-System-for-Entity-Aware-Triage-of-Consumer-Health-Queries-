"""
Phase 11 - Checkpoint selection and confidence calibration (guide section 14).

All design decisions are frozen before this point (guide 14.1): preprocessing,
tokenization, corpus, embeddings, architecture, hyperparameters, and the
Phase 10 checkpoint-selection rule. Test sets are not touched here or in this
phase - only severity VALIDATION logits are used, for two things:

  1. Fit a single scalar temperature T minimizing validation NLL
     (guide 14.2): P_calibrated = softmax(logits / T). This never changes
     which class wins the softmax in the usual case - only the confidence
     number attached to that decision.

  2. Choose the "low confidence - human review" threshold from validation
     behaviour, not a hard-coded 0.50/0.70 (guide 14.3) - see
     severity_eval.choose_review_threshold.

Usage
-----
  python scripts/phase11_calibrate.py --smoke --cpu
  python scripts/phase11_calibrate.py --cache /kaggle/input/.../feature_cache.npz

Outputs
-------
  checkpoints/calibration.json   (temperature + threshold; consumed by Phase 12
                                   and by any future inference/demo script)
  results/phase11_calibration.json / .md
  figures/reliability_plot.png   (best-effort; skipped with a warning if
                                   matplotlib is unavailable)
"""

import argparse
from pathlib import Path

import torch

from model import BanglaCareModel
from severity_eval import (
    _nll,
    choose_review_threshold,
    expected_calibration_error,
    fit_temperature,
    softmax,
)
from train_utils import (
    CHECKPOINTS,
    DEFAULT_CACHE,
    RESULTS,
    ROOT,
    get_device,
    load_checkpoint,
    load_featurizer,
    severity_loader,
    write_json,
    write_text,
)

FIGURES = ROOT / "figures"


@torch.no_grad()
def collect_val_logits(model, loader, device):
    model.eval()
    all_logits, all_labels = [], []
    for batch in loader:
        features = batch["features"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        logits, _ = model.severity_forward(features, batch["lengths"], mask)
        all_logits.append(logits.float().cpu())
        all_labels.append(batch["severity_ids"])
    return torch.cat(all_logits).numpy(), torch.cat(all_labels).numpy()


def maybe_plot_reliability(bins_before, bins_after, out_path):
    """Best-effort reliability diagram. Matplotlib is preinstalled on Kaggle
    but is not a hard dependency of this repo, so a missing import degrades
    to a printed warning rather than a crash - the numeric bins are already
    saved in the JSON/markdown reports regardless.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available - skipping figures/reliability_plot.png "
              "(numeric bins are still in results/phase11_calibration.json)")
        return False

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
    for ax, bins, label in [(axes[0], bins_before, "before (T=1)"), (axes[1], bins_after, "after")]:
        centers = [(b["lo"] + b["hi"]) / 2 for b in bins if b["count"] > 0]
        confs = [b["confidence"] for b in bins if b["count"] > 0]
        accs = [b["accuracy"] for b in bins if b["count"] > 0]
        ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="perfect calibration")
        ax.bar(centers, accs, width=0.06, alpha=0.7, label="accuracy")
        ax.scatter(confs, accs, color="crimson", zorder=3, s=15, label="bin mean")
        ax.set_title(f"Reliability ({label})")
        ax.set_xlabel("confidence")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("accuracy")
    axes[0].legend(fontsize=8)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 11 - calibration")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--joint-checkpoint", type=Path, default=CHECKPOINTS / "best_joint.pt")
    parser.add_argument("--out", type=Path, default=CHECKPOINTS / "calibration.json")
    parser.add_argument("--n-bins", type=int, default=15, help="reliability-diagram bin count")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    limit = args.limit if args.limit is not None else (200 if args.smoke else None)

    device = get_device(prefer_cuda=not args.cpu)
    print("BanglaCare - Phase 11: confidence calibration")
    print("=" * 60)
    print(f"device: {device}")
    print(f"loading Phase 10 checkpoint from {args.joint_checkpoint}")
    print("using severity VALIDATION only - test sets are untouched (guide 14.1)")

    featurizer = load_featurizer(args.cache)
    val_loader = severity_loader(featurizer, "val", args.batch_size, shuffle=False, limit=limit)

    model = BanglaCareModel().to(device)
    load_checkpoint(args.joint_checkpoint, model, device=device, strict=True)

    logits, labels = collect_val_logits(model, val_loader, device)
    print(f"collected {len(labels)} validation logits")

    temperature, fitted_nll = fit_temperature(logits, labels)
    uncalibrated_nll = _nll(logits, labels, 1.0)

    boundary_warning = None
    if temperature <= 0.051 or temperature >= 4.99:
        boundary_warning = (
            f"fitted T={temperature:.3f} sits at the search boundary - the true "
            "optimum may lie outside [0.05, 5.0]. Inspect the logits before trusting "
            "this calibration."
        )
        print(f"WARNING: {boundary_warning}")

    probs_before = softmax(logits, 1.0)
    probs_after = softmax(logits, temperature)
    ece_before, bins_before = expected_calibration_error(probs_before, labels, args.n_bins)
    ece_after, bins_after = expected_calibration_error(probs_after, labels, args.n_bins)

    preds = probs_after.argmax(axis=1)  # same as probs_before.argmax - T doesn't change ranking
    is_error = preds != labels
    confidences_after = probs_after.max(axis=1)
    threshold, threshold_stats = choose_review_threshold(confidences_after, is_error)

    print(f"\ntemperature: {temperature:.4f}")
    print(f"validation NLL: {uncalibrated_nll:.4f} -> {fitted_nll:.4f}")
    print(f"ECE: {ece_before:.4f} -> {ece_after:.4f}")
    print(f"review threshold: flag when confidence < {threshold:.4f}")
    print(
        f"  catches {threshold_stats.get('errors_caught_recall', 0):.1%} of validation errors, "
        f"needlessly flags {threshold_stats.get('correct_needlessly_flagged_rate', 0):.1%} "
        f"of correct predictions"
    )

    plotted = maybe_plot_reliability(bins_before, bins_after, FIGURES / "reliability_plot.png")

    calibration = {
        "temperature": temperature,
        "review_threshold": threshold,
        "fitted_on": "severity validation split only",
        "source_checkpoint": str(args.joint_checkpoint),
        "n_validation_examples": int(len(labels)),
        "boundary_warning": boundary_warning,
    }
    write_json(args.out, calibration)
    print(f"\nwrote {args.out}")

    summary = {
        "config": {"phase": 11, "n_bins": args.n_bins, "batch_size": args.batch_size},
        "calibration": calibration,
        "nll_before": uncalibrated_nll,
        "nll_after": fitted_nll,
        "ece_before": ece_before,
        "ece_after": ece_after,
        "threshold_stats": threshold_stats,
        "reliability_bins_before": bins_before,
        "reliability_bins_after": bins_after,
    }
    write_json(RESULTS / "phase11_calibration.json", summary)

    report = "\n".join([
        "# BanglaCare - Phase 11 calibration",
        "",
        f"- Checkpoint: `{args.joint_checkpoint}`",
        f"- Fitted on: severity validation only ({len(labels)} examples) - "
        f"test sets untouched (guide 14.1)",
        "",
        "## Temperature scaling",
        "",
        f"- **Temperature: {temperature:.4f}**",
        f"- Validation NLL: {uncalibrated_nll:.4f} -> {fitted_nll:.4f}",
        f"- ECE: {ece_before:.4f} -> {ece_after:.4f}",
        (f"- ⚠️ {boundary_warning}" if boundary_warning else "- Fit converged inside the search range."),
        "",
        "## Low-confidence review threshold",
        "",
        "Chosen by maximising Youden's J (guide 14.3: never hard-code 0.50/0.70) - "
        "the threshold that best separates the model's own errors from its "
        "correct predictions using calibrated confidence alone.",
        "",
        f"- **Flag for human review when confidence < {threshold:.4f}**",
        f"- Catches {threshold_stats.get('errors_caught_recall', 0):.1%} of validation errors",
        f"- Needlessly flags {threshold_stats.get('correct_needlessly_flagged_rate', 0):.1%} "
        f"of correct predictions",
        f"- {threshold_stats.get('pct_of_validation_flagged', 0):.1%} of validation would be flagged",
        "",
        f"## Reliability diagram: {'figures/reliability_plot.png' if plotted else 'skipped (matplotlib unavailable)'}",
    ])
    write_text(RESULTS / "phase11_calibration.md", report)
    print(f"wrote {RESULTS / 'phase11_calibration.md'}")


if __name__ == "__main__":
    main()
