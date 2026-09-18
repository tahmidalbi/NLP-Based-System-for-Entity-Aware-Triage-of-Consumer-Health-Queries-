"""
Phase 12 - Final evaluation (guide section 15).

Runs each test set exactly ONCE with the frozen Phase 10 checkpoint and the
Phase 11 calibration - this script must not be used to iterate on the model.
Predictions are saved to disk (guide 15: "so later tables/figures can be
regenerated without repeatedly modifying the model").

Produces:
  - Severity test metrics (guide 15.1): Macro-F1, accuracy, weighted-F1,
    per-class P/R/F1, Emergency Recall, confusion matrix, ECE - using the
    Phase 11 temperature for the reported confidence.
  - NER test metrics (guide 15.2): entity-level P/R/F1, per-entity F1,
    micro F1, token F1.
  - Structured error analysis (guide 15.3). The guide names eight categories;
    they are not equally automatable, so this script is explicit about which
    is which:
      mechanically exact  - severity confusion pairs, entity type of each NER
                             miss, rare-type (Specialist/Medical Procedure)
                             involvement, medicine/dosage cross-confusion
      heuristic proxy      - long queries (length percentile), multi-symptom
                             (>1 gold Symptom span), code-switching (mixed
                             Bangla/Latin script), negation (guide 5.1's own
                             keyword list)
      needs human eyes      - spelling variants: flagged as "cannot be
                             detected without a dictionary the project does
                             not have"; a stratified error sample is dumped
                             instead of a fabricated automatic verdict.

Usage
-----
  python scripts/phase12_evaluate.py --smoke --cpu
  python scripts/phase12_evaluate.py --cache /kaggle/input/.../feature_cache.npz

Outputs
-------
  results/phase12_severity_test.json / .md
  results/phase12_ner_test.json / .md
  results/phase12_predictions_severity.csv
  results/phase12_predictions_ner.json
  results/phase12_error_analysis.json / .md
  results/phase12_error_sample.csv   (stratified sample for manual review)
"""

import argparse
import csv
import json
from pathlib import Path

import torch

from labels import ENTITY_TYPES, NER_ID2LABEL, SEVERITY_ID2LABEL
from model import BanglaCareModel
from ner_eval import NEREvaluator, extract_spans
from ner_eval import format_report as format_ner_report
from severity_eval import SeverityEvaluator, softmax
from severity_eval import format_report as format_severity_report
from text_utils import BANGLA_RE, LATIN_RE
from train_utils import (
    CHECKPOINTS,
    DEFAULT_CACHE,
    RESULTS,
    ROOT,
    get_device,
    load_checkpoint,
    load_featurizer,
    ner_loader,
    severity_loader,
    write_json,
    write_text,
)

# guide 5.1's own conservative keep-list, reused verbatim here so "negation
# present" means the same thing in Phase 12 that it meant when Phase 3
# deliberately chose not to strip these words out.
NEGATION_WORDS = {"না", "নেই", "হয়নি"}

RARE_NER_TYPES = {"Specialist", "Medical Procedure"}  # smallest gold support, guide 11.4


def is_code_switched(text):
    """True if the text mixes Bangla and Latin script - guide 15.3's
    'Banglish/code-switching' category, defined the same way Phase 4's corpus
    stats already measure script mix (mixed_script_pct)."""
    return bool(BANGLA_RE.search(text)) and bool(LATIN_RE.search(text))


def has_negation(text):
    return any(word in text for word in NEGATION_WORDS)


# ---------------------------------------------------------------- severity


@torch.no_grad()
def run_severity_test(model, loader, texts, device):
    model.eval()
    evaluator = SeverityEvaluator()
    rows = []
    for batch, batch_texts in zip(loader, _chunks(texts, loader.batch_size)):
        features = batch["features"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        targets = batch["severity_ids"]

        logits, _ = model.severity_forward(features, batch["lengths"], mask)
        logits = logits.float().cpu()
        evaluator.add_batch(logits, targets)

        for text, logit_row, gold in zip(batch_texts, logits.numpy(), targets.tolist()):
            rows.append({"text": text, "gold_id": gold, "logits": logit_row.tolist()})

    return evaluator, rows


def _chunks(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def load_severity_texts(split):
    import pandas as pd
    df = pd.read_csv(ROOT / "data" / "processed" / f"severity_{split}.csv", usecols=["Text"])
    return df["Text"].astype(str).tolist()


def write_severity_predictions(path, rows, temperature):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "gold_label", "pred_label", "confidence", "is_error"])
        for row in rows:
            probs = softmax([row["logits"]], temperature)[0]
            pred_id = int(probs.argmax())
            gold_id = row["gold_id"]
            writer.writerow([
                row["text"],
                SEVERITY_ID2LABEL[gold_id],
                SEVERITY_ID2LABEL[pred_id],
                f"{probs[pred_id]:.4f}",
                int(pred_id != gold_id),
            ])


# --------------------------------------------------------------------- NER


@torch.no_grad()
def run_ner_test(model, loader, device, strict=False):
    model.eval()
    evaluator = NEREvaluator(strict=strict)
    rows = []
    for batch in loader:
        features = batch["features"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)

        emissions, _ = model.ner_forward(features, batch["lengths"], mask)
        predictions = model.ner_head.decode(emissions.float(), mask)
        evaluator.add_batch(predictions, batch["ner_tags"], batch["lengths"])

        gold_rows = batch["ner_tags"].tolist()
        for tokens, pred_ids, gold_ids, n in zip(
            batch["tokens"], predictions, gold_rows, batch["lengths"].tolist()
        ):
            gold_labels = [NER_ID2LABEL[i] for i in gold_ids[:n]]
            pred_labels = [NER_ID2LABEL[i] for i in pred_ids]
            rows.append({"tokens": tokens, "gold": gold_labels, "pred": pred_labels})

    return evaluator, rows


def write_ner_predictions(path, rows):
    write_json(path, rows)


# ---------------------------------------------------------------- error analysis


def analyze_severity_errors(rows, temperature, threshold, length_p90):
    """Mechanical + heuristic categorization of severity test errors."""
    confusion_pairs = {}
    categorized = []

    for row in rows:
        probs = softmax([row["logits"]], temperature)[0]
        pred_id = int(probs.argmax())
        gold_id = row["gold_id"]
        confidence = float(probs[pred_id])
        if pred_id == gold_id:
            continue

        text = row["text"]
        pair = f"{SEVERITY_ID2LABEL[gold_id]} -> {SEVERITY_ID2LABEL[pred_id]}"
        confusion_pairs[pair] = confusion_pairs.get(pair, 0) + 1

        n_tokens = len(text.split())
        categorized.append({
            "text": text,
            "gold": SEVERITY_ID2LABEL[gold_id],
            "pred": SEVERITY_ID2LABEL[pred_id],
            "confidence": confidence,
            "would_be_flagged_for_review": confidence < threshold,
            "long_query": n_tokens > length_p90,
            "code_switched": is_code_switched(text),
            "contains_negation": has_negation(text),
            "n_tokens": n_tokens,
        })

    return categorized, confusion_pairs


def analyze_ner_errors(rows):
    """Mechanical categorization of NER test errors, span by span.

    Every gold span not exactly matched is one of:
      missed          - no predicted span of any type overlaps it at all
      boundary_error   - a predicted span of the SAME type overlaps it, but
                          the exact boundaries differ
      type_confusion   - a predicted span of a DIFFERENT type overlaps it
                          (with special tracking for Medicine<->Dosage, guide
                          15.3's named "medicine/dosage confusion")
    Every predicted span not matched to any gold span is a spurious detection.
    """
    per_type_errors = {t: {"missed": 0, "boundary_error": 0, "type_confusion": 0} for t in ENTITY_TYPES}
    medicine_dosage_confusions = 0
    rare_type_errors = 0
    spurious = 0
    examples = []

    for row in rows:
        gold_spans = extract_spans(row["gold"])
        pred_spans = extract_spans(row["pred"])
        matched_gold = gold_spans & pred_spans
        unmatched_gold = gold_spans - matched_gold
        unmatched_pred = pred_spans - matched_gold
        consumed_pred = set()  # pred spans that explain some gold miss, by overlap

        for etype, start, end in unmatched_gold:
            gold_range = set(range(start, end))
            overlap = None
            for pspan in unmatched_pred:
                petype, pstart, pend = pspan
                if gold_range & set(range(pstart, pend)):
                    overlap = petype
                    consumed_pred.add(pspan)
                    break

            if overlap is None:
                per_type_errors[etype]["missed"] += 1
                kind = "missed"
            elif overlap == etype:
                per_type_errors[etype]["boundary_error"] += 1
                kind = "boundary_error"
            else:
                per_type_errors[etype]["type_confusion"] += 1
                kind = "type_confusion"
                if {overlap, etype} == {"Medicine", "Dosage"}:
                    medicine_dosage_confusions += 1

            if etype in RARE_NER_TYPES:
                rare_type_errors += 1

            if len(examples) < 200:
                examples.append({
                    "tokens": " ".join(row["tokens"]),
                    "gold_type": etype,
                    "error_kind": kind,
                    "confused_with": overlap,
                })

        # A predicted span that overlaps no gold span whatsoever - a pure
        # hallucination, distinct from a type/boundary confusion already
        # counted above via the gold side.
        spurious += len(unmatched_pred - consumed_pred)

    return {
        "per_type_errors": per_type_errors,
        "medicine_dosage_confusions": medicine_dosage_confusions,
        "rare_type_errors": rare_type_errors,
        "spurious_predicted_spans": spurious,
        "examples": examples,
    }


def write_error_sample(path, severity_categorized, ner_examples, sample_size=60):
    """Stratified sample for manual review (guide 15.3: 'manually inspect a
    structured sample'). Spelling variants specifically cannot be verified
    without a dictionary this project does not build, so they are left for
    this human pass rather than guessed at automatically.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["source", "text_or_tokens", "detail", "manual_category"])

        per_bucket = max(1, sample_size // 6)
        buckets = [
            ("long_query", [r for r in severity_categorized if r["long_query"]]),
            ("code_switched", [r for r in severity_categorized if r["code_switched"]]),
            ("contains_negation", [r for r in severity_categorized if r["contains_negation"]]),
            ("low_confidence", [r for r in severity_categorized if r["would_be_flagged_for_review"]]),
            ("other_severity_error", severity_categorized),
        ]
        for name, items in buckets:
            for r in items[:per_bucket]:
                writer.writerow(["severity", r["text"], f"{r['gold']} -> {r['pred']}", ""])

        for ex in ner_examples[:per_bucket]:
            writer.writerow(["ner", ex["tokens"], f"{ex['gold_type']} ({ex['error_kind']})", ""])


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 12 - final evaluation")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--joint-checkpoint", type=Path, default=CHECKPOINTS / "best_joint.pt")
    parser.add_argument("--calibration", type=Path, default=CHECKPOINTS / "calibration.json")
    parser.add_argument("--strict-iob", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    limit = args.limit if args.limit is not None else (200 if args.smoke else None)

    device = get_device(prefer_cuda=not args.cpu)
    print("BanglaCare - Phase 12: final evaluation")
    print("=" * 60)
    print(f"device: {device}")
    print(f"loading checkpoint {args.joint_checkpoint}")
    print("TEST SETS - run once. Do not iterate against these numbers.")

    if not args.calibration.exists():
        raise SystemExit(f"Missing {args.calibration} - run scripts/phase11_calibrate.py first.")
    calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
    temperature = calibration["temperature"]
    threshold = calibration["review_threshold"]
    print(f"calibration: T={temperature:.4f}, review threshold={threshold:.4f}")

    featurizer = load_featurizer(args.cache)
    model = BanglaCareModel().to(device)
    load_checkpoint(args.joint_checkpoint, model, device=device, strict=True)

    # ---------------------------------------------------------- severity
    sev_loader = severity_loader(featurizer, "test", args.batch_size, shuffle=False, limit=limit)
    sev_texts = load_severity_texts("test")[:limit] if limit else load_severity_texts("test")
    sev_evaluator, sev_rows = run_severity_test(model, sev_loader, sev_texts, device)
    sev_metrics = sev_evaluator.compute(temperature=temperature)

    print(f"\nseverity test macro-F1: {sev_metrics['macro_f1']:.4f}  "
          f"accuracy: {sev_metrics['accuracy']:.4f}  "
          f"emergency-recall: {sev_metrics['emergency_recall']:.4f}")

    write_json(RESULTS / "phase12_severity_test.json", sev_metrics)
    write_text(
        RESULTS / "phase12_severity_test.md",
        format_severity_report(sev_metrics, title="BanglaCare - Phase 12 severity TEST results"),
    )
    write_severity_predictions(RESULTS / "phase12_predictions_severity.csv", sev_rows, temperature)

    # -------------------------------------------------------------- NER
    ner_test_loader = ner_loader(featurizer, "test", args.batch_size, shuffle=False, limit=limit)
    ner_evaluator, ner_rows = run_ner_test(model, ner_test_loader, device, strict=args.strict_iob)
    ner_metrics = ner_evaluator.compute()

    print(f"NER test entity-F1: {ner_metrics['entity_f1']:.4f}  "
          f"P: {ner_metrics['entity_precision']:.4f}  R: {ner_metrics['entity_recall']:.4f}")

    write_json(RESULTS / "phase12_ner_test.json", ner_metrics)
    write_text(
        RESULTS / "phase12_ner_test.md",
        format_ner_report(ner_metrics, title="BanglaCare - Phase 12 NER TEST results"),
    )
    write_ner_predictions(RESULTS / "phase12_predictions_ner.json", ner_rows)

    # ---------------------------------------------------------- error analysis
    lengths = sorted(len(t.split()) for t in sev_texts)
    length_p90 = lengths[int(0.9 * (len(lengths) - 1))] if lengths else 0

    sev_categorized, confusion_pairs = analyze_severity_errors(
        sev_rows, temperature, threshold, length_p90
    )
    ner_error_analysis = analyze_ner_errors(ner_rows)

    error_analysis = {
        "severity": {
            "total_errors": len(sev_categorized),
            "confusion_pairs": confusion_pairs,
            "long_query_errors": sum(1 for r in sev_categorized if r["long_query"]),
            "code_switched_errors": sum(1 for r in sev_categorized if r["code_switched"]),
            "negation_present_errors": sum(1 for r in sev_categorized if r["contains_negation"]),
            "would_have_been_flagged_for_review": sum(
                1 for r in sev_categorized if r["would_be_flagged_for_review"]
            ),
            "length_p90_tokens": length_p90,
        },
        "ner": ner_error_analysis,
        "note_on_spelling_variants": (
            "Not automatically categorized: detecting a genuine spelling variant "
            "(vs. a different word) needs a reference dictionary or edit-distance "
            "index this project does not build. See phase12_error_sample.csv for "
            "manual review instead (guide 15.3)."
        ),
    }
    write_json(RESULTS / "phase12_error_analysis.json", error_analysis)
    write_error_sample(
        RESULTS / "phase12_error_sample.csv", sev_categorized, ner_error_analysis["examples"]
    )

    top_confusions = sorted(confusion_pairs.items(), key=lambda kv: -kv[1])[:5]
    report = "\n".join([
        "# BanglaCare - Phase 12 error analysis",
        "",
        f"- Severity test errors: {len(sev_categorized)} / {sev_metrics['n']}",
        f"- NER test: {ner_error_analysis['spurious_predicted_spans']} spurious spans, "
        f"{sum(v['missed'] for v in ner_error_analysis['per_type_errors'].values())} missed spans, "
        f"{sum(v['boundary_error'] for v in ner_error_analysis['per_type_errors'].values())} boundary errors",
        "",
        "## Severity confusion pairs (gold -> pred)",
        "",
        "| Confusion | Count |",
        "|---|---|",
        *[f"| {pair} | {count} |" for pair, count in top_confusions],
        "",
        "## Severity error categories (heuristic)",
        "",
        f"- Long queries (>{length_p90} tokens, 90th pct of test): "
        f"{error_analysis['severity']['long_query_errors']}",
        f"- Code-switched (Bangla+Latin mixed): {error_analysis['severity']['code_switched_errors']}",
        f"- Contains negation ({'/'.join(NEGATION_WORDS)}): "
        f"{error_analysis['severity']['negation_present_errors']}",
        f"- Would have been flagged for human review (confidence < {threshold:.4f}): "
        f"{error_analysis['severity']['would_have_been_flagged_for_review']}",
        "",
        "## NER errors by entity type",
        "",
        "| Type | Missed | Boundary error | Type confusion |",
        "|---|---|---|---|",
        *[
            f"| {t} | {v['missed']} | {v['boundary_error']} | {v['type_confusion']} |"
            for t, v in ner_error_analysis["per_type_errors"].items()
        ],
        "",
        f"- Medicine <-> Dosage confusions specifically: "
        f"{ner_error_analysis['medicine_dosage_confusions']}",
        f"- Errors involving rare types (Specialist, Medical Procedure): "
        f"{ner_error_analysis['rare_type_errors']}",
        "",
        "## Not automated",
        "",
        error_analysis["note_on_spelling_variants"],
        "",
        "A stratified sample for manual review is in "
        "`results/phase12_error_sample.csv` (long queries, code-switching, "
        "negation, low-confidence, and NER misses by type).",
    ])
    write_text(RESULTS / "phase12_error_analysis.md", report)

    print("\n" + "=" * 60)
    print("Phase 12 complete. Written to results/:")
    for name in [
        "phase12_severity_test.md", "phase12_ner_test.md", "phase12_error_analysis.md",
        "phase12_predictions_severity.csv", "phase12_predictions_ner.json",
        "phase12_error_sample.csv",
    ]:
        print(f"  {name}")


if __name__ == "__main__":
    main()
