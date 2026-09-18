"""
Phase 10 - Joint multi-task fine-tuning (guide section 13).

The final training stage. Everything is unfrozen and NER and severity batches
alternate 1:1, each supplying only the labels it actually has (guide 13.1):

  HealthNER batch  -> CRF NER loss        -> projection + BiLSTM + NER head
  Severity batch   -> 4-class CE          -> projection + BiLSTM + severity head
                                             (NER head protected by the detach
                                              inside model.severity_forward)

Discriminative learning rates (guide 13.3):
  shared projection + BiLSTM   2e-4
  NER head on NER batches      5e-4
  severity attention + MLP     1e-3

Model selection (guide 13.4): primarily severity validation Macro-F1, but a
checkpoint is REJECTED if NER validation F1 has fallen more than 1 point below
the Phase 8 level. Severity is the final routing objective, yet a model that
achieves it by destroying the entity extractor is not the system this project
is building.

The NER baseline for that guardrail is measured here at epoch 0, before any
joint step. Phase 9 froze the encoder and NER head, so that number is exactly
the Phase 8 checkpoint's validation F1 - no cross-file lookup needed.

An "epoch" is one pass over the SEVERITY train split (the smaller of the two,
~132 batches), with NER batches drawn from a repeating cycle. Guide 13.2: the
encoder already saw all of HealthNER during pretraining, so joint training is
about maintaining NER while adapting the encoder to severity.

Usage
-----
  python scripts/phase10_joint.py --smoke --cpu
  python scripts/phase10_joint.py --cache /kaggle/input/.../feature_cache.npz

Outputs
-------
  checkpoints/best_joint.pt
  results/phase10_joint_metrics.json / .md
  logs/phase10_history.json
"""

import argparse
from itertools import cycle
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW

from model import BanglaCareModel
from ner_eval import NEREvaluator
from ner_eval import format_report as format_ner_report
from severity_eval import SeverityEvaluator
from severity_eval import format_report as format_severity_report
from train_utils import (
    CHECKPOINTS,
    DEFAULT_CACHE,
    LOGS,
    RESULTS,
    AverageMeter,
    EarlyStopping,
    Timer,
    clip_and_step,
    describe_trainable,
    get_device,
    human_time,
    load_checkpoint,
    load_featurizer,
    make_amp,
    ner_loader,
    save_checkpoint,
    set_requires_grad,
    set_seed,
    severity_loader,
    trainable_parameters,
    write_json,
    write_text,
)


def build_model(device, warmup_checkpoint):
    """Phase 9 checkpoint with everything unfrozen (guide 13.1)."""
    model = BanglaCareModel().to(device)
    load_checkpoint(warmup_checkpoint, model, device=device, strict=True)
    set_requires_grad(model, True)
    return model


def build_optimizer(model, lr_shared, lr_ner, lr_severity, weight_decay):
    """Discriminative learning rates per parameter group (guide 13.3)."""
    return AdamW(
        [
            {"params": list(model.encoder.parameters()), "lr": lr_shared},
            {"params": list(model.ner_head.parameters()), "lr": lr_ner},
            {"params": list(model.severity_head.parameters()), "lr": lr_severity},
        ],
        weight_decay=weight_decay,
    )


def ner_step(model, batch, optimizer, autocast, scaler, device, params, clip):
    features = batch["features"].to(device, non_blocking=True)
    mask = batch["mask"].to(device, non_blocking=True)
    tags = batch["ner_tags"].to(device, non_blocking=True)

    optimizer.zero_grad(set_to_none=True)
    with autocast():
        emissions, _ = model.ner_forward(features, batch["lengths"], mask)
    # CRF stays in fp32: its partition function is a long logsumexp chain that
    # degrades badly in fp16 (same reasoning as Phase 8).
    loss = model.ner_head.loss(emissions.float(), tags, mask)

    scaler.scale(loss).backward()
    clip_and_step(scaler, optimizer, params, clip)
    return loss.item(), features.size(0)


def severity_step(model, batch, optimizer, criterion, autocast, scaler, device, params, clip):
    features = batch["features"].to(device, non_blocking=True)
    mask = batch["mask"].to(device, non_blocking=True)
    targets = batch["severity_ids"].to(device, non_blocking=True)

    optimizer.zero_grad(set_to_none=True)
    with autocast():
        logits, _ = model.severity_forward(features, batch["lengths"], mask)
    loss = criterion(logits.float(), targets)

    scaler.scale(loss).backward()
    clip_and_step(scaler, optimizer, params, clip)
    return loss.item(), features.size(0)


@torch.no_grad()
def evaluate_ner(model, loader, device, strict=False):
    model.eval()
    evaluator = NEREvaluator(strict=strict)
    for batch in loader:
        features = batch["features"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        emissions, _ = model.ner_forward(features, batch["lengths"], mask)
        predictions = model.ner_head.decode(emissions.float(), mask)
        evaluator.add_batch(predictions, batch["ner_tags"], batch["lengths"])
    return evaluator.compute()


@torch.no_grad()
def evaluate_severity(model, loader, device):
    model.eval()
    evaluator = SeverityEvaluator()
    for batch in loader:
        features = batch["features"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        logits, _ = model.severity_forward(features, batch["lengths"], mask)
        evaluator.add_batch(logits.float(), batch["severity_ids"])
    return evaluator


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 10 - joint multi-task fine-tuning")
    parser.add_argument("--epochs", type=int, default=15, help="guide 13.3: 10-15")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr-shared", type=float, default=2e-4)
    parser.add_argument("--lr-ner", type=float, default=5e-4)
    parser.add_argument("--lr-severity", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--warmup-checkpoint", type=Path, default=CHECKPOINTS / "best_warmup.pt")
    parser.add_argument("--out", type=Path, default=CHECKPOINTS / "best_joint.pt")
    parser.add_argument(
        "--ner-guardrail",
        type=float,
        default=0.01,
        help="max allowed drop in NER val F1 below the Phase 8 level (guide 13.4)",
    )
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--strict-iob", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    limit = args.limit if args.limit is not None else (200 if args.smoke else None)
    epochs = 1 if args.smoke else args.epochs

    set_seed(args.seed)
    device = get_device(prefer_cuda=not args.cpu)
    autocast, scaler, amp_on = make_amp(device, enabled=not args.no_amp)

    print("BanglaCare - Phase 10: joint multi-task fine-tuning")
    print("=" * 60)
    print(f"device: {device} | mixed precision: {amp_on} | seed: {args.seed}")
    print(f"loading Phase 9 warm-up from {args.warmup_checkpoint}")

    featurizer = load_featurizer(args.cache)
    ner_train = ner_loader(featurizer, "train", args.batch_size, limit=limit)
    ner_val = ner_loader(featurizer, "valid", args.batch_size, limit=limit)
    sev_train = severity_loader(featurizer, "train", args.batch_size, limit=limit)
    sev_val = severity_loader(featurizer, "val", args.batch_size, limit=limit)
    print(f"severity train batches: {len(sev_train)} (defines one joint epoch)")
    print(f"NER train batches available: {len(ner_train)} (drawn from a cycle)")

    model = build_model(device, args.warmup_checkpoint)
    for name, info in describe_trainable(model).items():
        print(f"  {name:<14} {info['trainable']:>9,} / {info['total']:>9,} trainable")

    params = trainable_parameters(model)
    optimizer = build_optimizer(
        model, args.lr_shared, args.lr_ner, args.lr_severity, args.weight_decay
    )
    criterion = nn.CrossEntropyLoss()
    stopper = EarlyStopping(patience=args.patience)

    # Guardrail baseline, measured before any joint step (see module docstring).
    baseline = evaluate_ner(model, ner_val, device, strict=args.strict_iob)["entity_f1"]
    floor = baseline - args.ner_guardrail
    print(f"\nNER guardrail: Phase 8 val entity-F1 = {baseline:.4f}, "
          f"checkpoints rejected below {floor:.4f}")

    config = {
        "phase": 10,
        "optimizer": "AdamW",
        "lr_shared": args.lr_shared,
        "lr_ner_head": args.lr_ner,
        "lr_severity_head": args.lr_severity,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "grad_clip": args.clip,
        "max_epochs": epochs,
        "early_stopping_patience": args.patience,
        "schedule": "1:1 alternating NER / severity batches",
        "mixed_precision": amp_on,
        "seed": args.seed,
        "selection_metric": "severity validation Macro-F1 with NER guardrail",
        "ner_baseline_f1": baseline,
        "ner_floor_f1": floor,
        "smoke": args.smoke,
    }

    history = []
    best_macro = -1.0
    best_epoch = None
    best_severity = None
    best_ner = None
    timer = Timer().__enter__()

    for epoch in range(1, epochs + 1):
        model.train()
        ner_meter, sev_meter = AverageMeter(), AverageMeter()
        ner_stream = cycle(ner_train)

        for sev_batch in sev_train:
            loss, n = ner_step(
                model, next(ner_stream), optimizer, autocast, scaler, device, params, args.clip
            )
            ner_meter.update(loss, n)

            loss, n = severity_step(
                model, sev_batch, optimizer, criterion, autocast, scaler, device, params, args.clip
            )
            sev_meter.update(loss, n)

        ner_metrics = evaluate_ner(model, ner_val, device, strict=args.strict_iob)
        sev_evaluator = evaluate_severity(model, sev_val, device)
        sev_metrics = sev_evaluator.compute()

        macro = sev_metrics["macro_f1"]
        ner_f1 = ner_metrics["entity_f1"]
        allowed = ner_f1 >= floor

        improved = macro > best_macro and allowed
        if improved:
            flag = "  <- best"
        elif macro > best_macro:
            flag = "  (rejected: NER below guardrail)"
        else:
            flag = ""

        print(
            f"epoch {epoch:>2}/{epochs}  NER NLL {ner_meter.avg:7.3f}  CE {sev_meter.avg:.4f}  "
            f"| val macro-F1 {macro:.4f}  emergency-R {sev_metrics['emergency_recall']:.4f}  "
            f"| NER-F1 {ner_f1:.4f}  [{human_time(timer.now)}]{flag}"
        )

        history.append({
            "epoch": epoch,
            "train_ner_loss": ner_meter.avg,
            "train_severity_loss": sev_meter.avg,
            "val_macro_f1": macro,
            "val_accuracy": sev_metrics["accuracy"],
            "val_emergency_recall": sev_metrics["emergency_recall"],
            "val_ner_entity_f1": ner_f1,
            "ner_guardrail_passed": bool(allowed),
            "accepted_as_best": bool(improved),
        })

        if improved:
            best_macro, best_epoch = macro, epoch
            best_severity, best_ner = sev_metrics, ner_metrics
            save_checkpoint(
                args.out, model, optimizer, epoch=epoch,
                metrics={"severity": sev_metrics, "ner": ner_metrics},
                config=config, seed=args.seed,
            )

        # Early stopping tracks severity Macro-F1 regardless of the guardrail,
        # so a run that keeps improving severity only by wrecking NER still
        # terminates instead of burning the full epoch budget.
        stopper.step(macro, epoch)
        if stopper.should_stop:
            print(f"early stopping: no improvement for {args.patience} epochs")
            break

    timer.__exit__()

    if best_epoch is None:
        raise SystemExit(
            "No checkpoint passed the NER guardrail. The severity task is "
            "destroying the entity extractor - lower --lr-shared (guide 20: "
            "'NER F1 collapses during joint training')."
        )

    print("=" * 60)
    print(f"best epoch {best_epoch} | severity macro-F1 {best_macro:.4f} "
          f"| NER F1 {best_ner['entity_f1']:.4f} (baseline {baseline:.4f}) "
          f"| total {human_time(timer.elapsed)}")
    print(f"checkpoint: {args.out}")

    print("\nper-class severity F1 at best epoch:")
    for name, m in best_severity["per_class"].items():
        print(f"  {name:<16} F1 {m['f1']:.4f}  R {m['recall']:.4f}  (support {m['support']})")

    write_json(RESULTS / "phase10_joint_metrics.json", {
        "config": config,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_macro,
        "best_val_ner_f1": best_ner["entity_f1"],
        "ner_baseline_f1": baseline,
        "train_seconds": timer.elapsed,
        "best_severity_metrics": best_severity,
        "best_ner_metrics": best_ner,
        "history": history,
    })
    write_json(LOGS / "phase10_history.json", history)

    report = format_severity_report(
        best_severity, title="BanglaCare - Phase 10 joint fine-tuning (severity validation)"
    )
    report += "\n\n---\n\n" + format_ner_report(
        best_ner, title="Phase 10 NER validation (guardrail check)"
    )
    report += (
        f"\n\n## Joint training\n\n"
        f"- Best epoch: {best_epoch} of {len(history)} run (max {epochs})\n"
        f"- Schedule: 1:1 alternating NER / severity batches\n"
        f"- Learning rates: shared {args.lr_shared}, NER head {args.lr_ner}, "
        f"severity head {args.lr_severity}\n"
        f"- NER guardrail: Phase 8 baseline {baseline:.4f}, floor {floor:.4f}, "
        f"selected checkpoint {best_ner['entity_f1']:.4f}\n"
        f"- Seed: {args.seed} | wall clock: {human_time(timer.elapsed)}\n"
        f"- Checkpoint: `{args.out}`\n"
    )
    write_text(RESULTS / "phase10_joint_metrics.md", report)
    print(f"\nwrote {RESULTS / 'phase10_joint_metrics.md'}")


if __name__ == "__main__":
    main()
