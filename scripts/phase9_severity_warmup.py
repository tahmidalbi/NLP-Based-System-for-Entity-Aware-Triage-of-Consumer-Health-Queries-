"""
Phase 9 - Severity-head warm-up (guide section 12).

Attaches the entity-aware attention + max-pooling + severity MLP to the
encoder pretrained in Phase 8, and lets ONLY the new severity head learn for a
short warm-up. Everything below it stays frozen (guide 12.1):

  FastText        fixed          Entity-aware attention   TRAIN
  Projection      freeze         Severity MLP             TRAIN
  BiLSTM          freeze
  NER emission    freeze
  CRF             freeze

Why: a randomly initialised severity head produces large, noisy gradients in
its first steps. Sending those straight into the medical encoder would undo
Phase 8's work. The warm-up lets the head reach a sane starting point first,
so Phase 10's joint training begins from a stable place.

Loss is plain 4-class cross-entropy with no class weights, SMOTE or resampling
(guide 1.3 / 12.2): the four classes are already 22%-27% of the data.

Fixed settings (guide 12.3): AdamW, lr 1e-3, batch 32, 2-3 epochs,
selection on validation Macro-F1.

Usage
-----
  python scripts/phase9_severity_warmup.py --smoke --cpu
  python scripts/phase9_severity_warmup.py --cache /kaggle/input/.../feature_cache.npz

Outputs
-------
  checkpoints/best_warmup.pt
  results/phase9_severity_metrics.json / .md
  logs/phase9_history.json
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW

from model import BanglaCareModel
from severity_eval import SeverityEvaluator, format_report
from train_utils import (
    CHECKPOINTS,
    DEFAULT_CACHE,
    LOGS,
    RESULTS,
    AverageMeter,
    Timer,
    clip_and_step,
    describe_trainable,
    get_device,
    human_time,
    load_checkpoint,
    load_featurizer,
    make_amp,
    save_checkpoint,
    set_requires_grad,
    set_seed,
    severity_loader,
    trainable_parameters,
    write_json,
    write_text,
)


def build_model(device, ner_checkpoint):
    """Phase 8 encoder + NER head, frozen; only the severity head trains.

    strict=False on the load because best_ner.pt was written while the severity
    head still held its random initialisation - that part of the state dict is
    intentionally discarded here.
    """
    model = BanglaCareModel().to(device)
    load_checkpoint(ner_checkpoint, model, device=device, strict=False)

    set_requires_grad(model.encoder, False)
    set_requires_grad(model.ner_head, False)
    set_requires_grad(model.severity_head, True)
    return model


def run_epoch(model, loader, optimizer, criterion, autocast, scaler, device, params, clip):
    """One warm-up pass. The encoder is frozen but still runs forward."""
    model.train()
    # The frozen BiLSTM/projection must stay in eval mode so their dropout is
    # off - otherwise the severity head is warmed up against a noisy encoder
    # that behaves differently at validation time.
    model.encoder.eval()
    model.ner_head.eval()

    meter = AverageMeter()
    for batch in loader:
        features = batch["features"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        targets = batch["severity_ids"].to(device, non_blocking=True)
        lengths = batch["lengths"]

        optimizer.zero_grad(set_to_none=True)
        with autocast():
            logits, _ = model.severity_forward(features, lengths, mask)
        loss = criterion(logits.float(), targets)

        scaler.scale(loss).backward()
        clip_and_step(scaler, optimizer, params, clip)
        meter.update(loss.item(), features.size(0))

    return meter.avg


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    meter = AverageMeter()
    evaluator = SeverityEvaluator()

    for batch in loader:
        features = batch["features"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        targets = batch["severity_ids"].to(device, non_blocking=True)
        lengths = batch["lengths"]

        logits, _ = model.severity_forward(features, lengths, mask)
        logits = logits.float()
        loss = criterion(logits, targets)

        evaluator.add_batch(logits, targets)
        meter.update(loss.item(), features.size(0))

    return meter.avg, evaluator


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 9 - severity head warm-up")
    parser.add_argument("--epochs", type=int, default=3, help="guide 12.3 says 2-3")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--ner-checkpoint", type=Path, default=CHECKPOINTS / "best_ner.pt")
    parser.add_argument("--out", type=Path, default=CHECKPOINTS / "best_warmup.pt")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--cpu", action="store_true")
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

    print("BanglaCare - Phase 9: severity-head warm-up")
    print("=" * 60)
    print(f"device: {device} | mixed precision: {amp_on} | seed: {args.seed}")
    print(f"loading Phase 8 encoder from {args.ner_checkpoint}")

    featurizer = load_featurizer(args.cache)
    train_loader = severity_loader(featurizer, "train", args.batch_size, limit=limit)
    val_loader = severity_loader(featurizer, "val", args.batch_size, limit=limit)
    print(f"train batches: {len(train_loader)} | val batches: {len(val_loader)}")

    model = build_model(device, args.ner_checkpoint)
    for name, info in describe_trainable(model).items():
        print(f"  {name:<14} {info['trainable']:>9,} / {info['total']:>9,} trainable")

    params = trainable_parameters(model)
    optimizer = AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    # No class weights: all four classes sit at 22%-27% (guide 1.3, 12.2).
    criterion = nn.CrossEntropyLoss()

    config = {
        "phase": 9,
        "optimizer": "AdamW",
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "grad_clip": args.clip,
        "epochs": epochs,
        "mixed_precision": amp_on,
        "seed": args.seed,
        "class_weighting": "none (classes are 22-27%)",
        "selection_metric": "validation Macro-F1",
        "frozen": ["encoder", "ner_head"],
        "smoke": args.smoke,
    }

    history = []
    best_macro = -1.0
    best_epoch = None
    best_metrics = None
    timer = Timer().__enter__()

    for epoch in range(1, epochs + 1):
        train_loss = run_epoch(
            model, train_loader, optimizer, criterion, autocast, scaler,
            device, params, args.clip,
        )
        val_loss, evaluator = evaluate(model, val_loader, criterion, device)
        metrics = evaluator.compute()
        macro = metrics["macro_f1"]

        improved = macro > best_macro
        flag = "  <- best" if improved else ""
        print(
            f"epoch {epoch}/{epochs}  train CE {train_loss:.4f}  val CE {val_loss:.4f}  "
            f"macro-F1 {macro:.4f}  acc {metrics['accuracy']:.4f}  "
            f"emergency-R {metrics['emergency_recall']:.4f}  "
            f"[{human_time(timer.now)}]{flag}"
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_macro_f1": macro,
            "val_accuracy": metrics["accuracy"],
            "val_emergency_recall": metrics["emergency_recall"],
        })

        if improved:
            best_macro, best_epoch, best_metrics = macro, epoch, metrics
            save_checkpoint(
                args.out, model, optimizer, epoch=epoch, metrics=metrics,
                config=config, seed=args.seed,
            )

    timer.__exit__()

    print("=" * 60)
    print(f"best epoch {best_epoch} | val macro-F1 {best_macro:.4f} "
          f"| total {human_time(timer.elapsed)}")
    print(f"checkpoint: {args.out}")
    print("\nper-class validation F1 at best epoch:")
    for name, m in best_metrics["per_class"].items():
        print(f"  {name:<16} F1 {m['f1']:.4f}  R {m['recall']:.4f}  (support {m['support']})")

    write_json(RESULTS / "phase9_severity_metrics.json", {
        "config": config,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_macro,
        "train_seconds": timer.elapsed,
        "best_metrics": best_metrics,
        "history": history,
    })
    write_json(LOGS / "phase9_history.json", history)

    report = format_report(best_metrics, title="BanglaCare - Phase 9 severity warm-up (validation)")
    report += (
        f"\n\n## Warm-up\n\n"
        f"- Best epoch: {best_epoch} of {epochs}\n"
        f"- Trainable: entity-aware attention + severity MLP only "
        f"(encoder and NER head frozen, guide 12.1)\n"
        f"- Loss: plain 4-class cross-entropy, no class weighting\n"
        f"- Seed: {args.seed} | wall clock: {human_time(timer.elapsed)}\n"
        f"- Checkpoint: `{args.out}`\n"
    )
    write_text(RESULTS / "phase9_severity_metrics.md", report)
    print(f"\nwrote {RESULTS / 'phase9_severity_metrics.md'}")


if __name__ == "__main__":
    main()
