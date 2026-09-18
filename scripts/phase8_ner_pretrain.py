"""
Phase 8 - NER pretraining (guide section 11).

Trains the shared projection + BiLSTM + NER emission layer + CRF on HealthNER
TRAIN, so the shared encoder learns medical structure before it ever sees the
much smaller severity dataset. The severity head exists in the model but is
untouched here (guide 11.1: "Severity head - not used yet").

Fixed settings (guide 11.3, not tunable):
  AdamW, lr 1e-3, weight decay 1e-4, batch 32, grad clip 1.0,
  max 20 epochs, early stopping patience 3 on validation entity-level F1,
  mixed precision on GPU, checkpoint on best validation entity-F1.

Loss is the CRF negative log-likelihood (guide 11.2) - the gold IOB sequence
is optimised, not independent per-token accuracy.

Usage
-----
  python scripts/phase8_ner_pretrain.py
  python scripts/phase8_ner_pretrain.py --smoke          # CPU debug, ~1 min
  python scripts/phase8_ner_pretrain.py --seed 52
  python scripts/phase8_ner_pretrain.py --cache /kaggle/input/banglacare-artifacts/feature_cache.npz

Outputs
-------
  checkpoints/best_ner.pt
  results/phase8_ner_metrics.json / .md
  logs/phase8_history.json
"""

import argparse
from pathlib import Path

import torch
from torch.optim import AdamW

from model import BanglaCareModel
from ner_eval import NEREvaluator, format_report
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
    load_featurizer,
    make_amp,
    ner_loader,
    save_checkpoint,
    set_requires_grad,
    set_seed,
    trainable_parameters,
    write_json,
    write_text,
)


def build_model(device):
    """Full model, with the severity head frozen and unused this phase."""
    model = BanglaCareModel().to(device)
    set_requires_grad(model.severity_head, False)
    return model


def run_epoch(model, loader, optimizer, autocast, scaler, device, params, max_norm=1.0):
    """One training pass over HealthNER TRAIN. Returns mean CRF NLL."""
    model.train()
    meter = AverageMeter()

    for batch in loader:
        features = batch["features"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        tags = batch["ner_tags"].to(device, non_blocking=True)
        lengths = batch["lengths"]

        optimizer.zero_grad(set_to_none=True)
        with autocast():
            emissions, _ = model.ner_forward(features, lengths, mask)

        # The CRF partition function is a long chain of logsumexp reductions;
        # running it in fp16 loses too much precision and can produce inf/nan.
        # The encoder still gets the mixed-precision speedup above.
        loss = model.ner_head.loss(emissions.float(), tags, mask)

        scaler.scale(loss).backward()
        clip_and_step(scaler, optimizer, params, max_norm)

        meter.update(loss.item(), features.size(0))

    return meter.avg


@torch.no_grad()
def evaluate(model, loader, device, strict=False):
    """Validation pass -> (mean CRF NLL, entity-level metrics dict)."""
    model.eval()
    meter = AverageMeter()
    evaluator = NEREvaluator(strict=strict)

    for batch in loader:
        features = batch["features"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        tags = batch["ner_tags"].to(device, non_blocking=True)
        lengths = batch["lengths"]

        emissions, _ = model.ner_forward(features, lengths, mask)
        emissions = emissions.float()

        loss = model.ner_head.loss(emissions, tags, mask)
        predictions = model.ner_head.decode(emissions, mask)

        evaluator.add_batch(predictions, batch["ner_tags"], lengths)
        meter.update(loss.item(), features.size(0))

    return meter.avg, evaluator.compute()


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 8 - NER pretraining")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--out", type=Path, default=CHECKPOINTS / "best_ner.pt")
    parser.add_argument("--no-amp", action="store_true", help="disable mixed precision")
    parser.add_argument("--cpu", action="store_true", help="force CPU")
    parser.add_argument(
        "--strict-iob",
        action="store_true",
        help="discard dangling I- tags instead of opening a span (see ner_eval)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="cap examples per split (debugging; --smoke implies 200)",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="tiny subset + 1 epoch, for debugging the loop on CPU before "
             "spending GPU quota",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    limit = args.limit if args.limit is not None else (200 if args.smoke else None)
    epochs = 1 if args.smoke else args.epochs

    set_seed(args.seed)
    device = get_device(prefer_cuda=not args.cpu)
    autocast, scaler, amp_on = make_amp(device, enabled=not args.no_amp)

    print("BanglaCare - Phase 8: NER pretraining")
    print("=" * 60)
    print(f"device: {device} | mixed precision: {amp_on} | seed: {args.seed}")
    if args.smoke:
        print("SMOKE MODE: 200 examples, 1 epoch - numbers are meaningless")

    featurizer = load_featurizer(args.cache)
    train_loader = ner_loader(featurizer, "train", args.batch_size, limit=limit)
    val_loader = ner_loader(featurizer, "valid", args.batch_size, limit=limit)
    print(f"train batches: {len(train_loader)} | val batches: {len(val_loader)}")

    model = build_model(device)
    for name, info in describe_trainable(model).items():
        print(f"  {name:<14} {info['trainable']:>9,} / {info['total']:>9,} trainable")

    params = trainable_parameters(model)
    optimizer = AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    stopper = EarlyStopping(patience=args.patience)

    config = {
        "phase": 8,
        "optimizer": "AdamW",
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "grad_clip": args.clip,
        "max_epochs": epochs,
        "early_stopping_patience": args.patience,
        "mixed_precision": amp_on,
        "seed": args.seed,
        "strict_iob": args.strict_iob,
        "selection_metric": "validation entity-level F1",
        "smoke": args.smoke,
    }

    history = []
    timer = Timer().__enter__()

    for epoch in range(1, epochs + 1):
        train_loss = run_epoch(
            model, train_loader, optimizer, autocast, scaler, device, params, args.clip
        )
        val_loss, metrics = evaluate(model, val_loader, device, strict=args.strict_iob)
        entity_f1 = metrics["entity_f1"]

        improved = stopper.step(entity_f1, epoch)
        flag = "  <- best" if improved else ""
        print(
            f"epoch {epoch:>2}/{epochs}  "
            f"train NLL {train_loss:8.3f}  val NLL {val_loss:8.3f}  "
            f"val entity-F1 {entity_f1:.4f}  "
            f"P {metrics['entity_precision']:.4f}  R {metrics['entity_recall']:.4f}  "
            f"[{human_time(timer.now)}]{flag}"
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_entity_f1": entity_f1,
                "val_entity_precision": metrics["entity_precision"],
                "val_entity_recall": metrics["entity_recall"],
                "val_macro_f1": metrics["macro_f1"],
                "val_per_type_f1": {k: v["f1"] for k, v in metrics["per_type"].items()},
            }
        )

        if improved:
            save_checkpoint(
                args.out, model, optimizer, epoch=epoch, metrics=metrics,
                config=config, seed=args.seed,
            )
            best_metrics = metrics

        if stopper.should_stop:
            print(f"early stopping: no improvement for {args.patience} epochs")
            break

    timer.__exit__()

    print("=" * 60)
    print(
        f"best epoch {stopper.best_epoch} | val entity-F1 {stopper.best:.4f} "
        f"| total {human_time(timer.elapsed)}"
    )
    print(f"checkpoint: {args.out}")

    # Per-type scores are mandatory reporting (guide 11.4) - HealthNER is
    # imbalanced, so a good overall F1 can hide a dead entity type.
    print("\nper-type validation F1 at best epoch:")
    for etype, m in best_metrics["per_type"].items():
        print(f"  {etype:<20} F1 {m['f1']:.4f}  (gold spans: {m['support']})")

    summary = {
        "config": config,
        "best_epoch": stopper.best_epoch,
        "best_val_entity_f1": stopper.best,
        "epochs_run": len(history),
        "train_seconds": timer.elapsed,
        "best_metrics": best_metrics,
        "history": history,
    }
    write_json(RESULTS / "phase8_ner_metrics.json", summary)
    write_json(LOGS / "phase8_history.json", history)

    report = format_report(best_metrics, title="BanglaCare - Phase 8 NER validation")
    report += (
        f"\n\n## Training\n\n"
        f"- Best epoch: {stopper.best_epoch} of {len(history)} run "
        f"(max {epochs}, patience {args.patience})\n"
        f"- Seed: {args.seed}\n"
        f"- Optimizer: AdamW lr={args.lr} weight_decay={args.weight_decay} "
        f"batch={args.batch_size} clip={args.clip}\n"
        f"- Mixed precision: {amp_on}\n"
        f"- Wall clock: {human_time(timer.elapsed)}\n"
        f"- Checkpoint: `{args.out}`\n"
    )
    write_text(RESULTS / "phase8_ner_metrics.md", report)
    print(f"\nwrote {RESULTS / 'phase8_ner_metrics.md'}")


if __name__ == "__main__":
    main()
