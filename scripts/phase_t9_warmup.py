"""
Phase T9 - Severity-head warm-up, transformer variant (TRANSFORMER_PLAN.md 9).

Mirrors phase9_severity_warmup.py: the transformer and NER head are frozen and
only the entity-aware attention + severity MLP train. Plain 4-class CE, no
class weights (classes are 22-27%), 2-3 epochs, selection on validation
Macro-F1.

The frozen encoder AND the model-level dropout that follows it are kept in
eval() so the head is warmed up against the same deterministic features it
will see at validation time (same reasoning as Phase 9).

Usage
-----
  python scripts/phase_t9_warmup.py --smoke --cpu
  python scripts/phase_t9_warmup.py

Outputs
-------
  checkpoints/best_warmup_transformer.pt
  results/transformer_phase9_severity_metrics.json / .md
  logs/transformer_phase9_history.json
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW

from dataset_transformer import DEFAULT_MODEL, WordEncoder, load_tokenizer, make_normalizer, severity_loader
from model_transformer import BanglaCareTransformer
from severity_eval import SeverityEvaluator, format_report
from train_utils import (
    CHECKPOINTS,
    LOGS,
    RESULTS,
    AverageMeter,
    Timer,
    clip_and_step,
    describe_trainable,
    get_device,
    human_time,
    load_checkpoint,
    make_amp,
    save_checkpoint,
    set_requires_grad,
    set_seed,
    trainable_parameters,
    write_json,
    write_text,
)


def build_model(device, model_name, ner_checkpoint):
    """T8 encoder + NER head, frozen; only the severity head trains.

    strict=False: best_ner_transformer.pt was saved while the severity head
    still held its random initialisation, which is intentionally discarded.
    """
    model = BanglaCareTransformer(model_name, pretrained=False).to(device)
    load_checkpoint(ner_checkpoint, model, device=device, strict=False)
    set_requires_grad(model.encoder, False)
    set_requires_grad(model.ner_head, False)
    set_requires_grad(model.severity_head, True)
    return model


def _to_device(batch, device):
    return (
        batch["input_ids"].to(device, non_blocking=True),
        batch["attention_mask"].to(device, non_blocking=True),
        batch["first_pos"].to(device, non_blocking=True),
        batch["word_mask"].to(device, non_blocking=True),
        batch["severity_ids"].to(device, non_blocking=True),
    )


def run_epoch(model, loader, optimizer, criterion, autocast, scaler, device, params, clip):
    model.train()
    model.encoder.eval()   # transformer dropout off
    model.dropout.eval()   # ...and the dropout applied to its output
    model.ner_head.eval()

    meter = AverageMeter()
    for batch in loader:
        input_ids, attention_mask, first_pos, word_mask, targets = _to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with autocast():
            logits, _ = model.severity_forward(input_ids, attention_mask, first_pos, word_mask)
        loss = criterion(logits.float(), targets)

        scaler.scale(loss).backward()
        clip_and_step(scaler, optimizer, params, clip)
        meter.update(loss.item(), input_ids.size(0))
    return meter.avg


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    meter = AverageMeter()
    evaluator = SeverityEvaluator()
    for batch in loader:
        input_ids, attention_mask, first_pos, word_mask, targets = _to_device(batch, device)
        logits, _ = model.severity_forward(input_ids, attention_mask, first_pos, word_mask)
        logits = logits.float()
        evaluator.add_batch(logits, targets)
        meter.update(criterion(logits, targets).item(), input_ids.size(0))
    return meter.avg, evaluator


def parse_args():
    p = argparse.ArgumentParser(description="Phase T9 - severity warm-up (transformer)")
    p.add_argument("--model-name", default=DEFAULT_MODEL)
    p.add_argument("--no-normalize", action="store_true")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ner-checkpoint", type=Path, default=CHECKPOINTS / "best_ner_transformer.pt")
    p.add_argument("--out", type=Path, default=CHECKPOINTS / "best_warmup_transformer.pt")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    limit = args.limit if args.limit is not None else (200 if args.smoke else None)
    epochs = 1 if args.smoke else args.epochs

    set_seed(args.seed)
    device = get_device(prefer_cuda=not args.cpu)
    autocast, scaler, amp_on = make_amp(device, enabled=not args.no_amp)

    print("BanglaCare - Phase T9: severity-head warm-up (transformer)")
    print("=" * 60)
    print(f"device: {device} | mixed precision: {amp_on} | seed: {args.seed}")
    print(f"loading T8 checkpoint from {args.ner_checkpoint}")

    tokenizer = load_tokenizer(args.model_name)
    encoder = WordEncoder(tokenizer, make_normalizer(not args.no_normalize))
    train_loader = severity_loader(encoder, "train", args.batch_size, limit=limit)
    val_loader = severity_loader(encoder, "val", args.batch_size, limit=limit)
    print(f"train batches: {len(train_loader)} | val batches: {len(val_loader)}")

    model = build_model(device, args.model_name, args.ner_checkpoint)
    for name, info in describe_trainable(model).items():
        print(f"  {name:<14} {info['trainable']:>11,} / {info['total']:>11,} trainable")

    params = trainable_parameters(model)
    optimizer = AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()  # no class weights: classes are 22-27%

    config = {
        "phase": "T9", "model_name": args.model_name, "normalize": not args.no_normalize,
        "optimizer": "AdamW", "lr": args.lr, "weight_decay": args.weight_decay,
        "batch_size": args.batch_size, "grad_clip": args.clip, "epochs": epochs,
        "mixed_precision": amp_on, "seed": args.seed,
        "class_weighting": "none (classes are 22-27%)",
        "selection_metric": "validation Macro-F1",
        "frozen": ["encoder", "ner_head"], "smoke": args.smoke,
    }

    history = []
    best_macro, best_epoch, best_metrics = -1.0, None, None
    timer = Timer().__enter__()

    for epoch in range(1, epochs + 1):
        train_loss = run_epoch(model, train_loader, optimizer, criterion, autocast, scaler,
                               device, params, args.clip)
        val_loss, evaluator = evaluate(model, val_loader, criterion, device)
        metrics = evaluator.compute()
        macro = metrics["macro_f1"]

        improved = macro > best_macro
        print(
            f"epoch {epoch}/{epochs}  train CE {train_loss:.4f}  val CE {val_loss:.4f}  "
            f"macro-F1 {macro:.4f}  acc {metrics['accuracy']:.4f}  "
            f"emergency-R {metrics['emergency_recall']:.4f}  "
            f"[{human_time(timer.now)}]{'  <- best' if improved else ''}"
        )
        history.append({
            "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
            "val_macro_f1": macro, "val_accuracy": metrics["accuracy"],
            "val_emergency_recall": metrics["emergency_recall"],
        })
        if improved:
            best_macro, best_epoch, best_metrics = macro, epoch, metrics
            save_checkpoint(args.out, model, optimizer, epoch=epoch, metrics=metrics,
                            config=config, seed=args.seed)

    timer.__exit__()

    print("=" * 60)
    print(f"best epoch {best_epoch} | val macro-F1 {best_macro:.4f} | total {human_time(timer.elapsed)}")
    print(f"checkpoint: {args.out}")
    print("\nper-class validation F1 at best epoch:")
    for name, m in best_metrics["per_class"].items():
        print(f"  {name:<16} F1 {m['f1']:.4f}  R {m['recall']:.4f}  (support {m['support']})")

    write_json(RESULTS / "transformer_phase9_severity_metrics.json", {
        "config": config, "best_epoch": best_epoch, "best_val_macro_f1": best_macro,
        "train_seconds": timer.elapsed, "best_metrics": best_metrics, "history": history,
    })
    write_json(LOGS / "transformer_phase9_history.json", history)

    report = format_report(best_metrics, title="BanglaCare - Phase T9 severity warm-up (validation)")
    report += (
        f"\n\n## Warm-up\n\n- Best epoch: {best_epoch} of {epochs}\n"
        f"- Trainable: entity-aware attention + severity MLP only\n"
        f"- Loss: plain 4-class cross-entropy, no class weighting\n"
        f"- Seed: {args.seed} | wall clock: {human_time(timer.elapsed)}\n"
        f"- Checkpoint: `{args.out}`\n"
    )
    write_text(RESULTS / "transformer_phase9_severity_metrics.md", report)
    print(f"\nwrote {RESULTS / 'transformer_phase9_severity_metrics.md'}")


if __name__ == "__main__":
    main()
