"""
Phase T8 - NER fine-tuning, transformer variant (TRANSFORMER_PLAN.md section 9).

Mirrors phase8_ner_pretrain.py: the encoder + NER head (emission Linear + CRF)
train on HealthNER TRAIN, the severity head is untouched. Differences forced by
the encoder swap: discriminative LRs (encoder 2e-5, heads 1e-3), linear
warm-up/decay schedule, batch 16, 5 epochs, patience 2.

The CRF loss runs in fp32 under AMP (emissions.float()) - same reason as
Phase 8: the partition function degrades to nan in fp16.

Usage
-----
  python scripts/phase_t8_ner.py --smoke --cpu
  python scripts/phase_t8_ner.py

Outputs
-------
  checkpoints/best_ner_transformer.pt
  results/transformer_phase8_ner_metrics.json / .md
  logs/transformer_phase8_history.json
"""

import argparse
from pathlib import Path

import torch
from torch.optim import AdamW

from dataset_transformer import DEFAULT_MODEL, WordEncoder, load_tokenizer, make_normalizer, ner_loader
from model_transformer import BanglaCareTransformer, linear_warmup_schedule, param_groups
from ner_eval import NEREvaluator, format_report
from train_utils import (
    CHECKPOINTS,
    LOGS,
    RESULTS,
    AverageMeter,
    EarlyStopping,
    Timer,
    clip_and_step,
    count_parameters,
    describe_trainable,
    get_device,
    human_time,
    make_amp,
    save_checkpoint,
    set_requires_grad,
    set_seed,
    trainable_parameters,
    write_json,
    write_text,
)


def build_model(device, model_name):
    """Transformer + heads, severity head frozen and unused this phase."""
    model = BanglaCareTransformer(model_name).to(device)
    set_requires_grad(model.severity_head, False)
    return model


def run_epoch(model, loader, optimizer, scheduler, autocast, scaler, device, params, max_norm):
    model.train()
    meter = AverageMeter()
    for batch in loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        first_pos = batch["first_pos"].to(device, non_blocking=True)
        word_mask = batch["word_mask"].to(device, non_blocking=True)
        tags = batch["ner_tags"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast():
            emissions, _ = model.ner_forward(input_ids, attention_mask, first_pos)
        loss = model.ner_head.loss(emissions.float(), tags, word_mask)  # fp32 CRF

        scaler.scale(loss).backward()
        clip_and_step(scaler, optimizer, params, max_norm)
        scheduler.step()
        meter.update(loss.item(), input_ids.size(0))
    return meter.avg


@torch.no_grad()
def evaluate(model, loader, device, strict=False):
    model.eval()
    meter = AverageMeter()
    evaluator = NEREvaluator(strict=strict)
    for batch in loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        first_pos = batch["first_pos"].to(device, non_blocking=True)
        word_mask = batch["word_mask"].to(device, non_blocking=True)
        tags = batch["ner_tags"].to(device, non_blocking=True)

        emissions, _ = model.ner_forward(input_ids, attention_mask, first_pos)
        emissions = emissions.float()
        loss = model.ner_head.loss(emissions, tags, word_mask)
        predictions = model.ner_head.decode(emissions, word_mask)

        evaluator.add_batch(predictions, batch["ner_tags"], batch["lengths"])
        meter.update(loss.item(), input_ids.size(0))
    return meter.avg, evaluator.compute()


def parse_args():
    p = argparse.ArgumentParser(description="Phase T8 - NER fine-tuning (transformer)")
    p.add_argument("--model-name", default=DEFAULT_MODEL)
    p.add_argument("--no-normalize", action="store_true", help="skip csebuetnlp normalizer")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr-encoder", type=float, default=2e-5)
    p.add_argument("--lr-head", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-frac", type=float, default=0.10)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--patience", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, default=CHECKPOINTS / "best_ner_transformer.pt")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--strict-iob", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--smoke", action="store_true", help="200 examples, 1 epoch (CPU debug)")
    return p.parse_args()


def main():
    args = parse_args()
    limit = args.limit if args.limit is not None else (200 if args.smoke else None)
    epochs = 1 if args.smoke else args.epochs

    set_seed(args.seed)
    device = get_device(prefer_cuda=not args.cpu)
    autocast, scaler, amp_on = make_amp(device, enabled=not args.no_amp)

    print("BanglaCare - Phase T8: NER fine-tuning (transformer)")
    print("=" * 60)
    print(f"model: {args.model_name} | device: {device} | mixed precision: {amp_on} | seed: {args.seed}")
    if args.smoke:
        print("SMOKE MODE: 200 examples, 1 epoch - numbers are meaningless")

    tokenizer = load_tokenizer(args.model_name)
    encoder = WordEncoder(tokenizer, make_normalizer(not args.no_normalize))
    train_loader = ner_loader(encoder, "train", args.batch_size, limit=limit)
    val_loader = ner_loader(encoder, "valid", args.batch_size, limit=limit)
    print(f"train batches: {len(train_loader)} | val batches: {len(val_loader)}")

    model = build_model(device, args.model_name)
    for name, info in describe_trainable(model).items():
        print(f"  {name:<14} {info['trainable']:>11,} / {info['total']:>11,} trainable")

    params = trainable_parameters(model)
    optimizer = AdamW(param_groups(
        [(model.encoder, args.lr_encoder), (model.ner_head, args.lr_head)], args.weight_decay
    ))
    scheduler = linear_warmup_schedule(optimizer, epochs * len(train_loader), args.warmup_frac)
    stopper = EarlyStopping(patience=args.patience)

    config = {
        "phase": "T8", "model_name": args.model_name, "normalize": not args.no_normalize,
        "optimizer": "AdamW", "lr_encoder": args.lr_encoder, "lr_head": args.lr_head,
        "weight_decay": args.weight_decay, "warmup_frac": args.warmup_frac,
        "batch_size": args.batch_size, "grad_clip": args.clip, "max_epochs": epochs,
        "early_stopping_patience": args.patience, "mixed_precision": amp_on,
        "seed": args.seed, "strict_iob": args.strict_iob,
        "selection_metric": "validation entity-level F1", "smoke": args.smoke,
        "total_parameters": count_parameters(model),
        "trainable_parameters": count_parameters(model, only_trainable=True),
    }

    history, best_metrics = [], None
    timer = Timer().__enter__()

    for epoch in range(1, epochs + 1):
        train_loss = run_epoch(model, train_loader, optimizer, scheduler, autocast, scaler,
                               device, params, args.clip)
        val_loss, metrics = evaluate(model, val_loader, device, strict=args.strict_iob)
        entity_f1 = metrics["entity_f1"]

        improved = stopper.step(entity_f1, epoch)
        print(
            f"epoch {epoch:>2}/{epochs}  train NLL {train_loss:8.3f}  val NLL {val_loss:8.3f}  "
            f"val entity-F1 {entity_f1:.4f}  P {metrics['entity_precision']:.4f}  "
            f"R {metrics['entity_recall']:.4f}  [{human_time(timer.now)}]"
            f"{'  <- best' if improved else ''}"
        )
        history.append({
            "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
            "val_entity_f1": entity_f1,
            "val_entity_precision": metrics["entity_precision"],
            "val_entity_recall": metrics["entity_recall"],
            "val_macro_f1": metrics["macro_f1"],
            "val_per_type_f1": {k: v["f1"] for k, v in metrics["per_type"].items()},
        })

        if improved:
            # No optimizer state: Adam moments for a 110M-parameter encoder are
            # ~1GB and nothing downstream resumes from them.
            save_checkpoint(args.out, model, None, epoch=epoch, metrics=metrics,
                            config=config, seed=args.seed)
            best_metrics = metrics
        if stopper.should_stop:
            print(f"early stopping: no improvement for {args.patience} epochs")
            break

    timer.__exit__()

    print("=" * 60)
    print(f"best epoch {stopper.best_epoch} | val entity-F1 {stopper.best:.4f} "
          f"| total {human_time(timer.elapsed)}")
    print(f"checkpoint: {args.out}")
    print("\nper-type validation F1 at best epoch:")
    for etype, m in best_metrics["per_type"].items():
        print(f"  {etype:<20} F1 {m['f1']:.4f}  (gold spans: {m['support']})")

    write_json(RESULTS / "transformer_phase8_ner_metrics.json", {
        "config": config, "best_epoch": stopper.best_epoch,
        "best_val_entity_f1": stopper.best, "epochs_run": len(history),
        "train_seconds": timer.elapsed, "best_metrics": best_metrics, "history": history,
    })
    write_json(LOGS / "transformer_phase8_history.json", history)

    report = format_report(best_metrics, title="BanglaCare - Phase T8 NER validation (transformer)")
    report += (
        f"\n\n## Training\n\n"
        f"- Encoder: {args.model_name}\n"
        f"- Best epoch: {stopper.best_epoch} of {len(history)} run "
        f"(max {epochs}, patience {args.patience})\n"
        f"- Seed: {args.seed}\n"
        f"- AdamW: encoder lr={args.lr_encoder}, head lr={args.lr_head}, "
        f"wd={args.weight_decay}, batch={args.batch_size}, clip={args.clip}, "
        f"linear schedule {args.warmup_frac:.0%} warm-up\n"
        f"- Parameters: {config['total_parameters']:,} total\n"
        f"- Mixed precision: {amp_on}\n"
        f"- Wall clock: {human_time(timer.elapsed)}\n"
        f"- Checkpoint: `{args.out}`\n"
    )
    write_text(RESULTS / "transformer_phase8_ner_metrics.md", report)
    print(f"\nwrote {RESULTS / 'transformer_phase8_ner_metrics.md'}")


if __name__ == "__main__":
    main()
