"""
Phase T10 - Joint multi-task fine-tuning, transformer variant
(TRANSFORMER_PLAN.md section 9).

Mirrors phase10_joint.py: everything unfrozen, NER and severity batches
alternate 1:1. Discriminative LRs: encoder 1e-5 (half the T8 rate), NER head
5e-4, severity head 1e-3. A linear warm-up/decay schedule spans all optimizer
steps (two per iteration: one NER step, one severity step).

Selection: severity validation Macro-F1, but a checkpoint is REJECTED if NER
validation F1 falls more than 1 point below the T8 level. The baseline is
measured here at epoch 0 (before any joint step), not read from a file.

An "epoch" is one pass over the SEVERITY train split; NER batches come from a
repeating cycle.

Usage
-----
  python scripts/phase_t10_joint.py --smoke --cpu
  python scripts/phase_t10_joint.py

Outputs
-------
  checkpoints/best_joint_transformer.pt
  results/transformer_phase10_joint_metrics.json / .md
  logs/transformer_phase10_history.json
"""

import argparse
from itertools import cycle
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW

from dataset_transformer import (
    DEFAULT_MODEL,
    WordEncoder,
    load_tokenizer,
    make_normalizer,
    ner_loader,
    severity_loader,
)
from model_transformer import BanglaCareTransformer, linear_warmup_schedule, param_groups
from ner_eval import NEREvaluator
from ner_eval import format_report as format_ner_report
from severity_eval import SeverityEvaluator
from severity_eval import format_report as format_severity_report
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
    load_checkpoint,
    make_amp,
    save_checkpoint,
    set_requires_grad,
    set_seed,
    trainable_parameters,
    write_json,
    write_text,
)


def build_model(device, model_name, warmup_checkpoint):
    model = BanglaCareTransformer(model_name, pretrained=False).to(device)
    load_checkpoint(warmup_checkpoint, model, device=device, strict=True)
    set_requires_grad(model, True)
    return model


def _ner_inputs(batch, device):
    return (
        batch["input_ids"].to(device, non_blocking=True),
        batch["attention_mask"].to(device, non_blocking=True),
        batch["first_pos"].to(device, non_blocking=True),
        batch["word_mask"].to(device, non_blocking=True),
    )


def ner_step(model, batch, optimizer, scheduler, autocast, scaler, device, params, clip):
    input_ids, attention_mask, first_pos, word_mask = _ner_inputs(batch, device)
    tags = batch["ner_tags"].to(device, non_blocking=True)

    optimizer.zero_grad(set_to_none=True)
    with autocast():
        emissions, _ = model.ner_forward(input_ids, attention_mask, first_pos)
    loss = model.ner_head.loss(emissions.float(), tags, word_mask)  # fp32 CRF

    scaler.scale(loss).backward()
    clip_and_step(scaler, optimizer, params, clip)
    scheduler.step()
    return loss.item(), input_ids.size(0)


def severity_step(model, batch, optimizer, scheduler, criterion, autocast, scaler, device,
                  params, clip):
    input_ids, attention_mask, first_pos, word_mask = _ner_inputs(batch, device)
    targets = batch["severity_ids"].to(device, non_blocking=True)

    optimizer.zero_grad(set_to_none=True)
    with autocast():
        logits, _ = model.severity_forward(input_ids, attention_mask, first_pos, word_mask)
    loss = criterion(logits.float(), targets)

    scaler.scale(loss).backward()
    clip_and_step(scaler, optimizer, params, clip)
    scheduler.step()
    return loss.item(), input_ids.size(0)


@torch.no_grad()
def evaluate_ner(model, loader, device, strict=False):
    model.eval()
    evaluator = NEREvaluator(strict=strict)
    for batch in loader:
        input_ids, attention_mask, first_pos, word_mask = _ner_inputs(batch, device)
        emissions, _ = model.ner_forward(input_ids, attention_mask, first_pos)
        predictions = model.ner_head.decode(emissions.float(), word_mask)
        evaluator.add_batch(predictions, batch["ner_tags"], batch["lengths"])
    return evaluator.compute()


@torch.no_grad()
def evaluate_severity(model, loader, device):
    model.eval()
    evaluator = SeverityEvaluator()
    for batch in loader:
        input_ids, attention_mask, first_pos, word_mask = _ner_inputs(batch, device)
        logits, _ = model.severity_forward(input_ids, attention_mask, first_pos, word_mask)
        evaluator.add_batch(logits.float(), batch["severity_ids"])
    return evaluator


def parse_args():
    p = argparse.ArgumentParser(description="Phase T10 - joint fine-tuning (transformer)")
    p.add_argument("--model-name", default=DEFAULT_MODEL)
    p.add_argument("--no-normalize", action="store_true")
    p.add_argument("--epochs", type=int, default=5, help="plan: 3-5")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr-encoder", type=float, default=1e-5)
    p.add_argument("--lr-ner", type=float, default=5e-4)
    p.add_argument("--lr-severity", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-frac", type=float, default=0.10)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--patience", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--warmup-checkpoint", type=Path, default=CHECKPOINTS / "best_warmup_transformer.pt")
    p.add_argument("--out", type=Path, default=CHECKPOINTS / "best_joint_transformer.pt")
    p.add_argument("--ner-guardrail", type=float, default=0.01,
                   help="max allowed drop in NER val F1 below the T8 level")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--strict-iob", action="store_true")
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

    print("BanglaCare - Phase T10: joint multi-task fine-tuning (transformer)")
    print("=" * 60)
    print(f"device: {device} | mixed precision: {amp_on} | seed: {args.seed}")
    print(f"loading T9 warm-up from {args.warmup_checkpoint}")

    tokenizer = load_tokenizer(args.model_name)
    encoder = WordEncoder(tokenizer, make_normalizer(not args.no_normalize))
    ner_train = ner_loader(encoder, "train", args.batch_size, limit=limit)
    ner_val = ner_loader(encoder, "valid", args.batch_size, limit=limit)
    sev_train = severity_loader(encoder, "train", args.batch_size, limit=limit)
    sev_val = severity_loader(encoder, "val", args.batch_size, limit=limit)
    print(f"severity train batches: {len(sev_train)} (defines one joint epoch)")
    print(f"NER train batches available: {len(ner_train)} (drawn from a cycle)")

    model = build_model(device, args.model_name, args.warmup_checkpoint)
    for name, info in describe_trainable(model).items():
        print(f"  {name:<14} {info['trainable']:>11,} / {info['total']:>11,} trainable")

    params = trainable_parameters(model)
    optimizer = AdamW(param_groups(
        [(model.encoder, args.lr_encoder), (model.ner_head, args.lr_ner),
         (model.severity_head, args.lr_severity)],
        args.weight_decay,
    ))
    scheduler = linear_warmup_schedule(optimizer, epochs * len(sev_train) * 2, args.warmup_frac)
    criterion = nn.CrossEntropyLoss()
    stopper = EarlyStopping(patience=args.patience)

    baseline = evaluate_ner(model, ner_val, device, strict=args.strict_iob)["entity_f1"]
    floor = baseline - args.ner_guardrail
    print(f"\nNER guardrail: T8 val entity-F1 = {baseline:.4f}, "
          f"checkpoints rejected below {floor:.4f}")

    config = {
        "phase": "T10", "model_name": args.model_name, "normalize": not args.no_normalize,
        "optimizer": "AdamW", "lr_encoder": args.lr_encoder, "lr_ner_head": args.lr_ner,
        "lr_severity_head": args.lr_severity, "weight_decay": args.weight_decay,
        "warmup_frac": args.warmup_frac, "batch_size": args.batch_size,
        "grad_clip": args.clip, "max_epochs": epochs,
        "early_stopping_patience": args.patience,
        "schedule": "1:1 alternating NER / severity batches",
        "mixed_precision": amp_on, "seed": args.seed,
        "selection_metric": "severity validation Macro-F1 with NER guardrail",
        "ner_baseline_f1": baseline, "ner_floor_f1": floor, "smoke": args.smoke,
        "total_parameters": count_parameters(model),
    }

    history = []
    best_macro, best_epoch, best_severity, best_ner = -1.0, None, None, None
    timer = Timer().__enter__()

    for epoch in range(1, epochs + 1):
        model.train()
        ner_meter, sev_meter = AverageMeter(), AverageMeter()
        ner_stream = cycle(ner_train)

        for sev_batch in sev_train:
            loss, n = ner_step(model, next(ner_stream), optimizer, scheduler, autocast, scaler,
                               device, params, args.clip)
            ner_meter.update(loss, n)
            loss, n = severity_step(model, sev_batch, optimizer, scheduler, criterion, autocast,
                                    scaler, device, params, args.clip)
            sev_meter.update(loss, n)

        ner_metrics = evaluate_ner(model, ner_val, device, strict=args.strict_iob)
        sev_metrics = evaluate_severity(model, sev_val, device).compute()

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
            "epoch": epoch, "train_ner_loss": ner_meter.avg, "train_severity_loss": sev_meter.avg,
            "val_macro_f1": macro, "val_accuracy": sev_metrics["accuracy"],
            "val_emergency_recall": sev_metrics["emergency_recall"],
            "val_ner_entity_f1": ner_f1, "ner_guardrail_passed": bool(allowed),
            "accepted_as_best": bool(improved),
        })

        if improved:
            best_macro, best_epoch = macro, epoch
            best_severity, best_ner = sev_metrics, ner_metrics
            save_checkpoint(args.out, model, None, epoch=epoch,
                            metrics={"severity": sev_metrics, "ner": ner_metrics},
                            config=config, seed=args.seed)

        stopper.step(macro, epoch)
        if stopper.should_stop:
            print(f"early stopping: no improvement for {args.patience} epochs")
            break

    timer.__exit__()

    if best_epoch is None:
        raise SystemExit(
            "No checkpoint passed the NER guardrail. Severity training is damaging "
            "the entity extractor - lower --lr-encoder."
        )

    print("=" * 60)
    print(f"best epoch {best_epoch} | severity macro-F1 {best_macro:.4f} "
          f"| NER F1 {best_ner['entity_f1']:.4f} (baseline {baseline:.4f}) "
          f"| total {human_time(timer.elapsed)}")
    print(f"checkpoint: {args.out}")
    print("\nper-class severity F1 at best epoch:")
    for name, m in best_severity["per_class"].items():
        print(f"  {name:<16} F1 {m['f1']:.4f}  R {m['recall']:.4f}  (support {m['support']})")

    write_json(RESULTS / "transformer_phase10_joint_metrics.json", {
        "config": config, "best_epoch": best_epoch, "best_val_macro_f1": best_macro,
        "best_val_ner_f1": best_ner["entity_f1"], "ner_baseline_f1": baseline,
        "train_seconds": timer.elapsed, "best_severity_metrics": best_severity,
        "best_ner_metrics": best_ner, "history": history,
    })
    write_json(LOGS / "transformer_phase10_history.json", history)

    report = format_severity_report(
        best_severity, title="BanglaCare - Phase T10 joint fine-tuning (severity validation)")
    report += "\n\n---\n\n" + format_ner_report(
        best_ner, title="Phase T10 NER validation (guardrail check)")
    report += (
        f"\n\n## Joint training\n\n"
        f"- Encoder: {args.model_name}\n"
        f"- Best epoch: {best_epoch} of {len(history)} run (max {epochs})\n"
        f"- Learning rates: encoder {args.lr_encoder}, NER head {args.lr_ner}, "
        f"severity head {args.lr_severity}\n"
        f"- NER guardrail: T8 baseline {baseline:.4f}, floor {floor:.4f}, "
        f"selected checkpoint {best_ner['entity_f1']:.4f}\n"
        f"- Seed: {args.seed} | wall clock: {human_time(timer.elapsed)}\n"
        f"- Checkpoint: `{args.out}`\n"
    )
    write_text(RESULTS / "transformer_phase10_joint_metrics.md", report)
    print(f"\nwrote {RESULTS / 'transformer_phase10_joint_metrics.md'}")


if __name__ == "__main__":
    main()
