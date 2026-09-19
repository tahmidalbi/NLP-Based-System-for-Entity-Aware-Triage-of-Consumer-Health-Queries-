# BanglaCare - Phase T10 joint fine-tuning (severity validation)

- Examples: 526
- Temperature: 1.0000  (uncalibrated)

## Headline

- **Macro-F1: 0.9328**
- Accuracy: 0.9297
- Weighted-F1: 0.9302
- **Emergency Recall: 0.9191**
- ECE: 0.0195 (mean confidence 0.9421)

## Per class

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Emergency | 0.9542 | 0.9191 | 0.9363 | 136 |
| Urgent | 0.8471 | 0.9172 | 0.8808 | 145 |
| Routine | 0.9580 | 0.8976 | 0.9268 | 127 |
| General Query | 0.9832 | 0.9915 | 0.9873 | 118 |

## Confusion matrix

Rows = gold, columns = predicted.

| gold \ pred | Emergency | Urgent | Routine | General Query |
|---|---|---|---|---|
| **Emergency** | 125 | 11 | 0 | 0 |
| **Urgent** | 6 | 133 | 5 | 1 |
| **Routine** | 0 | 12 | 114 | 1 |
| **General Query** | 0 | 1 | 0 | 117 |

---

# Phase T10 NER validation (guardrail check)

- Sequences: 3178 (115037 tokens)
- Gold spans: 8436 | predicted spans: 8452
- IOB decoding: conlleval-style (dangling I- opens a span)

## Entity-level (primary)

- **Entity F1 (micro): 0.6239**
- Precision: 0.6233
- Recall: 0.6245
- Macro F1 over types present in gold: 0.6686

## Per entity type

| Entity type | Precision | Recall | F1 | Gold | Pred |
|---|---|---|---|---|---|
| Symptom | 0.4701 | 0.4723 | 0.4712 | 3460 | 3476 |
| Health Condition | 0.5714 | 0.5127 | 0.5405 | 788 | 707 |
| Medicine | 0.8394 | 0.8512 | 0.8453 | 1781 | 1806 |
| Age | 0.7859 | 0.8084 | 0.7970 | 595 | 612 |
| Dosage | 0.5033 | 0.5052 | 0.5043 | 762 | 765 |
| Specialist | 0.8610 | 0.8734 | 0.8671 | 695 | 705 |
| Medical Procedure | 0.6325 | 0.6789 | 0.6549 | 355 | 381 |

## Token level (non-O positions, comparison metric only)

- Precision: 0.7631
- Recall: 0.7440
- F1: 0.7534

## Joint training

- Encoder: csebuetnlp/banglabert
- Best epoch: 2 of 5 run (max 5)
- Learning rates: encoder 1e-05, NER head 0.0005, severity head 0.001
- NER guardrail: T8 baseline 0.6322, floor 0.6222, selected checkpoint 0.6239
- Seed: 42 | wall clock: 8m55s
- Checkpoint: `/kaggle/working/checkpoints/best_joint_transformer.pt`
