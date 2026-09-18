# BanglaCare - Phase 10 joint fine-tuning (severity validation)

- Examples: 526
- Temperature: 1.0000  (uncalibrated)

## Headline

- **Macro-F1: 0.9182**
- Accuracy: 0.9144
- Weighted-F1: 0.9153
- **Emergency Recall: 0.9044**
- ECE: 0.0491 (mean confidence 0.9496)

## Per class

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Emergency | 0.9762 | 0.9044 | 0.9389 | 136 |
| Urgent | 0.8355 | 0.8759 | 0.8552 | 145 |
| Routine | 0.8788 | 0.9134 | 0.8958 | 127 |
| General Query | 0.9914 | 0.9746 | 0.9829 | 118 |

## Confusion matrix

Rows = gold, columns = predicted.

| gold \ pred | Emergency | Urgent | Routine | General Query |
|---|---|---|---|---|
| **Emergency** | 123 | 13 | 0 | 0 |
| **Urgent** | 3 | 127 | 14 | 1 |
| **Routine** | 0 | 11 | 116 | 0 |
| **General Query** | 0 | 1 | 2 | 115 |

---

# Phase 10 NER validation (guardrail check)

- Sequences: 3178 (115037 tokens)
- Gold spans: 8436 | predicted spans: 8430
- IOB decoding: conlleval-style (dangling I- opens a span)

## Entity-level (primary)

- **Entity F1 (micro): 0.6048**
- Precision: 0.6050
- Recall: 0.6046
- Macro F1 over types present in gold: 0.6527

## Per entity type

| Entity type | Precision | Recall | F1 | Gold | Pred |
|---|---|---|---|---|---|
| Symptom | 0.4433 | 0.4575 | 0.4503 | 3460 | 3571 |
| Health Condition | 0.5331 | 0.4810 | 0.5057 | 788 | 711 |
| Medicine | 0.8168 | 0.8237 | 0.8202 | 1781 | 1796 |
| Age | 0.7734 | 0.7916 | 0.7824 | 595 | 609 |
| Dosage | 0.5386 | 0.4856 | 0.5107 | 762 | 687 |
| Specialist | 0.8728 | 0.8691 | 0.8709 | 695 | 692 |
| Medical Procedure | 0.6209 | 0.6366 | 0.6287 | 355 | 364 |

## Token level (non-O positions, comparison metric only)

- Precision: 0.7460
- Recall: 0.7314
- F1: 0.7386

## Joint training

- Best epoch: 7 of 10 run (max 15)
- Schedule: 1:1 alternating NER / severity batches
- Learning rates: shared 0.0002, NER head 0.0005, severity head 0.001
- NER guardrail: Phase 8 baseline 0.6099, floor 0.5999, selected checkpoint 0.6048
- Seed: 42 | wall clock: 4m07s
- Checkpoint: `/kaggle/working/bc/checkpoints/best_joint.pt`
