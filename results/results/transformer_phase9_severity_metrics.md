# BanglaCare - Phase T9 severity warm-up (validation)

- Examples: 526
- Temperature: 1.0000  (uncalibrated)

## Headline

- **Macro-F1: 0.8591**
- Accuracy: 0.8517
- Weighted-F1: 0.8532
- **Emergency Recall: 0.8235**
- ECE: 0.0666 (mean confidence 0.8013)

## Per class

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Emergency | 0.8750 | 0.8235 | 0.8485 | 136 |
| Urgent | 0.7233 | 0.7931 | 0.7566 | 145 |
| Routine | 0.8537 | 0.8268 | 0.8400 | 127 |
| General Query | 1.0000 | 0.9831 | 0.9915 | 118 |

## Confusion matrix

Rows = gold, columns = predicted.

| gold \ pred | Emergency | Urgent | Routine | General Query |
|---|---|---|---|---|
| **Emergency** | 112 | 23 | 1 | 0 |
| **Urgent** | 13 | 115 | 17 | 0 |
| **Routine** | 3 | 19 | 105 | 0 |
| **General Query** | 0 | 2 | 0 | 116 |

## Warm-up

- Best epoch: 2 of 3
- Trainable: entity-aware attention + severity MLP only
- Loss: plain 4-class cross-entropy, no class weighting
- Seed: 42 | wall clock: 15s
- Checkpoint: `/kaggle/working/checkpoints/best_warmup_transformer.pt`
