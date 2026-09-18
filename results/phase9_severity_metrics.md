# BanglaCare - Phase 9 severity warm-up (validation)

- Examples: 526
- Temperature: 1.0000  (uncalibrated)

## Headline

- **Macro-F1: 0.7540**
- Accuracy: 0.7471
- Weighted-F1: 0.7439
- **Emergency Recall: 0.8162**
- ECE: 0.0466 (mean confidence 0.7312)

## Per class

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Emergency | 0.6894 | 0.8162 | 0.7475 | 136 |
| Urgent | 0.6260 | 0.5310 | 0.5746 | 145 |
| Routine | 0.7222 | 0.7165 | 0.7194 | 127 |
| General Query | 0.9828 | 0.9661 | 0.9744 | 118 |

## Confusion matrix

Rows = gold, columns = predicted.

| gold \ pred | Emergency | Urgent | Routine | General Query |
|---|---|---|---|---|
| **Emergency** | 111 | 20 | 5 | 0 |
| **Urgent** | 39 | 77 | 27 | 2 |
| **Routine** | 11 | 25 | 91 | 0 |
| **General Query** | 0 | 1 | 3 | 114 |

## Warm-up

- Best epoch: 3 of 3
- Trainable: entity-aware attention + severity MLP only (encoder and NER head frozen, guide 12.1)
- Loss: plain 4-class cross-entropy, no class weighting
- Seed: 42 | wall clock: 5s
- Checkpoint: `/kaggle/working/bc/checkpoints/best_warmup.pt`
