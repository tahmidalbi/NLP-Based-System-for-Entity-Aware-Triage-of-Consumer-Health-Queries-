# BanglaCare - Phase 12 severity TEST results

- Examples: 526
- Temperature: 1.2142  (calibrated)

## Headline

- **Macro-F1: 0.9218**
- Accuracy: 0.9183
- Weighted-F1: 0.9181
- **Emergency Recall: 0.9407**
- ECE: 0.0322 (mean confidence 0.9338)

## Per class

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Emergency | 0.9407 | 0.9407 | 0.9407 | 135 |
| Urgent | 0.8611 | 0.8435 | 0.8522 | 147 |
| Routine | 0.8837 | 0.9048 | 0.8941 | 126 |
| General Query | 1.0000 | 1.0000 | 1.0000 | 118 |

## Confusion matrix

Rows = gold, columns = predicted.

| gold \ pred | Emergency | Urgent | Routine | General Query |
|---|---|---|---|---|
| **Emergency** | 127 | 8 | 0 | 0 |
| **Urgent** | 8 | 124 | 15 | 0 |
| **Routine** | 0 | 12 | 114 | 0 |
| **General Query** | 0 | 0 | 0 | 118 |