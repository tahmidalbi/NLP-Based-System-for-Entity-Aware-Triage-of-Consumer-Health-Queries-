# BanglaCare - Phase 12 severity TEST results

- Examples: 526
- Temperature: 1.6358  (calibrated)

## Headline

- **Macro-F1: 0.9099**
- Accuracy: 0.9068
- Weighted-F1: 0.9062
- **Emergency Recall: 0.9556**
- ECE: 0.0250 (mean confidence 0.9254)

## Per class

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Emergency | 0.9149 | 0.9556 | 0.9348 | 135 |
| Urgent | 0.8686 | 0.8095 | 0.8380 | 147 |
| Routine | 0.8626 | 0.8968 | 0.8794 | 126 |
| General Query | 0.9915 | 0.9831 | 0.9872 | 118 |

## Confusion matrix

Rows = gold, columns = predicted.

| gold \ pred | Emergency | Urgent | Routine | General Query |
|---|---|---|---|---|
| **Emergency** | 129 | 6 | 0 | 0 |
| **Urgent** | 11 | 119 | 17 | 0 |
| **Routine** | 1 | 11 | 113 | 1 |
| **General Query** | 0 | 1 | 1 | 116 |