# BanglaCare - Phase 12 NER TEST results

- Sequences: 3179 (114008 tokens)
- Gold spans: 8462 | predicted spans: 8473
- IOB decoding: conlleval-style (dangling I- opens a span)

## Entity-level (primary)

- **Entity F1 (micro): 0.6252**
- Precision: 0.6248
- Recall: 0.6256
- Macro F1 over types present in gold: 0.6689

## Per entity type

| Entity type | Precision | Recall | F1 | Gold | Pred |
|---|---|---|---|---|---|
| Symptom | 0.4721 | 0.4634 | 0.4677 | 3528 | 3463 |
| Health Condition | 0.5357 | 0.5254 | 0.5305 | 786 | 771 |
| Medicine | 0.8570 | 0.8639 | 0.8604 | 1852 | 1867 |
| Age | 0.7778 | 0.7834 | 0.7806 | 554 | 558 |
| Dosage | 0.5132 | 0.5207 | 0.5169 | 747 | 758 |
| Specialist | 0.8707 | 0.8821 | 0.8764 | 687 | 696 |
| Medical Procedure | 0.6028 | 0.7045 | 0.6497 | 308 | 360 |

## Token level (non-O positions, comparison metric only)

- Precision: 0.7600
- Recall: 0.7349
- F1: 0.7472