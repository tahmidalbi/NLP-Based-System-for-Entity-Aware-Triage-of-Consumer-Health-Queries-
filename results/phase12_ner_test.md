# BanglaCare - Phase 12 NER TEST results

- Sequences: 3179 (114008 tokens)
- Gold spans: 8462 | predicted spans: 8465
- IOB decoding: conlleval-style (dangling I- opens a span)

## Entity-level (primary)

- **Entity F1 (micro): 0.6209**
- Precision: 0.6208
- Recall: 0.6210
- Macro F1 over types present in gold: 0.6669

## Per entity type

| Entity type | Precision | Recall | F1 | Gold | Pred |
|---|---|---|---|---|---|
| Symptom | 0.4639 | 0.4660 | 0.4649 | 3528 | 3544 |
| Health Condition | 0.5084 | 0.5013 | 0.5048 | 786 | 775 |
| Medicine | 0.8445 | 0.8472 | 0.8458 | 1852 | 1858 |
| Age | 0.7879 | 0.7780 | 0.7829 | 554 | 547 |
| Dosage | 0.5720 | 0.5315 | 0.5510 | 747 | 694 |
| Specialist | 0.8839 | 0.8865 | 0.8852 | 687 | 689 |
| Medical Procedure | 0.5894 | 0.6851 | 0.6336 | 308 | 358 |

## Token level (non-O positions, comparison metric only)

- Precision: 0.7479
- Recall: 0.7313
- F1: 0.7395