# BanglaCare - Phase T8 NER validation (transformer)

- Sequences: 3178 (115037 tokens)
- Gold spans: 8436 | predicted spans: 8815
- IOB decoding: conlleval-style (dangling I- opens a span)

## Entity-level (primary)

- **Entity F1 (micro): 0.6322**
- Precision: 0.6186
- Recall: 0.6464
- Macro F1 over types present in gold: 0.6771

## Per entity type

| Entity type | Precision | Recall | F1 | Gold | Pred |
|---|---|---|---|---|---|
| Symptom | 0.4703 | 0.5032 | 0.4862 | 3460 | 3702 |
| Health Condition | 0.5368 | 0.5558 | 0.5461 | 788 | 816 |
| Medicine | 0.8341 | 0.8692 | 0.8513 | 1781 | 1856 |
| Age | 0.7950 | 0.8084 | 0.8017 | 595 | 605 |
| Dosage | 0.5237 | 0.5210 | 0.5224 | 762 | 758 |
| Specialist | 0.8547 | 0.8719 | 0.8632 | 695 | 709 |
| Medical Procedure | 0.6558 | 0.6817 | 0.6685 | 355 | 369 |

## Token level (non-O positions, comparison metric only)

- Precision: 0.7518
- Recall: 0.7651
- F1: 0.7584

## Training

- Encoder: csebuetnlp/banglabert
- Best epoch: 5 of 5 run (max 5, patience 2)
- Seed: 42
- AdamW: encoder lr=2e-05, head lr=0.001, wd=0.01, batch=16, clip=1.0, linear schedule 10% warm-up
- Parameters: 111,029,778 total
- Mixed precision: True
- Wall clock: 29m54s
- Checkpoint: `/kaggle/working/checkpoints/best_ner_transformer.pt`
