# BanglaCare - Phase 8 NER validation

- Sequences: 3178 (115037 tokens)
- Gold spans: 8436 | predicted spans: 8190
- IOB decoding: conlleval-style (dangling I- opens a span)

## Entity-level (primary)

- **Entity F1 (micro): 0.6099**
- Precision: 0.6190
- Recall: 0.6010
- Macro F1 over types present in gold: 0.6550

## Per entity type

| Entity type | Precision | Recall | F1 | Gold | Pred |
|---|---|---|---|---|---|
| Symptom | 0.4741 | 0.4468 | 0.4601 | 3460 | 3261 |
| Health Condition | 0.4814 | 0.5266 | 0.5030 | 788 | 862 |
| Medicine | 0.8333 | 0.8113 | 0.8222 | 1781 | 1734 |
| Age | 0.7815 | 0.7933 | 0.7873 | 595 | 604 |
| Dosage | 0.5324 | 0.4633 | 0.4954 | 762 | 663 |
| Specialist | 0.8776 | 0.8662 | 0.8718 | 695 | 686 |
| Medical Procedure | 0.6237 | 0.6676 | 0.6449 | 355 | 380 |

## Token level (non-O positions, comparison metric only)

- Precision: 0.7761
- Recall: 0.6936
- F1: 0.7325

## Training

- Best epoch: 9 of 12 run (max 20, patience 3)
- Seed: 42
- Optimizer: AdamW lr=0.001 weight_decay=0.0001 batch=32 clip=1.0
- Mixed precision: True
- Wall clock: 21m40s
- Checkpoint: `/kaggle/working/bc/checkpoints/best_ner.pt`
