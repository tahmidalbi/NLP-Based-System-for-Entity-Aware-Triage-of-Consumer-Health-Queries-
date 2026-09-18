# Subword length audit (csebuetnlp/banglabert, cap 512)

| Split | n | median | p95 | p99 | max | over cap |
|---|---|---|---|---|---|---|
| ner_train | 25426 | 40 | 110 | 162 | 556 | 2 (0.01%) |
| ner_valid | 3178 | 40 | 111 | 160 | 433 | 0 (0.00%) |
| ner_test | 3179 | 40 | 110 | 169 | 369 | 0 (0.00%) |
| severity_train | 4211 | 17 | 33 | 55 | 70 | 0 (0.00%) |
| severity_val | 526 | 17 | 31 | 58 | 62 | 0 (0.00%) |
| severity_test | 526 | 17 | 31 | 51 | 56 | 0 (0.00%) |

**Decision:** Plain truncation is fine (<1% of NER examples exceed the cap). Log the truncated counts (the loaders print them).