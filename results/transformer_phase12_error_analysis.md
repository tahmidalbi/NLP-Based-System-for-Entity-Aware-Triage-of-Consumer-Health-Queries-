# BanglaCare - Phase 12 error analysis

- Severity test errors: 43 / 526
- NER test: 1080 spurious spans, 901 missed spans, 1993 boundary errors

## Severity confusion pairs (gold -> pred)

| Confusion | Count |
|---|---|
| Urgent -> Routine | 15 |
| Routine -> Urgent | 12 |
| Emergency -> Urgent | 8 |
| Urgent -> Emergency | 8 |

## Severity error categories (heuristic)

- Long queries (>17 tokens, 90th pct of test): 3
- Code-switched (Bangla+Latin mixed): 0
- Contains negation (নেই/না/হয়নি): 18
- Would have been flagged for human review (confidence < 0.9392): 39

## NER errors by entity type

| Type | Missed | Boundary error | Type confusion |
|---|---|---|---|
| Symptom | 500 | 1302 | 91 |
| Health Condition | 158 | 99 | 116 |
| Medicine | 110 | 115 | 27 |
| Age | 12 | 101 | 7 |
| Dosage | 62 | 281 | 15 |
| Specialist | 18 | 59 | 4 |
| Medical Procedure | 41 | 36 | 14 |

- Medicine <-> Dosage confusions specifically: 18
- Errors involving rare types (Specialist, Medical Procedure): 172

## Not automated

Not automatically categorized: detecting a genuine spelling variant (vs. a different word) needs a reference dictionary or edit-distance index this project does not build. See phase12_error_sample.csv for manual review instead (guide 15.3).

A stratified sample for manual review is in `results/phase12_error_sample.csv` (long queries, code-switching, negation, low-confidence, and NER misses by type).