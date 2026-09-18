# BanglaCare - Phase 12 error analysis

- Severity test errors: 49 / 526
- NER test: 1112 spurious spans, 898 missed spans, 2017 boundary errors

## Severity confusion pairs (gold -> pred)

| Confusion | Count |
|---|---|
| Urgent -> Routine | 17 |
| Routine -> Urgent | 11 |
| Urgent -> Emergency | 11 |
| Emergency -> Urgent | 6 |
| Routine -> Emergency | 1 |

## Severity error categories (heuristic)

- Long queries (>17 tokens, 90th pct of test): 6
- Code-switched (Bangla+Latin mixed): 5
- Contains negation (না/হয়নি/নেই): 20
- Would have been flagged for human review (confidence < 0.9196): 37

## NER errors by entity type

| Type | Missed | Boundary error | Type confusion |
|---|---|---|---|
| Symptom | 459 | 1328 | 97 |
| Health Condition | 159 | 108 | 125 |
| Medicine | 118 | 140 | 25 |
| Age | 18 | 98 | 7 |
| Dosage | 76 | 256 | 18 |
| Specialist | 25 | 49 | 4 |
| Medical Procedure | 43 | 38 | 16 |

- Medicine <-> Dosage confusions specifically: 19
- Errors involving rare types (Specialist, Medical Procedure): 175

## Not automated

Not automatically categorized: detecting a genuine spelling variant (vs. a different word) needs a reference dictionary or edit-distance index this project does not build. See phase12_error_sample.csv for manual review instead (guide 15.3).

A stratified sample for manual review is in `results/phase12_error_sample.csv` (long queries, code-switching, negation, low-confidence, and NER misses by type).