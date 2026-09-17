# BanglaCare - Phase 4 Medical Corpus Construction Report

## Sources included
- **BanglaHealth**: source_sentence + paraphrased_sentence, all 200,000 rows
- **Bangla-HealthNER**: TRAIN text only (25,426 rows)
- **BanglaCHQ-Summ**: questions + summaries, all splits (train/valid/test)
- **Severity dataset**: TRAIN Text only (Action Needed never included)

## Deduplication funnel
- Raw candidate lines: 434337 ({'banglahealth_source': 200000, 'banglahealth_paraphrase': 200000, 'healthner_train': 25426, 'chq_summ_question': 2350, 'chq_summ_summary': 2350, 'severity_train': 4211})
- Removed empty lines: 0
- Removed exact duplicate lines: 135462
- Removed exact held-out matches: 377
- Near-dup check scope: 29585 candidates (HealthNER-train + Severity-train only)
- Removed near-duplicate held-out matches: 51
- **Final corpus lines: 298447**
- Final lines per source: {'banglahealth_source': 132162, 'banglahealth_paraphrase': 133833, 'healthner_train': 25362, 'chq_summ_question': 574, 'chq_summ_summary': 2344, 'severity_train': 4172}

## Corpus statistics
- Unique whitespace tokens: 131406
- Script mix: {'n': 298447, 'contains_bangla_pct': 98.1, 'contains_latin_pct': 5.14, 'mixed_script_pct': 3.24, 'contains_digit_pct': 16.25, 'contains_url_pct': 0.0}

## Near-duplicate scope justification
Near-duplicate checking against the held-out fingerprint list (guide step 6.3.10) is computationally scoped to HealthNER-train and Severity-train candidates only. These are the only two sources that share dataset lineage with the held-out HealthNER-val/test and Severity-val/test texts (same collection, split from the same pool), so they are the only sources where near-duplicate leakage is plausible. BanglaHealth and BanglaCHQ-Summ are entirely separate datasets/collection processes; exact-duplicate removal (step 6.3.9) already catches any coincidental verbatim overlap with held-out text, and near-identical overlap beyond that is not a realistic risk given the different sources.