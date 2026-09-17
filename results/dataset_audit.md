# BanglaCare - Phase 1 Dataset Audit Report

## 1. Bangla-HealthNER
### train
- Rows: 25426
- Missing/empty text: 0
- Token/label count mismatches: 0
- Token length (median/p90/p95/p99/max): 31.0/64.0/82.0/112.0/507
- Contains Bangla: 77.7% | Latin: 44.9% | Mixed-script: 22.6% | Digits: 46.52% | URLs: 0.0%
- Exact duplicate rows: 12 (0.05%) in 6 groups
- Entity span counts (B- tags): {'Symptom': 26560, 'Health Condition': 6597, 'Medicine': 14229, 'Specialist': 5642, 'Age': 4436, 'Dosage': 6497, 'Medical Procedure': 3089}
- Rows with zero Bangla characters (pure English/Banglish): 5670 (22.3%)

### valid
- Rows: 3178
- Missing/empty text: 0
- Token/label count mismatches: 0
- Token length (median/p90/p95/p99/max): 30.0/65.0/83.0/113.0/342
- Contains Bangla: 76.75% | Latin: 44.68% | Mixed-script: 21.43% | Digits: 46.66% | URLs: 0.0%
- Exact duplicate rows: 0 (0.0%) in 0 groups
- Entity span counts (B- tags): {'Medicine': 1781, 'Symptom': 3460, 'Age': 595, 'Dosage': 762, 'Medical Procedure': 355, 'Health Condition': 788, 'Specialist': 695}
- Rows with zero Bangla characters (pure English/Banglish): 739 (23.25%)

### test
- Rows: 3179
- Missing/empty text: 0
- Token/label count mismatches: 0
- Token length (median/p90/p95/p99/max): 30.0/65.0/82.0/116.0/321
- Contains Bangla: 77.76% | Latin: 44.26% | Mixed-script: 22.02% | Digits: 47.4% | URLs: 0.0%
- Exact duplicate rows: 0 (0.0%) in 0 groups
- Entity span counts (B- tags): {'Age': 554, 'Symptom': 3528, 'Medicine': 1852, 'Health Condition': 786, 'Dosage': 747, 'Specialist': 687, 'Medical Procedure': 308}
- Rows with zero Bangla characters (pure English/Banglish): 707 (22.24%)

- Cross-split exact text overlap: {'train_valid': 4, 'train_test': 3, 'valid_test': 0}
- Full label set (15 labels): ['B-Age', 'B-Dosage', 'B-Health Condition', 'B-Medical Procedure', 'B-Medicine', 'B-Specialist', 'B-Symptom', 'I-Age', 'I-Dosage', 'I-Health Condition', 'I-Medical Procedure', 'I-Medicine', 'I-Specialist', 'I-Symptom', 'O']
- Entity types (7): ['Age', 'Dosage', 'Health Condition', 'Medical Procedure', 'Medicine', 'Specialist', 'Symptom']

## 2. Bangla Healthcare Severity Dataset
- Rows: 5263
- Columns: ['Text', 'Categories', 'Action Needed']
- **Authoritative source: Bangla-Healthcare_Dataset.xlsx (CSV export is corrupted, see csv_encoding_issue)**
- CSV encoding issue: {'csv_is_valid_utf8': False, 'csv_bangla_text_appears_mojibaked_to_question_marks': True, 'recommendation': 'Load the Severity dataset from the .xlsx file, not the .csv file.'}
- Missing/empty Text: 0
- Missing Action Needed: 0
- Token length (median/p90/p95/p99/max): 12.0/17.0/23.0/40.0/52
- Contains Bangla: 100.0% | Latin: 1.52% | Mixed-script: 1.52% | Digits: 3.14%
- Exact duplicate rows: 90 (1.71%) in 45 groups
- Class counts: {'Urgent': 1461, 'Emergency': 1353, 'Routine': 1268, 'General Query': 1181}
- Class percentages: {'Urgent': 27.76, 'Emergency': 25.71, 'Routine': 24.09, 'General Query': 22.44}
- All classes within 22-27%: False
- **Leakage rule: Action Needed column exists but MUST NOT be used as input feature, auxiliary target, prompt, or corpus sentence (annotation-derived, leaks the severity label).**

## 3. BanglaCHQ-Summ
### train
- Rows: 1880 | Columns: ['id', 'question', 'indices', 'summary']
- Missing/empty question: 0 | Missing/empty summary: 0
- Question token length (median/p90/p95/p99/max): 62.0/97.0/104.0/116.21000000000004/169
- Summary token length (median/p90/p95/p99/max): 28.0/44.0/50.0/65.0/86
- Exact duplicate questions: 0 (0.0%)

### valid
- Rows: 235 | Columns: ['id', 'question', 'indices', 'summary']
- Missing/empty question: 0 | Missing/empty summary: 0
- Question token length (median/p90/p95/p99/max): 59.0/96.0/106.0/113.66/141
- Summary token length (median/p90/p95/p99/max): 30.0/45.0/49.0/63.619999999999976/86
- Exact duplicate questions: 0 (0.0%)

### test
- Rows: 235 | Columns: ['id', 'question', 'indices', 'summary']
- Missing/empty question: 0 | Missing/empty summary: 0
- Question token length (median/p90/p95/p99/max): 64.0/97.6/109.0/113.66/147
- Summary token length (median/p90/p95/p99/max): 29.0/44.0/51.0/66.66/84
- Exact duplicate questions: 0 (0.0%)

## 4. BanglaHealth (Hugging Face)
- Rows: 200000 | Columns: ['sl', 'id', 'source_sentence', 'paraphrased_sentence']
- Missing/empty source_sentence: 0 | Missing/empty paraphrased_sentence: 0
- Source sentence token length (median/p90/p95/p99/max): 11.0/21.0/25.0/38.0/341
- Exact duplicate source sentences: 124997 (62.5%)
- Unique source sentences: 132575 (mean 1.51 paraphrases per unique source)
- Top 5 most-repeated source sentences: {'লেখক অধ্যাপক মেডিসিন বিভাগ বারডেম হাসপাতাল ঢাকা': 74, 'সভাপতি ডায়াবেটিস নিউট্রিশনিস্ট সোসাইটি অব বাংলাদেশ পপুলার ডায়াগনস্টিক সেন্টার শ্যামলী ও অ্যাডভান্স হাসপাতাল ঢাকা': 36, 'লেখক  চিফ নিউট্রিশন অফিসার ও বিভাগীয় প্রধান অব বারডেম': 30, 'ডা ফারাহ দোলা বিশেষ ভারপ্রাপ্ত কর্মকর্তা স্বাস্থ্য অধিদপ্তর': 27, 'লেখক  মুখ ও দন্তরোগ বিশেষজ্ঞ': 26}
- **Warning: Most-repeated source sentences look like author/byline boilerplate (e.g. 'লেখক ... বিভাগ ... হাসপাতাল') rather than health content. Consider filtering very-short, highly-repeated lines during Phase 4 corpus construction so they do not dominate FastText training.**
- Note: Use only as unlabeled corpus text (Phase 4). Do not derive severity or NER labels from it.

## Key Findings and Flags for Phase 2+
1. **Severity dataset CSV is corrupted** (Bangla text mojibaked to `?`). Always load `Bangla-Healthcare_Dataset.xlsx`, never the `.csv` export.
2. Severity class balance is close but not perfectly within 22-27%: Urgent is 27.76% (slightly above 27%). Still close enough that stratified splitting without rebalancing (per the guide) is appropriate.
3. Severity dataset has 45 exact-duplicate Text groups (90 rows) — these MUST be grouped together during the Phase 2 duplicate-aware split so they don't leak across train/val/test.
4. HealthNER has a small amount of exact-duplicate text **across the official splits** ({'train_valid': 4, 'train_test': 3, 'valid_test': 0}). The official splits are still used as-is per the guide, but this is worth noting as a minor, pre-existing leakage source outside our control.
5. Roughly a fifth of HealthNER text is pure English/Banglish with zero Bangla characters (consistent with the project's code-switching design goal, not a defect).
6. BanglaHealth's most-repeated "source_sentence" values are author/byline boilerplate, not paraphrasable health content — flag for filtering consideration in Phase 4 medical corpus construction.
7. All four raw sources are otherwise clean: zero missing/empty text fields, and HealthNER has a perfect 0-mismatch token/label alignment across all three splits (the Phase 3 sanity gate already passes on the raw data).