# BanglaCare - Phase 2 Split Report

## HealthNER (frozen official splits)
- train: 25426 rows
- valid: 3178 rows
- test: 3179 rows

## Severity (duplicate-aware stratified 80/10/10)
- Total rows: 5263
- Total duplicate-aware groups: 5192
- Exact/near-duplicate groups (size > 1): 69
- Near-duplicate pairs found (Jaccard >= 0.90, not already exact): 72
- Duplicate groups with inconsistent Categories labels: 3
  Examples (row indices -> label set):
    [227, 2537] -> ['Emergency', 'Urgent']
    [234, 2577] -> ['Routine', 'Urgent']
    [237, 2629] -> ['General Query', 'Urgent']

- Split sizes: {'train': 4211, 'val': 526, 'test': 526}
- Split class %: {'train': {'Urgent': 27.76, 'Emergency': 25.69, 'Routine': 24.1, 'General Query': 22.44}, 'val': {'Urgent': 27.57, 'Emergency': 25.86, 'Routine': 24.14, 'General Query': 22.43}, 'test': {'Urgent': 27.95, 'Emergency': 25.67, 'Routine': 23.95, 'General Query': 22.43}}

## Held-out fingerprint list
- 7403 unique normalized texts (HealthNER val+test U Severity val+test)
- Saved to data/splits/held_out_fingerprints.json
- These texts must never enter the Phase 4 medical corpus.