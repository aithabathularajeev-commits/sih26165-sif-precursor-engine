# Dataset Split Report — SIH26165

- Source: `data/label_mapping.csv`
- Total rows: 75
- Split: 80% train / 20% val (random_state=42)
- Stratified on: `primary_class` (first class in each row's `class_ids`, since sklearn's stratify does not support true multi-label stratification)

## Per-class row counts (multi-label, so columns sum to more than row count)

| Class | Full dataset | Train | Val |
|---|---|---|---|
| C1 | 9 | 7 | 2 |
| C2 | 9 | 7 | 2 |
| C3 | 13 | 11 | 2 |
| C4 | 30 | 24 | 6 |
| C5 | 10 | 8 | 2 |
| C6 | 7 | 5 | 2 |

## Known limitation

With only 75 rows and 6 overlapping classes, some classes (notably C6, the smallest at 7 total examples) may end up with very few validation examples. Report per-class precision/recall, not just aggregate accuracy — an aggregate number will hide poor performance on the rare classes. See README.md Section 6.