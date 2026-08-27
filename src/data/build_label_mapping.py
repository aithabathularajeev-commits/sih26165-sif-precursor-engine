"""
build_label_mapping.py — SIH26165

Constructs data/label_mapping.csv, the missing link between training_data.csv
(raw precursor text) and labels_schema.json (the 6-class/11-subtag taxonomy).

Source of truth for the row->label assignments: README.md Section 4
("Row-level multi-label mapping"), which was hand-derived from the case
studies. This script does NOT re-derive labels from text (that would be
circular / unverifiable) — it encodes the README's published mapping table
as structured data and joins it against training_data.csv BY ROW POSITION.

Row-position join was verified before writing this script:
  - training_data.csv has exactly 75 rows in the same case-by-case order
    (Case 1 x4, Case 2 x7, Case 3 x8, ... Case 12 x7) as the README table.
  - Spot-checked Case 4 (rows 20-24): precursor_text order in the CSV matches
    the README table's text order exactly, row for row.
If training_data.csv is ever re-ordered or re-generated, this join breaks
silently — the assert block below guards against that by checking case_id
sequence and row count on every run.
"""

import csv
import json
import os

CLASS_IDS = ["C1", "C2", "C3", "C4", "C5", "C6"]
SUBTAG_IDS = [
    "C1-S1", "C2-S1", "C2-S2", "C3-S1", "C3-S2",
    "C4-S1", "C4-S2", "C4-S3", "C5-S1", "C6-S1", "C6-S2",
]

# (row#, case_id, [class_ids], [sub_tag_ids]) — transcribed verbatim from
# README.md Section 4 table, in row order 1-75.
ROW_LABELS = [
    (1, "Case 1", ["C3"], ["C3-S1"]),
    (2, "Case 1", ["C3"], ["C3-S1"]),
    (3, "Case 1", ["C4"], ["C4-S2"]),
    (4, "Case 1", ["C6"], ["C6-S2"]),
    (5, "Case 2", ["C1"], ["C1-S1"]),
    (6, "Case 2", ["C2"], ["C2-S1"]),
    (7, "Case 2", ["C4"], ["C4-S2"]),
    (8, "Case 2", ["C6"], ["C6-S2"]),
    (9, "Case 2", ["C6"], ["C6-S2"]),
    (10, "Case 2", ["C4"], ["C4-S1"]),
    (11, "Case 2", ["C4"], ["C4-S2"]),
    (12, "Case 3", ["C3"], ["C3-S2"]),
    (13, "Case 3", ["C3"], ["C3-S1"]),
    (14, "Case 3", ["C3"], ["C3-S1"]),
    (15, "Case 3", ["C3"], ["C3-S1"]),
    (16, "Case 3", ["C6"], ["C6-S2"]),
    (17, "Case 3", ["C6"], ["C6-S2"]),
    (18, "Case 3", ["C5"], ["C5-S1"]),
    (19, "Case 3", ["C1"], ["C1-S1"]),
    (20, "Case 4", ["C4"], ["C4-S1"]),
    (21, "Case 4", ["C4"], ["C4-S2"]),
    (22, "Case 4", ["C1", "C2"], ["C1-S1", "C2-S1"]),
    (23, "Case 4", ["C4"], ["C4-S2"]),
    (24, "Case 4", ["C4"], ["C4-S1"]),
    (25, "Case 5", ["C4"], ["C4-S1"]),
    (26, "Case 5", ["C5"], ["C5-S1"]),
    (27, "Case 5", ["C4"], ["C4-S1"]),
    (28, "Case 5", ["C5"], ["C5-S1"]),
    (29, "Case 5", ["C5"], ["C5-S1"]),
    (30, "Case 5", ["C1"], ["C1-S1"]),
    (31, "Case 5", ["C1"], ["C1-S1"]),
    (32, "Case 5", ["C1"], ["C1-S1"]),
    (33, "Case 5", ["C5"], ["C5-S1"]),
    (34, "Case 6", ["C2"], ["C2-S2"]),
    (35, "Case 6", ["C4"], ["C4-S2"]),
    (36, "Case 6", ["C4"], ["C4-S2"]),
    (37, "Case 6", ["C5"], ["C5-S1"]),
    (38, "Case 6", ["C3"], ["C3-S1"]),
    (39, "Case 7", ["C1"], ["C1-S1"]),
    (40, "Case 7", ["C4"], ["C4-S2"]),
    (41, "Case 7", ["C3"], ["C3-S1"]),
    (42, "Case 7", ["C3"], ["C3-S2"]),
    (43, "Case 7", ["C6"], ["C6-S2"]),
    (44, "Case 8", ["C4"], ["C4-S1"]),
    (45, "Case 8", ["C4"], ["C4-S1"]),
    (46, "Case 8", ["C2"], ["C2-S1"]),
    (47, "Case 8", ["C4"], ["C4-S2"]),
    (48, "Case 8", ["C2"], ["C2-S2"]),
    (49, "Case 8", ["C3"], ["C3-S1"]),
    (50, "Case 8", ["C1"], ["C1-S1"]),
    (51, "Case 8", ["C1"], ["C1-S1"]),
    (52, "Case 9", ["C4"], ["C4-S1"]),
    (53, "Case 9", ["C4"], ["C4-S1"]),
    (54, "Case 9", ["C4"], ["C4-S1"]),
    (55, "Case 9", ["C4"], ["C4-S1"]),
    (56, "Case 9", ["C5"], ["C5-S1"]),
    (57, "Case 9", ["C5"], ["C5-S1"]),
    (58, "Case 9", ["C4"], ["C4-S2"]),
    (59, "Case 9", ["C5"], ["C5-S1"]),
    (60, "Case 10", ["C3"], ["C3-S1"]),
    (61, "Case 10", ["C4"], ["C4-S3"]),
    (62, "Case 10", ["C4"], ["C4-S1", "C4-S3"]),
    (63, "Case 10", ["C5"], ["C5-S1"]),
    (64, "Case 10", ["C3"], ["C3-S1"]),
    (65, "Case 11", ["C2"], ["C2-S2"]),
    (66, "Case 11", ["C2"], ["C2-S2"]),
    (67, "Case 11", ["C2"], ["C2-S2"]),
    (68, "Case 11", ["C2", "C4"], ["C2-S1", "C4-S1"]),
    (69, "Case 12", ["C3"], ["C3-S1"]),
    (70, "Case 12", ["C4"], ["C4-S3"]),
    (71, "Case 12", ["C4"], ["C4-S3"]),
    (72, "Case 12", ["C4"], ["C4-S1"]),
    (73, "Case 12", ["C4"], ["C4-S1"]),
    (74, "Case 12", ["C4"], ["C4-S3"]),
    (75, "Case 12", ["C4", "C6"], ["C4-S1", "C6-S1"]),
]


def main():
    repo_root = os.path.dirname(os.path.abspath(__file__))
    src_csv = os.path.join(repo_root, "data", "training_data.csv")
    schema_path = os.path.join(repo_root, "data", "labels_schema.json")
    out_path = os.path.join(repo_root, "data", "label_mapping.csv")

    with open(src_csv, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    with open(schema_path, encoding="utf-8") as f:
        schema = json.load(f)
    case_ref = schema["case_reference"]

    # --- integrity checks: fail loudly rather than silently mis-joining ---
    assert len(rows) == len(ROW_LABELS) == 75, (
        f"Row count mismatch: training_data.csv has {len(rows)} rows, "
        f"ROW_LABELS has {len(ROW_LABELS)}. The README-derived mapping table "
        f"assumes exactly 75 rows in a specific order — do not proceed until "
        f"this is reconciled."
    )
    for i, (row, (row_num, case_id, classes, subtags)) in enumerate(zip(rows, ROW_LABELS), start=1):
        assert row["case_id"] == case_id, (
            f"Row {i}: training_data.csv has case_id='{row['case_id']}' but "
            f"README mapping expects '{case_id}'. training_data.csv row order "
            f"no longer matches the README Section 4 table — the position-based "
            f"join is invalid. Re-verify before trusting label_mapping.csv."
        )
        for c in classes:
            assert c in CLASS_IDS
        for s in subtags:
            assert s in SUBTAG_IDS, f"Row {i}: unknown sub_tag_id '{s}'"

    out_rows = []
    for row, (row_num, case_id, classes, subtags) in zip(rows, ROW_LABELS):
        oisd_id = case_ref.get(case_id, {}).get("oisd_id", "")
        rec = {
            "row_id": row_num,
            "case_id": case_id,
            "source_case_oisd_id": oisd_id,
            "precursor_text": row["precursor_text"],
            "severity": row["severity"],
            "class_ids": "|".join(classes),
            "sub_tag_ids": "|".join(subtags),
        }
        for c in CLASS_IDS:
            rec[f"class_{c}"] = 1 if c in classes else 0
        for s in SUBTAG_IDS:
            rec[f"sub_{s}"] = 1 if s in subtags else 0
        out_rows.append(rec)

    fieldnames = (
        ["row_id", "case_id", "source_case_oisd_id", "precursor_text", "severity",
         "class_ids", "sub_tag_ids"]
        + [f"class_{c}" for c in CLASS_IDS]
        + [f"sub_{s}" for s in SUBTAG_IDS]
    )

    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    # --- sanity summary ---
    print(f"Wrote {len(out_rows)} rows -> {out_path}")
    print("\nPer-class row counts (multi-label; sums exceed 75):")
    for c in CLASS_IDS:
        n = sum(r[f"class_{c}"] for r in out_rows)
        print(f"  {c}: {n}")
    print("\nPer-subtag row counts:")
    for s in SUBTAG_IDS:
        n = sum(r[f"sub_{s}"] for r in out_rows)
        print(f"  {s}: {n}")
    multi = sum(1 for r in out_rows if "|" in r["class_ids"])
    print(f"\nMulti-label rows (>1 class): {multi}")


if __name__ == "__main__":
    main()
