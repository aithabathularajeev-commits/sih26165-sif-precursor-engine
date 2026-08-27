"""
prepare_dataset.py — SIH26165 SIF Precursor Detection Engine

Turns data/label_mapping.csv into a ready-to-train (X, Y) train/val split.

Usage (run from repo root):
    python src/data/prepare_dataset.py

Input:
    data/label_mapping.csv
        Columns used:
          - precursor_text        (X: the NLP input)
          - class_C1 ... class_C6 (Y: 6-label multi-hot target — PRIMARY training target)
          - sub_C1-S1 ... sub_C6-S2 (Y_sub: 11-label multi-hot — optional, for explainability head)

Output:
    data/processed/train.csv
    data/processed/val.csv
    data/processed/split_report.md   (human-readable summary of what happened and why)

Design notes (read before changing this file):
    - Dataset is intentionally tiny (75 rows / 12 source incidents) — see README.md
      Section 6 "Known limitations". This script does NOT do anything fancy like
      k-fold CV by default because with this few examples per class, a single
      deliberate stratified split is more transparent and easier to reason about
      than cross-validation results that will bounce around wildly.
    - This is a MULTI-LABEL problem (a row can belong to >1 class). Standard
      sklearn `stratify=` only supports single-label stratification, so we
      stratify on a derived `primary_class` (the first class listed in
      `class_ids` for that row) as a practical approximation. This is documented
      here explicitly so nobody mistakes it for true multi-label stratification.
    - C4 has ~4x the examples of C6 (see label_mapping.csv sanity check). This
      script does NOT rebalance classes — it just reports the imbalance in
      split_report.md. Handle class weighting at the MODEL stage (e.g.
      class_weight='balanced' in sklearn, or pos_weight in a PyTorch BCE loss),
      not by duplicating/dropping data here.
"""

import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split

CLASS_COLS = ["class_C1", "class_C2", "class_C3", "class_C4", "class_C5", "class_C6"]
SUBTAG_COLS = [
    "sub_C1-S1", "sub_C2-S1", "sub_C2-S2", "sub_C3-S1", "sub_C3-S2",
    "sub_C4-S1", "sub_C4-S2", "sub_C4-S3", "sub_C5-S1", "sub_C6-S1", "sub_C6-S2",
]


def load_label_mapping(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in ["precursor_text", "class_ids"] + CLASS_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"label_mapping.csv is missing expected columns: {missing}")
    return df


def add_primary_class(df: pd.DataFrame) -> pd.DataFrame:
    """Derive a single-label stratification key from the multi-label class_ids column.
    Example: 'C1|C2' -> 'C1' (first-listed class is used as the stratification proxy)."""
    df = df.copy()
    df["primary_class"] = df["class_ids"].apply(lambda s: s.split("|")[0])
    return df


def split_dataset(df: pd.DataFrame, val_size: float, random_state: int):
    return train_test_split(
        df,
        test_size=val_size,
        random_state=random_state,
        stratify=df["primary_class"],
    )


def class_counts(df: pd.DataFrame) -> pd.Series:
    return df[CLASS_COLS].sum().astype(int)


def write_split_report(path: str, full_df, train_df, val_df, val_size, random_state):
    lines = []
    lines.append("# Dataset Split Report — SIH26165\n")
    lines.append(f"- Source: `data/label_mapping.csv`")
    lines.append(f"- Total rows: {len(full_df)}")
    lines.append(f"- Split: {int((1-val_size)*100)}% train / {int(val_size*100)}% val "
                 f"(random_state={random_state})")
    lines.append(f"- Stratified on: `primary_class` (first class in each row's `class_ids`, "
                 f"since sklearn's stratify does not support true multi-label stratification)")
    lines.append("")
    lines.append("## Per-class row counts (multi-label, so columns sum to more than row count)\n")
    lines.append("| Class | Full dataset | Train | Val |")
    lines.append("|---|---|---|---|")
    full_c, train_c, val_c = class_counts(full_df), class_counts(train_df), class_counts(val_df)
    for c in CLASS_COLS:
        lines.append(f"| {c.replace('class_','')} | {full_c[c]} | {train_c[c]} | {val_c[c]} |")
    lines.append("")
    lines.append("## Known limitation\n")
    lines.append(
        "With only 75 rows and 6 overlapping classes, some classes (notably C6, the "
        "smallest at 7 total examples) may end up with very few validation examples. "
        "Report per-class precision/recall, not just aggregate accuracy — an aggregate "
        "number will hide poor performance on the rare classes. See README.md Section 6."
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="Prepare train/val split for SIH26165 SIF precursor model.")
    parser.add_argument("--input", default="data/label_mapping.csv", help="Path to label_mapping.csv")
    parser.add_argument("--outdir", default="data/processed", help="Output directory for train/val CSVs")
    parser.add_argument("--val-size", type=float, default=0.2, help="Validation fraction (default 0.2)")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = load_label_mapping(args.input)
    df = add_primary_class(df)

    train_df, val_df = split_dataset(df, args.val_size, args.random_state)

    keep_cols = ["row_id", "case_id", "source_case_oisd_id", "precursor_text",
                 "severity", "class_ids", "sub_tag_ids"] + CLASS_COLS + SUBTAG_COLS

    train_out = os.path.join(args.outdir, "train.csv")
    val_out = os.path.join(args.outdir, "val.csv")
    report_out = os.path.join(args.outdir, "split_report.md")

    train_df[keep_cols].to_csv(train_out, index=False)
    val_df[keep_cols].to_csv(val_out, index=False)
    write_split_report(report_out, df, train_df, val_df, args.val_size, args.random_state)

    print(f"Loaded {len(df)} rows from {args.input}")
    print(f"Train: {len(train_df)} rows -> {train_out}")
    print(f"Val:   {len(val_df)} rows -> {val_out}")
    print(f"Report -> {report_out}")
    print("\nPer-class counts (train / val):")
    tc, vc = class_counts(train_df), class_counts(val_df)
    for c in CLASS_COLS:
        print(f"  {c.replace('class_',''):>3}: train={tc[c]:>2}  val={vc[c]:>2}")


if __name__ == "__main__":
    main()
