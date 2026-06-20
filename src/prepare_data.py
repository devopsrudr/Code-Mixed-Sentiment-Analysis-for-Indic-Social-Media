"""
prepare_data.py
----------------
Loads the raw Hindi-English code-mixed sentiment corpus (Joshi et al., 2016 —
Facebook comments dataset, distributed in tab-separated format) and produces
clean, stratified train/val/test CSV splits ready for transformer fine-tuning.

Raw format (tab-separated, no header):
    <id>  <text>  <label>  <label_repeated>
    label: 0 = negative, 1 = neutral, 2 = positive

Usage:
    python prepare_data.py --raw_path ../data/joshi2016_raw_data.txt --out_dir ../data
"""

import argparse
import csv
import re
import os
import pandas as pd
from sklearn.model_selection import train_test_split

LABEL_MAP = {0: "negative", 1: "neutral", 2: "positive"}


def load_raw(raw_path: str) -> pd.DataFrame:
    """Parse the tab-separated raw file into a DataFrame, skipping malformed lines."""
    rows = []
    skipped = 0
    with open(raw_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) != 4:
                skipped += 1
                continue
            _id, text, label, _label2 = parts
            try:
                label = int(label)
            except ValueError:
                skipped += 1
                continue
            if label not in LABEL_MAP:
                skipped += 1
                continue
            rows.append({"id": _id, "text": text, "label": label})
    print(f"Parsed {len(rows)} rows, skipped {skipped} malformed lines.")
    return pd.DataFrame(rows)


def clean_text(text: str) -> str:
    """Light cleaning, preserving code-mixed structure (no translation/transliteration)."""
    text = text.strip()
    # Collapse excessive repeated characters (e.g. "soooo" stays readable but "sooooooo" -> capped)
    text = re.sub(r"(.)\1{3,}", r"\1\1\1", text)
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_path", type=str, default="../data/joshi2016_raw_data.txt")
    parser.add_argument("--out_dir", type=str, default="../data")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = load_raw(args.raw_path)
    df["text"] = df["text"].apply(clean_text)

    # Drop empty / near-empty texts and exact duplicates
    before = len(df)
    df = df[df["text"].str.len() >= 2]
    df = df.drop_duplicates(subset="text")
    print(f"Dropped {before - len(df)} empty/duplicate rows -> {len(df)} remain.")

    df["label_name"] = df["label"].map(LABEL_MAP)

    print("\nClass distribution:")
    print(df["label_name"].value_counts())
    print("\nClass distribution (%):")
    print((df["label_name"].value_counts(normalize=True) * 100).round(1))

    # Stratified 80/10/10 split
    train_df, temp_df = train_test_split(
        df, test_size=0.2, stratify=df["label"], random_state=args.seed
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, stratify=temp_df["label"], random_state=args.seed
    )

    os.makedirs(args.out_dir, exist_ok=True)
    train_df.to_csv(os.path.join(args.out_dir, "train.csv"), index=False, quoting=csv.QUOTE_ALL)
    val_df.to_csv(os.path.join(args.out_dir, "val.csv"), index=False, quoting=csv.QUOTE_ALL)
    test_df.to_csv(os.path.join(args.out_dir, "test.csv"), index=False, quoting=csv.QUOTE_ALL)

    print(f"\nSaved splits to {args.out_dir}/")
    print(f"  train.csv : {len(train_df)} rows")
    print(f"  val.csv   : {len(val_df)} rows")
    print(f"  test.csv  : {len(test_df)} rows")


if __name__ == "__main__":
    main()
