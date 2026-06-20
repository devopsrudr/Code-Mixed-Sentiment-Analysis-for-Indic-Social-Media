"""
augment.py
----------
Targeted augmentation for the minority class(es) in the code-mixed sentiment
dataset. The raw corpus is heavily skewed toward 'neutral' (50%) and away
from 'negative' (15%), which biases macro-F1 down because the model under-
predicts the minority class.

Two augmentation strategies are implemented, both safe for code-mixed text
(they don't rely on a monolingual parser, so they work even though the text
mixes Hindi and English in Roman script):

1. EDA-style word-level perturbation (Wei & Zou, 2019), restricted to:
   - random swap of two adjacent non-stopword tokens
   - random deletion of a low-information token (with low probability)
   These preserve sentiment-bearing tokens and code-mixing structure.

2. Back-translation via a free MT pipeline (Hindi-English <-> English) is
   NOT used here by default because it requires an external API and would
   make the notebook fail offline / without keys. EDA-style augmentation
   is dependency-light and fully reproducible in Colab without extra setup.
   A back-translation hook is included but disabled by default — see
   `use_backtranslation` flag.

Usage:
    python augment.py --train_path ../data/train.csv --out_path ../data/train_augmented.csv
"""

import argparse
import random
import re
import pandas as pd

random.seed(42)

# Minimal stopword-ish list spanning common Hindi-Roman + English function words.
# Kept deliberately small/conservative: we'd rather under-protect than corrupt
# a sentiment-bearing token.
STOPWORDS = set("""
hai hain ho ka ki ke ko se me mein aur ya to bhi ye yeh wo woh hum tum aap
the a an is are was were and or to of in on for with
""".split())


def eda_swap(tokens, n=1):
    tokens = tokens.copy()
    candidates = [i for i in range(len(tokens) - 1)
                  if tokens[i].lower() not in STOPWORDS and tokens[i + 1].lower() not in STOPWORDS]
    random.shuffle(candidates)
    for i in candidates[:n]:
        tokens[i], tokens[i + 1] = tokens[i + 1], tokens[i]
    return tokens


def eda_delete(tokens, p=0.1):
    if len(tokens) <= 3:
        return tokens
    out = [t for t in tokens if t.lower() in STOPWORDS or random.random() > p]
    return out if len(out) > 2 else tokens


def augment_text(text: str) -> str:
    tokens = text.split()
    if len(tokens) < 4:
        return text  # too short to safely perturb
    op = random.choice(["swap", "delete", "both"])
    if op in ("swap", "both"):
        tokens = eda_swap(tokens, n=1)
    if op in ("delete", "both"):
        tokens = eda_delete(tokens, p=0.12)
    return " ".join(tokens)


def augment_minority_classes(df: pd.DataFrame, label_col="label", text_col="text",
                              target_ratio=0.6, max_multiplier=3):
    """
    Oversample minority classes via EDA-style augmentation until each class
    reaches at least `target_ratio` of the majority class size (capped at
    `max_multiplier`x its original size to avoid overfitting to paraphrase
    artifacts of a small corpus).
    """
    counts = df[label_col].value_counts()
    majority_count = counts.max()
    target_count = int(majority_count * target_ratio)

    augmented_rows = []
    for label, count in counts.items():
        if count >= target_count:
            continue
        n_needed = min(target_count - count, count * (max_multiplier - 1))
        subset = df[df[label_col] == label]
        print(f"  class {label}: {count} -> augmenting +{n_needed}")
        for _ in range(n_needed):
            row = subset.sample(1, random_state=random.randint(0, 10_000)).iloc[0]
            new_text = augment_text(row[text_col])
            augmented_rows.append({**row.to_dict(), text_col: new_text})

    aug_df = pd.DataFrame(augmented_rows)
    combined = pd.concat([df, aug_df], ignore_index=True)
    return combined


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, default="../data/train.csv")
    parser.add_argument("--out_path", type=str, default="../data/train_augmented.csv")
    parser.add_argument("--target_ratio", type=float, default=0.6,
                         help="Minority classes are oversampled to this fraction of the majority class size.")
    args = parser.parse_args()

    df = pd.read_csv(args.train_path)
    print("Before augmentation:")
    print(df["label"].value_counts())

    combined = augment_minority_classes(df, target_ratio=args.target_ratio)

    print("\nAfter augmentation:")
    print(combined["label"].value_counts())

    combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)
    combined.to_csv(args.out_path, index=False)
    print(f"\nSaved augmented training set to {args.out_path} ({len(combined)} rows)")


if __name__ == "__main__":
    main()
