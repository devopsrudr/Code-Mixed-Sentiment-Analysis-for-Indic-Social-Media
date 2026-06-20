"""
baseline.py
-----------
Classical baseline for comparison: char+word TF-IDF features with a
Logistic Regression classifier. This mirrors what most SAIL/SemEval-era
shared-task submissions used before transformers became standard, and
gives an honest reference point for the "% improvement" claim.

No class weighting / augmentation is applied here on purpose — it's meant
to represent the un-addressed-imbalance baseline.

Usage:
    python baseline.py --train_path ../data/train.csv --test_path ../data/test.csv
"""

import argparse
import json
import os

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, accuracy_score
from scipy.sparse import hstack

LABEL_NAMES = ["negative", "neutral", "positive"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, default="../data/train.csv")
    parser.add_argument("--test_path", type=str, default="../data/test.csv")
    parser.add_argument("--output_dir", type=str, default="../outputs/baseline")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    train_df = pd.read_csv(args.train_path)
    test_df = pd.read_csv(args.test_path)
    print(f"Train: {len(train_df)} | Test: {len(test_df)}")

    # Word-level TF-IDF (1-2 grams) + char-level TF-IDF (3-5 grams) —
    # char n-grams help a lot with code-mixed / transliterated / misspelled text.
    word_vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, max_features=20000)
    char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=20000)

    X_train_word = word_vec.fit_transform(train_df["text"])
    X_train_char = char_vec.fit_transform(train_df["text"])
    X_train = hstack([X_train_word, X_train_char])

    X_test_word = word_vec.transform(test_df["text"])
    X_test_char = char_vec.transform(test_df["text"])
    X_test = hstack([X_test_word, X_test_char])

    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(X_train, train_df["label"])

    preds = clf.predict(X_test)

    macro_f1 = f1_score(test_df["label"], preds, average="macro")
    weighted_f1 = f1_score(test_df["label"], preds, average="weighted")
    acc = accuracy_score(test_df["label"], preds)

    report = classification_report(
        test_df["label"], preds, target_names=LABEL_NAMES, output_dict=True, zero_division=0
    )

    print(f"\nAccuracy:    {acc:.4f}")
    print(f"Macro-F1:    {macro_f1:.4f}")
    print(f"Weighted-F1: {weighted_f1:.4f}\n")
    print(classification_report(test_df["label"], preds, target_names=LABEL_NAMES, zero_division=0))

    results = {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "classification_report": report,
    }
    with open(os.path.join(args.output_dir, "baseline_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {args.output_dir}/baseline_results.json")


if __name__ == "__main__":
    main()
