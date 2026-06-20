"""
train.py
--------
Fine-tunes a multilingual transformer (mBERT or XLM-R) for 3-way sentiment
classification (negative / neutral / positive) on Hindi-English code-mixed
text.

Two imbalance-handling strategies are combined (this is what drives the
macro-F1 improvement over the baseline):
  1. Data-level: train on the augmented training set produced by augment.py
  2. Loss-level: class-weighted cross-entropy, weights computed from the
     (pre-augmentation) label frequencies so the rarer 'negative' class
     contributes proportionally more to the loss

Usage (Colab):
    !python train.py \
        --model_name bert-base-multilingual-cased \
        --train_path ../data/train_augmented.csv \
        --val_path ../data/val.csv \
        --output_dir ../outputs/mbert_run \
        --epochs 4 --batch_size 16 --lr 2e-5

    !python train.py \
        --model_name xlm-roberta-base \
        --train_path ../data/train_augmented.csv \
        --val_path ../data/val.csv \
        --output_dir ../outputs/xlmr_run \
        --epochs 4 --batch_size 16 --lr 2e-5
"""

import argparse
import json
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, precision_recall_fscore_support, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)

LABEL_NAMES = ["negative", "neutral", "positive"]


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class CodeMixDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


class WeightedLossTrainer(Trainer):
    """Trainer subclass that applies class-weighted cross-entropy loss."""

    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        weight = self.class_weights.to(logits.device) if self.class_weights is not None else None
        loss_fct = nn.CrossEntropyLoss(weight=weight)
        loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    macro_f1 = f1_score(labels, preds, average="macro")
    weighted_f1 = f1_score(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    precision, recall, f1_per_class, _ = precision_recall_fscore_support(
        labels, preds, average=None, labels=[0, 1, 2], zero_division=0
    )
    metrics = {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    }
    for i, name in enumerate(LABEL_NAMES):
        metrics[f"f1_{name}"] = f1_per_class[i]
        metrics[f"precision_{name}"] = precision[i]
        metrics[f"recall_{name}"] = recall[i]
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="bert-base-multilingual-cased",
                         help="e.g. bert-base-multilingual-cased or xlm-roberta-base")
    parser.add_argument("--train_path", type=str, default="../data/train_augmented.csv")
    parser.add_argument("--val_path", type=str, default="../data/val.csv")
    parser.add_argument("--output_dir", type=str, default="../outputs/run")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--no_class_weights", action="store_true",
                         help="Disable class-weighted loss (ablation / baseline mode).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading tokenizer/model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=3)

    train_df = pd.read_csv(args.train_path)
    val_df = pd.read_csv(args.val_path)
    print(f"Train: {len(train_df)} rows | Val: {len(val_df)} rows")

    train_ds = CodeMixDataset(train_df["text"], train_df["label"], tokenizer, args.max_length)
    val_ds = CodeMixDataset(val_df["text"], val_df["label"], tokenizer, args.max_length)

    class_weights = None
    if not args.no_class_weights:
        # NOTE: computed on the *original* label distribution philosophy —
        # i.e. pass the un-augmented train.csv label column if you want pure
        # inverse-frequency weights; here we compute on whatever train_path
        # was given so weighting reflects the actual data the model sees.
        weights = compute_class_weight(
            class_weight="balanced", classes=np.array([0, 1, 2]), y=train_df["label"].values
        )
        class_weights = torch.tensor(weights, dtype=torch.float)
        print(f"Class weights (neg/neu/pos): {weights.round(3)}")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=2,
        report_to="none",
        fp16=torch.cuda.is_available(),
        warmup_ratio=0.1,
        seed=args.seed,
    )

    trainer = WeightedLossTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        class_weights=class_weights,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    trainer.train()

    print("\nFinal validation metrics:")
    metrics = trainer.evaluate()
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # Save model, tokenizer, and metrics
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    with open(os.path.join(args.output_dir, "val_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nModel + metrics saved to {args.output_dir}")


if __name__ == "__main__":
    main()
