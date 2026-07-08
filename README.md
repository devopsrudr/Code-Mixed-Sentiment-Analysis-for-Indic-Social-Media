# Code-Mixed Sentiment Analysis for Indic Social Media

A multilingual NLP project for sentiment classification of Hindi-English (Hinglish) code-mixed social media text using Transformer models.

---

## Project Overview

People in India often write social media posts by mixing Hindi and English in the same sentence.

For example:

> "Movie toh mast thi but ending was boring."

This type of text is called **code-mixed text**, and it is difficult for traditional NLP systems because it contains multiple languages, informal spelling, and inconsistent grammar.

In this project, I fine-tuned multilingual Transformer models to classify code-mixed text into three sentiment classes:

- Positive
- Neutral
- Negative

I also compared the Transformer models with a traditional machine learning baseline to understand whether deep learning actually provides an improvement on this dataset.

---

## Why I Built This Project

I wanted to understand how multilingual Transformer models perform on low-resource Indic NLP tasks.

Instead of focusing only on achieving the highest score, I wanted to build a project that is:

- Easy to reproduce
- Scientifically fair
- Easy to understand
- Based on proper model comparison

---

## Dataset

This project uses the **Hindi-English Code-Mixed Sentiment Dataset** released by **Joshi et al. (COLING 2016).**

The dataset contains approximately **3,900 manually labelled Facebook comments**.

### Why I selected this dataset

I selected this dataset because:

- It is publicly available.
- Anyone can reproduce this project without requesting dataset access.
- It is widely used in early code-mixed sentiment research.
- It is easy to download and use.

Although larger datasets such as **SemEval-2020 SentiMix** exist, they require additional registration and preprocessing.

For this project, I preferred **reproducibility over dataset size**.

---

## Project Pipeline

```
Dataset
      ↓
Data Cleaning
      ↓
Train / Validation / Test Split
      ↓
EDA Data Augmentation
      ↓
Class Weight Calculation
      ↓
Model Training
      ↓
Evaluation
      ↓
Comparison with Baseline
```

---

## Data Preprocessing

The following preprocessing steps were applied:

- Removed duplicate samples
- Removed empty rows
- Normalized whitespace
- Reduced repeated characters
- Performed stratified train-validation-test split

The stratified split helps maintain similar class distribution across all datasets.

---

## Handling Class Imbalance

The dataset is naturally imbalanced.

Instead of ignoring this problem, I used two techniques.

### 1. Easy Data Augmentation (EDA)

EDA performs simple text modifications such as:

- Random word swap
- Random word deletion

Minority classes were augmented until they reached around **60% of the majority class size**.

To avoid excessive duplication, augmentation was limited to **3× the original class size**.

### Why EDA?

Although back translation is a popular augmentation method, I intentionally selected EDA because:

- It works completely offline.
- It does not require paid APIs.
- Anyone can reproduce the project easily.
- It keeps the project simple and lightweight.

---

### 2. Class Weighted Loss

During Transformer fine-tuning, I used **class-weighted cross entropy loss**.

This gives more importance to minority classes during training and helps reduce the effect of class imbalance.

---

## Models

Three different models were evaluated.

### TF-IDF + Logistic Regression

This is the traditional machine learning baseline.

I included it to fairly compare whether Transformer models actually improve performance.

---

### mBERT

Multilingual BERT fine-tuned for three-class sentiment classification.

---

### XLM-R

XLM-RoBERTa fine-tuned using the same training strategy.

XLM-R has stronger multilingual pretraining and achieved the best performance in this project.

---

## Results

| Model | Macro-F1 | Accuracy |
|--------|---------:|---------:|
| TF-IDF + Logistic Regression | 0.6243 | 0.6881 |
| mBERT | 0.6088 | 0.6366 |
| XLM-R | **0.6659** | **0.6881** |

### Improvement over Baseline

| Model | Macro-F1 Improvement |
|--------|--------------------:|
| mBERT | -2.49% |
| XLM-R | **+6.66%** |

---

## Key Findings

### Traditional machine learning is still competitive

The TF-IDF baseline performed surprisingly well on this relatively small dataset.

This shows that traditional NLP methods are still strong baselines for low-resource problems.

---

### mBERT did not outperform the baseline

Although mBERT is a multilingual Transformer, it achieved lower performance than the baseline.

Possible reasons include:

- Small dataset size
- Limited fine-tuning data
- Better multilingual representations in newer models

Negative results are also valuable because they help us understand model limitations.

---

### XLM-R achieved the best performance

XLM-R produced the highest Macro-F1 score.

Its stronger multilingual pretraining likely helped it better understand Hindi-English code-mixed text.

---

### Macro-F1 is more important than Accuracy

Since the dataset is imbalanced, I mainly evaluated the models using **Macro-F1** instead of Accuracy.

Macro-F1 gives equal importance to every class and provides a more reliable evaluation for imbalanced datasets.

---

## Why I Compared with a Baseline

Many projects report only Transformer results.

I intentionally included a TF-IDF baseline because it answers an important research question:

**Does a Transformer actually improve performance?**

Without a baseline, it is difficult to know whether the improvement is meaningful.

---

## Limitations

This project has some limitations.

- Small dataset (~3,900 samples)
- Only Hindi-English code-mixed text
- Single augmentation strategy
- No statistical significance testing
- Multiple random seeds were not evaluated

These limitations provide opportunities for future improvements.

---

## Future Work

Possible future improvements include:

- Evaluate on SemEval-2020 SentiMix
- Compare with IndicBERT and MuRIL
- Experiment with back translation
- Train with multiple random seeds
- Extend to multilingual Indic sentiment analysis
- Explore sequence-to-sequence models for Indic machine translation

---

## Skills Demonstrated

This project helped me gain practical experience with:

- Natural Language Processing (NLP)
- Multilingual NLP
- Hugging Face Transformers
- PyTorch
- mBERT
- XLM-R
- TF-IDF
- Logistic Regression
- Data Augmentation
- Class Imbalance Handling
- Macro-F1 Evaluation
- Reproducible Machine Learning Pipelines

---

## Reproducibility

This repository is designed to be easy to reproduce.

- Public dataset
- Automatic data download
- No paid APIs
- Simple project structure
- Reproducible training pipeline

Anyone can clone the repository and reproduce the complete workflow.

---

## Final Thoughts

The objective of this project was not only to improve model performance but also to understand how multilingual Transformer models behave on low-resource Indic code-mixed text.

One important lesson from this work is that **a stronger model does not always guarantee better performance**. Careful experimentation, fair comparison, and honest reporting are equally important in research.
