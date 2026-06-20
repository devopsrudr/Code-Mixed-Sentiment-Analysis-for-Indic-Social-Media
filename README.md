# Code-Mixed Sentiment Analysis for Indic Social Media

Fine-tuned multilingual transformer models (**mBERT**, **XLM-R**) for sentiment classification
of Hindi-English code-mixed ("Hinglish") social media text, with targeted data augmentation
and class-weighted loss to address severe class imbalance.

## TL;DR results

| Model | Macro-F1 | Accuracy | Notes |
|---|---|---|---|
| TF-IDF + Logistic Regression (baseline) | 0.624 | 0.688 | word(1-2gram) + char(3-5gram) TF-IDF |
| mBERT, fine-tuned (augmented + class-weighted) | *run notebook* | *run notebook* | `bert-base-multilingual-cased` |
| XLM-R, fine-tuned (augmented + class-weighted) | *run notebook* | *run notebook* | `xlm-roberta-base` |

> The baseline numbers above are real, reproducible results from this repo's `src/baseline.py`
> on the actual dataset (see `outputs/baseline/baseline_results.json`). The transformer numbers
> require a GPU to fine-tune (recommended: Colab T4, ~10–15 min/model) — run
> `notebooks/codemix_sentiment_analysis.ipynb` end-to-end to populate them, then update this
> table with your actual run's numbers before using this in a portfolio/resume context.

## Project structure

```
codemix-sentiment/
├── data/
│   └── joshi2016_raw_data.txt      # raw labeled corpus (see Dataset section)
├── notebooks/
│   └── codemix_sentiment_analysis.ipynb   # main, run this end-to-end in Colab
├── src/
│   ├── prepare_data.py             # raw -> clean train/val/test CSV splits
│   ├── augment.py                  # minority-class oversampling (EDA-style)
│   ├── baseline.py                 # TF-IDF + Logistic Regression baseline
│   └── train.py                    # mBERT / XLM-R fine-tuning w/ class-weighted loss
└── outputs/                        # metrics, trained models (gitignored if large)
```

## Dataset

This project uses the **Joshi et al. (2016)** Hindi-English code-mixed Facebook comments
corpus — ~3,900 sentences manually annotated for sentiment (negative / neutral / positive),
originally introduced in:

> Joshi, A., Prabhu, A., Shrivastava, M., & Varma, V. (2016). *Towards sub-word level
> compositions for sentiment analysis of Hindi-English code mixed text.* COLING 2016.

This is one of the foundational, well-cited datasets in Hindi-English code-mixed sentiment
research and is what most SAIL 2017 / early SemEval-era systems are compared against. The raw
file is fetched directly from a public GitHub mirror by `notebooks/codemix_sentiment_analysis.ipynb`.

**Why this dataset and not SAIL 2017 / SemEval-2020 Task 9 directly?** Both are good options
and arguably more "canonical" for citing this exact task, but:
- SAIL 2017's original data distribution links (Google Drive / Codalab from 2017) are unreliable today.
- SemEval-2020 Task 9 (SentiMix) is a strong, larger (20k tweets) alternative, but requires a
  free CodaLab account/registration to download — not a one-line `wget`.

The Joshi et al. 2016 corpus is real, labeled, citable, and reliably downloadable via a stable
GitHub raw URL, which matters if you want the notebook to actually run for someone else without
manual intervention. **If you want to swap in SemEval-2020 Task 9** after registering for it,
just point `prepare_data.py --raw_path` at your downloaded file and adjust the parser to match
its CoNLL-style token-per-line format (different from this corpus's one-row-per-sentence format).

**Class distribution** (imbalanced, which is the point of this project):

| Label | Count | % |
|---|---|---|
| neutral | 1,957 | 50.5% |
| positive | 1,352 | 34.9% |
| negative | 570 | 14.7% |

## Method

1. **Data prep** (`prepare_data.py`): parse raw tab-separated file, light cleaning
   (whitespace normalization, repeated-character capping), drop empty/duplicate rows,
   stratified 80/10/10 train/val/test split.

2. **Imbalance handling — two complementary strategies:**
   - **Data-level** (`augment.py`): EDA-style augmentation (Wei & Zou, 2019) restricted to
     adjacent non-stopword token swaps and low-probability token deletion. Chosen over
     back-translation because it requires no external API/dependency and is fully
     reproducible offline, which matters for a notebook other people will actually run.
     Minority classes are oversampled to ~60% of the majority class size, capped at 3x
     original size to avoid overfitting to repeated paraphrase artifacts.
   - **Loss-level** (`train.py`): class-weighted cross-entropy with inverse-frequency
     weights, applied via a custom `Trainer` subclass.

3. **Models**: `bert-base-multilingual-cased` (mBERT) and `xlm-roberta-base` (XLM-R),
   fine-tuned for 3-way sequence classification.

4. **Baseline**: word(1-2gram) + char(3-5gram) TF-IDF features with Logistic Regression —
   representative of pre-transformer SAIL/SemEval-era submissions, trained on the
   *non*-augmented data to give an honest "what imbalance costs you" reference point.

5. **Ablation**: the notebook also trains XLM-R with class-weighted loss but *without*
   augmentation, to isolate how much of the improvement comes from each intervention.

## Honest notes on the results

- On the TF-IDF baseline, augmentation alone barely moved macro-F1 (0.624 → 0.621 in our
  runs) — within noise. Classical bag-of-features models don't benefit much from oversampled
  near-duplicates the way transformers with proper regularization tend to. The real gains in
  this project come from the **combination** of transfer learning (mBERT/XLM-R pretraining) +
  class-weighted loss + augmentation together, not augmentation in isolation. The notebook's
  ablation cell (Section 9) measures this directly — report your own ablation numbers rather
  than assuming the textbook story holds for this exact dataset.
- The negative class is genuinely the hard one: the TF-IDF baseline gets 0.37 recall on
  negative vs 0.83 on neutral. Watch this per-class breakdown, not just macro-F1, when judging
  whether your imbalance fix actually worked.
- This corpus (~3,900 examples) is small by modern standards. Treat results as directionally
  meaningful, not as a tight estimate — rerun with different seeds if you want error bars.

## How to run

1. Open `notebooks/codemix_sentiment_analysis.ipynb` in Google Colab.
2. Runtime → Change runtime type → **T4 GPU**.
3. Run all cells top to bottom. Total runtime: ~25–40 minutes (mostly the two fine-tuning runs
   plus the ablation run).
4. Final results land in `outputs/final_results_summary.json`.

## Citation

If you use the dataset, cite:
```bibtex
@inproceedings{joshi2016subword,
  title={Towards sub-word level compositions for sentiment analysis of Hindi-English code mixed text},
  author={Joshi, Aditya and Prabhu, Ameya and Shrivastava, Manish and Varma, Vasudeva},
  booktitle={Proceedings of COLING 2016},
  pages={2482--2491},
  year={2016}
}
```# Code-Mixed-Sentiment-Analysis-for-Indic-Social-Media
