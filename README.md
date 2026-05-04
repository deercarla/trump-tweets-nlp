# 📈 From Social Media to Wall Street
### *Do Trump's Posts Predict the S&P 500?*

> **NLP Final Assignment** · Natural Language Processing and Text Analytics (CDSCO1002U)  
> Copenhagen Business School — MSc. Business Administration and Data Science

**Authors:** Carla de Erausquin · Christoph Schilling · Diogo Semedo Amaro · José Lobo

---

## 🧭 Overview

This project investigates whether features derived from Donald Trump's social media posts — spanning **2009 to 2026** across **X (formerly Twitter)** and **Truth Social** — contain information predictive of **S&P 500 movements**.

We build a full NLP pipeline from raw text to a model-ready feature matrix, combining lexicon-based sentiment, transformer-based financial sentiment, automatic topic modelling, sentence embeddings, and a novel semantic novelty score. We then test whether these signals help predict next-day market direction using a suite of classification and regression models.

**Key finding:** Trump's posts contain a modest but consistent directional signal. Sentiment *dispersion* — how variable his tone is within a day — outperforms mean sentiment as a predictor. The signal concentrates on days with large market moves when Trump posted about economically relevant topics.

---

## 📊 Results at a Glance

| Setting | Best Model | Accuracy | ROC-AUC |
|---|---|---|---|
| Next-day classification (all days) | Gradient Boosting | 53.3% | 0.538 |
| Next-day classification (large moves, event days) | Logistic Regression | **55.3%** | **0.549** |
| Pre-market classification | Random Forest | 52.9% | 0.537 |
| Next-day regression | — | — | R² ≈ 0 |

> Dummy baseline: 50.0% accuracy. Return magnitude is not predictable under any specification.

---

## 🗂️ Repository Structure

```
nlp-final-project/
│
├── 1_data/                          # Raw data sources
│   ├── trump_tweets_raw.pkl         # Trump Twitter Archive + Truth Social
│   └── sp500_raw.csv                # Yahoo Finance S&P 500 daily data
│
├── 2_preprocessing/
│   ├── Preprocessing.ipynb          # Tokenisation, lemmatisation, POS tagging
│   └── processed/
│       ├── tweets_processed.pkl     # Cleaned tweet-level data (90,551 posts)
│       ├── daily_posts.csv          # Daily aggregated post counts
│       └── sp500_aligned.csv        # Market data aligned to tweet dates
│
├── 3_feature_engineering/
│   ├── Feature_Engineering.ipynb   # Full feature pipeline (VADER, FinBERT,
│   │                                #   BERTopic, embeddings, novelty score)
│   └── features/
│       ├── tweets_features.pkl      # Tweet-level features (74,742 posts)
│       ├── daily_features.csv       # Daily aggregated features
│       ├── feature_matrix.csv       # Model-ready matrix (~1,800 trading days)
│       ├── tweet_embeddings.npy     # Sentence embeddings (74742 × 384)
│       └── daily_mean_embeddings.npy
│
├── 4_eda/
│   ├── EDA.ipynb                    # Full exploratory analysis (18 figures)
│   └── figures/                     # All plots saved as PNG
│
├── 5_modelling/
│   ├── modelling.ipynb              # Classification & regression pipeline
│   └── results/                     # Saved result CSVs and importance tables
│
└── paper/
    └── Trump_Posts_vs_SP500.pdf     # Final paper
```

---

## 🔧 Pipeline

```
Raw Posts (90,551)
       │
       ▼
  Preprocessing
  ─────────────────────────────────────────────────
  • Timezone normalisation (US Eastern)
  • Repost exclusion → 74,742 original posts
  • Tokenisation · POS tagging · Lemmatisation
  • Stopword removal for frequency analysis
  • Raw text preserved for sentiment & embeddings
       │
       ▼
  Feature Engineering
  ─────────────────────────────────────────────────
  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────────┐
  │   VADER         │  │   FinBERT       │  │   BERTopic       │
  │   compound      │  │   polarity      │  │   11 topic flags │
  │   label         │  │   label         │  │   (data-driven)  │
  └─────────────────┘  └─────────────────┘  └──────────────────┘
  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────────┐
  │   Structural    │  │   Sentence      │  │   Novelty Score  │
  │   caps ratio    │  │   Embeddings    │  │   cosine dist    │
  │   engagement    │  │   384-dim       │  │   14-day window  │
  │   timing flags  │  │   MiniLM-L6-v2  │  │                  │
  └─────────────────┘  └─────────────────┘  └──────────────────┘
       │
       ▼
  Daily Aggregation → Merge with S&P 500 → Feature Matrix
       │
       ▼
  Modelling
  ─────────────────────────────────────────────────
  • Chronological split: train 2009–2020 / val 2022–2023 / test 2024–2025
  • Logistic Regression · Linear SVM · Random Forest · Gradient Boosting
  • ElasticNet · Random Forest Regressor
  • Incremental feature sets: market-only → +simple NLP → +full NLP → +semantic
  • Advanced: large-move classification on event days (walk-forward)
```

---

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/[your-repo]/nlp-final-project.git
cd nlp-final-project

# Create virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Key dependencies

| Package | Purpose |
|---|---|
| `vaderSentiment` | Lexicon-based social media sentiment |
| `transformers` | FinBERT (ProsusAI/finbert) |
| `sentence-transformers` | all-MiniLM-L6-v2 embeddings |
| `bertopic` | Automatic topic modelling |
| `umap-learn` | Dimensionality reduction for visualisation |
| `scikit-learn` | Modelling pipeline |
| `pandas` · `numpy` | Data manipulation |
| `matplotlib` · `seaborn` | Visualisation |

---

## 🚀 How to Run

Run the notebooks **in order**:

```
1. 2_preprocessing/Preprocessing.ipynb
2. 3_feature_engineering/Feature_Engineering.ipynb
3. 4_eda/EDA.ipynb
4. 5_modelling/modelling.ipynb
```

> ⚠️ **FinBERT** (~10–20 min on CPU) and **sentence embeddings** (~5–15 min) are cached to disk after the first run. Set `FINBERT_ENABLED = False` to skip.

> ⚠️ **UMAP** projection (~5–10 min) is also cached. Re-running notebooks is fast after initial computation.

---

## 📦 Data Sources

| Source | Description |
|---|---|
| [Trump Twitter Archive](https://datadrivendecisionlab.com/resources?resource=trump-tweets-dataset) | 90,551 posts, 2009–2026, Twitter + Truth Social |
| [Yahoo Finance](https://finance.yahoo.com/quote/%5EGSPC/) | S&P 500 daily OHLCV + returns, 2009–2026 |

---

## 🔑 Key Features Engineered

**Sentiment**
- `vader_compound` / `vader_std` — VADER mean and dispersion (most predictive NLP feature)
- `finbert_polarity` / `finbert_std` — FinBERT financial sentiment

**Topics** *(BERTopic, data-driven)*
- `topic_trump_identity` · `topic_elections` · `topic_trade_geopolitics`
- `topic_economy_markets` · `topic_rallies_nationalism` · and 6 more

**Structural**
- `caps_ratio` · `exclamation_count` · `log_engagement` · timing flags

**Semantic**
- `novelty_score` — cosine distance from 14-day rolling embedding baseline
- `emb_diversity` — within-day embedding standard deviation

**Market controls**
- `return_lag1/2` · `vol_5d` · `vol_20d`

---

## 📄 Citation

```bibtex
@misc{erausquin2026trump,
  title     = {From Social Media to Wall Street: Do Trump's Posts Predict the S\&P 500?},
  author    = {de Erausquin, Carla and Schilling, Christoph and
               Semedo Amaro, Diogo and Lobo, José},
  year      = {2026},
  note      = {NLP Final Assignment, Copenhagen Business School},
}
```

---

## 📚 Key References

- Hutto & Gilbert (2014) — VADER sentiment analysis
- Araci (2019) — FinBERT financial sentiment
- Reimers & Gurevych (2019) — Sentence-BERT embeddings
- Grootendorst (2022) — BERTopic
- Molnár et al. (2021) — Trump tweets and financial markets
- Frydendahl & Stenger (2019) — Trump tweets and firm-level stock prices (CBS thesis)
- Kim et al. (2021) — Musk tweets and Tesla stock

---

<div align="center">
  <sub>Copenhagen Business School · MSc. Business Administration and Data Science · 2026</sub>
</div>
