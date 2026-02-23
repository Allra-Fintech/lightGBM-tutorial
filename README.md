# LightGBM Ads Tutorial

End-to-end LightGBM tutorial for keyword-ads performance prediction, using a synthetic dataset that mirrors real search-ads data.

## What's covered

| # | Section | Task |
|---|---|---|
| 1 | Dataset generation | Synthetic `(given_word, keyword)` pairs with realistic CTR and conversion signals |
| 2 | Feature preparation | Native categorical handling via `pandas.Categorical` |
| 3 | Model A — CTR | `LGBMRegressor`, impression-weighted, early stopping |
| 4 | Model B — Conversion | `LGBMClassifier`, AUC / PR-AUC evaluation |
| 5 | Keyword ranking | Score-based ranking using both models combined |
| 6 | LambdaMART | `LGBMRanker` with proper group-based train/test split, NDCG@k |
| 7 | Feature importance | Gain-based importance for all three models |

## Dataset schema

Each row is one `(given_word, keyword)` observation.

| Column | Type | Description |
|---|---|---|
| `given_word` | categorical | Seed / query word (e.g. `"loan"`) |
| `keyword` | categorical | Candidate keyword (e.g. `"refinance"`) |
| `similarity` | float | Similarity score between the two words (0–1) |
| `competition` | float | Auction competition level (0–1) |
| `impressions` | int | Number of times the ad was shown |
| `clicks` | int | Number of clicks |
| `cpc` | float | Cost per click |
| `cost` | float | Total spend (`clicks × cpc`) |
| `device` | categorical | `"mobile"` or `"desktop"` |
| `hour` | int | Hour of day (0–23) |
| `ctr` | float | Click-through rate (`clicks / impressions`) — **regression target** |
| `has_conversion` | int | 1 if at least one conversion occurred — **classification target** |
| `conversions` | int | Raw conversion count |

## Models

### Model A — CTR Regression

- **Objective:** `regression` (L2)
- **Target:** `ctr` (float, 0–1)
- **Key detail:** samples are weighted by `impressions` so high-traffic rows carry more influence
- **Eval metric:** RMSE

#### LGBMRegressor parameters

| Parameter | Value | Description |
|---|---|---|
| `n_estimators` | 2000 | Max number of trees to build. Early stopping will cut this short if validation score stops improving. |
| `learning_rate` | 0.03 | How much each tree corrects the previous ones. Lower = more careful, requires more trees. |
| `num_leaves` | 63 | Max leaves per tree. Controls model complexity — higher = more complex. |
| `subsample` | 0.8 | Use 80% of rows randomly per tree. Reduces overfitting. |
| `colsample_bytree` | 0.8 | Use 80% of features randomly per tree. Reduces overfitting. |
| `random_state` | 42 | Fixed seed for reproducibility. |
| `verbose` | -1 | Suppress all training logs. |

Reference: [LGBMRegressor API](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRegressor.html) · [Full parameter list](https://lightgbm.readthedocs.io/en/latest/Parameters.html)

### Model B — Conversion Classification

- **Objective:** `binary`
- **Target:** `has_conversion` (0 or 1)
- **Eval metrics:** AUC, PR-AUC

#### LGBMClassifier parameters

| Parameter | Value | Description |
|---|---|---|
| `n_estimators` | 3000 | More trees than the regressor — classification benefits from extra iterations. |
| `learning_rate` | 0.03 | Same as regressor. |
| `num_leaves` | 63 | Same as regressor. |
| `subsample` | 0.8 | Same as regressor. |
| `colsample_bytree` | 0.8 | Same as regressor. |
| `random_state` | 42 | Same as regressor. |
| `verbose` | -1 | Same as regressor. |

#### Regressor vs Classifier

| | `LGBMRegressor` | `LGBMClassifier` |
|---|---|---|
| Task | Predict a continuous number (CTR) | Predict a probability (conversion) |
| Output | `.predict()` → float | `.predict_proba()` → 0–1 probability |
| Default objective | `regression` (L2 loss) | `binary` (log loss) |

Reference: [LGBMClassifier API](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html) · [Full parameter list](https://lightgbm.readthedocs.io/en/latest/Parameters.html)

### Keyword Ranking (score-based)

`rank_keywords_for_given(given_word, candidates, base_features)` scores a list of candidate keywords by:

```
score = pred_ctr × pred_conv_prob
```

Swap this formula for ROAS, profit, or any other business metric.

### LambdaMART Ranker

- **Objective:** `lambdarank`
- **Eval metric:** NDCG@3, NDCG@5, NDCG@10
- **Relevance label:** `ctr` binned into integer grades 0–4 (swap for ROAS / conversions in production)
- **Split strategy:** group-based — all rows sharing a `given_word` stay in the same split (required for correct NDCG evaluation)

### Real Similarity

`similarity` is the most important feature in the model. It is computed in section 1 using `sentence-transformers` — cosine similarity between embeddings of `given_word` and each `keyword`.

| Model | Quality | Speed | Cost |
|---|---|---|---|
| `all-MiniLM-L6-v2` | Good | Fast, local | Free |
| `text-embedding-3-small` | Great | API call | OpenAI pricing |
| `word2vec` / `GloVe` | OK | Very fast, local | Free |

## Choosing the right setup

Rows marked ✅ are implemented in this notebook; rows marked 📖 are for reference only.

| | Success metric | Target variable | LightGBM objective | Eval metric |
|---|---|---|---|---|
| ✅ | CTR | `ctr` (float) | `regression` | RMSE / MAE |
| ✅ | Conversion | `has_conversion` (0/1) | `binary` | AUC / PR-AUC |
| ✅ | Keyword ranking | any relevance label | `lambdarank` | NDCG@k |
| 📖 | ROAS / Profit | continuous value | `regression` or `tweedie` | RMSE |
| 📖 | Click volume | `clicks` (count) | `poisson` | — |

## Requirements

- Python 3.9+
- lightgbm
- scikit-learn
- pandas
- numpy
- sentence-transformers

Install:

```bash
pip install lightgbm scikit-learn pandas numpy sentence-transformers
```

**macOS only:** LightGBM requires OpenMP, which is not bundled with the pip package. Install it via Homebrew before running:

```bash
brew install libomp
```

Then restart the notebook kernel after installing.

## How to run

**Option 1 — Jupyter (recommended)**

```bash
pip install jupyter
jupyter notebook lightgbm_ads_tutorial.ipynb
```

Run cells with `Shift+Enter`.

**Option 2 — VS Code**

Open `lightgbm_ads_tutorial.ipynb` directly. Install the Jupyter extension when prompted, then click **Run All**.

**Option 3 — Script**

```bash
jupyter nbconvert --to script lightgbm_ads_tutorial.ipynb
python lightgbm_ads_tutorial.py
```

## File structure

```
lightgbm/
├── lightgbm_ads_tutorial.ipynb   # Main notebook
├── given_words.json              # 50 e-commerce seed / query terms
├── keywords.json                 # 1,000 e-commerce ad keywords (20 categories)
├── .gitignore                    # Excludes .ipynb_checkpoints/
└── README.md                     # This file
```

### Vocabulary files

| File | Count | Description |
|---|---|---|
| `given_words.json` | 50 | Broad seed terms: `phone`, `laptop`, `yoga`, `camping`, etc. |
| `keywords.json` | 1,000 | Specific ad keywords across 20 e-commerce categories |

**Categories covered in `keywords.json`:**
electronics, fashion, footwear, watches, headphones, cameras, home & furniture, skincare, vitamins & supplements, running & fitness, yoga, gaming, coffee, baby, pets, jewelry, luggage & travel, desk & office, makeup, perfume, sunglasses, backpacks, wallets, rings, necklaces, shirts, pants, jackets, sneakers, boots, handbags, mattresses, blenders, vacuums, printers, monitors, keyboards, protein & supplements, cycling, camping, guitars, books, toys, candles, plants, gifts, fishing
