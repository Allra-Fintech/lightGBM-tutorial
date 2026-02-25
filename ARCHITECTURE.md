# LightGBM Ads Model — Serving Architecture

## Overview

This document describes the recommended infrastructure for serving the LightGBM keyword-ads model, along with evaluated alternatives. The service runs on an existing EKS cluster with S3 as a data lake and Redshift as the data warehouse.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     Data Layer                          │
│                                                         │
│   Redshift (DW)          S3 (Data Lake)                 │
│   └── ad performance     └── raw logs, exports          │
└──────────────────┬──────────────────────────────────────┘
                   │ training data
                   ▼
┌─────────────────────────────────────────────────────────┐
│                   Training (EKS CronJob)                │
│                                                         │
│   1. Pull data from Redshift / S3                       │
│   2. Precompute keyword embeddings (sentence-transformers│
│   3. Train LGBMRegressor, LGBMClassifier, LGBMRanker    │
│   4. Save model artifacts → S3                          │
└──────────────────┬──────────────────────────────────────┘
                   │ model.txt + embeddings.npy
                   ▼
┌─────────────────────────────────────────────────────────┐
│               Model Store (S3)                          │
│                                                         │
│   s3://bucket/models/lgbm/                              │
│   ├── latest/                                           │
│   │   ├── ctr_model.txt                                 │
│   │   ├── conv_model.txt                                │
│   │   ├── ranker_model.txt                              │
│   │   └── keyword_embeddings.npy                        │
│   └── 2026-02-23/   (versioned backup)                  │
└──────────────────┬──────────────────────────────────────┘
                   │ load on startup
                   ▼
┌─────────────────────────────────────────────────────────┐
│              Inference Service (EKS)                    │
│                                                         │
│   FastAPI Deployment (2+ pods)                          │
│   ├── POST /rank     → rank keywords for a given word   │
│   ├── POST /predict/ctr        → CTR score              │
│   └── POST /predict/conversion → conversion probability │
│                                                         │
│   HPA: scale by CPU / requests per second               │
└─────────────────────────────────────────────────────────┘
```

---

## Component Details

### Training — EKS CronJob

Runs on a schedule (e.g. nightly or weekly) to retrain models on fresh data.

```yaml
# k8s/training/cronjob.yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: lgbm-training
spec:
  schedule: "0 2 * * *"   # 2 AM daily
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: trainer
            image: your-ecr/lgbm-trainer:latest
            env:
            - name: S3_BUCKET
              value: your-bucket
            - name: REDSHIFT_DSN
              valueFrom:
                secretKeyRef:
                  name: redshift-secret
                  key: dsn
```

**Training script responsibilities:**
1. Pull `(given_word, keyword, impressions, clicks, cpc, cost, hour, ctr, has_conversion)` from Redshift
2. Encode all unique keywords once with `sentence-transformers` → save `keyword_embeddings.npy` to S3
3. Train CTR regressor, conversion classifier, LambdaMART ranker
4. Save model files to S3 with date-versioned path and overwrite `latest/`

---

### Model Store — S3

```
s3://your-bucket/
└── models/
    └── lgbm/
        ├── latest/
        │   ├── ctr_model.txt
        │   ├── conv_model.txt
        │   ├── ranker_model.txt
        │   └── keyword_embeddings.npy
        ├── 2026-02-23/
        └── 2026-02-16/
```

Enable S3 versioning on the bucket for additional safety. The `latest/` prefix is what inference pods load from.

---

### Inference Service — FastAPI on EKS

Models are loaded **once at pod startup** from S3, not per request.

```python
# Startup: load models from S3
ctr_model   = lgb.Booster(model_file=download("ctr_model.txt"))
conv_model  = lgb.Booster(model_file=download("conv_model.txt"))
kw_embeddings = np.load(download("keyword_embeddings.npy"))

# Request: encode only the given_word (keywords already cached)
@app.post("/rank")
def rank(given_word: str, candidates: list[str]):
    given_vec = embed_model.encode([given_word], normalize_embeddings=True)
    sims      = cosine_similarity(given_vec, kw_embeddings[candidate_indices])[0]
    # build features, predict, return ranked list
    ...
```

```yaml
# k8s/inference/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: lgbm-inference
spec:
  replicas: 2
  template:
    spec:
      containers:
      - name: api
        image: your-ecr/lgbm-inference:latest
        resources:
          requests:
            cpu: "500m"
            memory: "1Gi"
          limits:
            cpu: "2"
            memory: "2Gi"
```

---

### Sentence-Transformer Handling

The `all-MiniLM-L6-v2` model (~80MB) is **too slow to run at inference time** for real-time APIs.

**Recommended approach (Option B):**

| Step | Where | What |
|---|---|---|
| Training time | EKS CronJob | Encode all keywords → `keyword_embeddings.npy` → S3 |
| Inference time | FastAPI pod | Load cached embeddings from S3, encode only the `given_word` |

This means each request only encodes a single word — fast enough for real-time use.

---

## Kubernetes Resource Layout

```
k8s/
├── training/
│   └── cronjob.yaml
├── inference/
│   ├── deployment.yaml
│   ├── service.yaml        # internal LoadBalancer
│   └── hpa.yaml            # scale on CPU > 60%
└── configmap.yaml          # S3 bucket, model path, env config
```

---

## Alternatives Considered

### Option 1 — FastAPI on EKS ✅ Recommended

| | |
|---|---|
| Complexity | Low |
| Cost | Low (no extra infra) |
| Latency | Low |
| Pros | Full control, fits existing stack, no new services |
| Cons | Serving code is self-managed |

---

### Option 2 — BentoML on EKS

ML-specific serving framework that packages model + preprocessing as a deployable container.

| | |
|---|---|
| Complexity | Medium |
| Cost | Low |
| Latency | Low |
| Pros | Built-in batching, versioning, monitoring; good for multiple models |
| Cons | Extra dependency; adds learning curve |

**Best for:** teams planning to serve multiple models or wanting production ML features out of the box.

---

### Option 3 — MLflow Model Serving on EKS

Uses MLflow as both experiment tracker and model server.

| | |
|---|---|
| Complexity | Medium |
| Cost | Medium (MLflow tracking server on EKS) |
| Latency | Low |
| Pros | Unified experiment tracking + serving; one-command model serving |
| Cons | Requires running MLflow server; heavier infra for a single model |

**Best for:** teams that want full experiment tracking and reproducibility across many training runs.

---

### Option 4 — AWS SageMaker Endpoint

Fully managed ML serving outside EKS.

| | |
|---|---|
| Complexity | Low |
| Cost | High (per-hour endpoint pricing) |
| Latency | Low |
| Pros | Zero infra management; auto-scaling built in |
| Cons | Redundant cost if EKS is already running; vendor lock-in |

**Best for:** teams without EKS or who want zero-ops serving.

---

### Option 5 — NVIDIA Triton Inference Server on EKS

High-performance inference server with dynamic batching.

| | |
|---|---|
| Complexity | High |
| Cost | Medium |
| Latency | Very low |
| Pros | Very high throughput; dynamic batching; supports LightGBM via FIL backend |
| Cons | Complex setup; designed for GPU/deep learning; overkill for LightGBM |

**Best for:** very high QPS requirements (thousands of requests per second).

---

### Option 6 — AWS Lambda (Serverless)

Run inference as a serverless function behind API Gateway.

| | |
|---|---|
| Complexity | Low |
| Cost | Very low (pay per invocation) |
| Latency | High (cold start) |
| Pros | Zero idle cost; no infra |
| Cons | Cold start loads model from S3 each time; 250MB package limit is tight with sentence-transformers; inconsistent with EKS-first stack |

**Best for:** very infrequent requests where cold start latency is acceptable.

---

## Options Comparison

| Option | Complexity | Cost | Latency | Fits EKS stack |
|---|---|---|---|---|
| **FastAPI on EKS** | Low | Low | Low | ✅ |
| BentoML on EKS | Medium | Low | Low | ✅ |
| MLflow on EKS | Medium | Medium | Low | ✅ |
| SageMaker | Low | High | Low | ⚠️ Separate service |
| Triton on EKS | High | Medium | Very low | ✅ |
| Lambda | Low | Very low | High | ⚠️ Outside EKS |

---

## Data Flow: Naver Search Ads API Integration

With real data from the Naver Search Ads API, the training pipeline becomes:

```
Naver Search Ads API
├── /stats          → impressions, clicks, ctr, cpc, conversions, salesAmt
├── /ncc/keywords   → keyword, bidAmt, nccQi (quality score), isBrand, isSeason
└── /keywordstool   → related keyword suggestions

        │
        ▼ (scheduled ETL)

Redshift (DW)
└── ads_performance table

        │
        ▼ (training job)

LightGBM models → S3 → Inference API
```

### Recommended new features from the API

| API field | Add as | Expected impact |
|---|---|---|
| `nccQi` / Ad Relevance (2026) | Feature | High — direct quality signal |
| `avgRnk` | Feature | Medium — position affects CTR |
| `bidAmt` | Feature | Medium — bid level signals intent |
| `salesAmt` | Target (ROAS) | High — better business metric than CTR |
| `isBrand` / `isSeason` | Feature | Low-medium — keyword type signals |
