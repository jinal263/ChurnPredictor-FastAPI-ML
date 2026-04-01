# 🏦 CC Underwriting — Churn Prediction API

>; FastAPI · scikit-learn · SQLAlchemy · HTMX · Docker

Predicts whether a customer will **churn** (leave a subscription) using a trained Random Forest model. Exposes a REST API, stores every prediction, and serves a live HTMX dashboard — no JavaScript framework needed.

![Python](https://img.shields.io/badge/Python-3.10+-blue) ![FastAPI](https://img.shields.io/badge/FastAPI-latest-green) ![scikit--learn](https://img.shields.io/badge/scikit--learn-latest-orange) ![SQLAlchemy](https://img.shields.io/badge/SQLAlchemy-2.x-red) ![Docker](https://img.shields.io/badge/Docker-ready-blue)

---

## 📋 Table of Contents
1. [Quick Start](#-quick-start)
2. [Architecture](#-architecture)
3. [ML Pipeline](#-ml-pipeline)
4. [API Endpoints](#-api-endpoints)
5. [Database Schema](#-database-schema)
6. [Security Features](#-security-features)
7. [Project Structure](#-project-structure)
8. [Tech Stack](#-tech-stack)
9. [Error Codes](#-error-codes)
10. [Dataset](#-dataset)

---

## 🚀 Quick Start

```bash
git clone https://github.com/your-username/cc-underwriting.git
cd cc-underwriting

python -m venv venv && venv\Scripts\activate   # Windows
# source venv/bin/activate                     # Mac/Linux

pip install -r requirements.txt
python cli_train.py        # train initial model
python run.py              # start server → http://localhost:8000
```

| URL | Purpose |
|-----|---------|
| `http://localhost:8000` | Dashboard |
| `http://localhost:8000/docs` | Swagger UI |
| `http://localhost:8000/redoc` | API Reference |
| `http://localhost:8000/health` | Health check |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│                  CC Underwriting                    │
│                                                     │
│   FRONTEND          BACKEND           ML LAYER      │
│   ─────────         ────────          ────────      │
│   HTMX              FastAPI           Random Forest │
│   Tailwind CSS      Pydantic v2       27 Features   │
│   Jinja2            Rate Limiter      Versioned     │
│   Google Fonts      CORS / Timeout    Pickle Store  │
│                                                     │
│          ┌──────────────────────────────┐           │
│          │  SQLAlchemy 2.x  (Async)     │           │
│          │  SQLite + aiosqlite driver   │           │
│          └──────────────────────────────┘           │
└─────────────────────────────────────────────────────┘
```

### Request Flow

```
 User clicks "Predict"
        │
        ▼
  ┌─────────────┐     exceeded?    ┌──────────────┐
  │ Rate Limiter│ ───────────────► │ 429 ERR_008  │
  │ 60 req/min  │                  └──────────────┘
  └──────┬──────┘
         │ ok
         ▼
  ┌─────────────┐     invalid?     ┌──────────────┐
  │ Pydantic v2 │ ───────────────► │ 422 ERR_002  │
  │ Validation  │                  └──────────────┘
  └──────┬──────┘
         │ valid
         ▼
  ┌─────────────┐
  │  Feature    │  17 raw inputs → 27 engineered features
  │ Engineering │
  └──────┬──────┘
         │
         ▼
  ┌─────────────┐
  │   Scaler    │  Normalise all numeric features
  └──────┬──────┘
         │
         ▼
  ┌─────────────┐
  │   Random    │  200 trees vote → probability
  │   Forest    │  ≥ 0.50 = Churn  │  < 0.50 = Retain
  └──────┬──────┘
         │
         ▼
  ┌─────────────┐
  │  Risk Label │  <35% Low · 35–65% Medium · >65% High
  └──────┬──────┘
         │
         ▼
  ┌─────────────┐
  │  Save to DB │  → JSON response → HTMX updates UI
  └─────────────┘
```

---

## 🤖 ML Pipeline

### How Random Forest Works

```
  17 Customer Fields
         │
         ▼
  ┌──────────────────────────────────────────────────┐
  │              Feature Engineering                 │
  │                                                  │
  │  Raw (10)          Engineered (6)                │
  │  ─────────         ─────────────                 │
  │  age               charges_per_tenure            │
  │  income            usage_per_dollar              │
  │  credit_score      complaint_rate                │
  │  tenure_months     support_per_product           │
  │  monthly_charges   financial_stress              │
  │  num_products      engagement_score              │
  │  support_calls                                   │
  │  complaints_last_6m  Cyclic (3)                  │
  │  avg_monthly_usage_gb  signup_month_sin/cos      │
  │  payment_delay_days    is_q4_signup              │
  │                                                  │
  │  Ordinal (2)       One-Hot (6)                   │
  │  education 0–3     gender · contract · marital   │
  └──────────────┬───────────────────────────────────┘
                 │  27 features
                 ▼
  ┌──────────────────────────────────────────────────┐
  │            StandardScaler                        │
  │  Normalises all numbers to the same scale        │
  └──────────────┬───────────────────────────────────┘
                 │
                 ▼
  ┌──────────────────────────────────────────────────┐
  │            Random Forest (200 Trees)             │
  │                                                  │
  │   Tree 1   Tree 2   Tree 3  ···  Tree 200        │
  │   Churn    Retain   Churn        Churn           │
  │      └────────┴────────┴──────────┘             │
  │              Majority Vote                       │
  │         e.g. 142/200 = 71% Churn                │
  └──────────────┬───────────────────────────────────┘
                 │
                 ▼
         Probability + Risk Label
         ┌──────────────────────┐
         │  < 35%  → Low Risk   │
         │  35–65% → Med Risk   │
         │  > 65%  → High Risk  │
         └──────────────────────┘
```

### Model Config & Performance

| Hyperparameter | Value | Why |
|---|---|---|
| `n_estimators` | 200 | More trees = more stable |
| `max_depth` | 12 | Prevents overfitting |
| `min_samples_split` | 10 | Avoids tiny node splits |
| `class_weight` | balanced | Handles 73/27 imbalance |
| `oob_score` | True | Free accuracy estimate |
| `criterion` | gini | Measures split quality |

| Metric | Score | Meaning |
|---|---|---|
| Accuracy | ~81% | 81/100 correct |
| ROC-AUC | ~0.88 | Near-perfect separation |
| F1 Score | ~0.76 | Balanced precision/recall |

### Versioned Model Store

Every training run → timestamped folder, **never overwritten**:

```
models_store/
└── v20260401_124353/
    ├── model.pkl            ← trained Random Forest
    ├── scaler.pkl           ← StandardScaler
    ├── imputer_stats.pkl    ← medians for missing data
    ├── label_encoders.pkl   ← category encodings
    ├── final_features.pkl   ← ordered list of 27 features
    └── manifest.pkl         ← version, accuracy, ROC-AUC, date
```

---

## ⚡ API Endpoints

```
GET    /health                    → server status + active model
POST   /api/predict               → single prediction (17 fields)
POST   /api/predict/batch         → up to 500 customers at once
POST   /api/train/start           → start background model training
GET    /api/train/{id}            → poll training status
GET    /api/train/                → list all model versions + metrics
GET    /api/applications/         → paginated prediction history
DELETE /api/applications/{id}     → delete a record
```

### Prediction Response

```json
{
  "id": 142,
  "customer_id": "CUST_001",
  "churn_prediction": true,
  "churn_probability": 0.723,
  "risk_label": "High",
  "model_version": "v20260401_124353",
  "created_at": "2026-04-01T12:43:53"
}
```

### Key Validation Rules (Pydantic v2)

| Field | Rule | Field | Rule |
|---|---|---|---|
| `age` | int 18–100 | `gender` | Male / Female / Other |
| `credit_score` | int 300–850 | `contract` | Month-to-month / 1yr / 2yr |
| `income` | float > 0 | `signup_month` | int 1–12 |
| `num_products` | int 1–10 | `education` | HS / Bachelor / Master / PhD |

---

## 🗄️ Database Schema

### `applications` — every prediction

| Column | Type | Description |
|---|---|---|
| `id` | PK | Auto-increment |
| `customer_id` | String | Optional reference |
| *(17 input fields)* | Mixed | All raw customer data |
| `churn_prediction` | Boolean | True = Will Churn |
| `churn_probability` | Float | 0.0 – 1.0 |
| `risk_label` | String | Low / Medium / High |
| `model_version` | String | Which model predicted |
| `created_at` | DateTime | Timestamp |

### `training_runs` — every model version

| Column | Type | Description |
|---|---|---|
| `id` | PK | Auto-increment |
| `model_version` | String | e.g. `v20260401_124353` |
| `status` | String | running / done / failed |
| `accuracy` / `roc_auc` / `f1_score` / `oob_score` | Float | Performance metrics |
| `is_active` | Boolean | Currently loaded model |
| `created_at` / `finished_at` | DateTime | Timing |

> **Switch databases:** Change `DATABASE_URL` in `.env` to move from SQLite → PostgreSQL/MySQL with zero code changes.

---

## 🔒 Security Features

| Feature | Detail |
|---|---|
| **Rate limiting** | 60 req/min per IP · sliding window · `X-RateLimit-*` headers on every response |
| **Request timeout** | 30s max → HTTP 504 · prevents hung threads |
| **CORS** | Open in dev · lock via `ALLOWED_ORIGINS` env var for production |
| **Background training** | `FastAPI BackgroundTasks` · immediate response + poll `/api/train/{id}` |
| **Pagination** | Default 10/page · returns `total`, `has_next`, `has_prev` |

---

## 📁 Project Structure

```
cc_underwriting/
├── run.py                      ← starts Uvicorn on :8000
├── cli_train.py                ← train without the web server
├── Dockerfile
├── requirements.txt
│
├── app/
│   ├── main.py                 ← FastAPI app, middleware, handlers
│   ├── database.py             ← async SQLAlchemy engine
│   ├── models.py               ← ORM table definitions
│   ├── schemas.py              ← Pydantic v2 schemas
│   ├── errors.py               ← ERR_001–ERR_013 registry
│   ├── api/
│   │   ├── predict.py          ← /api/predict endpoints
│   │   ├── train.py            ← /api/train endpoints
│   │   └── applications.py     ← /api/applications endpoints
│   ├── ml/
│   │   ├── feature_engineering.py  ← 17 → 27 features
│   │   ├── train_pipeline.py       ← RandomizedSearchCV pipeline
│   │   └── predict.py              ← load pickles + run inference
│   └── middleware/
│       └── rate_limit.py       ← sliding window limiter
│
├── templates/
│   ├── base.html               ← nav + CDN imports
│   ├── index.html              ← dashboard layout
│   └── partials/
│       ├── result.html         ← HTMX prediction card
│       └── history.html        ← HTMX history table
│
└── models_store/               ← git-ignored · versioned pickles
```

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **API** | FastAPI + Uvicorn | REST endpoints, async ASGI server |
| **Validation** | Pydantic v2 | Request/response schema enforcement |
| **ML** | scikit-learn | Random Forest, StandardScaler, RandomizedSearchCV |
| **Data** | NumPy + Pandas | Synthetic data generation, transformations |
| **Serialisation** | Pickle | Save/load trained model files |
| **Database** | SQLAlchemy 2.x + aiosqlite | Async ORM + SQLite driver |
| **Migrations** | Alembic (ready) | Schema change management |
| **Frontend** | HTMX + Tailwind CSS + Jinja2 | Partial updates, styling, server-side rendering |
| **Fonts** | Syne · DM Sans · DM Mono | Headings · body · numbers/code |
| **Container** | Docker | Any-server deployment |

---

## ❌ Error Codes

All errors return: `{ "error_code": "ERR_XXX", "message": "...", "detail": "..." }`

| Code | HTTP | Meaning |
|---|---|---|
| `ERR_001` | 422 | Missing required field |
| `ERR_002` | 422 | Invalid field value |
| `ERR_003` | 422 | Invalid field type |
| `ERR_004` | 503 | No trained model — run `cli_train.py` first |
| `ERR_005` | 500 | Model inference failed |
| `ERR_006` | 500 | Model training failed |
| `ERR_007` | 404 | Training run not found |
| `ERR_008` | 429 | Rate limit exceeded |
| `ERR_009` | 400 | Batch > 500 customers |
| `ERR_010` | 404 | Prediction record not found |
| `ERR_011` | 400 | CSV empty or malformed |
| `ERR_012` | 400 | CSV missing required columns |
| `ERR_013` | 500 | Internal server error |

---

## 📊 Dataset

**Synthetic** · 10,000 rows · ~27% churn rate · generated by `train_pipeline.py`

> Real customer data is private. Synthetic data follows the same statistical patterns — safe to build and test with.

| Column | Description | Column | Description |
|---|---|---|---|
| `age` | 18–85 | `gender` | Male / Female / Other |
| `income` | Annual (£) | `education` | HS / Bachelor / Master / PhD |
| `credit_score` | 300–850 | `marital_status` | Single / Married / Divorced |
| `tenure_months` | Months as customer | `contract` | Monthly / 1yr / 2yr |
| `monthly_charges` | Monthly fee | `signup_month` | 1–12 |
| `num_products` | Products held | `signup_year` | Year |
| `support_calls` | Call count | **`churn`** | **TARGET: 0 = stay · 1 = leave** |
| `complaints_last_6m` | Recent complaints | | |

**Key churn drivers:** more complaints → higher churn · month-to-month contracts churn more · higher charges = more churn · longer tenure = more loyal · single product holders churn most.


