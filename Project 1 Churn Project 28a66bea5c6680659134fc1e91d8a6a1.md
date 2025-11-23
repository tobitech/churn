# Project 1: Churn Project

Awesome. Let’s build a job-ready churn project end-to-end in ~4 weeks (10–15 hrs/week). We’ll start with the IBM Telco Customer Churn dataset for a fast v1, then optionally layer a time-aware variant.

## Project goal and success criteria

- Goal: Predict churn risk with calibrated probabilities to target top-k customers for retention.
- Success criteria (aims, adjust as you go):
    - PR-AUC meaningfully above baseline (baseline = churn rate).
    - Recall@top-10% ≥ 70% at acceptable precision (or set your own ops target).
    - Clear business case via cost/benefit thresholding and a small demo API.

# Week-by-week plan

## Week 1 – Setup, EDA, and a solid baseline

- Repo + env + data
    - Create a clean repo with: data/, notebooks/, src/, models/, reports/, requirements.txt, README.md.
    - Install: pandas, numpy, scikit-learn, lightgbm, shap, matplotlib, seaborn, mlflow (or Weights & Biases), fastapi, uvicorn, pydantic, joblib, pyyaml.
    - Dataset: IBM Telco Customer Churn (Kaggle). File: Telco-Customer-Churn.csv.
- EDA and leakage checks
    - Clean TotalCharges (string with blanks), convert to numeric; drop customerID.
    - Understand churn rate; inspect missingness; sanity-check categorical levels.
    - Identify potential leakage (none obvious here; still keep preprocessing inside a Pipeline).
- Baseline model
    - Train/test split (stratified).
    - scikit-learn Pipeline with ColumnTransformer:
        - Numeric: impute median → StandardScaler.
        - Categorical: OneHotEncoder(handle_unknown='ignore', min_frequency=0.01).
    - Model: LogisticRegression(max_iter=1000, class_weight='balanced').
    - Metrics: PR-AUC, ROC-AUC, precision/recall at top-k% (k = 5, 10, 20), confusion matrix.
    - Save model and a first README with results and key plots.

## Week 2 – Strong models, tuning, and pipelines

- Add LightGBM in a Pipeline; RandomizedSearchCV with StratifiedKFold (n_splits=5).
- Keep preprocessing inside the Pipeline to avoid leakage.
- Track experiments with MLflow or W&B (log params, metrics, artifact: model).
- Compare: logistic vs LightGBM; report cross-val PR-AUC and test PR-AUC.
- Add simple feature engineering if helpful (e.g., tenure buckets, charge ratios).

## Week 3 – Calibration, cost/benefit thresholding, and explainability

- Calibrate probabilities (CalibratedClassifierCV with 'isotonic' and 'sigmoid', pick best by Brier score).
- Choose threshold by business objective:
    - Cost to contact vs. expected retained revenue and save rate.
    - Report expected profit across thresholds; pick one and justify.
- Interpretability:
    - SHAP summary plot and per-customer explanations.
    - Partial dependence/ICE for top drivers (e.g., tenure, contract type).

## Week 4 – Deployment and presentation

- Package and persist the Pipeline with joblib.
- FastAPI service with a Pydantic schema; returns churn probability and top reasons.
- Optional: a lightweight Streamlit dashboard to score a CSV and rank customers.
- Dockerize and deploy (Render, Fly.io, Cloud Run, or Hugging Face Spaces).
- README: problem → approach → metrics → ROI → how to run → link to demo and a 60–90s video.

Minimal repo structure

- data/raw/, data/processed/
- notebooks/01_eda.ipynb
- src/data/make_dataset.py
- src/models/train.py, evaluate.py, predict.py
- src/serve/app.py (FastAPI)
- models/ (saved artifacts)
- reports/figures/
- requirements.txt, config.yaml, Makefile, README.md, tests/

Starter code: baseline model (run end-to-end)
Create src/models/train.py with this minimal baseline:

```jsx
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, confusion_matrix
import joblib

def top_k_metrics(y_true, y_proba, k=0.1):
# Select top-k fraction as positives
thresh = np.quantile(y_proba, 1 - k)
y_pred_topk = (y_proba >= thresh).astype(int)
tp = np.sum((y_true == 1) & (y_pred_topk == 1))
fp = np.sum((y_true == 0) & (y_pred_topk == 1))
fn = np.sum((y_true == 1) & (y_pred_topk == 0))
precision = tp / (tp + fp + 1e-9)
recall = tp / (tp + fn + 1e-9)
return {"threshold": float(thresh), "precision_at_k": float(precision), "recall_at_k": float(recall)}

def load_telco(csv_path):
df = pd.read_csv(csv_path)
df["Churn"] = (df["Churn"] == "Yes").astype(int)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].replace(" ", np.nan), errors="coerce")
# Drop obvious ID
df = df.drop(columns=["customerID"])
return df

def main():
data_path = Path("data/raw/Telco-Customer-Churn.csv")
df = load_telco(data_path)

`target = "Churn"
X = df.drop(columns=[target])
y = df[target].values

cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
num_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ("num", Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=0.01), cat_cols)
    ]
)

model = LogisticRegression(max_iter=1000, class_weight="balanced", n_jobs=None)

pipe = Pipeline(steps=[("pre", preprocessor), ("clf", model)])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

pipe.fit(X_train, y_train)
y_proba = pipe.predict_proba(X_test)[:, 1]

roc = roc_auc_score(y_test, y_proba)
pr = average_precision_score(y_test, y_proba)  # PR-AUC
top10 = top_k_metrics(y_test, y_proba, k=0.10)
top20 = top_k_metrics(y_test, y_proba, k=0.20)

print({"roc_auc": roc, "pr_auc": pr, "top10": top10, "top20": top20})
print("Confusion matrix at default 0.5 threshold:")
print(confusion_matrix(y_test, (y_proba >= 0.5).astype(int)))

Path("models").mkdir(exist_ok=True)
joblib.dump(pipe, "models/logreg_pipeline.joblib")
print("Saved model to models/logreg_pipeline.joblib")`

if **name** == "**main**":
main()
```

Next steps after running the baseline

- Save the printed metrics to your README with churn rate and PR baseline (baseline PR-AUC ≈ churn rate).
- Add a LightGBM model with RandomizedSearchCV and cross-val; compare metrics.
- Add calibration and business thresholding:

Example threshold-to-profit function (put in src/models/evaluate.py):

```jsx
import numpy as np

def expected_profit(y_true, y_proba, threshold, contact_cost=2.0, retained_value=50.0, save_rate=0.2):
# If we contact predicted churners:
# TP: contact a churner → retain with probability save_rate → +retained_value, but always pay contact_cost
# FP: contact a non-churner → 0 revenue, pay contact_cost
y_pred = (y_proba >= threshold).astype(int)
tp = np.sum((y_true == 1) & (y_pred == 1))
fp = np.sum((y_true == 0) & (y_pred == 1))
total_contacts = tp + fp
revenue = tp * save_rate * retained_value
cost = total_contacts * contact_cost
return revenue - cost

def best_threshold(y_true, y_proba, contact_cost=2.0, retained_value=50.0, save_rate=0.2):
thresholds = np.linspace(0.05, 0.95, 19)
profits = [expected_profit(y_true, y_proba, t, contact_cost, retained_value, save_rate) for t in thresholds]
i = int(np.argmax(profits))
return thresholds[i], profits[i]

Deployment skeleton (FastAPI)
Create src/serve/app.py:

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()
model = joblib.load("models/best_pipeline.joblib")

class Customer(BaseModel):
# Include all model features, example subset:
gender: str
SeniorCitizen: int
Partner: str
Dependents: str
tenure: int
PhoneService: str
MultipleLines: str
InternetService: str
OnlineSecurity: str
OnlineBackup: str
DeviceProtection: str
TechSupport: str
StreamingTV: str
StreamingMovies: str
Contract: str
PaperlessBilling: str
PaymentMethod: str
MonthlyCharges: float
TotalCharges: float

@app.post("/predict")
def predict(cust: Customer):
df = pd.DataFrame([cust.dict()])
proba = float(model.predict_proba(df)[:, 1][0])
return {"churn_probability": proba}
```

Run locally:

- python -m venv .venv && source .venv/bin/activate (or Windows equivalent)
- pip install -r requirements.txt
- Place Telco-Customer-Churn.csv in data/raw/
- python src/models/train.py
- uvicorn src.serve.app:app --reload

What to showcase in the repo

- README with:
    - Problem and business framing.
    - Metrics (PR-AUC, ROC-AUC, top-k), calibration plot, SHAP plots.
    - Cost/benefit analysis and chosen threshold.
    - How to run training and the API; link to a live demo if deployed.
- Reproducibility:
    - requirements.txt or poetry, seed control, and simple Makefile targets (make train, make evaluate, make serve).
    - Experiment tracking screenshot (MLflow/W&B).
- Interpretability figures in reports/figures.
- A short video walkthrough (screen recording) linked from the README.

Optional stretch once v1 is done

- Time-aware variant using a dataset with timestamps (e.g., Olist). Do month-based splits and evaluate drift.
- Add monitoring mock-up: daily PR-AUC and data drift using simple KS tests across key features.
- Uplift modeling version (if you can get a dataset with treatment/control).

Your move

- Confirm you’re okay starting with the Telco dataset and the above plan.
- If yes, set up the repo and run the baseline train.py. Share the printed metrics and I’ll help you choose the next best improvements (LightGBM config, calibration, and threshold tuning).