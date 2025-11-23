# Churn Prediction Project

Goal: predict customer churn risk with calibrated probabilities to target top-k customers for retention.

Dataset: IBM Telco Customer Churn (Telco-Customer-Churn.csv) — place in `data/raw/` (download from Kaggle).

Next steps:
- Set up a Python environment and install dependencies: `pip install -r requirements.txt`.
- Run EDA in `notebooks/01_eda.ipynb` to inspect data, churn rate, missingness, and feature types.
- Build a baseline Logistic Regression pipeline in `src/models/train.py`.

Running EDA:
- From project root: `cd /Users/tobiomotayo/Developer/ml/churn`
- Activate venv: `source .venv/bin/activate`
- Launch Jupyter: `jupyter notebook` and open `notebooks/01_eda.ipynb`
- Outputs: summary saved to `reports/eda_summary.md`; figures to `reports/figures/churn_rate.png` and `reports/figures/missingness_top10.png`.

Repo structure (planned):
- `data/raw/`, `data/processed/`
- `notebooks/`
- `src/data/`, `src/models/`, `src/serve/`
- `models/`
- `reports/figures/`
- `requirements.txt`, `todos.md`
