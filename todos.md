# Churn Project Tasks & Status

Legend: Todo = not started, Doing = in progress, Done = acceptance criteria met.

## 0. Project Setup
- [Todo] Create project structure (`data/raw`, `data/processed`, `notebooks`, `src/data`, `src/models`, `src/serve`, `models`, `reports/figures`) and README stub. **Accept:** directories exist; README has goal, dataset link placeholder.
- [Todo] Set up Python env and dependencies (`pandas`, `numpy`, `scikit-learn`, `lightgbm`, `matplotlib`, `seaborn`, `shap`, `mlflow` or `wandb`, `fastapi`, `uvicorn`, `pydantic`, `joblib`, `pyyaml`). **Accept:** `requirements.txt` (or poetry/conda) created and installs without errors.
- [Todo] Place dataset `Telco-Customer-Churn.csv` in `data/raw/`. **Accept:** file present and loadable with `pandas.read_csv`.

## 1. EDA & Data Checks
- [Todo] Clean and inspect data (TotalCharges to numeric, drop `customerID`, inspect missingness and churn rate, check categorical levels). **Accept:** notebook `notebooks/01_eda.ipynb` saved with findings, churn rate, and notes on leakage (none expected) and missingness plan.
- [Todo] Define feature lists and preprocessing choices. **Accept:** documented list of numeric vs categorical fields and handling plan in notebook or `README`.

## 2. Baseline Model
- [Todo] Implement baseline Logistic Regression pipeline with preprocessing in `src/models/train.py`; include train/test split (stratified). **Accept:** script runs end-to-end and saves model to `models/logreg_pipeline.joblib`.
- [Todo] Compute metrics (PR-AUC, ROC-AUC, precision/recall at top 5/10/20%, confusion matrix) and record churn rate/baseline PR. **Accept:** metrics captured in `reports/` (text or markdown) and summarized in `README`.

## 3. Stronger Models & Tuning
- [Todo] Add LightGBM model with hyperparameter search (RandomizedSearchCV, StratifiedKFold=5) inside pipeline. **Accept:** script/notebook run saved results; comparison table vs logistic.
- [Todo] Enable experiment tracking (MLflow or W&B) logging params, metrics, artifacts. **Accept:** run logged with at least one model artifact and metrics.

## 4. Calibration & Thresholding
- [Todo] Calibrate probabilities (CalibratedClassifierCV with sigmoid and isotonic) and pick best by Brier score. **Accept:** chosen calibrated model saved; Brier scores reported.
- [Todo] Implement business thresholding (expected profit function) to pick contact threshold; report chosen threshold and rationale. **Accept:** threshold choice documented with profit table/plot in `reports/`.

## 5. Interpretability
- [Todo] Generate SHAP summary plot and per-customer example; PDP/ICE for top drivers (e.g., tenure, contract). **Accept:** figures saved to `reports/figures` and referenced in `README`.

## 6. Serving
- [Todo] Package best pipeline with joblib (`models/best_pipeline.joblib`) and version it. **Accept:** file saved with provenance noted (model, date, metrics).
- [Todo] Build FastAPI endpoint with Pydantic schema returning probability (and optionally top reasons). **Accept:** `src/serve/app.py` runs via `uvicorn`; sample request/response validated.
- [Todo] Optional: Streamlit dashboard for CSV scoring and ranking. **Accept:** app runs locally and ranks customers.

## 7. Deployment & Ops
- [Todo] Dockerize service and provide run instructions (Render/Fly/Cloud Run/Spaces). **Accept:** `Dockerfile` works locally; README has deploy steps.
- [Todo] Add monitoring mock (daily PR-AUC + drift checks) or at least describe approach. **Accept:** simple script or doc outlining metrics and checks.

## 8. Documentation & Reproducibility
- [Todo] Update README with problem framing, metrics, calibration, threshold choice, how to run train/evaluate/serve, and (optional) video link. **Accept:** README covers above and references figures/metrics.
- [Todo] Add Makefile (or simple scripts) for `train`, `evaluate`, `serve`, `format/test`. **Accept:** running targets works end-to-end.
- [Todo] Add tests/smoke checks (e.g., data load, pipeline fit, API response). **Accept:** tests run and pass locally.
