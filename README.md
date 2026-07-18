# AI Bill Anomaly Detection System

An AI-powered system that analyzes invoice data to detect suspicious or abnormal bills using Machine Learning (Isolation Forest), and presents results through an interactive Streamlit dashboard.

Built as an 8-milestone portfolio project demonstrating a full data science workflow: synthetic data generation, cleaning, EDA, feature engineering, unsupervised anomaly detection, risk scoring, dashboarding, and testing.

## Problem Statement

Manual invoice verification is slow and error-prone, often missing duplicate invoices, unusually high amounts, or incorrect tax calculations. This project automates anomaly detection so auditors can prioritize which bills need review instead of checking every invoice by hand.

## Results

| Metric | Value |
|---|---|
| Dataset size | 10,200 invoices (synthetic, with known injected anomalies) |
| True anomaly rate | 8.25% |
| Model | Isolation Forest (200 estimators, 34 engineered features) |
| ROC-AUC | 0.872 |
| PR-AUC | 0.745 |
| High-risk bucket precision | 100% true anomalies (188 invoices) |
| Medium+High risk recall | 68.8% of all anomalies caught by reviewing only 8.7% of invoices |
| Key engineered signal | `Amount_Deviation`: 0.81 (normal) vs. 3.07 (anomalous) — 3.7x separation |

**The business case in one line:** an auditor reviewing only the Medium + High risk bucket checks 8.7% of invoices and still catches roughly 69% of real anomalies — turning an all-or-nothing manual audit into a prioritized, tractable review queue.

## Tech Stack

- **Language:** Python
- **Data:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn, Plotly
- **Machine Learning:** Scikit-learn (Isolation Forest, StandardScaler)
- **Dashboard:** Streamlit
- **Testing:** Pytest

## Project Structure

```
notebooks/
  01_dataset_design.ipynb          Synthetic invoice dataset with injected anomalies
  02_data_cleaning.ipynb           Noise cleaning, anomaly signal preserved
  03_eda.ipynb                     Exploratory analysis, reusable Plotly chart functions
  04_feature_engineering.ipynb     Relative/contextual anomaly features
  05_isolation_forest.ipynb        Model training, comparison, feature importance
  06_risk_scoring_evaluation.ipynb 0-100 risk score, ROC/PR evaluation, business case

data/
  raw/                             Original synthetic dataset
  processed/                       Cleaned, feature-engineered, and scored datasets

models/
  isolation_forest.pkl             Trained model
  scaler.pkl                       Fitted StandardScaler
  model_feature_columns.json       Authoritative list of model input features

dashboard/
  app.py                           5-tab Streamlit dashboard

tests/
  test_pipeline.py                 31 pytest checks across the full pipeline
```

## Setup

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Running the Project

**Regenerate the full pipeline** (run notebooks in order, 01 through 06):
```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/01_dataset_design.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/02_data_cleaning.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/03_eda.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/04_feature_engineering.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/05_isolation_forest.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/06_risk_scoring_evaluation.ipynb
```
Or open them in Jupyter and run all cells interactively.

**Run the dashboard:**
```bash
streamlit run dashboard/app.py
```

**Run the tests:**
```bash
pytest tests/ -v
```

## Methodology Notes (worth knowing for interviews)

- **Unsupervised, not supervised.** `True_Anomaly` exists only because the dataset is synthetic (for validation). Isolation Forest never sees it during training — only `data/processed/invoices_scored.csv` onward uses it, purely for evaluation.
- **Cleaning preserves signal.** Data cleaning removes genuine noise (accidental duplicates, missing values, inconsistent formatting) but never touches the injected anomalies — outliers are flagged (`IQR_Outlier_Flag`), never dropped or capped.
- **Raw columns don't separate anomalies well.** EDA showed weak correlation between raw fields and anomaly status. This motivated relative/contextual features (`Amount_Deviation`, `Vendor_Frequency`) which do separate the classes clearly.
- **Contamination affects the cutoff, not the ranking.** `auto` vs. informed (0.0825) contamination produced identical ROC-AUC/PR-AUC — proving the parameter only shifts where the binary threshold sits, not the quality of the model's underlying anomaly ranking.
- **A feature-importance false lead, caught and explained.** Naive permutation importance on the trained model was dominated by sparse one-hot category columns — a scaling artifact, not genuine signal. Documented in `05_isolation_forest.ipynb` rather than left unexamined.

## Deployment

The dashboard is deployable for free on **Streamlit Community Cloud**:

1. Push this repository to GitHub (public repo, or connect a private one via Streamlit Cloud's GitHub integration).
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **New app**, select this repository, branch `main`, and set the main file path to `dashboard/app.py`.
4. Deploy. Streamlit Cloud installs from `requirements.txt` automatically.
5. Note: the app reads `data/processed/invoices_final.csv` at a relative path — make sure that file is committed to the repo (see the note on `.gitignore` below) or regenerated via a startup script, since Streamlit Cloud only has access to what's in the repo.

`.streamlit/config.toml` sets a consistent theme so the deployed app matches the local version.

### Note on `data/processed/`

This folder is gitignored by default (standard practice — processed data is regenerable from raw data + notebooks, not hand-maintained). If deploying to Streamlit Cloud, either:
- Force-add the final CSV so it's available at deploy time (`git add -f data/processed/invoices_final.csv`), or
- Add a startup step that runs the notebooks before the app starts.

## Scope

This project is for **educational and demonstration purposes**. It is not connected to a real invoicing system, GST verification API, or fraud database.

**Not included (see Future Scope):**
- OCR-based invoice reading
- Real GST/vendor verification
- Fraud classification beyond anomaly scoring
- Database connectivity
- Cloud-native deployment (beyond the free Streamlit Cloud option above)

## Future Scope

- OCR for automatic invoice reading
- Deep Learning–based fraud detection
- Explainable AI (SHAP) — noted in `05_isolation_forest.ipynb` as a fix for the permutation-importance limitation found during development
- Vendor risk profiling
- GST verification via government API
- PDF report generation
- FastAPI backend
- Docker + AWS deployment

## Roadmap

- [x] Milestone 0: Project setup
- [x] Milestone 1: Dataset design
- [x] Milestone 2: Data cleaning
- [x] Milestone 3: Exploratory Data Analysis
- [x] Milestone 4: Feature engineering
- [x] Milestone 5: Isolation Forest model
- [x] Milestone 6: Risk scoring & evaluation
- [x] Milestone 7: Streamlit dashboard
- [x] Milestone 8: Testing, documentation & deployment

## License

MIT
