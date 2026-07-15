# AI Bill Anomaly Detection System

An AI-powered system that analyzes invoice data to detect suspicious or abnormal bills using Machine Learning (Isolation Forest), and presents results through an interactive Streamlit dashboard.

## Problem Statement

Manual invoice verification is slow and error-prone, often missing duplicate invoices, unusually high amounts, or incorrect tax calculations. This project automates anomaly detection to help auditors prioritize which bills need review.

## Status

🚧 Work in progress — following an 8-milestone build plan.

## Tech Stack

- Python, Pandas, NumPy
- Scikit-learn (Isolation Forest)
- Matplotlib, Plotly
- Streamlit

## Setup

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run dashboard/app.py
```

## Project Structure

```
data/         raw and processed invoice data
notebooks/    EDA and experimentation
src/          data cleaning, feature engineering, model code
dashboard/    Streamlit app
```

## Roadmap

- [x] Milestone 0: Project setup
- [x] Milestone 1: Dataset design
- [x] Milestone 2: Data cleaning
- [ ] Milestone 3: Exploratory Data Analysis
- [ ] Milestone 4: Feature engineering
- [ ] Milestone 5: Isolation Forest model
- [ ] Milestone 6: Risk scoring
- [ ] Milestone 7: Streamlit dashboard
- [ ] Milestone 8: Documentation & deployment

## License

MIT
