"""
Milestone 8 — Testing
AI Bill Anomaly Detection System

Sanity-check tests for the full pipeline: raw data -> cleaned data -> features ->
model -> scored/risk output. These aren't exhaustive unit tests of every function;
they're the checks that matter most for a data/ML pipeline — catching broken data
contracts between notebooks before they silently corrupt the dashboard.

Run with:  pytest tests/ -v
"""

import json
from pathlib import Path

import joblib
import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent
DATA_RAW = ROOT / "data" / "raw" / "invoices.csv"
DATA_CLEANED = ROOT / "data" / "processed" / "invoices_cleaned.csv"
DATA_FEATURES = ROOT / "data" / "processed" / "invoices_features.csv"
DATA_SCORED = ROOT / "data" / "processed" / "invoices_scored.csv"
DATA_FINAL = ROOT / "data" / "processed" / "invoices_final.csv"
MODEL_PATH = ROOT / "models" / "isolation_forest.pkl"
SCALER_PATH = ROOT / "models" / "scaler.pkl"
FEATURE_COLS_PATH = ROOT / "models" / "model_feature_columns.json"

REQUIRED_RAW_COLUMNS = {
    "Invoice_ID", "Vendor_Name", "Department", "Invoice_Date",
    "Amount", "GST", "Quantity", "Category", "True_Anomaly",
}


# ---------------------------------------------------------------------------
# File existence
# ---------------------------------------------------------------------------
class TestArtifactsExist:
    def test_raw_data_exists(self):
        assert DATA_RAW.exists(), f"Missing {DATA_RAW}. Run notebook 01 first."

    def test_cleaned_data_exists(self):
        assert DATA_CLEANED.exists(), f"Missing {DATA_CLEANED}. Run notebook 02 first."

    def test_features_data_exists(self):
        assert DATA_FEATURES.exists(), f"Missing {DATA_FEATURES}. Run notebook 04 first."

    def test_scored_data_exists(self):
        assert DATA_SCORED.exists(), f"Missing {DATA_SCORED}. Run notebook 05 first."

    def test_final_data_exists(self):
        assert DATA_FINAL.exists(), f"Missing {DATA_FINAL}. Run notebook 06 first."

    def test_model_artifacts_exist(self):
        assert MODEL_PATH.exists(), "Missing trained Isolation Forest model."
        assert SCALER_PATH.exists(), "Missing fitted StandardScaler."
        assert FEATURE_COLS_PATH.exists(), "Missing model_feature_columns.json."


# ---------------------------------------------------------------------------
# Raw dataset shape and schema
# ---------------------------------------------------------------------------
class TestRawData:
    @pytest.fixture(scope="class")
    def df(self):
        return pd.read_csv(DATA_RAW)

    def test_has_required_columns(self, df):
        missing = REQUIRED_RAW_COLUMNS - set(df.columns)
        assert not missing, f"Raw data missing columns: {missing}"

    def test_has_rows(self, df):
        assert len(df) > 0

    def test_invoice_id_unique(self, df):
        assert df["Invoice_ID"].is_unique, "Invoice_ID should be unique per invoice"

    def test_true_anomaly_is_binary(self, df):
        assert set(df["True_Anomaly"].unique()) <= {0, 1}

    def test_anomaly_rate_reasonable(self, df):
        # Sanity bound, not a strict spec -- catches gross generation bugs
        rate = df["True_Anomaly"].mean()
        assert 0.01 < rate < 0.30, f"Anomaly rate {rate:.2%} outside sane bounds"


# ---------------------------------------------------------------------------
# Cleaned dataset
# ---------------------------------------------------------------------------
class TestCleanedData:
    @pytest.fixture(scope="class")
    def df(self):
        return pd.read_csv(DATA_CLEANED, parse_dates=["Invoice_Date"])

    def test_no_missing_values(self, df):
        nulls = df.isnull().sum()
        assert nulls.sum() == 0, f"Unexpected missing values:\n{nulls[nulls > 0]}"

    def test_no_exact_duplicate_rows(self, df):
        assert df.duplicated().sum() == 0

    def test_amount_is_positive(self, df):
        assert (df["Amount"] > 0).all()

    def test_quantity_is_positive_int(self, df):
        assert (df["Quantity"] > 0).all()
        assert pd.api.types.is_integer_dtype(df["Quantity"])

    def test_date_parsed_correctly(self, df):
        assert pd.api.types.is_datetime64_any_dtype(df["Invoice_Date"])

    def test_anomaly_count_preserved_from_raw(self, df):
        raw = pd.read_csv(DATA_RAW)
        # Cleaning must not destroy the anomaly signal -- counts should match
        # (allowing for the intentional removal of a handful of accidental exact dupes
        # that were injected on top of the raw anomaly set during cleaning-demo noise)
        assert df["True_Anomaly"].sum() >= raw["True_Anomaly"].sum() * 0.95


# ---------------------------------------------------------------------------
# Feature-engineered dataset
# ---------------------------------------------------------------------------
class TestFeatureData:
    @pytest.fixture(scope="class")
    def df(self):
        return pd.read_csv(DATA_FEATURES)

    @pytest.fixture(scope="class")
    def feature_cols(self):
        with open(FEATURE_COLS_PATH) as f:
            return json.load(f)

    def test_readable_department_column_retained(self, df):
        # Regression test: an earlier version of this pipeline dropped the human-readable
        # Department/Category columns during one-hot encoding, which silently broke the
        # dashboard's filters. This must not happen again.
        assert "Department" in df.columns
        assert "Category" in df.columns

    def test_model_feature_columns_exist_in_data(self, df, feature_cols):
        missing = set(feature_cols) - set(df.columns)
        assert not missing, f"Feature columns missing from data: {missing}"

    def test_target_leakage_not_in_feature_columns(self, feature_cols):
        assert "True_Anomaly" not in feature_cols, (
            "True_Anomaly must never be a model input -- it's unsupervised."
        )

    def test_identifiers_not_in_feature_columns(self, feature_cols):
        for leaky_col in ["Invoice_ID", "Vendor_Name", "Invoice_Date"]:
            assert leaky_col not in feature_cols

    def test_feature_matrix_is_numeric(self, df, feature_cols):
        non_numeric = [c for c in feature_cols if not pd.api.types.is_numeric_dtype(df[c])]
        assert not non_numeric, f"Non-numeric model features: {non_numeric}"

    def test_amount_deviation_signal(self, df):
        # Regression check on the key engineered signal validated in Milestone 4:
        # anomalous invoices should show meaningfully higher Amount_Deviation
        normal_mean = df.loc[df["True_Anomaly"] == 0, "Amount_Deviation"].mean()
        anomaly_mean = df.loc[df["True_Anomaly"] == 1, "Amount_Deviation"].mean()
        assert anomaly_mean > normal_mean * 2, (
            "Amount_Deviation should be substantially higher for true anomalies"
        )


# ---------------------------------------------------------------------------
# Model artifacts
# ---------------------------------------------------------------------------
class TestModel:
    @pytest.fixture(scope="class")
    def model(self):
        return joblib.load(MODEL_PATH)

    @pytest.fixture(scope="class")
    def scaler(self):
        return joblib.load(SCALER_PATH)

    @pytest.fixture(scope="class")
    def feature_cols(self):
        with open(FEATURE_COLS_PATH) as f:
            return json.load(f)

    def test_model_predicts_on_real_data(self, model, scaler, feature_cols):
        df = pd.read_csv(DATA_FEATURES)
        X = df[feature_cols]
        X_scaled = scaler.transform(X)
        preds = model.predict(X_scaled)
        assert set(preds).issubset({-1, 1})
        assert len(preds) == len(df)

    def test_scaler_feature_count_matches_model_features(self, scaler, feature_cols):
        assert scaler.n_features_in_ == len(feature_cols)


# ---------------------------------------------------------------------------
# Final risk-scored dataset (what the dashboard actually reads)
# ---------------------------------------------------------------------------
class TestFinalData:
    @pytest.fixture(scope="class")
    def df(self):
        return pd.read_csv(DATA_FINAL, parse_dates=["Invoice_Date"])

    def test_risk_score_in_valid_range(self, df):
        assert df["Risk_Score"].between(0, 100).all()

    def test_risk_level_values_valid(self, df):
        assert set(df["Risk_Level"].unique()) <= {"Low", "Medium", "High"}

    def test_risk_level_matches_score_bands(self, df):
        low = df[df["Risk_Level"] == "Low"]["Risk_Score"]
        med = df[df["Risk_Level"] == "Medium"]["Risk_Score"]
        high = df[df["Risk_Level"] == "High"]["Risk_Score"]
        assert low.empty or low.max() <= 30
        assert med.empty or med.between(30, 70).all() or (med.min() > 30 and med.max() <= 70)
        assert high.empty or high.min() > 70

    def test_high_risk_bucket_concentrates_true_anomalies(self, df):
        # The whole point of the risk score: High bucket should be far more
        # concentrated with real anomalies than the overall base rate.
        overall_rate = df["True_Anomaly"].mean()
        high_rate = df.loc[df["Risk_Level"] == "High", "True_Anomaly"].mean()
        assert high_rate > overall_rate, (
            "High risk bucket should concentrate true anomalies above the base rate"
        )

    def test_dashboard_required_columns_present(self, df):
        # Columns the Streamlit dashboard directly depends on
        required = {
            "Invoice_ID", "Vendor_Name", "Department", "Category", "Invoice_Date",
            "Amount", "GST", "Quantity", "Risk_Score", "Risk_Level", "True_Anomaly",
        }
        missing = required - set(df.columns)
        assert not missing, f"Dashboard-required columns missing: {missing}"

    def test_no_missing_values_in_key_columns(self, df):
        key_cols = ["Amount", "GST", "Quantity", "Risk_Score", "Risk_Level", "Department"]
        assert df[key_cols].isnull().sum().sum() == 0
