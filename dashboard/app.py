"""
AI Bill Anomaly Detection System — Streamlit Dashboard
Milestone 7

Loads the final scored dataset from Milestone 6 (data/processed/invoices_final.csv)
and presents: dataset overview, expenditure visualizations, risk analysis,
invoice search, and bill statistics.

Run with:  streamlit run dashboard/app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Bill Anomaly Detection",
    page_icon="🧾",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_PATH = Path(__file__).parent.parent / "data" / "processed" / "invoices_final.csv"

RISK_COLORS = {"Low": "#2E7D32", "Medium": "#F9A825", "High": "#C62828"}
RISK_ORDER = ["Low", "Medium", "High"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
@st.cache_data
def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Invoice_Date"])
    return df


# ---------------------------------------------------------------------------
# Reusable chart functions
# (same functions authored in notebooks/03_eda.ipynb — kept in sync here so
# the dashboard and the EDA notebook produce identical visuals)
# ---------------------------------------------------------------------------
def plot_department_spend(dataframe: pd.DataFrame):
    spend = (
        dataframe.groupby("Department")["Amount"]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )
    fig = px.bar(
        spend, x="Amount", y="Department", orientation="h",
        title="Total Expenditure by Department", color="Amount",
        color_continuous_scale="Viridis",
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=420)
    return fig


def plot_monthly_trend(dataframe: pd.DataFrame):
    monthly = dataframe.copy()
    monthly["Month"] = monthly["Invoice_Date"].dt.to_period("M").astype(str)
    monthly_spend = monthly.groupby("Month")["Amount"].sum().reset_index()
    fig = px.line(
        monthly_spend, x="Month", y="Amount", markers=True,
        title="Monthly Expenditure Trend",
    )
    fig.update_xaxes(tickangle=45)
    fig.update_layout(height=420)
    return fig


def plot_amount_distribution(dataframe: pd.DataFrame):
    fig = px.histogram(
        dataframe, x="Amount", nbins=60, color="Risk_Level",
        color_discrete_map=RISK_COLORS, category_orders={"Risk_Level": RISK_ORDER},
        title="Invoice Amount Distribution by Risk Level",
    )
    fig.update_layout(height=420)
    return fig


def plot_vendor_frequency(dataframe: pd.DataFrame, top_n: int = 15):
    counts = dataframe["Vendor_Name"].value_counts().head(top_n).reset_index()
    counts.columns = ["Vendor_Name", "Invoice_Count"]
    fig = px.bar(
        counts, x="Invoice_Count", y="Vendor_Name", orientation="h",
        title=f"Top {top_n} Vendors by Invoice Count", color="Invoice_Count",
        color_continuous_scale="Magma",
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=420)
    return fig


def plot_gst_distribution(dataframe: pd.DataFrame):
    fig = px.histogram(
        dataframe, x="GST", color="Risk_Level", barmode="group",
        color_discrete_map=RISK_COLORS, category_orders={"Risk_Level": RISK_ORDER},
        title="GST Rate Distribution by Risk Level",
    )
    fig.update_layout(height=420)
    return fig


def plot_risk_distribution(dataframe: pd.DataFrame):
    counts = dataframe["Risk_Level"].value_counts().reindex(RISK_ORDER).reset_index()
    counts.columns = ["Risk_Level", "Count"]
    fig = px.bar(
        counts, x="Risk_Level", y="Count", color="Risk_Level",
        color_discrete_map=RISK_COLORS, category_orders={"Risk_Level": RISK_ORDER},
        title="Invoice Count by Risk Level", text="Count",
    )
    fig.update_layout(height=380, showlegend=False)
    return fig


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
if not DATA_PATH.exists():
    st.error(
        f"Could not find `{DATA_PATH}`. Run notebooks 01 through 06 first to "
        "generate `data/processed/invoices_final.csv`."
    )
    st.stop()

df = load_data(DATA_PATH)

# ---------------------------------------------------------------------------
# Sidebar — filters
# ---------------------------------------------------------------------------
st.sidebar.title("🧾 Filters")

departments = sorted(df["Department"].unique())
selected_departments = st.sidebar.multiselect(
    "Department", departments, default=departments
)

risk_levels = st.sidebar.multiselect(
    "Risk Level", RISK_ORDER, default=RISK_ORDER
)

date_min, date_max = df["Invoice_Date"].min(), df["Invoice_Date"].max()
date_range = st.sidebar.date_input(
    "Invoice Date Range", value=(date_min, date_max),
    min_value=date_min, max_value=date_max,
)

st.sidebar.markdown("---")
show_ground_truth = st.sidebar.checkbox(
    "Show ground-truth anomaly labels (demo only)",
    value=False,
    help=(
        "This dataset is synthetic with known injected anomalies (True_Anomaly), "
        "used to validate the model. A real deployment would NOT have this column — "
        "it's shown here only to demonstrate the model's accuracy."
    ),
)

# Apply filters
mask = (
    df["Department"].isin(selected_departments)
    & df["Risk_Level"].isin(risk_levels)
)
if isinstance(date_range, tuple) and len(date_range) == 2:
    start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
    mask &= df["Invoice_Date"].between(start, end)

filtered = df[mask].copy()

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("🧾 AI Bill Anomaly Detection System")
st.caption(
    "Machine Learning–based invoice anomaly detection using Isolation Forest. "
    "Flags high-risk invoices for prioritized manual review."
)

tab_overview, tab_viz, tab_risk, tab_search, tab_stats = st.tabs(
    ["📊 Overview", "📈 Visualizations", "🚨 Risk Analysis", "🔍 Invoice Search", "📋 Statistics"]
)

# ---------------------------------------------------------------------------
# Tab 1: Overview
# ---------------------------------------------------------------------------
with tab_overview:
    st.subheader("Dataset Overview")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Invoices", f"{len(filtered):,}")
    c2.metric("Total Amount", f"₹{filtered['Amount'].sum():,.0f}")
    c3.metric("High Risk Invoices", f"{(filtered['Risk_Level'] == 'High').sum():,}")
    c4.metric(
        "Avg Risk Score",
        f"{filtered['Risk_Score'].mean():.1f}" if len(filtered) else "—",
    )

    st.plotly_chart(plot_risk_distribution(filtered), use_container_width=True)

    st.subheader("Top 10 Highest-Risk Invoices")
    display_cols = ["Invoice_ID", "Vendor_Name", "Department", "Amount",
                     "GST", "Risk_Score", "Risk_Level"]
    if show_ground_truth:
        display_cols.append("True_Anomaly")

    top_risk = filtered.sort_values("Risk_Score", ascending=False).head(10)
    st.dataframe(top_risk[display_cols], use_container_width=True, hide_index=True)

    if show_ground_truth and len(filtered):
        precision_at_10 = top_risk["True_Anomaly"].mean() * 100
        st.caption(
            f"Of these top 10 flagged invoices, {precision_at_10:.0f}% are confirmed "
            "true anomalies (ground truth, demo dataset only)."
        )

# ---------------------------------------------------------------------------
# Tab 2: Visualizations
# ---------------------------------------------------------------------------
with tab_viz:
    st.subheader("Expenditure & Distribution Analysis")

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(plot_department_spend(filtered), use_container_width=True)
    with col2:
        st.plotly_chart(plot_vendor_frequency(filtered), use_container_width=True)

    st.plotly_chart(plot_monthly_trend(filtered), use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.plotly_chart(plot_amount_distribution(filtered), use_container_width=True)
    with col4:
        st.plotly_chart(plot_gst_distribution(filtered), use_container_width=True)

# ---------------------------------------------------------------------------
# Tab 3: Risk Analysis
# ---------------------------------------------------------------------------
with tab_risk:
    st.subheader("Risk Level Breakdown")

    for level in RISK_ORDER:
        subset = filtered[filtered["Risk_Level"] == level]
        pct = len(subset) / len(filtered) * 100 if len(filtered) else 0
        st.markdown(
            f"**{level} Risk** — {len(subset):,} invoices ({pct:.1f}% of filtered data)"
        )
        st.progress(min(pct / 100, 1.0))

    st.markdown("---")
    st.subheader("Invoices by Risk Level")

    chosen_level = st.selectbox("Select risk level to inspect", RISK_ORDER, index=2)
    level_df = filtered[filtered["Risk_Level"] == chosen_level].sort_values(
        "Risk_Score", ascending=False
    )

    display_cols = ["Invoice_ID", "Vendor_Name", "Department", "Category",
                     "Amount", "GST", "Quantity", "Risk_Score", "Risk_Level"]
    if show_ground_truth:
        display_cols.append("True_Anomaly")

    st.dataframe(level_df[display_cols], use_container_width=True, hide_index=True)
    st.caption(f"{len(level_df):,} invoices in {chosen_level} risk bucket")

# ---------------------------------------------------------------------------
# Tab 4: Invoice Search
# ---------------------------------------------------------------------------
with tab_search:
    st.subheader("Search Invoices")

    search_type = st.radio(
        "Search by", ["Invoice ID", "Vendor Name"], horizontal=True
    )

    if search_type == "Invoice ID":
        query = st.text_input("Enter Invoice ID (e.g. INV100001)")
        results = filtered[filtered["Invoice_ID"].str.contains(query, case=False, na=False)] \
            if query else pd.DataFrame(columns=filtered.columns)
    else:
        query = st.text_input("Enter Vendor Name (e.g. Vendor_012)")
        results = filtered[filtered["Vendor_Name"].str.contains(query, case=False, na=False)] \
            if query else pd.DataFrame(columns=filtered.columns)

    if query:
        st.write(f"**{len(results)}** matching invoice(s) found.")
        display_cols = ["Invoice_ID", "Vendor_Name", "Department", "Category",
                         "Invoice_Date", "Amount", "GST", "Quantity",
                         "Risk_Score", "Risk_Level"]
        if show_ground_truth:
            display_cols.append("True_Anomaly")
        st.dataframe(results[display_cols], use_container_width=True, hide_index=True)
    else:
        st.info("Enter a search term above to look up invoices.")

# ---------------------------------------------------------------------------
# Tab 5: Statistics
# ---------------------------------------------------------------------------
with tab_stats:
    st.subheader("Bill Statistics")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Amount Summary**")
        st.dataframe(filtered["Amount"].describe().to_frame("Amount"), use_container_width=True)
    with col2:
        st.markdown("**Risk Score Summary**")
        st.dataframe(filtered["Risk_Score"].describe().to_frame("Risk_Score"), use_container_width=True)

    st.markdown("**Department-wise Summary**")
    dept_summary = filtered.groupby("Department").agg(
        Invoice_Count=("Invoice_ID", "count"),
        Total_Amount=("Amount", "sum"),
        Avg_Amount=("Amount", "mean"),
        High_Risk_Count=("Risk_Level", lambda x: (x == "High").sum()),
    ).sort_values("Total_Amount", ascending=False)
    st.dataframe(dept_summary, use_container_width=True)

    st.markdown("**Category-wise Summary**")
    cat_summary = filtered.groupby("Category").agg(
        Invoice_Count=("Invoice_ID", "count"),
        Total_Amount=("Amount", "sum"),
        Avg_Amount=("Amount", "mean"),
    ).sort_values("Total_Amount", ascending=False)
    st.dataframe(cat_summary, use_container_width=True)

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("---")
st.caption(
    "AI Bill Anomaly Detection System · Isolation Forest · Built for educational/"
    "demonstration purposes as part of an 8-milestone ML portfolio project."
)
