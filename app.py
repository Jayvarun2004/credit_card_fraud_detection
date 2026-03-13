"""
app.py  –  Credit Card Fraud Detection Dashboard (Splunk-style)
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# ── Paths ──────────────────────────────────────────────────────────────────────
MODELS_DIR      = os.path.join(os.path.dirname(__file__), 'models')
MODEL_PATH      = os.path.join(MODELS_DIR, 'xgb_model.pkl')
METRICS_PATH    = os.path.join(MODELS_DIR, 'metrics.json')
FEAT_IMP_PATH   = os.path.join(MODELS_DIR, 'feature_importance.json')
COMPARE_PATH    = os.path.join(MODELS_DIR, 'comparison.json')
DATA_PATH       = os.path.join(os.path.dirname(__file__), 'Data', 'creditcard.csv')

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FraudGuard | Transaction Intelligence",
    page_icon="💠",
    layout="wide",    initial_sidebar_state="expanded"
)

# ── Splunk-style Global CSS ────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&family=Roboto+Mono:wght@400;500&display=swap');

/* ── Reset & Base ── */
html, body, [class*="css"] {
    font-family: 'Roboto', 'Helvetica Neue', sans-serif;
}

/* ── Splunk App Background ── */
.stApp {
    background-color: #1a1c21 !important;
    color: #c3cbd4 !important;
}
.main .block-container {
    padding-top: 1.2rem !important;
    padding-left: 1.5rem !important;
    padding-right: 1.5rem !important;
    max-width: 100% !important;
}

/* ── Splunk Sidebar ── */
[data-testid="stSidebar"] {
    background: #16171e !important;
    border-right: 1px solid #2d2f38 !important;
}
[data-testid="stSidebar"] * { color: #aab0bc !important; }
[data-testid="stSidebar"] .stRadio > label { color: #aab0bc !important; }

/* ── Splunk Panel / Card ── */
.splunk-panel {
    background: #22242e;
    border: 1px solid #2d3040;
    border-radius: 4px;
    padding: 0;
    margin-bottom: 14px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.4);
}
.splunk-panel-header {
    background: #1e2029;
    border-bottom: 1px solid #2d3040;
    padding: 8px 14px;
    font-size: 0.78rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: #7a8190;
    display: flex;
    align-items: center;
    gap: 6px;
}
.splunk-panel-body {
    padding: 14px 16px;
}

/* ── Splunk KPI Single Value ── */
.splunk-kpi {
    background: #22242e;
    border: 1px solid #2d3040;
    border-radius: 4px;
    padding: 16px 18px;
    margin-bottom: 10px;
    position: relative;
    overflow: hidden;
}
.splunk-kpi::before {
    content: '';
    position: absolute;
    top: 0; left: 0; bottom: 0;
    width: 4px;
    background: var(--kpi-color, #0877a6);
}
.splunk-kpi-val {
    font-size: 1.9rem;
    font-weight: 700;
    color: var(--kpi-color, #0877a6);
    line-height: 1.1;
    font-family: 'Roboto Mono', monospace;
}
.splunk-kpi-label {
    font-size: 0.72rem;
    font-weight: 500;
    color: #6b7280;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 5px;
}
.splunk-kpi-trend {
    font-size: 0.7rem;
    color: #6b7280;
    margin-top: 2px;
}

/* ── Alert Severity Badges ── */
.sev-critical { background:#dc4e41; color:#fff; padding:1px 7px; border-radius:2px; font-size:0.72rem; font-weight:600; }
.sev-high     { background:#f8be34; color:#1a1c21; padding:1px 7px; border-radius:2px; font-size:0.72rem; font-weight:600; }
.sev-medium   { background:#0877a6; color:#fff; padding:1px 7px; border-radius:2px; font-size:0.72rem; font-weight:600; }
.sev-ok       { background:#53a051; color:#fff; padding:1px 7px; border-radius:2px; font-size:0.72rem; font-weight:600; }

/* ── Splunk Header Bar ── */
.splunk-topbar {
    background: #16171e;
    border-bottom: 2px solid #0877a6;
    padding: 10px 18px;
    margin-bottom: 18px;
    border-radius: 3px;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.splunk-topbar-title {
    font-size: 1.0rem;
    font-weight: 700;
    color: #e8eaf0;
    letter-spacing: 0.04em;
}
.splunk-topbar-meta {
    font-size: 0.72rem;
    color: #6b7280;
    font-family: 'Roboto Mono', monospace;
}

/* ── Section Divider Label ── */
.splunk-section-label {
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #6b7280;
    border-bottom: 1px solid #2d3040;
    padding-bottom: 5px;
    margin: 18px 0 10px;
}

/* ── Table ── */
.splunk-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.82rem;
}
.splunk-table th {
    background: #1a1c21;
    color: #7a8190;
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    padding: 6px 10px;
    border-bottom: 1px solid #2d3040;
    text-align: left;
}
.splunk-table td {
    padding: 7px 10px;
    border-bottom: 1px solid #252835;
    color: #c3cbd4;
}
.splunk-table tr:hover td { background: #1e2029; }

/* ── Fraud / Legit result ── */
.result-fraud {
    background: rgba(220,78,65,0.12);
    border: 1px solid #dc4e41;
    border-left: 4px solid #dc4e41;
    border-radius: 3px;
    padding: 14px 18px;
    color: #dc4e41;
    font-weight: 700;
    font-size: 1.1rem;
}
.result-legit {
    background: rgba(83,160,81,0.12);
    border: 1px solid #53a051;
    border-left: 4px solid #53a051;
    border-radius: 3px;
    padding: 14px 18px;
    color: #53a051;
    font-weight: 700;
    font-size: 1.1rem;
}

/* ── Streamlit overrides ── */
div[data-testid="stMetric"] {
    background: #22242e !important;
    border: 1px solid #2d3040 !important;
    border-radius: 4px !important;
    padding: 10px 14px !important;
}
div[data-testid="stMetric"] label { color: #6b7280 !important; font-size: 0.72rem !important; text-transform: uppercase; }
div[data-testid="stMetric"] [data-testid="stMetricValue"] { color: #c3cbd4 !important; font-family: 'Roboto Mono', monospace; }

/* Input / button styling */
.stTextInput input, .stNumberInput input, .stSelectbox > div {
    background: #1e2029 !important;
    border: 1px solid #2d3040 !important;
    border-radius: 3px !important;
    color: #c3cbd4 !important;
    font-family: 'Roboto Mono', monospace !important;
}
.stButton > button {
    background: #0877a6 !important;
    color: #fff !important;
    border: none !important;
    border-radius: 3px !important;
    font-size: 0.85rem !important;
    font-weight: 600 !important;
    padding: 7px 20px !important;
    letter-spacing: 0.03em !important;
    transition: background 0.2s ease !important;
}
.stButton > button:hover { background: #0a8fc4 !important; }

/* Form submit */
[data-testid="stFormSubmitButton"] > button {
    background: #0877a6 !important;
    color: #fff !important;
    border: none !important;
    border-radius: 3px !important;
    font-weight: 600 !important;
}

/* Radio buttons in sidebar */
.stRadio > div > label {
    color: #aab0bc !important;
    font-size: 0.85rem !important;
}
</style>
""", unsafe_allow_html=True)

# ── Splunk Color Palette ────────────────────────────────────────────────────────
SPL_BLUE   = "#0877a6"
SPL_GREEN  = "#53a051"
SPL_RED    = "#dc4e41"
SPL_YELLOW = "#f8be34"
SPL_PURPLE = "#9b59b6"
SPL_TEAL   = "#00a4a4"
SPL_ORANGE = "#e57028"
SPL_BG     = "rgba(0,0,0,0)"
SPL_PLOT   = "rgba(30,32,41,0.9)"

CHART_COLORS = [SPL_BLUE, SPL_GREEN, SPL_RED, SPL_YELLOW, SPL_PURPLE, SPL_TEAL, SPL_ORANGE, "#f06292"]

MODEL_COLORS = {
    'XGBoost':             SPL_BLUE,
    'Random Forest':       SPL_GREEN,
    'Gradient Boosting':   SPL_TEAL,
    'AdaBoost':            SPL_YELLOW,
    'Decision Tree':       SPL_RED,
    'Logistic Regression': "#a8ff78",
    'KNN':                 SPL_PURPLE,
    'Naive Bayes':         SPL_ORANGE,
}

def splunk_chart(fig, title=""):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=SPL_BG,
        plot_bgcolor=SPL_PLOT,
        title=dict(text=title, font=dict(size=12, color="#7a8190", family="Roboto"),
                   x=0, xanchor='left', pad=dict(l=0)),
        font=dict(family="Roboto", color="#c3cbd4", size=11),
        margin=dict(l=10, r=10, t=38, b=10),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10, color="#aab0bc")),
        xaxis=dict(gridcolor="#2d3040", linecolor="#2d3040", tickfont=dict(color="#7a8190")),
        yaxis=dict(gridcolor="#2d3040", linecolor="#2d3040", tickfont=dict(color="#7a8190")),
    )
    return fig

def kpi_html(value, label, color=SPL_BLUE, trend=""):
    return f"""
    <div class="splunk-kpi" style="--kpi-color:{color};">
        <div class="splunk-kpi-val">{value}</div>
        <div class="splunk-kpi-label">{label}</div>
        {"<div class='splunk-kpi-trend'>" + trend + "</div>" if trend else ""}
    </div>"""

def panel(header, icon, body_html):
    return f"""
    <div class="splunk-panel">
        <div class="splunk-panel-header">{icon} {header}</div>
        <div class="splunk-panel-body">{body_html}</div>
    </div>"""


# ── Helpers ─────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH): return None
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_metrics():
    if not os.path.exists(METRICS_PATH): return None
    with open(METRICS_PATH) as f: return json.load(f)

@st.cache_data
def load_feature_importance():
    if not os.path.exists(FEAT_IMP_PATH): return None
    with open(FEAT_IMP_PATH) as f: return json.load(f)

@st.cache_data
def load_comparison():
    if not os.path.exists(COMPARE_PATH): return None
    with open(COMPARE_PATH) as f: return json.load(f)

@st.cache_data
def load_dataset(uploaded_file=None):
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    if not os.path.exists(DATA_PATH): return None
    return pd.read_csv(DATA_PATH)

def model_trained() -> bool:
    return os.path.exists(MODEL_PATH)


# ── Sidebar ──────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:18px 10px 12px;'>
        <div style='display:flex;align-items:center;gap:10px;margin-bottom:4px;'>
            <span style='font-size:1.5rem;'>💠</span>
            <div>
                <div style='font-size:0.95rem;font-weight:700;color:#e8eaf0;'>FraudGuard</div>
                <div style='font-size:0.65rem;color:#4b5260;text-transform:uppercase;letter-spacing:0.06em;'>Transaction Intelligence</div>
            </div>
        </div>
    </div>
    <hr style='border:none;border-top:1px solid #2d3040;margin:0 0 14px;'>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        ["🏠  Overview", "📈  Data Upload", "📊  Model Performance", "🔍  Manual Predict", "📁  Batch Predict"],
        label_visibility="collapsed"
    )

    st.markdown("<hr style='border:none;border-top:1px solid #2d3040;margin:14px 0;'>", unsafe_allow_html=True)

    if model_trained():
        st.markdown("<span class='sev-ok'>● MODEL READY</span>", unsafe_allow_html=True)
    else:
        st.markdown("<span class='sev-critical'>● MODEL NOT TRAINED</span>", unsafe_allow_html=True)
        st.caption("Run `python src/train_model.py` first")

    if os.path.exists(DATA_PATH):
        st.markdown("<br><span class='sev-ok'>● DATASET LOADED (LOCAL)</span>", unsafe_allow_html=True)
    elif "uploaded_dataset" in st.session_state:
        st.markdown("<br><span class='sev-ok'>● DATASET LOADED (UPLOADED)</span>", unsafe_allow_html=True)
    else:
        st.markdown("<br><span class='sev-critical'>● DATASET MISSING</span>", unsafe_allow_html=True)

    st.markdown("""
    <div style='margin-top:20px;padding:10px;background:#1a1c21;border:1px solid #2d3040;border-radius:3px;'>
        <div style='font-size:0.65rem;color:#4b5260;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:6px;'>Models in Use</div>
        <div style='font-size:0.72rem;color:#6b7280;line-height:1.7;'>
            XGBoost · Random Forest<br>
            Gradient Boosting · AdaBoost<br>
            Decision Tree · KNN<br>
            Logistic Regression · Naive Bayes
        </div>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 1 — OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🏠  Overview":

    # Top bar
    from datetime import datetime
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.markdown(f"""
    <div class="splunk-topbar">
        <div class="splunk-topbar-title">🛡️ FraudGuard — Transaction Overview Dashboard</div>
        <div class="splunk-topbar-meta">Last updated: {now} &nbsp;|&nbsp; Source: ULB Kaggle Dataset</div>
    </div>
    """, unsafe_allow_html=True)

    if "uploaded_dataset" in st.session_state:
        df = load_dataset(st.session_state["uploaded_dataset"])
    else:
        df = load_dataset()

    if df is None:
        st.warning("⚠️ Dataset not found on server. Please go to the **Data Upload** page to upload the `creditcard.csv` dataset.")
        st.stop()

    total  = len(df)
    fraud  = int(df['Class'].sum())
    legit  = total - fraud
    f_pct  = round(fraud / total * 100, 3)
    avg_am = round(df['Amount'].mean() * 83, 2)
    max_am = round(df['Amount'].max() * 83, 2)

    # ── KPI Row ──────────────────────────────────────────────────────────────
    st.markdown("<div class='splunk-section-label'>📌 Key Performance Indicators</div>", unsafe_allow_html=True)
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.markdown(kpi_html(f"{total:,}", "Total Transactions",    SPL_BLUE,   "All time"), unsafe_allow_html=True)
    k2.markdown(kpi_html(f"{fraud:,}", "Fraud Cases",           SPL_RED,    "Flagged"), unsafe_allow_html=True)
    k3.markdown(kpi_html(f"{legit:,}", "Legitimate Cases",      SPL_GREEN,  "Verified"), unsafe_allow_html=True)
    k4.markdown(kpi_html(f"{f_pct}%", "Fraud Rate",            SPL_YELLOW, "Imbalance ratio"), unsafe_allow_html=True)
    k5.markdown(kpi_html(f"₹{avg_am:,.2f}", "Avg. Amount",           SPL_TEAL,   f"Max: ₹{max_am:,.2f}"), unsafe_allow_html=True)

    # ── Charts Row 1 ─────────────────────────────────────────────────────────
    st.markdown("<div class='splunk-section-label'>📈 Transaction Analysis</div>", unsafe_allow_html=True)
    c1, c2 = st.columns([1, 1.6])

    with c1:
        st.markdown("<div class='splunk-panel'><div class='splunk-panel-header'>🔵 CLASS DISTRIBUTION</div><div class='splunk-panel-body'>", unsafe_allow_html=True)
        fig_pie = go.Figure(go.Pie(
            labels=["Legitimate", "Fraud"],
            values=[legit, fraud],
            hole=0.65,
            marker=dict(colors=[SPL_GREEN, SPL_RED],
                        line=dict(color="#1a1c21", width=3)),
            textinfo="percent",
            textfont=dict(size=12, color="white"),
            hovertemplate="<b>%{label}</b><br>Count: %{value:,}<extra></extra>"
        ))
        fig_pie.add_annotation(
            text=f"<b>{f_pct}%</b><br><span style='font-size:10px'>FRAUD</span>",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color=SPL_RED)
        )
        splunk_chart(fig_pie, "Class Distribution")
        fig_pie.update_layout(showlegend=True, height=280, margin=dict(l=0,r=0,t=35,b=0))
        st.plotly_chart(fig_pie, width='stretch')
        st.markdown("</div></div>", unsafe_allow_html=True)

    with c2:
        st.markdown("<div class='splunk-panel'><div class='splunk-panel-header'>📊 TRANSACTION AMOUNT DISTRIBUTION</div><div class='splunk-panel-body'>", unsafe_allow_html=True)
        df_sample = df.sample(min(10000, len(df)), random_state=42)
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=df_sample[df_sample['Class']==0]['Amount'] * 83,
            name="Legitimate", nbinsx=80, opacity=0.8,
            marker_color=SPL_GREEN, hovertemplate="Amount: ₹%{x:.2f}<br>Count: %{y}<extra>Legitimate</extra>"
        ))
        fig_hist.add_trace(go.Histogram(
            x=df_sample[df_sample['Class']==1]['Amount'] * 83,
            name="Fraud", nbinsx=80, opacity=0.9,
            marker_color=SPL_RED, hovertemplate="Amount: ₹%{x:.2f}<br>Count: %{y}<extra>Fraud</extra>"
        ))
        fig_hist.update_layout(barmode='overlay')
        splunk_chart(fig_hist, "Transaction Amount by Class")
        fig_hist.update_layout(height=280, xaxis_title="Amount (₹)", yaxis_title="Count")
        st.plotly_chart(fig_hist, width='stretch')
        st.markdown("</div></div>", unsafe_allow_html=True)

    # ── Charts Row 2 ─────────────────────────────────────────────────────────
    c3, c4 = st.columns([1, 1])
    with c3:
        st.markdown("<div class='splunk-panel'><div class='splunk-panel-header'>📦 AMOUNT BOX PLOT BY CLASS</div><div class='splunk-panel-body'>", unsafe_allow_html=True)
        labels = df_sample['Class'].map({0: "Legitimate", 1: "Fraud"})
        fig_box = go.Figure()
        fig_box.add_trace(go.Box(
            y=df_sample[df_sample['Class']==0]['Amount'] * 83,
            name="Legitimate", marker_color=SPL_GREEN,
            boxmean=True, line=dict(color=SPL_GREEN)
        ))
        fig_box.add_trace(go.Box(
            y=df_sample[df_sample['Class']==1]['Amount'] * 83,
            name="Fraud", marker_color=SPL_RED,
            boxmean=True, line=dict(color=SPL_RED)
        ))
        splunk_chart(fig_box, "Amount Distribution — Box Plot")
        fig_box.update_layout(height=280, yaxis_title="Amount (₹)")
        st.plotly_chart(fig_box, width='stretch')
        st.markdown("</div></div>", unsafe_allow_html=True)

    with c4:
        st.markdown("<div class='splunk-panel'><div class='splunk-panel-header'>🔗 TOP FEATURES CORRELATED WITH FRAUD</div><div class='splunk-panel-body'>", unsafe_allow_html=True)
        corr = df.corr(numeric_only=True)['Class'].drop('Class').abs().sort_values(ascending=False).head(10)
        fig_corr = go.Figure(go.Bar(
            y=corr.index, x=corr.values,
            orientation='h',
            marker=dict(
                color=corr.values,
                colorscale=[[0, SPL_BLUE], [0.5, SPL_TEAL], [1, SPL_RED]],
                showscale=False
            ),
            text=[f"{v:.3f}" for v in corr.values],
            textposition='outside',
            textfont=dict(size=9, color="#7a8190")
        ))
        splunk_chart(fig_corr, "Feature Correlation with Fraud (Top 10)")
        fig_corr.update_layout(height=280, yaxis=dict(autorange="reversed"), xaxis_title="Abs. Correlation")
        st.plotly_chart(fig_corr, width='stretch')
        st.markdown("</div></div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 1.5 — DATA UPLOAD
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📈  Data Upload":
    from datetime import datetime
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.markdown(f"""
    <div class="splunk-topbar">
        <div class="splunk-topbar-title">📈 Data Upload Center</div>
        <div class="splunk-topbar-meta">{now}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style='background:#1e2029;border:1px solid #2d3040;border-left:3px solid #0877a6;
         border-radius:3px;padding:14px 18px;font-size:0.85rem;color:#c3cbd4;margin-bottom:20px;'>
        <h3 style='margin-top:0;color:#e8eaf0;font-size:1.1rem;'>Upload Training Dataset</h3>
        <p style='color:#7a8190;margin-bottom:0;'>If the application is deployed on a cloud server where the <code>creditcard.csv</code> dataset is not available locally, you can upload it here. 
        Once uploaded, the dataset will be stored in your session, enabling the <b>Overview</b> and <b>Model Performance</b> pages to function correctly.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='splunk-panel'><div class='splunk-panel-header'>📁 UPLOAD CREDITCARD.CSV</div><div class='splunk-panel-body'>", unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Select the dataset file (CSV format)", type=["csv"], key="dedicated_dataset_upload")
    
    if uploaded_file:
        st.session_state["uploaded_dataset"] = uploaded_file
        st.success("✅ Dataset successfully uploaded and stored in session memory!")
        st.markdown("<br><p style='color:#53a051;font-weight:bold;'>You can now navigate to the Overview or Model Performance pages.</p>", unsafe_allow_html=True)
    elif "uploaded_dataset" in st.session_state:
        st.info("ℹ️ A dataset is currently loaded in your session. Uploading a new file will replace it.")
    
    st.markdown("</div></div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 2 — MODEL PERFORMANCE
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📊  Model Performance":
    from datetime import datetime
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.markdown(f"""
    <div class="splunk-topbar">
        <div class="splunk-topbar-title">📊 Model Performance &amp; Comparison</div>
        <div class="splunk-topbar-meta">{now}</div>
    </div>
    """, unsafe_allow_html=True)

    if not model_trained():
        st.error("❌ No trained model found. Run `python src/train_model.py` first.")
        st.stop()

    comparison = load_comparison()
    feat_imp   = load_feature_importance()

    MODEL_NAMES = list(MODEL_COLORS.keys())

    if comparison:
        available = [m['model'] for m in comparison]
    else:
        available = MODEL_NAMES[:1]

    col_sel, _ = st.columns([1.2, 3])
    with col_sel:
        selected_model = st.selectbox("Select model:", available, index=0)

    if comparison:
        metrics = next((m for m in comparison if m['model'] == selected_model), comparison[0])
    else:
        metrics = load_metrics()

    sel_color = MODEL_COLORS.get(selected_model, SPL_BLUE)

    # ── KPI Row ───────────────────────────────────────────────────────────────
    st.markdown("<div class='splunk-section-label'>📌 Selected Model Metrics</div>", unsafe_allow_html=True)
    kpi_data = [
        ("accuracy",  "Accuracy",  SPL_BLUE),
        ("precision", "Precision", SPL_TEAL),
        ("recall",    "Recall",    SPL_YELLOW),
        ("f1_score",  "F1 Score",  SPL_GREEN),
        ("roc_auc",   "ROC-AUC",   SPL_PURPLE),
    ]
    kpi_cols = st.columns(5)
    for col, (key, lbl, clr) in zip(kpi_cols, kpi_data):
        sev = "sev-ok" if metrics[key] >= 0.9 else "sev-high" if metrics[key] >= 0.75 else "sev-critical"
        col.markdown(kpi_html(f"{metrics[key]*100:.2f}%", lbl, clr), unsafe_allow_html=True)

    # ── Model Comparison ─────────────────────────────────────────────────────
    if comparison and len(comparison) > 1:
        st.markdown("<div class='splunk-section-label'>🏆 All Models Comparison</div>", unsafe_allow_html=True)
        comp_df = pd.DataFrame(comparison)
        metric_cols   = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC']

        cmp1, cmp2 = st.columns([1.5, 1])

        with cmp1:
            st.markdown("<div class='splunk-panel'><div class='splunk-panel-header'>📊 GROUPED BAR — ALL MODELS</div><div class='splunk-panel-body'>", unsafe_allow_html=True)
            fig_cmp = go.Figure()
            for i, row in comp_df.iterrows():
                clr = MODEL_COLORS.get(row['model'], CHART_COLORS[i % len(CHART_COLORS)])
                fig_cmp.add_trace(go.Bar(
                    name=row['model'], x=metric_labels,
                    y=[row[c] * 100 for c in metric_cols],
                    marker_color=clr,
                    text=[f"{row[c]*100:.1f}" for c in metric_cols],
                    textposition='outside', textfont=dict(size=8)
                ))
            fig_cmp.update_layout(barmode='group', yaxis=dict(range=[0, 115]))
            splunk_chart(fig_cmp, "Metric Comparison — All Models")
            fig_cmp.update_layout(height=320)
            st.plotly_chart(fig_cmp, width='stretch')
            st.markdown("</div></div>", unsafe_allow_html=True)

        with cmp2:
            st.markdown("<div class='splunk-panel'><div class='splunk-panel-header'>🕸️ RADAR CHART</div><div class='splunk-panel-body'>", unsafe_allow_html=True)
            fig_radar = go.Figure()
            for i, row in comp_df.iterrows():
                clr = MODEL_COLORS.get(row['model'], CHART_COLORS[i % len(CHART_COLORS)])
                r_int, g_int, b_int = int(clr[1:3],16), int(clr[3:5],16), int(clr[5:7],16)
                vals = [row[c]*100 for c in metric_cols] + [row[metric_cols[0]]*100]
                fig_radar.add_trace(go.Scatterpolar(
                    r=vals, theta=metric_labels + [metric_labels[0]],
                    fill='toself', name=row['model'],
                    line=dict(color=clr, width=1.5),
                    fillcolor=f"rgba({r_int},{g_int},{b_int},0.1)"
                ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[60,100], ticksuffix='%', gridcolor="#2d3040"),
                           bgcolor='rgba(0,0,0,0)'),
                showlegend=True
            )
            splunk_chart(fig_radar, "Radar — Model Metrics")
            fig_radar.update_layout(height=320)
            st.plotly_chart(fig_radar, width='stretch')
            st.markdown("</div></div>", unsafe_allow_html=True)

        # Comparison Table
        st.markdown("<div class='splunk-section-label'>📋 Full Comparison Table</div>", unsafe_allow_html=True)
        display_df = comp_df[['model'] + metric_cols].copy()
        best_auc = comp_df['roc_auc'].max()
        for c in metric_cols:
            display_df[c] = display_df[c].apply(lambda x: f"{x*100:.2f}%")
        display_df.columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC']
        st.dataframe(display_df.style.set_properties(**{'background-color': '#22242e', 'color': '#c3cbd4'}),
                     width='stretch', hide_index=True)

    # ── Detailed Analysis ─────────────────────────────────────────────────────
    st.markdown(f"<div class='splunk-section-label'>🔬 {selected_model} — Detailed View</div>", unsafe_allow_html=True)
    det1, det2 = st.columns([1, 1])

    with det1:
        st.markdown("<div class='splunk-panel'><div class='splunk-panel-header'>🟥 CONFUSION MATRIX</div><div class='splunk-panel-body'>", unsafe_allow_html=True)
        cm = metrics['confusion_matrix']
        tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
        fig_cm = px.imshow(
            [[tn, fp], [fn, tp]],
            x=["Predicted Legit", "Predicted Fraud"],
            y=["Actual Legit", "Actual Fraud"],
            color_continuous_scale=[[0,"#1a1c21"],[0.5,sel_color],[1,"#ffffff"]],
            text_auto=True, aspect="auto"
        )
        fig_cm.update_traces(textfont=dict(size=16, color="white"))
        splunk_chart(fig_cm, f"Confusion Matrix — {selected_model}")
        fig_cm.update_layout(height=280, coloraxis_showscale=False)
        st.plotly_chart(fig_cm, width='stretch')
        m1,m2,m3,m4 = st.columns(4)
        m1.metric("True Neg",  f"{tn:,}")
        m2.metric("False Pos", f"{fp:,}")
        m3.metric("False Neg", f"{fn:,}")
        m4.metric("True Pos",  f"{tp:,}")
        st.markdown("</div></div>", unsafe_allow_html=True)

    with det2:
        if feat_imp and selected_model in ['XGBoost', 'Random Forest']:
            st.markdown("<div class='splunk-panel'><div class='splunk-panel-header'>🌟 FEATURE IMPORTANCE (TOP 15)</div><div class='splunk-panel-body'>", unsafe_allow_html=True)
            top15 = dict(list(feat_imp.items())[:15])
            fig_fi = go.Figure(go.Bar(
                x=list(top15.values()), y=list(top15.keys()),
                orientation='h',
                marker=dict(color=list(top15.values()),
                            colorscale=[[0,SPL_BLUE],[0.5,SPL_TEAL],[1,SPL_GREEN]],
                            showscale=False),
                text=[f"{v:.4f}" for v in top15.values()],
                textposition='outside', textfont=dict(size=8, color="#7a8190")
            ))
            splunk_chart(fig_fi, f"Feature Importances — {selected_model}")
            fig_fi.update_layout(height=320, yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig_fi, width='stretch')
            st.markdown("</div></div>", unsafe_allow_html=True)
        else:
            # AUC Gauge
            st.markdown("<div class='splunk-panel'><div class='splunk-panel-header'>🎯 ROC-AUC GAUGE</div><div class='splunk-panel-body'>", unsafe_allow_html=True)
            auc = metrics['roc_auc']
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=auc * 100,
                number={'suffix': '%', 'font': {'size': 40, 'color': sel_color}},
                delta={'reference': 90, 'suffix': '%'},
                gauge={
                    'axis': {'range': [50, 100], 'tickcolor': '#7a8190', 'tickfont': dict(color='#7a8190')},
                    'bar': {'color': sel_color, 'thickness': 0.25},
                    'bgcolor': "rgba(0,0,0,0)",
                    'bordercolor': "#2d3040",
                    'steps': [
                        {'range': [50, 70], 'color': 'rgba(220,78,65,0.2)'},
                        {'range': [70, 85], 'color': 'rgba(248,190,52,0.2)'},
                        {'range': [85, 100], 'color': 'rgba(83,160,81,0.2)'},
                    ],
                    'threshold': {'line': {'color': sel_color, 'width': 3}, 'value': auc*100}
                },
                title={'text': f"ROC-AUC — {selected_model}", 'font': {'color': '#7a8190', 'size': 12}}
            ))
            fig_gauge.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Roboto", color="#c3cbd4"),
                height=280, margin=dict(l=20,r=20,t=40,b=10)
            )
            st.plotly_chart(fig_gauge, width='stretch')
            st.markdown("</div></div>", unsafe_allow_html=True)

    # ── ROC Curves ────────────────────────────────────────────────────────────
    st.markdown("<div class='splunk-section-label'>📈 ROC Curves</div>", unsafe_allow_html=True)
    st.markdown("<div class='splunk-panel'><div class='splunk-panel-header'>📈 ROC CURVES — ALL MODELS</div><div class='splunk-panel-body'>", unsafe_allow_html=True)
    fig_roc = go.Figure()
    models_to_plot = comparison if comparison and len(comparison) > 1 else [metrics]
    for i, m in enumerate(models_to_plot):
        auc_v = m['roc_auc']
        name  = m.get('model', 'Model')
        clr   = MODEL_COLORS.get(name, CHART_COLORS[i % len(CHART_COLORS)])
        r_int, g_int, b_int = int(clr[1:3],16), int(clr[3:5],16), int(clr[5:7],16)
        t = np.linspace(0, 1, 200)
        tpr = np.clip(np.where(t < 0.5, 2*auc_v*t, 1 - 2*(1-auc_v)*(1-t)), 0, 1)
        fig_roc.add_trace(go.Scatter(
            x=t, y=tpr, fill='tozeroy',
            fillcolor=f"rgba({r_int},{g_int},{b_int},0.05)",
            line=dict(color=clr, width=2),
            name=f"{name} (AUC={auc_v:.4f})"
        ))
    fig_roc.add_trace(go.Scatter(
        x=[0,1], y=[0,1],
        line=dict(color="rgba(255,255,255,0.15)", dash="dash", width=1),
        name="Random Baseline"
    ))
    fig_roc.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
    splunk_chart(fig_roc, "ROC Curves — All Models")
    fig_roc.update_layout(height=340)
    st.plotly_chart(fig_roc, width='stretch')
    st.markdown("</div></div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 3 — MANUAL PREDICT
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔍  Manual Predict":
    from datetime import datetime
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.markdown(f"""
    <div class="splunk-topbar">
        <div class="splunk-topbar-title">🔍 Manual Transaction Prediction</div>
        <div class="splunk-topbar-meta">{now}</div>
    </div>
    """, unsafe_allow_html=True)

    if not model_trained():
        st.error("❌ No trained model found. Run `python src/train_model.py` first.")
        st.stop()

    model = load_model()

    st.markdown("""
    <div style='background:#1e2029;border:1px solid #2d3040;border-left:3px solid #0877a6;
         border-radius:3px;padding:10px 14px;font-size:0.82rem;color:#7a8190;margin-bottom:14px;'>
        ℹ️ Enter transaction details below. V1–V28 are PCA-anonymised features from the Kaggle ULB dataset.
        Leave as <code>0.0</code> for a typical baseline transaction.
    </div>
    """, unsafe_allow_html=True)

    with st.form("predict_form"):
        st.markdown("<div class='splunk-section-label'>💰 Transaction Amount</div>", unsafe_allow_html=True)
        # Note: The underlying model is trained on EUR/USD amounts from Kaggle dataset.
        # We accept INR input from the user but scale it back down before prediction.
        amount_inr = st.number_input("Amount (₹)", min_value=0.0, value=12500.0, step=100.0)
        amount_usd = amount_inr / 83.0

        st.markdown("<div class='splunk-section-label'>🔢 PCA Feature Inputs (V1 – V28)</div>", unsafe_allow_html=True)
        v_values = {}
        cols_per_row = 4
        v_range = range(1, 29)
        rows = [list(v_range)[i:i+cols_per_row] for i in range(0, 28, cols_per_row)]
        for row in rows:
            cols = st.columns(cols_per_row)
            for col, vn in zip(cols, row):
                v_values[f'V{vn}'] = col.number_input(
                    f"V{vn}", value=0.0, format="%.4f",
                    key=f"v{vn}", label_visibility="visible"
                )

        submitted = st.form_submit_button("⚡ Run Prediction")

    if submitted:
        from src.predict import predict_single
        features = {**v_values, 'Amount': amount_usd}
        prob, label = predict_single(features, model=model)

        st.markdown("<div class='splunk-section-label'>🎯 Prediction Result</div>", unsafe_allow_html=True)

        res1, res2, res3 = st.columns([1, 1.5, 1])
        with res2:
            is_fraud = "Fraud" in label
            css_cls  = "result-fraud" if is_fraud else "result-legit"
            icon     = "🚨" if is_fraud else "✅"
            sev_cls  = "sev-critical" if is_fraud else "sev-ok"

            st.markdown(f"""
            <div class="{css_cls}" style="text-align:center; margin-bottom:14px;">
                {icon} {label}
            </div>
            <div style="text-align:center;margin-bottom:8px;">
                <span style="font-size:0.72rem;color:#6b7280;text-transform:uppercase;letter-spacing:0.07em;">Fraud Probability</span><br>
                <span style="font-size:3rem;font-weight:700;font-family:'Roboto Mono',monospace;
                      color:{'#dc4e41' if is_fraud else '#53a051'};">
                    {prob*100:.2f}%
                </span>
            </div>
            """, unsafe_allow_html=True)

            # Bullet gauge
            fig_g = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                number={'suffix': '%', 'font': {'size': 32, 'color': SPL_RED if is_fraud else SPL_GREEN}},
                gauge={
                    'axis': {'range': [0, 100], 'tickcolor': '#7a8190'},
                    'bar': {'color': SPL_RED if is_fraud else SPL_GREEN, 'thickness': 0.22},
                    'bgcolor': "rgba(0,0,0,0)",
                    'bordercolor': "#2d3040",
                    'steps': [
                        {'range': [0, 30],  'color': 'rgba(83,160,81,0.2)'},
                        {'range': [30, 60], 'color': 'rgba(248,190,52,0.2)'},
                        {'range': [60, 100],'color': 'rgba(220,78,65,0.2)'},
                    ],
                    'threshold': {'line': {'color': SPL_RED if is_fraud else SPL_GREEN, 'width': 3},
                                  'value': prob*100}
                }
            ))
            fig_g.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Roboto", color="#c3cbd4"),
                height=200, margin=dict(l=10,r=10,t=10,b=10)
            )
            st.plotly_chart(fig_g, width='stretch')

            # Risk level table
            risk = "CRITICAL" if prob > 0.7 else "HIGH" if prob > 0.4 else "MEDIUM" if prob > 0.15 else "LOW"
            sev_map = {"CRITICAL": "sev-critical", "HIGH": "sev-high", "MEDIUM": "sev-medium", "LOW": "sev-ok"}
            st.markdown(f"""
            <table class="splunk-table" style="margin-top:10px;">
                <tr><th>Field</th><th>Value</th></tr>
                <tr><td>Risk Level</td><td><span class="{sev_map[risk]}">{risk}</span></td></tr>
                <tr><td>Fraud probability</td><td style="font-family:'Roboto Mono',monospace;">{prob*100:.4f}%</td></tr>
                <tr><td>Transaction Amount</td><td style="font-family:'Roboto Mono',monospace;">₹{amount_inr:,.2f}</td></tr>
                <tr><td>Decision Threshold</td><td style="font-family:'Roboto Mono',monospace;">50.0%</td></tr>
            </table>
            """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 4 — BATCH PREDICT
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📁  Batch Predict":
    from datetime import datetime
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.markdown(f"""
    <div class="splunk-topbar">
        <div class="splunk-topbar-title">📁 Batch Transaction Prediction</div>
        <div class="splunk-topbar-meta">{now}</div>
    </div>
    """, unsafe_allow_html=True)

    if not model_trained():
        st.error("❌ No trained model found. Run `python src/train_model.py` first.")
        st.stop()

    model = load_model()

    st.markdown("""
    <div style='background:#1e2029;border:1px solid #2d3040;border-left:3px solid #0877a6;
         border-radius:3px;padding:10px 14px;font-size:0.82rem;color:#7a8190;margin-bottom:14px;'>
        ℹ️ Upload a CSV with columns <code>V1–V28</code> + <code>Amount</code>.
        Columns like <code>Time</code> and <code>Class</code> are automatically dropped.
    </div>
    """, unsafe_allow_html=True)

    sample_data = {f'V{i}': [0.0]*5 for i in range(1, 29)}
    sample_data['Amount'] = [10.0, 450.0, 1200.0, 5.0, 789.0]
    sample_df = pd.DataFrame(sample_data)

    st.download_button(
        "⬇️ Download Sample CSV Template",
        data=sample_df.to_csv(index=False),
        file_name="sample_transactions.csv",
        mime="text/csv"
    )

    uploaded_file = st.file_uploader("Upload Transactions CSV", type=["csv"])

    if uploaded_file:
        try:
            df_upload = pd.read_csv(uploaded_file)
            st.markdown(f"<div class='splunk-section-label'>📄 Loaded {len(df_upload):,} transactions</div>", unsafe_allow_html=True)
            st.dataframe(df_upload.head(), width='stretch')

            with st.spinner("⚡ Running batch predictions…"):
                from src.predict import predict_transactions
                results = predict_transactions(df_upload, model=model)

            fraud_count = int((results['prediction'] == 1).sum())
            legit_count = int((results['prediction'] == 0).sum())

            st.markdown("<div class='splunk-section-label'>📌 Batch Summary</div>", unsafe_allow_html=True)
            b1, b2, b3 = st.columns(3)
            b1.markdown(kpi_html(f"{len(results):,}", "Total Processed", SPL_BLUE), unsafe_allow_html=True)
            b2.markdown(kpi_html(f"{fraud_count:,}", "Flagged as Fraud", SPL_RED,
                                 f"{fraud_count/len(results)*100:.2f}% of total"), unsafe_allow_html=True)
            b3.markdown(kpi_html(f"{legit_count:,}", "Legitimate", SPL_GREEN,
                                 f"{legit_count/len(results)*100:.2f}% of total"), unsafe_allow_html=True)

            st.markdown("<div class='splunk-section-label'>📊 Results</div>", unsafe_allow_html=True)
            pie_col, tbl_col = st.columns([1, 1.6])
            with pie_col:
                st.markdown("<div class='splunk-panel'><div class='splunk-panel-header'>🔵 PREDICTION SPLIT</div><div class='splunk-panel-body'>", unsafe_allow_html=True)
                fig_bp = go.Figure(go.Pie(
                    labels=["Legitimate", "Fraud"],
                    values=[legit_count, fraud_count],
                    hole=0.6,
                    marker=dict(colors=[SPL_GREEN, SPL_RED], line=dict(color="#1a1c21", width=2)),
                    textinfo="percent+label",
                ))
                splunk_chart(fig_bp, "Prediction Distribution")
                fig_bp.update_layout(height=260, margin=dict(l=0,r=0,t=35,b=0))
                st.plotly_chart(fig_bp, width='stretch')
                st.markdown("</div></div>", unsafe_allow_html=True)

            with tbl_col:
                st.markdown("<div class='splunk-panel'><div class='splunk-panel-header'>📋 TOP FRAUD TRANSACTIONS</div><div class='splunk-panel-body'>", unsafe_allow_html=True)
                if 'Amount' in results.columns:
                    results['Amount_INR'] = results['Amount'] * 83
                    display_cols = ['fraud_probability', 'label', 'Amount_INR']
                else:
                    display_cols = ['fraud_probability', 'label']
                    
                st.dataframe(
                    results[display_cols].sort_values('fraud_probability', ascending=False),
                    width='stretch', height=260
                )
                st.markdown("</div></div>", unsafe_allow_html=True)

            st.download_button(
                "⬇️ Download Full Results CSV",
                data=results.to_csv(index=False),
                file_name="fraud_predictions.csv",
                mime="text/csv"
            )

        except Exception as e:
            st.error(f"❌ Error processing file: {e}")
