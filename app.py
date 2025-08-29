import streamlit as st
import pandas as pd
import numpy as np
import joblib
import statsmodels.api as sm
from tensorflow.keras.models import load_model
import base64
import plotly.express as px

def _image_to_data_uri(image_path: str, mime: str = "image/png") -> str:
    try:
        with open(image_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
        return f"data:{mime};base64,{encoded}"
    except Exception:
        return image_path  # fallback to path if embedding fails

# -----------------------------
# Load Models & Scaler
# -----------------------------
@st.cache_resource
def load_models():
    mlr_data = joblib.load("mlr_model.pkl")
    mlr_model = mlr_data["model"]
    mlr_features = mlr_data["feature_names"]
    scaler = mlr_data["scaler"]

    ann_data = joblib.load("ann_model.joblib")
    ann_model = load_model("ann_model.keras")
    ann_input_shape = ann_data["input_shape"]

    return mlr_model, mlr_features, scaler, ann_model, ann_input_shape

mlr_model, mlr_features, scaler, ann_model, ann_input_shape = load_models()

# -----------------------------
# Streamlit Config
# -----------------------------
st.set_page_config(
    page_title="TTU Student GPA Predictor", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -----------------------------
# Custom CSS - Force Light Theme & Professional Styling
# -----------------------------
st.markdown(
    """
    <style>
        /* Force light theme and override dark mode */
        .main {
            background: #f8f9fa !important;
            color: #212529 !important;
        }
        
        /* Override Streamlit's dark mode */
        [data-testid="stAppViewContainer"] {
            background: #f8f9fa !important;
        }
        
        /* Force all text to be visible */
        .stMarkdown, .stText, .stMetric, .stRadio, .stNumberInput, .stButton, .stFileUploader {
            color: #212529 !important;
        }
        
        /* Header banner */
        .header-banner {
            background: linear-gradient(135deg, #f59e0b, #fbbf24);
            padding: 2.4rem;
            text-align: center;
            color: #1f2937;
            border-radius: 16px;
            margin-bottom: 2rem;
            box-shadow: 0 8px 32px rgba(245, 158, 11, 0.4);
            border: 3px solid #d97706;
            position: relative;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 1rem;
        }
        
        .header-banner .text-content {
            flex: 1;
            text-align: center;
        }
        
        .header-banner h1 {
            margin: 0;
            font-size: 3.2rem;
            font-weight: 900;
            text-shadow: 0 2px 4px rgba(0,0,0,0.2);
            letter-spacing: 2px;
            color: #1f2937;
        }
        
        .header-banner h3 {
            margin: 1rem 0 0 0;
            font-size: 1.85rem;
            font-weight: 700;
            opacity: 1;
            text-shadow: 0 1px 2px rgba(0,0,0,0.2);
            color: #374151;
        }

        /* Responsive text sizing */
        @media (max-width: 1200px) {
            .header-banner h1 { font-size: 2.8rem; }
            .header-banner h3 { font-size: 1.65rem; }
        }
        @media (max-width: 992px) {
            .header-banner { padding: 1.9rem; }
            .header-banner h1 { font-size: 2.4rem; }
            .header-banner h3 { font-size: 1.45rem; }
        }
        @media (max-width: 640px) {
            .header-banner { padding: 1.25rem; }
            .header-banner h1 { font-size: 1.75rem; letter-spacing: 1px; }
            .header-banner h3 { font-size: 1.2rem; }
        }

        /* Dashboard title */
        .dashboard-title {
            text-align: center;
            color: #1e3a8a;
            font-size: 2rem;
            font-weight: 700;
            margin: 1.5rem 0;
            text-shadow: 0 1px 2px rgba(0,0,0,0.1);
        }

        /* (all your other CSS remains unchanged...) */
    </style>
    """,
    unsafe_allow_html=True,
)

# Header banner WITHOUT logos
st.markdown(
    """
    <div class="header-banner">
        <div class="text-content">
            <h1>TAKORADI TECHNICAL UNIVERSITY</h1>
            <h3>Student Final CGPA Predictor</h3>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <style>
    /* Target the whole metric block */
    div[data-testid="stMetric"] {
        text-align: center;
    }

    /* First line inside metric (label) */
    div[data-testid="stMetric"] label, 
    div[data-testid="stMetric"] p {
        font-size: 28px !important;
        font-weight: 700 !important;
        color: #111827 !important;
    }

    /* Value text inside metric */
    div[data-testid="stMetric"] > div > div {
        font-size: 26px !important;
        font-weight: 900 !important;
        color: #1e3a8a !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# Dashboard Rendering Function
# -----------------------------
def render_dashboard(df=None, mlr_metrics=None, ann_metrics=None):
    st.markdown('<h2 class="dashboard-title">Student Performance Dashboard</h2>', unsafe_allow_html=True)

    # Calculate stats if df exists
    total_students = df.shape[0] if df is not None else 0
    best_model = "MLR" if mlr_metrics and mlr_metrics["R2"] > (ann_metrics["R2"] if ann_metrics else -1) else "ANN"
    best_acc = max(mlr_metrics["R2"], ann_metrics["R2"]) if (mlr_metrics and ann_metrics) else 0

    col1, col2, col3, col4 = st.columns(4)
    with col1: 
        st.metric("Total Students Analyzed", total_students)
    with col2: 
        st.metric("Available Models", "2 (MLR, ANN)")
    with col3: 
        st.metric("Best Accuracy", f"{best_acc:.2%} ({best_model})")
    with col4: 
        st.metric("Last Training Run", "2 days ago")

# -----------------------------
# Load Default Dataset (Optional)
# -----------------------------
try:
    base_df = pd.read_csv("Dataset.csv")
except:
    base_df = None

mlr_metrics = {"R2": 0.93, "MAE": 0.051, "RMSE": 0.071}
ann_metrics = {"R2": -0.597, "MAE": 0.293, "RMSE": 0.339}

render_dashboard(base_df, mlr_metrics, ann_metrics)

# -----------------------------
# Mode Selection
# -----------------------------
st.markdown("<br>", unsafe_allow_html=True)
mode = st.radio(
    "Choose Prediction Mode:",
    ["Single Prediction", "Batch Prediction (CSV)"],
    horizontal=True
)

# -----------------------------
# SINGLE PREDICTION
# -----------------------------
if "Single" in mode:
    st.subheader("Enter Student Marks for Prediction")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        social = st.number_input("Social Studies", 0.0, 100.0, 72.0, format="%.1f")
    with c2:
        science = st.number_input("Integrated Science", 0.0, 100.0, 75.0, format="%.1f")
    with c3:
        english = st.number_input("English Language", 0.0, 100.0, 59.0, format="%.1f")
    with c4:
        maths = st.number_input("Mathematics", 0.0, 100.0, 85.0, format="%.1f")

    if st.button("Predict Final CGPA"):
        input_df = pd.DataFrame([{
            "Social Studies": social,
            "Integrated Science": science,
            "English Language": english,
            "Mathematics": maths
        }])

        mlr_pred = mlr_model.predict(sm.add_constant(input_df[mlr_features], has_constant="add"))[0]
        input_for_ann_scaling = pd.DataFrame(0.0, index=[0], columns=scaler.feature_names_in_)
        input_for_ann_scaling["Social Studies"] = social
        input_for_ann_scaling["Integrated Science"] = science
        input_for_ann_scaling["English Language"] = english
        input_for_ann_scaling["Mathematics"] = maths
        ann_input_scaled = scaler.transform(input_for_ann_scaling)
        ann_pred = ann_model.predict(ann_input_scaled, verbose=0)[0][0]

        st.markdown(
            f'<div class="result-row">'
            f'<div class="result-card">MLR Prediction<div class="prediction-value">{mlr_pred:.2f}</div></div>'
            f'<div class="result-card">ANN Prediction<div class="prediction-value">{ann_pred:.2f}</div></div>'
            f'</div>',
            unsafe_allow_html=True
        )

# -----------------------------
# BATCH PREDICTION
# -----------------------------
else:
    st.subheader("Upload CSV for Batch Prediction")
    file = st.file_uploader("Upload CSV file", type=["csv"])

    if file:
        df_in = pd.read_csv(file)
        st.write("### Preview Data", df_in.head())

        mlr_preds = mlr_model.predict(sm.add_constant(df_in[mlr_features], has_constant="add")).ravel()
        df_for_ann_scaling = pd.DataFrame(0.0, index=df_in.index, columns=scaler.feature_names_in_)
        for col in df_in.columns:
            if col in df_for_ann_scaling.columns:
                df_for_ann_scaling[col] = df_in[col]
        ann_preds = ann_model.predict(scaler.transform(df_for_ann_scaling), verbose=0).ravel()

        out = df_in.copy()
        out["FGPA_MLR_Pred"] = mlr_preds
        out["FGPA_ANN_Pred"] = ann_preds

        format_dict = {col: "{:.1f}" for col in ["Social Studies", "Integrated Science", "English Language", "Mathematics"]}
        format_dict.update({"FGPA_MLR_Pred": "{:.2f}", "FGPA_ANN_Pred": "{:.2f}"})

        st.success("Batch prediction complete")
        st.dataframe(out.head(20).style.format(format_dict))

        st.download_button(
            "Download Predictions CSV",
            data=out.to_csv(index=False, float_format="%.2f").encode("utf-8"),
            file_name="predictions.csv",
            mime="text/csv"
        )

        # -----------------------------
        # Summary Panel & Final CGPA Statistics
        # -----------------------------
        st.markdown("##  Summary Panel")

        # Dataset Overview
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Records", out.shape[0])
        with col2:
            st.metric("Total Missing Values", out.isnull().sum().sum())
        with col3:
            st.metric("Number of Columns", out.shape[1])

        # Missing Values per Column
        missing_cols = out.isnull().sum()
        missing_cols = missing_cols[missing_cols > 0]
        if not missing_cols.empty:
            st.subheader(" Missing Values per Column")
            st.table(missing_cols)

        # Final CGPA Statistics (use predictions if "Final CGPA" not in dataset)
        target_col = "Final CGPA" if "Final CGPA" in out.columns else "FGPA_MLR_Pred"
        st.subheader(" Final CGPA Statistics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Mean CGPA", round(out[target_col].mean(), 2))
            st.metric("Median CGPA", round(out[target_col].median(), 2))
        with col2:
            st.metric("Std Dev", round(out[target_col].std(), 2))
            st.metric("Minimum", round(out[target_col].min(), 2))
        with col3:
            st.metric("Maximum", round(out[target_col].max(), 2))

        # Performance Distribution
        st.subheader(" Performance Distribution")
        excellent = out[out[target_col] >= 3.5].shape[0]
        good = out[(out[target_col] >= 3.0) & (out[target_col] < 3.5)].shape[0]
        average = out[(out[target_col] >= 2.0) & (out[target_col] < 3.0)].shape[0]
        below_avg = out[out[target_col] < 2.0].shape[0]
        total = out.shape[0]

        dist_df = pd.DataFrame({
            "Category": ["Excellent (≥3.5)", "Good (3.0-3.49)", "Average (2.0-2.99)", "Below Average (<2.0)"],
            "Count": [excellent, good, average, below_avg],
            "Percentage": [
                round(excellent/total*100, 1),
                round(good/total*100, 1),
                round(average/total*100, 1),
                round(below_avg/total*100, 1)
            ]
        })
        st.table(dist_df)

        # Final CGPA Distribution Graph
        fig = px.histogram(
            out, x=target_col, nbins=20,
            title="Final CGPA Distribution",
            color_discrete_sequence=["#004080"]
        )
        st.plotly_chart(fig, use_container_width=True)
