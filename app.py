import streamlit as st
import pandas as pd
import numpy as np
import joblib
import statsmodels.api as sm
from tensorflow.keras.models import load_model

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
st.set_page_config(page_title="TTU Student GPA Predictor", layout="wide")

# -----------------------------
# Custom CSS
# -----------------------------
st.markdown(
    """
    <style>
        /* Background */
        .main {
            background: linear-gradient(to right bottom, #fff8e1, #fff); /* Yellow tint blend */
            background-attachment: fixed;
        }

        /* Header banner */
        .wine-bg {
            background: #722f37;
            padding: 1.2rem;
            text-align: center;
            color: white;
            border-radius: 16px;
            margin-bottom: 25px;
        }

        /* Metric cards */
        [data-testid="stMetric"] {
            background: #fff9c4;
            border-left: 6px solid #722f37;
            border-radius: 14px;
            padding: 20px;
            text-align: center;
            margin: 10px;
            color: #722f37;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }

        /* Summary panel cards */
        .card {
            background: white;
            border-radius: 14px;
            padding: 20px;
            margin: 10px 0;
            box-shadow: 0 4px 12px rgba(0,0,0,0.08);
            border-left: 6px solid #facc15; /* Yellow left border */
        }

        /* Bold label */
        .bold { font-weight: 700; }

        /* Radio (Prediction Mode) */
        div[role="radiogroup"] {
            background: #fff9c4;
            padding: 12px 20px;
            border-radius: 14px;
            border: 2px solid #722f37;
            margin-bottom: 25px;
        }

        /* Number input fields shorter */
        .stNumberInput input {
            height: 2.2rem !important;
        }

        /* Prediction button */
        .stButton>button {
            background: #722f37;
            color: white;
            border-radius: 12px;
            padding: 0.6rem 1.2rem;
            font-weight: bold;
            border: none;
        }
        .stButton>button:hover {
            background: #facc15;
            color: #722f37;
        }

        /* Prediction result cards */
        .result-card {
            background: #fff9c4;
            border-radius: 14px;
            padding: 20px;
            margin: 10px;
            text-align: center;
            font-weight: bold;
            color: #722f37;
            box-shadow: 0 4px 12px rgba(0,0,0,0.12);
        }
    </style>

    <div class="wine-bg">
        <h1>TAKORADI TECHNICAL UNIVERSITY</h1>
        <h3>Student Final CGPA Predictor</h3>
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Dashboard Rendering Function
# -----------------------------
def render_dashboard(df=None, mlr_metrics=None, ann_metrics=None):
    st.markdown("<h2 style='text-align:center;'>Student Performance Dashboard</h2>", unsafe_allow_html=True)

    # Calculate stats if df exists
    total_students = df.shape[0] if df is not None else 0
    best_model = "MLR" if mlr_metrics and mlr_metrics["R2"] > (ann_metrics["R2"] if ann_metrics else -1) else "ANN"
    best_acc = max(mlr_metrics["R2"], ann_metrics["R2"]) if (mlr_metrics and ann_metrics) else 0

    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("Total Students Analyzed", total_students)
    with col2: st.metric("Available Models", "2 (MLR, ANN)")
    with col3: st.metric("Best Accuracy", f"{best_acc:.2%} ({best_model})")
    with col4: st.metric("Last Training Run", "2 days ago")

    # Summary panel
    if df is not None and "Final_CGPA" in df.columns:
        st.markdown("### Summary Panel")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="card"><span class="bold">Total Records:</span> ' + str(len(df)) + '</div>', unsafe_allow_html=True)
            st.markdown('<div class="card"><span class="bold">Missing Values per Column:</span>', unsafe_allow_html=True)
            st.dataframe(df.isna().sum())
            st.markdown('</div>', unsafe_allow_html=True)
        with col2:
            desc = df["Final_CGPA"].describe()
            st.markdown(
                f"""
                <div class="card">
                <h4>Final CGPA Statistics</h4>
                Mean: {desc['mean']:.2f} <br>
                Median: {df['Final_CGPA'].median():.2f} <br>
                Std Dev: {desc['std']:.2f}
                </div>
                """, unsafe_allow_html=True
            )

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
    col1, col2 = st.columns(2)
    with col1:
        social = st.number_input("Social Studies", 0.0, 100.0, 72.0, format="%.1f")
        science = st.number_input("Integrated Science", 0.0, 100.0, 75.0, format="%.1f")
    with col2:
        english = st.number_input("English Language", 0.0, 100.0, 59.0, format="%.1f")
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

        col1, col2 = st.columns(2)
        with col1: st.markdown(f"<div class='result-card'>MLR Prediction<br>{mlr_pred:.2f}</div>", unsafe_allow_html=True)
        with col2: st.markdown(f"<div class='result-card'>ANN Prediction<br>{ann_pred:.2f}</div>", unsafe_allow_html=True)

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