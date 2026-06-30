# ML-Model
ML model Built
pip install -r requirements.txt
streamlit run app.py
"""
app.py
Main Streamlit application — the entry point for the AutoML platform.
Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd

from preprocessing import (
    detect_problem_type, get_missing_value_summary, handle_missing_values,
    detect_outliers_iqr, cap_outliers, encode_target, check_class_imbalance, apply_smote
)
from models import train_and_evaluate_all
from utils import get_metric_explanation, plain_language_summary, save_model, predict_on_new_data


st.set_page_config(page_title="AutoML Platform", layout="wide")
st.title("🤖 AutoML Platform")
st.caption("Upload your data, pick what you want to predict, and let the system find the best model — no coding required.")

# ---------------------------------------------------------------------------
# Session state init
# ---------------------------------------------------------------------------
if "df" not in st.session_state:
    st.session_state.df = None
if "trained_pipelines" not in st.session_state:
    st.session_state.trained_pipelines = None
if "leaderboard" not in st.session_state:
    st.session_state.leaderboard = None
if "label_encoder" not in st.session_state:
    st.session_state.label_encoder = None
if "problem_type" not in st.session_state:
    st.session_state.problem_type = None

# ---------------------------------------------------------------------------
# STEP 1 — Upload data
# ---------------------------------------------------------------------------
st.header("1. Upload your data")
uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx", "xls"])

if uploaded_file is not None:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    st.session_state.df = df

if st.session_state.df is not None:
    df = st.session_state.df
    st.success(f"Loaded {df.shape[0]} rows and {df.shape[1]} columns.")
    st.dataframe(df.head(10), use_container_width=True)

    # -----------------------------------------------------------------------
    # STEP 2 — Select target & problem type
    # -----------------------------------------------------------------------
    st.header("2. What do you want to predict?")
    target_col = st.selectbox("Select your target column", options=df.columns)

    auto_detected = detect_problem_type(df, target_col)
    st.info(f"Auto-detected problem type: **{auto_detected.title()}**")

    override = st.checkbox("Override auto-detected problem type")
    if override:
        problem_type = st.radio("Choose problem type", ["classification", "regression"],
                                 index=0 if auto_detected == "classification" else 1)
    else:
        problem_type = auto_detected

    st.session_state.problem_type = problem_type

    # -----------------------------------------------------------------------
    # STEP 3 — Data quality: missing values
    # -----------------------------------------------------------------------
    st.header("3. Data quality check")
    missing_summary = get_missing_value_summary(df)

    if not missing_summary.empty:
        st.warning(f"Found missing values in {len(missing_summary)} column(s).")
        st.dataframe(missing_summary, use_container_width=True)

        missing_strategy_label = st.radio(
            "How should missing values be handled?",
            ["Fill in automatically (recommended)", "Drop rows with missing values", "Drop specific columns"]
        )
        strategy_map = {
            "Fill in automatically (recommended)": "impute",
            "Drop rows with missing values": "drop_rows",
            "Drop specific columns": "drop_columns"
        }
        missing_strategy = strategy_map[missing_strategy_label]

        cols_to_drop = []
        if missing_strategy == "drop_columns":
            cols_to_drop = st.multiselect("Select columns to drop", options=missing_summary["column"].tolist())

        df_clean = handle_missing_values(df, missing_strategy, cols_to_drop)
    else:
        st.success("No missing values found.")
        df_clean = df.copy()

    # -----------------------------------------------------------------------
    # STEP 4 — Train models
    # -----------------------------------------------------------------------
    st.header("4. Train models")

    feature_cols = [c for c in df_clean.columns if c != target_col]
    numeric_cols = df_clean[feature_cols].select_dtypes(include="number").columns.tolist()
    categorical_cols = df_clean[feature_cols].select_dtypes(exclude="number").columns.tolist()

    st.write(f"**Features:** {len(numeric_cols)} numeric, {len(categorical_cols)} categorical")

    use_smote = False
    if problem_type == "classification":
        y_check = df_clean[target_col]
        imbalance_info = check_class_imbalance(y_check)
        if imbalance_info["is_imbalanced"]:
            st.warning("Your target classes are imbalanced. This can bias the model toward the majority class.")
            use_smote = st.checkbox("Apply SMOTE to balance classes before training")

    if st.button("🚀 Train all models", type="primary"):
        X = df_clean[feature_cols]
        y_raw = df_clean[target_col]

        label_encoder = None
        if problem_type == "classification":
            y, label_encoder = encode_target(y_raw)
        else:
            y = y_raw.values

        st.session_state.label_encoder = label_encoder

        with st.spinner("Training and comparing models... this should only take a few seconds."):
            leaderboard, trained_pipelines, _ = train_and_evaluate_all(
                X, y, problem_type, numeric_cols, categorical_cols
            )

        st.session_state.leaderboard = leaderboard
        st.session_state.trained_pipelines = trained_pipelines
        st.success("Training complete!")

    # -----------------------------------------------------------------------
    # STEP 5 — Leaderboard & results
    # -----------------------------------------------------------------------
    if st.session_state.leaderboard is not None:
        st.header("5. Results leaderboard")
        leaderboard = st.session_state.leaderboard

        display_cols = [c for c in leaderboard.columns if c != "Error"]
        st.dataframe(leaderboard[display_cols], use_container_width=True)

        with st.expander("ℹ️ What do these metrics mean?"):
            metric_cols = [c for c in leaderboard.columns if c in
                           ["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC", "RMSE", "MAE", "MAPE (%)", "R2"]]
            for m in metric_cols:
                st.write(f"**{m}**: {get_metric_explanation(m)}")

        best_row = leaderboard.iloc[0].to_dict()
        st.subheader("🏆 Recommended model")
        st.markdown(plain_language_summary(best_row, problem_type))

        # -------------------------------------------------------------------
        # STEP 6 — Download best model
        # -------------------------------------------------------------------
        st.header("6. Download your model")
        best_model_name = best_row["Model"]
        best_pipeline = st.session_state.trained_pipelines[best_model_name]

        if st.button("💾 Save best model"):
            path = save_model(best_pipeline, best_model_name)
            with open(path, "rb") as f:
                st.download_button(
                    label="Download trained model (.joblib)",
                    data=f,
                    file_name=f"{best_model_name.replace(' ', '_').lower()}_model.joblib"
                )

        # -------------------------------------------------------------------
        # STEP 7 — Predict on new data
        # -------------------------------------------------------------------
        st.header("7. Run predictions on new data")
        st.caption(f"Upload a new file with the same feature columns (without '{target_col}') to get predictions.")
        new_file = st.file_uploader("Upload new data for prediction", type=["csv", "xlsx", "xls"], key="predict_upload")

        if new_file is not None:
            if new_file.name.endswith(".csv"):
                new_df = pd.read_csv(new_file)
            else:
                new_df = pd.read_excel(new_file)

            model_choice = st.selectbox(
                "Choose which model to use for prediction",
                options=leaderboard["Model"].tolist(),
                index=0
            )
            chosen_pipeline = st.session_state.trained_pipelines[model_choice]

            try:
                result_df = predict_on_new_data(chosen_pipeline, new_df, st.session_state.label_encoder)
                st.success("Predictions generated.")
                st.dataframe(result_df, use_container_width=True)

                csv = result_df.to_csv(index=False).encode("utf-8")
                st.download_button("Download predictions as CSV", data=csv, file_name="predictions.csv")
            except Exception as e:
                st.error(f"Couldn't generate predictions — make sure the uploaded columns match the training data. ({e})")

else:
    st.info("👆 Upload a CSV or Excel file to get started.")
pip install -r requirements.txt
streamlit run app.py    
