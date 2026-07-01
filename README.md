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

    """
models.py
Defines the algorithm registry (classification + regression), trains every
candidate model inside a consistent pipeline, and produces a leaderboard
of evaluation metrics.
"""

import time
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVC
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
)

from preprocessing import build_preprocessing_pipeline


# ---------------------------------------------------------------------------
# Algorithm registry
# Each entry: rank, display name, rating (stars), short rationale, and the
# sklearn-compatible estimator with sensible default hyperparameters for
# small tabular datasets (<10k rows).
# ---------------------------------------------------------------------------

CLASSIFICATION_MODELS = {
    "Random Forest": {
        "rank": 1,
        "rating": "★★★★★",
        "why": "Robust default, handles mixed data well, minimal tuning needed.",
        "estimator": RandomForestClassifier(
            n_estimators=300, max_depth=12, min_samples_leaf=2,
            random_state=42, n_jobs=-1
        )
    },
    "XGBoost": {
        "rank": 2,
        "rating": "★★★★★",
        "why": "Typically the strongest raw accuracy on small/medium tabular data.",
        "estimator": XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            random_state=42, eval_metric="logloss"
        )
    },
    "Logistic Regression": {
        "rank": 3,
        "rating": "★★★★",
        "why": "Highly interpretable baseline, fast, good when relationships are linear.",
        "estimator": LogisticRegression(max_iter=1000, random_state=42)
    },
    "LightGBM": {
        "rank": 4,
        "rating": "★★★★",
        "why": "Similar strength to XGBoost; can be less stable on very small datasets.",
        "estimator": LGBMClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            random_state=42, verbose=-1
        )
    },
    "SVM": {
        "rank": 5,
        "rating": "★★★",
        "why": "Effective on small, clean datasets but slower and less interpretable.",
        "estimator": SVC(probability=True, random_state=42)
    },
}

REGRESSION_MODELS = {
    "Random Forest": {
        "rank": 1,
        "rating": "★★★★★",
        "why": "Robust default, minimal tuning, handles non-linear relationships well.",
        "estimator": RandomForestRegressor(
            n_estimators=300, max_depth=12, min_samples_leaf=2,
            random_state=42, n_jobs=-1
        )
    },
    "XGBoost": {
        "rank": 2,
        "rating": "★★★★★",
        "why": "Highest typical accuracy ceiling for small/medium tabular regression.",
        "estimator": XGBRegressor(
            n_estimators=300, max_depth=6, learning_rate=0.1, random_state=42
        )
    },
    "Ridge Regression": {
        "rank": 3,
        "rating": "★★★★",
        "why": "Interpretable, fast, strong baseline, resistant to overfitting.",
        "estimator": Ridge(alpha=1.0, random_state=42)
    },
    "Gradient Boosting": {
        "rank": 4,
        "rating": "★★★★",
        "why": "Strong accuracy but slower to train than XGBoost on larger sets.",
        "estimator": GradientBoostingRegressor(
            n_estimators=300, max_depth=4, learning_rate=0.1, random_state=42
        )
    },
    "Lasso": {
        "rank": 5,
        "rating": "★★★",
        "why": "Useful when automatic feature selection (sparsity) matters.",
        "estimator": Lasso(alpha=0.01, random_state=42)
    },
}


def get_model_registry(problem_type: str) -> dict:
    return CLASSIFICATION_MODELS if problem_type == "classification" else REGRESSION_MODELS


def train_and_evaluate_all(
    X: pd.DataFrame,
    y,
    problem_type: str,
    numeric_cols: list,
    categorical_cols: list,
    scale_numeric: bool = True,
    test_size: float = 0.2
):
    """
    Trains every model in the registry inside a full preprocessing + model
    pipeline, evaluates on a held-out test split, and returns:
        - leaderboard: DataFrame sorted by primary metric
        - trained_pipelines: dict of {model_name: fitted Pipeline}
    """
    registry = get_model_registry(problem_type)

    stratify = y if problem_type == "classification" else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=stratify
    )

    results = []
    trained_pipelines = {}

    for name, config in registry.items():
        preprocessor = build_preprocessing_pipeline(numeric_cols, categorical_cols, scale_numeric)
        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", config["estimator"])
        ])

        start = time.time()
        try:
            pipeline.fit(X_train, y_train)
            train_time = round(time.time() - start, 3)

            y_pred = pipeline.predict(X_test)

            if problem_type == "classification":
                metrics = _classification_metrics(pipeline, X_test, y_test, y_pred)
            else:
                metrics = _regression_metrics(y_test, y_pred)

            metrics.update({
                "Model": name,
                "Rating": config["rating"],
                "Training Time (s)": train_time
            })
            results.append(metrics)
            trained_pipelines[name] = pipeline

        except Exception as e:
            results.append({
                "Model": name,
                "Rating": config["rating"],
                "Training Time (s)": None,
                "Error": str(e)
            })

    leaderboard = pd.DataFrame(results)

    primary_metric = "F1 Score" if problem_type == "classification" else "R2"
    if primary_metric in leaderboard.columns:
        leaderboard = leaderboard.sort_values(primary_metric, ascending=False).reset_index(drop=True)

    return leaderboard, trained_pipelines, (X_test, y_test)


def _classification_metrics(pipeline, X_test, y_test, y_pred) -> dict:
    n_classes = len(np.unique(y_test))
    average = "binary" if n_classes == 2 else "macro"

    metrics = {
        "Accuracy": round(accuracy_score(y_test, y_pred), 4),
        "Precision": round(precision_score(y_test, y_pred, average=average, zero_division=0), 4),
        "Recall": round(recall_score(y_test, y_pred, average=average, zero_division=0), 4),
        "F1 Score": round(f1_score(y_test, y_pred, average=average, zero_division=0), 4),
    }

    try:
        if hasattr(pipeline, "predict_proba"):
            y_proba = pipeline.predict_proba(X_test)
            if n_classes == 2:
                metrics["ROC AUC"] = round(roc_auc_score(y_test, y_proba[:, 1]), 4)
            else:
                metrics["ROC AUC"] = round(roc_auc_score(y_test, y_proba, multi_class="ovr"), 4)
    except Exception:
        metrics["ROC AUC"] = None

    return metrics


def _regression_metrics(y_test, y_pred) -> dict:
    rmse = round(np.sqrt(mean_squared_error(y_test, y_pred)), 4)
    mae = round(mean_absolute_error(y_test, y_pred), 4)
    r2 = round(r2_score(y_test, y_pred), 4)

    try:
        mape = round(mean_absolute_percentage_error(y_test, y_pred) * 100, 2)
    except Exception:
        mape = None

    return {"RMSE": rmse, "MAE": mae, "MAPE (%)": mape, "R2": r2}

pip install -r requirements.txt
streamlit run app.py   

"""
preprocessing.py
Handles data quality checks, missing value treatment, encoding, scaling,
and class imbalance correction for the AutoML platform.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


def detect_problem_type(df: pd.DataFrame, target_col: str) -> str:
    """
    Auto-detects whether the target column implies a classification
    or regression problem.

    Heuristic:
    - Non-numeric dtype -> classification
    - Numeric but few unique values relative to row count -> classification
    - Otherwise -> regression
    """
    target = df[target_col]
    n_rows = len(df)
    n_unique = target.nunique(dropna=True)

    if not pd.api.types.is_numeric_dtype(target):
        return "classification"

    # Numeric column with low cardinality is likely a coded category (e.g., 0/1, 1-5 rating)
    if n_unique <= 20 or (n_unique / n_rows) < 0.05:
        return "classification"

    return "regression"


def get_missing_value_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Returns a summary table of missing values per column."""
    missing = df.isnull().sum()
    pct = (missing / len(df) * 100).round(2)
    summary = pd.DataFrame({
        "column": df.columns,
        "missing_count": missing.values,
        "missing_pct": pct.values,
        "dtype": df.dtypes.astype(str).values
    })
    return summary[summary["missing_count"] > 0].sort_values("missing_pct", ascending=False)


def handle_missing_values(df: pd.DataFrame, strategy: str, columns_to_drop: list = None) -> pd.DataFrame:
    """
    Applies the user-chosen missing value strategy.

    strategy options:
        - "drop_rows": drop any row with a missing value
        - "impute": mean/median for numeric, mode for categorical
        - "drop_columns": drop the specified columns_to_drop entirely
    """
    df = df.copy()

    if strategy == "drop_columns" and columns_to_drop:
        df = df.drop(columns=columns_to_drop)
        # Any remaining missing values get imputed as a safety net
        strategy = "impute"

    if strategy == "drop_rows":
        df = df.dropna()

    elif strategy == "impute":
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(exclude=[np.number]).columns

        for col in numeric_cols:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].median())

        for col in categorical_cols:
            if df[col].isnull().any():
                mode_val = df[col].mode()
                fill_val = mode_val[0] if not mode_val.empty else "Unknown"
                df[col] = df[col].fillna(fill_val)

    return df


def detect_outliers_iqr(df: pd.DataFrame, numeric_cols: list) -> pd.DataFrame:
    """Returns a summary of outlier counts per numeric column using IQR method."""
    rows = []
    for col in numeric_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        n_outliers = ((df[col] < lower) | (df[col] > upper)).sum()
        rows.append({"column": col, "outlier_count": n_outliers, "lower_bound": lower, "upper_bound": upper})
    return pd.DataFrame(rows)


def cap_outliers(df: pd.DataFrame, numeric_cols: list) -> pd.DataFrame:
    """Caps outliers to the IQR bounds (winsorization) instead of dropping rows."""
    df = df.copy()
    for col in numeric_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        df[col] = df[col].clip(lower=lower, upper=upper)
    return df


def build_preprocessing_pipeline(numeric_cols: list, categorical_cols: list, scale_numeric: bool = True):
    """
    Builds a sklearn ColumnTransformer that one-hot encodes categoricals
    and optionally scales numeric features. Used inside model pipelines
    so the same transform is applied consistently at train and predict time.
    """
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric_transformer = Pipeline(steps=numeric_steps)

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols)
        ],
        remainder="drop"
    )

    return preprocessor


def encode_target(y: pd.Series):
    """
    Label encodes a classification target if it's non-numeric.
    Returns the encoded target and the fitted encoder (or None for regression/numeric targets).
    """
    if not pd.api.types.is_numeric_dtype(y):
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        return y_encoded, le
    return y.values, None


def check_class_imbalance(y: pd.Series, threshold: float = 0.3) -> dict:
    """
    Checks if a classification target is imbalanced.
    Returns whether imbalance is detected and the class distribution.
    """
    counts = y.value_counts(normalize=True)
    is_imbalanced = (counts.min() < threshold)
    return {
        "is_imbalanced": bool(is_imbalanced),
        "distribution": counts.to_dict()
    }


def apply_smote(X, y):
    """Applies SMOTE oversampling to balance classes. X must already be numeric (post-encoding)."""
    from imblearn.over_sampling import SMOTE
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)
    return X_res, y_res

