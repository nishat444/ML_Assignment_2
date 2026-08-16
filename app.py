"""
Streamlit app: Dry Bean Classification — BITS WILP ML Assignment 2

Features:
- CSV upload (test data)
- Model selection dropdown
- Evaluation metrics display
- Confusion matrix + classification report
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
METRICS_PATH = MODEL_DIR / "metrics.json"
LABEL_ENCODER_PATH = MODEL_DIR / "label_encoder.joblib"
FEATURE_NAMES_PATH = MODEL_DIR / "feature_names.json"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "kNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Random Forest (Ensemble)": "random_forest_ensemble.joblib",
}


@st.cache_resource
def load_artifacts():
    label_encoder = joblib.load(LABEL_ENCODER_PATH)
    feature_names = json.loads(FEATURE_NAMES_PATH.read_text(encoding="utf-8"))
    saved_metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    models = {}
    for name, fname in MODEL_FILES.items():
        models[name] = joblib.load(MODEL_DIR / fname)
    return models, label_encoder, feature_names, saved_metrics


def evaluate(model, X, y_true, class_names):
    y_pred = model.predict(X)
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X)
    else:
        y_proba = np.eye(len(class_names))[y_pred]

    labels = list(range(len(class_names)))
    try:
        auc = float(
            roc_auc_score(
                y_true,
                y_proba,
                multi_class="ovr",
                average="weighted",
                labels=labels,
            )
        )
    except Exception:
        auc = float("nan")

    metrics = {
        "Accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "AUC": round(auc, 4) if auc == auc else None,
        "Precision": round(
            float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
            4,
        ),
        "Recall": round(
            float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
            4,
        ),
        "F1": round(float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
        "MCC": round(float(matthews_corrcoef(y_true, y_pred)), 4),
    }
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        zero_division=0,
        output_dict=True,
    )
    return metrics, cm, report, y_pred


def plot_confusion_matrix(cm, class_names, model_name: str):
    plt.close("all")
    fig, ax = plt.subplots(figsize=(7.5, 6.2), layout="constrained")
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        cbar_kws={"shrink": 0.75},
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {model_name}", pad=14)
    ax.tick_params(axis="x", labelrotation=45)
    ax.tick_params(axis="y", labelrotation=0)
    plt.setp(ax.get_xticklabels(), ha="right")
    return fig


def main():
    st.set_page_config(
        page_title="Dry Bean Classifier | ML Assignment 2",
        page_icon="🫘",
        layout="wide",
    )

    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.2rem; }
        div[data-testid="stMetric"] {
            background-color: #e8eef5;
            border: 1px solid #9fb3c8;
            border-radius: 10px;
            padding: 0.55rem 0.75rem;
        }
        div[data-testid="stMetric"] label,
        div[data-testid="stMetric"] [data-testid="stMetricLabel"],
        div[data-testid="stMetric"] [data-testid="stMetricValue"],
        div[data-testid="stMetric"] [data-testid="stMetricDelta"] {
            color: #102a43 !important;
        }
        div[data-testid="stMetric"] [data-testid="stMetricValue"] {
            font-weight: 700 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("Dry Bean Variety Classification")
    st.caption(
        "BITS WILP — Machine Learning Assignment 2 | "
        "UCI Dry Bean Dataset (7 classes, 16 features)"
    )

    models, label_encoder, feature_names, saved_metrics = load_artifacts()
    class_names = [str(c) for c in label_encoder.classes_]

    with st.sidebar:
        st.header("Dataset help")
        st.markdown("**Expected CSV columns**")
        st.code(", ".join(feature_names + ["Class"]), language=None)
        st.markdown(
            "Upload **test data only** (features + `Class` label column). "
            "A ready-made `test_data.csv` is included in the repository."
        )

    st.subheader("1. Upload Test Dataset (CSV)")
    uploaded = st.file_uploader("Choose a CSV file", type=["csv"])

    use_bundled = st.checkbox(
        "Use bundled repository test_data.csv",
        value=uploaded is None,
    )

    if uploaded is not None:
        df = pd.read_csv(uploaded)
        data_source = "Uploaded file"
    elif use_bundled and (ROOT / "test_data.csv").exists():
        df = pd.read_csv(ROOT / "test_data.csv")
        data_source = "Bundled test_data.csv"
    else:
        st.info("Upload a CSV or enable the bundled test data checkbox to continue.")
        st.stop()

    st.success(f"Loaded **{len(df)}** rows from {data_source}")
    with st.expander("Preview data", expanded=False):
        st.dataframe(df.head(20), use_container_width=True)

    missing = [c for c in feature_names if c not in df.columns]
    if missing:
        st.error(f"Missing required feature columns: {missing}")
        st.stop()
    if "Class" not in df.columns:
        st.error("CSV must include a `Class` column with true labels for evaluation.")
        st.stop()

    X = df[feature_names]
    try:
        y_true = label_encoder.transform(df["Class"].astype(str))
    except ValueError as exc:
        st.error(
            f"Unknown class labels in CSV. Expected one of: {class_names}. Detail: {exc}"
        )
        st.stop()

    st.subheader("2. Select a model")
    model_name = st.selectbox(
        "Choose a classifier to evaluate on the loaded test data",
        list(MODEL_FILES.keys()),
        key="model_selector",
        help="Metrics, confusion matrix, and classification report below update for this model.",
    )
    st.info(f"Showing live results for **{model_name}** on the loaded CSV.")

    model = models[model_name]
    metrics, cm, report, y_pred = evaluate(model, X, y_true, class_names)

    st.subheader(f"3. Evaluation Metrics — {model_name}")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
    c2.metric("AUC", f"{metrics['AUC']:.4f}" if metrics["AUC"] is not None else "N/A")
    c3.metric("Precision", f"{metrics['Precision']:.4f}")
    c4.metric("Recall", f"{metrics['Recall']:.4f}")
    c5.metric("F1 Score", f"{metrics['F1']:.4f}")
    c6.metric("MCC", f"{metrics['MCC']:.4f}")

    st.markdown("#### Comparison table (same test set, all models)")
    st.caption(
        "This table lists every model. The highlighted row is the one you selected. "
        "Pick Naive Bayes to see a large drop versus Logistic Regression / Random Forest."
    )
    comparison_rows = []
    for name, m in saved_metrics.items():
        comparison_rows.append(
            {
                "Selected": "← current" if name == model_name else "",
                "ML Model Name": name,
                "Accuracy": m["Accuracy"],
                "AUC": m["AUC"],
                "Precision": m["Precision"],
                "Recall": m["Recall"],
                "F1": m["F1"],
                "MCC": m["MCC"],
            }
        )
    comparison_df = pd.DataFrame(comparison_rows)
    st.dataframe(
        comparison_df,
        use_container_width=True,
        hide_index=True,
    )

    st.subheader(f"4. Confusion Matrix & Classification Report — {model_name}")
    fig = plot_confusion_matrix(cm, class_names, model_name)
    st.pyplot(fig, clear_figure=True, use_container_width=True)
    plt.close(fig)

    report_df = pd.DataFrame(report).transpose()
    st.markdown(f"**Classification Report — {model_name}**")
    st.dataframe(report_df.style.format("{:.4f}"), use_container_width=True)

    pred_df = df.copy()
    pred_df["Predicted_Class"] = label_encoder.inverse_transform(y_pred)
    with st.expander("Prediction sample (first 50 rows)"):
        st.dataframe(
            pred_df[["Class", "Predicted_Class"] + feature_names[:4]].head(50),
            use_container_width=True,
        )

    st.markdown("---")
    st.caption(
        "Models: Logistic Regression · Decision Tree · kNN · Gaussian Naive Bayes · Random Forest"
    )


if __name__ == "__main__":
    main()
