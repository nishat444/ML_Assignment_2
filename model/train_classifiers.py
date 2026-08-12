"""
Train classification models on UCI Dry Bean Dataset and persist artifacts.

Models: Logistic Regression, Decision Tree, KNN, Gaussian Naive Bayes, Random Forest
Metrics: Accuracy, AUC, Precision, Recall, F1, MCC
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
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
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

# This file lives under model/; project root is one level up.
ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = Path(__file__).resolve().parent
DATA_PATH = ROOT / "dry_bean_full.csv"
TEST_DATA_PATH = ROOT / "test_data.csv"
METRICS_PATH = MODEL_DIR / "metrics.json"
LABEL_ENCODER_PATH = MODEL_DIR / "label_encoder.joblib"
FEATURE_NAMES_PATH = MODEL_DIR / "feature_names.json"

RANDOM_STATE = 42
TEST_SIZE = 0.2


def build_models() -> dict:
    return {
        "Logistic Regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=2000,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=12,
            min_samples_leaf=5,
            random_state=RANDOM_STATE,
        ),
        "kNN": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", KNeighborsClassifier(n_neighbors=7)),
            ]
        ),
        "Naive Bayes": GaussianNB(),
        "Random Forest (Ensemble)": RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
    }


def compute_metrics(y_true, y_pred, y_proba, labels) -> dict:
    # Multi-class AUC via One-vs-Rest with weighted average
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

    return {
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
        "F1": round(
            float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
            4,
        ),
        "MCC": round(float(matthews_corrcoef(y_true, y_pred)), 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            target_names=[str(x) for x in labels],
            zero_division=0,
            output_dict=True,
        ),
    }


def main() -> None:
    if not DATA_PATH.exists():
        import sys

        sys.path.insert(0, str(ROOT))
        from download_data import main as download_main

        download_main()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    feature_cols = [c for c in df.columns if c != "Class"]
    X = df[feature_cols]
    y_raw = df["Class"].astype(str)

    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # Save test data with original class labels for Streamlit upload demos
    test_df = X_test.copy()
    test_df["Class"] = le.inverse_transform(y_test)
    test_df.to_csv(TEST_DATA_PATH, index=False)

    joblib.dump(le, LABEL_ENCODER_PATH)
    FEATURE_NAMES_PATH.write_text(json.dumps(feature_cols, indent=2), encoding="utf-8")

    models = build_models()
    all_metrics: dict = {}
    class_labels = list(range(len(le.classes_)))

    print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")
    print(f"Features ({len(feature_cols)}): {feature_cols}")
    print(f"Classes: {list(le.classes_)}")
    print("-" * 72)

    for name, model in models.items():
        print(f"Training: {name}")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)
        else:
            # Fallback: one-hot of predictions (should not happen for our models)
            y_proba = np.eye(len(le.classes_))[y_pred]

        metrics = compute_metrics(y_test, y_pred, y_proba, class_labels)
        # Store human-readable class names for the app
        metrics["class_names"] = [str(c) for c in le.classes_]
        all_metrics[name] = metrics

        safe_name = (
            name.lower()
            .replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
            .replace("-", "")
        )
        model_path = MODEL_DIR / f"{safe_name}.joblib"
        joblib.dump(model, model_path)

        print(
            f"  Acc={metrics['Accuracy']:.4f}  AUC={metrics['AUC']}  "
            f"Prec={metrics['Precision']:.4f}  Rec={metrics['Recall']:.4f}  "
            f"F1={metrics['F1']:.4f}  MCC={metrics['MCC']:.4f}"
        )
        print(f"  Saved -> {model_path.name}")

    METRICS_PATH.write_text(json.dumps(all_metrics, indent=2), encoding="utf-8")
    print("-" * 72)
    print(f"Metrics saved -> {METRICS_PATH}")
    print(f"Test data saved -> {TEST_DATA_PATH} ({len(test_df)} rows)")


if __name__ == "__main__":
    main()
