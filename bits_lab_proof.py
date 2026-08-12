"""
BITS Virtual Lab proof script.
Run this on BITS Virtual Lab, then screenshot the terminal output.
"""
from pathlib import Path

print("=" * 60)
print("BITS WILP | Machine Learning Assignment 2")
print("Dry Bean Classification - Lab Execution Proof")
print("=" * 60)

root = Path(__file__).resolve().parent
required = [
    "app.py",
    "requirements.txt",
    "README.md",
    "test_data.csv",
    "model/logistic_regression.joblib",
    "model/decision_tree.joblib",
    "model/knn.joblib",
    "model/naive_bayes.joblib",
    "model/random_forest_ensemble.joblib",
    "model/metrics.json",
]
print("\nChecking repository files...")
ok = True
for rel in required:
    exists = (root / rel).exists()
    print(f"  [{'OK' if exists else 'MISSING'}] {rel}")
    ok = ok and exists

print("\nLoading models and evaluating on test_data.csv ...")
try:
    import json
    import joblib
    import pandas as pd
    from sklearn.metrics import accuracy_score

    le = joblib.load(root / "model" / "label_encoder.joblib")
    feats = json.loads((root / "model" / "feature_names.json").read_text(encoding="utf-8"))
    df = pd.read_csv(root / "test_data.csv")
    X = df[feats]
    y = le.transform(df["Class"].astype(str))
    model = joblib.load(root / "model" / "logistic_regression.joblib")
    acc = accuracy_score(y, model.predict(X))
    print(f"  Logistic Regression test Accuracy = {acc:.4f}")
    print("  Execution on this machine: SUCCESS")
except Exception as exc:
    print(f"  ERROR: {exc}")
    ok = False

print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
print("Take a SCREENSHOT of this terminal now for PDF submission.")
print("=" * 60)
