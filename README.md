# Dry Bean Variety Classification — Machine Learning Assignment 2
# Work Integrated Learning Programmes Division | M.Tech (AIML / DSE) | BITS Pilani WILP

## a. Problem statement

Dry beans are traded by variety, but visual sorting at scale is slow and error-prone. This project builds a multi-class supervised learning pipeline that predicts the dry-bean variety from 16 geometric and shape descriptors extracted from bean images. Five classical classifiers are trained on the same UCI Dry Bean Dataset, compared with a common held-out test set, and exposed through an interactive Streamlit web application for CSV-based evaluation.

**Task type:** Multi-class classification (7 bean varieties)  
**Goal:** Compare Logistic Regression, Decision Tree, kNN, Gaussian Naive Bayes, and Random Forest on Accuracy, AUC, Precision, Recall, F1, and Matthews Correlation Coefficient (MCC), then deploy the models for interactive testing.

---

## b. Dataset description

| Item | Detail |
|------|--------|
| Name | Dry Bean Dataset |
| Source | [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/602/dry+bean+dataset) (Koklu & Ozkan, 2020) |
| Instances | 13,611 |
| Features | 16 numeric predictors + 1 target (`Class`) |
| Task | Multi-class classification |
| Classes | BARBUNYA, BOMBAY, CALI, DERMASON, HOROZ, SEKER, SIRA |
| Train / Test split | 80% / 20%, stratified (`random_state=42`) → **2,723 test rows** saved as `test_data.csv` |

**Feature list:** Area, Perimeter, MajorAxisLength, MinorAxisLength, AspectRation, Eccentricity, ConvexArea, EquivDiameter, Extent, Solidity, roundness, Compactness, ShapeFactor1, ShapeFactor2, ShapeFactor3, ShapeFactor4.

The dataset meets assignment constraints (**≥ 12 features**, **≥ 500 instances**).

---

## c. Github Repository Link

**https://github.com/nishat444/ML_Assignment_2**

**Live Streamlit App Link (after Streamlit Community Cloud deploy):**  
> `https://<YOUR_APP_NAME>.streamlit.app`

---

## d. Models used

All models were trained on the same stratified train split. Metrics below are computed on the held-out test set (`test_data.csv`). For multi-class evaluation:

- **Precision / Recall / F1** → weighted average  
- **AUC** → One-vs-Rest (OvR), weighted average  

### Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9214 | 0.9934 | 0.9222 | 0.9214 | 0.9216 | 0.9050 |
| Decision Tree | 0.9115 | 0.9649 | 0.9114 | 0.9115 | 0.9113 | 0.8930 |
| kNN | 0.9137 | 0.9837 | 0.9144 | 0.9137 | 0.9139 | 0.8956 |
| Naive Bayes | 0.7639 | 0.9644 | 0.7654 | 0.7639 | 0.7615 | 0.7154 |
| Random Forest (Ensemble) | 0.9174 | 0.9920 | 0.9176 | 0.9174 | 0.9173 | 0.9001 |

### Observations on model performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Strongest overall result on this dataset. With standardized features, the linear decision boundaries separate most varieties well (Accuracy ≈ 0.92, AUC ≈ 0.99, MCC ≈ 0.91). Fast to train and highly competitive with the ensemble. |
| Decision Tree | Competitive accuracy (~0.91) but lower AUC than linear / ensemble models, which is expected for a single tree. Slightly more brittle near overlapping classes (e.g., DERMASON vs SIRA). |
| kNN | Very close to the tree and forest after feature scaling (Accuracy ≈ 0.91, AUC ≈ 0.98). Distance-based voting works well on the continuous shape descriptors, but inference is slower than parametric models. |
| Naive Bayes | Weakest classifier here (Accuracy ≈ 0.76). Gaussian NB assumes feature independence; geometric descriptors are strongly correlated (Area / Perimeter / ConvexArea, etc.), which hurts calibration of class decisions despite a still-high OvR AUC. |
| Random Forest (Ensemble) | Near-best accuracy and AUC (0.9174 / 0.9920). Averaging many trees reduces variance versus a single Decision Tree and nearly matches Logistic Regression. A robust default choice for tabular multi-class problems. |
| **Overall Winner for your dataset?** | **Logistic Regression** — highest Accuracy (0.9214), AUC (0.9934), F1 (0.9216), and MCC (0.9050) on the held-out test set, with Random Forest a very close second. |

---

## Repository structure

```text
ML_Assignment_2/
├── app.py                  # Streamlit frontend
├── requirements.txt
├── README.md
├── test_data.csv           # Held-out test set used in experiments
├── train_models.py         # End-to-end training & metric export
├── download_data.py        # Downloads UCI Dry Bean Dataset
├── dry_bean_full.csv       # Full dataset (optional local cache)
└── model/
    ├── *.joblib            # Saved fitted models + label encoder
    ├── metrics.json        # Pre-computed evaluation metrics
    ├── feature_names.json
    └── train_classifiers.py
```

---

## How to run locally

```bash
pip install -r requirements.txt
python download_data.py      # first time only
python train_models.py       # trains models & refreshes metrics / test_data.csv
streamlit run app.py
```

## Streamlit app features

1. **CSV upload** for test data (or use bundled `test_data.csv`)
2. **Model selection** dropdown (all five trained models)
3. **Evaluation metrics** — Accuracy, AUC, Precision, Recall, F1, MCC
4. **Confusion matrix** and **classification report**

## Deploy on Streamlit Community Cloud

1. Push this folder to a **public** GitHub repository  
2. Open [https://streamlit.io/cloud](https://streamlit.io/cloud) and sign in with GitHub  
3. **New app** → select the repository → branch `main` → main file `app.py`  
4. Deploy and copy the live URL into section **c** above and into your submission PDF  

## BITS Virtual Lab screenshot

Perform training / Streamlit run once on the BITS Virtual Lab and include **one screenshot** in the submission PDF (assignment requirement — 1 mark).

---

## Academic integrity note

This repository is original coursework for BITS WILP ML Assignment 2. Models, metrics, and UI are produced for this submission.
