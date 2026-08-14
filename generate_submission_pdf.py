"""Generate the assignment submission PDF: links, one lab screenshot, README (Step 5)."""
from __future__ import annotations

from pathlib import Path

from fpdf import FPDF
from PIL import Image

ROOT = Path(__file__).resolve().parent
PARENT = ROOT.parent
OUT = PARENT / "ML_Assignment_2_Submission.pdf"

GITHUB = "https://github.com/nishat444/ML_Assignment_2"
STREAMLIT = "https://nishat444-ml-assignment-2-app-jwdbuo.streamlit.app/"
# Assignment asks for ONE screenshot of BITS Virtual Lab execution.
SCREENSHOT = PARENT / "BITS_Lab_2.png"


class PDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 11)
        self.cell(
            0,
            8,
            "BITS WILP | Machine Learning Assignment 2",
            align="C",
            new_x="LMARGIN",
            new_y="NEXT",
        )
        self.ln(3)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")


def clean(text: str) -> str:
    replacements = {
        "\u2014": "-",
        "\u2013": "-",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2022": "-",
        "\u2192": "->",
        "\u2248": "~",
        "\u2265": ">=",
        "\u00b7": "-",
        "`": "'",
        "*": "",
    }
    for a, b in replacements.items():
        text = text.replace(a, b)
    return text.encode("latin-1", errors="replace").decode("latin-1")


def write_wrapped(pdf: PDF, text: str, size: int = 10, bold: bool = False):
    pdf.set_x(pdf.l_margin)
    pdf.set_font("Helvetica", "B" if bold else "", size)
    usable = pdf.w - pdf.l_margin - pdf.r_margin
    pdf.multi_cell(usable, 5, clean(text))


def write_table_line(pdf: PDF, text: str):
    pdf.set_x(pdf.l_margin)
    pdf.set_font("Courier", "", 7)
    usable = pdf.w - pdf.l_margin - pdf.r_margin
    pdf.multi_cell(usable, 4, clean(text[:180]))


def main():
    if not SCREENSHOT.exists():
        raise FileNotFoundError(SCREENSHOT)

    pdf = PDF()
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()

    write_wrapped(pdf, "Mandatory Submission Links", size=14, bold=True)
    pdf.ln(3)

    write_wrapped(pdf, "1. GitHub Repository Link", size=12, bold=True)
    write_wrapped(pdf, GITHUB)
    pdf.ln(4)

    write_wrapped(pdf, "2. Live Streamlit App Link", size=12, bold=True)
    write_wrapped(pdf, STREAMLIT)
    pdf.ln(4)

    write_wrapped(pdf, "3. Screenshot", size=12, bold=True)
    write_wrapped(
        pdf,
        "Assignment execution on BITS Virtual Lab (python train_models.py). "
        "Screenshot is on the next page.",
    )

    pdf.add_page()
    write_wrapped(pdf, "3. Screenshot — BITS Virtual Lab execution", size=12, bold=True)
    pdf.ln(2)
    usable_w = pdf.w - pdf.l_margin - pdf.r_margin
    usable_h = pdf.h - pdf.get_y() - 18
    with Image.open(SCREENSHOT) as im:
        iw, ih = im.size
    ratio = min(usable_w / iw, usable_h / ih)
    pdf.image(
        str(SCREENSHOT),
        x=pdf.l_margin,
        y=pdf.get_y(),
        w=iw * ratio,
        h=ih * ratio,
    )

    pdf.add_page()
    write_wrapped(pdf, "4. GitHub README content (Section 3 - Step 5)", size=14, bold=True)
    pdf.ln(3)

    write_wrapped(pdf, "a. Problem statement", size=12, bold=True)
    write_wrapped(
        pdf,
        "Dry beans are traded by variety, but visual sorting at scale is slow and "
        "error-prone. This project builds a multi-class supervised learning pipeline "
        "that predicts the dry-bean variety from 16 geometric and shape descriptors "
        "extracted from bean images. Five classical classifiers are trained on the same "
        "UCI Dry Bean Dataset, compared on a common held-out test set, and exposed "
        "through an interactive Streamlit web application for CSV-based evaluation.",
    )
    write_wrapped(pdf, "Task type: Multi-class classification (7 bean varieties).")
    write_wrapped(
        pdf,
        "Goal: Compare Logistic Regression, Decision Tree, kNN, Gaussian Naive Bayes, "
        "and Random Forest on Accuracy, AUC, Precision, Recall, F1, and MCC, then "
        "deploy the models for interactive testing.",
    )
    pdf.ln(3)

    write_wrapped(pdf, "b. Dataset description", size=12, bold=True)
    for line in [
        "Name: Dry Bean Dataset",
        "Source: UCI Machine Learning Repository (Koklu & Ozkan, 2020)",
        "https://archive.ics.uci.edu/dataset/602/dry+bean+dataset",
        "Instances: 13,611",
        "Features: 16 numeric predictors + 1 target (Class)",
        "Task: Multi-class classification",
        "Classes: BARBUNYA, BOMBAY, CALI, DERMASON, HOROZ, SEKER, SIRA",
        "Train/Test split: 80%/20%, stratified (random_state=42); 2,723 test rows in test_data.csv",
        "Feature list: Area, Perimeter, MajorAxisLength, MinorAxisLength, AspectRation,",
        "Eccentricity, ConvexArea, EquivDiameter, Extent, Solidity, roundness,",
        "Compactness, ShapeFactor1, ShapeFactor2, ShapeFactor3, ShapeFactor4.",
        "The dataset meets assignment constraints (>= 12 features, >= 500 instances).",
    ]:
        write_wrapped(pdf, line)
    pdf.ln(3)

    write_wrapped(pdf, "c. Github Repository Link", size=12, bold=True)
    write_wrapped(pdf, GITHUB)
    write_wrapped(pdf, "Live Streamlit App: " + STREAMLIT)
    pdf.ln(3)

    write_wrapped(pdf, "d. Models used", size=12, bold=True)
    write_wrapped(
        pdf,
        "All models were trained on the same stratified train split. Metrics are on "
        "the held-out test set (test_data.csv). Precision / Recall / F1: weighted "
        "average. AUC: One-vs-Rest (OvR), weighted average.",
    )
    pdf.ln(2)

    write_wrapped(pdf, "Comparison Table", size=11, bold=True)
    for line in [
        "ML Model Name                 Accuracy  AUC     Precision  Recall   F1      MCC",
        "Logistic Regression           0.9214    0.9934  0.9222     0.9214   0.9216  0.9050",
        "Decision Tree                 0.9115    0.9649  0.9114     0.9115   0.9113  0.8930",
        "kNN                           0.9137    0.9837  0.9144     0.9137   0.9139  0.8956",
        "Naive Bayes                   0.7639    0.9644  0.7654     0.7639   0.7615  0.7154",
        "Random Forest (Ensemble)      0.9174    0.9920  0.9176     0.9174   0.9173  0.9001",
    ]:
        write_table_line(pdf, line)
    pdf.ln(3)

    write_wrapped(pdf, "Observations on model performance", size=11, bold=True)
    write_wrapped(
        pdf,
        "Logistic Regression: Strongest overall result. With standardized features, "
        "linear decision boundaries separate most varieties well (Accuracy 0.9214, "
        "AUC 0.9934, MCC 0.9050). Fast to train and highly competitive with the ensemble.",
    )
    write_wrapped(
        pdf,
        "Decision Tree: Competitive accuracy (0.9115) but lower AUC than linear / "
        "ensemble models, which is expected for a single tree. Slightly more brittle "
        "near overlapping classes (e.g. DERMASON vs SIRA).",
    )
    write_wrapped(
        pdf,
        "kNN: Close to the tree and forest after feature scaling (Accuracy 0.9137, "
        "AUC 0.9837). Distance-based voting works well on continuous shape descriptors, "
        "but inference is slower than parametric models.",
    )
    write_wrapped(
        pdf,
        "Naive Bayes: Weakest classifier (Accuracy 0.7639). Gaussian NB assumes feature "
        "independence; geometric descriptors are strongly correlated (Area / Perimeter / "
        "ConvexArea), which hurts class decisions despite a still-high OvR AUC.",
    )
    write_wrapped(
        pdf,
        "Random Forest (Ensemble): Near-best accuracy and AUC (0.9174 / 0.9920). "
        "Averaging many trees reduces variance versus a single Decision Tree and nearly "
        "matches Logistic Regression.",
    )
    write_wrapped(
        pdf,
        "Overall Winner for this dataset: Logistic Regression — highest Accuracy "
        "(0.9214), AUC (0.9934), F1 (0.9216), and MCC (0.9050) on the held-out test set, "
        "with Random Forest a very close second.",
        bold=True,
    )

    pdf.output(OUT)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
