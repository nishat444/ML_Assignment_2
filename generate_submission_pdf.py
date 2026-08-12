"""Generate ML Assignment 2 submission PDF."""
from __future__ import annotations

from pathlib import Path

from fpdf import FPDF

ROOT = Path(__file__).resolve().parent
README = (ROOT / "README.md").read_text(encoding="utf-8")
OUT = ROOT.parent / "ML_Assignment_2_Submission.pdf"

GITHUB = "https://github.com/nishat444/ML_Assignment_2"
STREAMLIT_FILE = ROOT / "streamlit_url.txt"
STREAMLIT = ""
if STREAMLIT_FILE.exists():
    for line in STREAMLIT_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and line.startswith("http"):
            STREAMLIT = line
            break


class PDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 11)
        self.cell(0, 8, "BITS WILP | Machine Learning Assignment 2", align="C", new_x="LMARGIN", new_y="NEXT")
        self.ln(4)

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
        "#": "",
    }
    for a, b in replacements.items():
        text = text.replace(a, b)
    return text.encode("latin-1", errors="replace").decode("latin-1")


def write_wrapped(pdf: PDF, text: str, size: int = 10, bold: bool = False):
    pdf.set_x(pdf.l_margin)
    style = "B" if bold else ""
    pdf.set_font("Helvetica", style, size)
    usable = pdf.w - pdf.l_margin - pdf.r_margin
    pdf.multi_cell(usable, 5, clean(text))


def main():
    pdf = PDF()
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()

    write_wrapped(pdf, "Mandatory Submission Links", size=14, bold=True)
    pdf.ln(2)
    write_wrapped(
        pdf,
        "1. GitHub Repository Link\n"
        f"{GITHUB}\n\n"
        "2. Live Streamlit App Link\n"
        f"{STREAMLIT if STREAMLIT else '[PASTE LIVE STREAMLIT URL AFTER DEPLOY]'}\n\n"
        "3. Screenshot of assignment execution on BITS Virtual Lab is on the next page.\n\n"
        "4. README.md content is included after the screenshot page.",
    )

    pdf.add_page()
    write_wrapped(pdf, "3. BITS Virtual Lab Screenshot", size=14, bold=True)
    pdf.ln(2)
    write_wrapped(
        pdf,
        "Insert ONE screenshot showing assignment execution on BITS Virtual Lab.\n\n"
        "Steps:\n"
        "1. Login to BITS Virtual Lab\n"
        "2. git clone https://github.com/nishat444/ML_Assignment_2.git\n"
        "3. cd ML_Assignment_2\n"
        "4. pip install -r requirements.txt\n"
        "5. python bits_lab_proof.py\n"
        "6. Screenshot the terminal output and paste it in the box below.",
    )
    pdf.ln(4)
    y = pdf.get_y()
    pdf.set_draw_color(100, 100, 100)
    box_h = min(110, pdf.h - y - 25)
    pdf.rect(pdf.l_margin, y, pdf.w - pdf.l_margin - pdf.r_margin, box_h)
    pdf.set_xy(pdf.l_margin, y + box_h / 2 - 5)
    write_wrapped(pdf, "[ PASTE BITS VIRTUAL LAB SCREENSHOT HERE ]", size=11)

    pdf.add_page()
    write_wrapped(pdf, "4. README.md Content", size=14, bold=True)
    pdf.ln(2)

    in_code = False
    for raw in README.splitlines():
        line = raw.rstrip()
        if line.startswith("```"):
            in_code = not in_code
            continue
        if not line.strip():
            pdf.ln(2)
            continue
        if in_code or line.lstrip().startswith("|"):
            pdf.set_x(pdf.l_margin)
            pdf.set_font("Courier", "", 7)
            usable = pdf.w - pdf.l_margin - pdf.r_margin
            pdf.multi_cell(usable, 4, clean(line[:180]))
            continue
        if line.startswith("#"):
            write_wrapped(pdf, line.lstrip("# ").strip(), size=11, bold=True)
        else:
            write_wrapped(pdf, line, size=9)

    pdf.output(OUT)
    print(f"Wrote {OUT}")
    print(f"Streamlit URL: {STREAMLIT or 'NOT SET'}")


if __name__ == "__main__":
    main()
