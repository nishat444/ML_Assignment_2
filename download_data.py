"""Download UCI Dry Bean Dataset and save as CSV."""
import io
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

URL = "https://archive.ics.uci.edu/static/public/602/dry+bean+dataset.zip"
OUT = Path(__file__).resolve().parent / "dry_bean_full.csv"


def main() -> None:
    print("Downloading Dry Bean Dataset from UCI...")
    raw = urllib.request.urlopen(URL, timeout=120).read()
    zf = zipfile.ZipFile(io.BytesIO(raw))
    xlsx_bytes = zf.read("DryBeanDataset/Dry_Bean_Dataset.xlsx")
    df = pd.read_excel(io.BytesIO(xlsx_bytes))
    df.to_csv(OUT, index=False)
    print(f"Saved {OUT} shape={df.shape}")
    print("Columns:", list(df.columns))
    print(df["Class"].value_counts())


if __name__ == "__main__":
    main()
