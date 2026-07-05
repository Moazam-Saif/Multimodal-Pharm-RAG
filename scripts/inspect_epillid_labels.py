"""
Phase 1 - step 0h: inspect ePillID's all_labels.csv - our new ground truth.

We read it directly out of the zip (via zipfile.open) without extracting
the whole archive to disk yet - useful when you just need one file out
of a much bigger zip.
"""

import zipfile
import pandas as pd

ZIP_PATH = "data/raw/epillid_data.zip"
CSV_INSIDE_ZIP = "ePillID_data/all_labels.csv"


def main() -> None:
    with zipfile.ZipFile(ZIP_PATH) as zf:
        with zf.open(CSV_INSIDE_ZIP) as f:
            df = pd.read_csv(f)

    print(f"Total rows: {len(df)}")
    print(f"\n=== Columns ===")
    for col in df.columns:
        print(col)

    print(f"\n=== First 10 rows ===")
    print(df.head(10))

    print(f"\n=== Unique label counts (if a 'label' column exists) ===")
    for candidate in ["label", "pilltype_id", "ndc9", "is_ref", "is_front"]:
        if candidate in df.columns:
            print(f"\n{candidate}: {df[candidate].nunique()} unique values")
            print(df[candidate].value_counts().head())


if __name__ == "__main__":
    main()
