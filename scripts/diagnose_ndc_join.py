"""
Phase 1 - step 0n: diagnose the 0% NDC join, same way we diagnosed the
path mismatch earlier - print raw values side by side, don't theorize
first.
"""

import re
import zipfile
import pandas as pd

EPILLID_ZIP = "data/raw/epillid_data.zip"
EPILLID_LABELS = "ePillID_data/all_labels.csv"
PILLBOX_CSV = "data/raw/pillbox_metadata.csv"


def digits_only(s: str) -> str:
    return re.sub(r"\D", "", s)


def main() -> None:
    with zipfile.ZipFile(EPILLID_ZIP) as zf:
        with zf.open(EPILLID_LABELS) as f:
            epillid = pd.read_csv(f)

    pillbox = pd.read_csv(PILLBOX_CSV, low_memory=False)

    is_ndc_style = epillid["label"].str.contains("-", na=False)
    ndc_style = epillid[is_ndc_style].copy()
    ndc_style["ndc_extracted"] = ndc_style["label"].str.split("_").str[0]
    ndc_style["ndc_digits"] = ndc_style["ndc_extracted"].apply(digits_only)

    print("=== Raw label -> extracted NDC -> digits, first 10 ePillID rows ===")
    for _, row in ndc_style.head(10).iterrows():
        print(f"  {row['label']!r:45s} -> ndc={row['ndc_extracted']!r:20s} -> digits={row['ndc_digits']!r}")

    pillbox["ndc_digits"] = pillbox["product_code"].dropna().astype(str).apply(digits_only)

    print("\n=== Raw product_code -> digits, first 10 Pillbox rows ===")
    for _, row in pillbox.head(10).iterrows():
        print(f"  {row['product_code']!r:20s} -> digits={row['ndc_digits']!r}")

    # Compare digit LENGTHS - if epillid digits are consistently longer/
    # shorter than pillbox digits, that's a segment-count difference we
    # can normalize; if lengths are similar but values differ, it's
    # something else entirely.
    print(f"\nePillID ndc_digits length distribution:\n{ndc_style['ndc_digits'].str.len().value_counts()}")
    print(f"\nPillbox ndc_digits length distribution:\n{pillbox['ndc_digits'].str.len().value_counts()}")


if __name__ == "__main__":
    main()
