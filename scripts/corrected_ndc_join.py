"""
Phase 1 - step 0r: the corrected NDC join.

Root cause chain we diagnosed, in order:
1. ePillID NDCs are 11 digits (labeler-product-package); Pillbox's
   product_code is only 8-9 digits (labeler-product, no package segment)
   -> fix: truncate ePillID's NDC to just labeler+product
2. Even after truncating, match rate was only ~3% -> too low to be a
   truncation-length issue alone
3. Checked labeler-code overlap alone: 59.6% as strings, but that's
   still suspiciously low for codes that should mostly be shared
4. Tested leading-zero theory: comparing labelers as INTEGERS instead
   of strings jumped overlap to 98.2% -> confirmed real cause

This script combines both fixes: split NDC into labeler+product segments
independently, convert EACH segment to int (drops leading zeros
correctly, per-segment - critical, since int() on the WHOLE concatenated
string would incorrectly merge padding across segment boundaries).
"""

import zipfile
import pandas as pd

EPILLID_ZIP = "data/raw/epillid_data.zip"
EPILLID_LABELS = "ePillID_data/all_labels.csv"
PILLBOX_CSV = "data/raw/pillbox_metadata.csv"


def normalize_ndc(ndc_str: str) -> str | None:
    """Split an NDC on '-', int-convert each segment (drops leading
    zeros), keep only the first two segments (labeler, product) since
    that's the max Pillbox's product_code ever contains. Rejoin with
    a separator that can't appear in a digit, so segments can't bleed
    into each other.

    Returns None for malformed values (e.g. non-numeric segments like
    "0019-N601" - confirmed via inspect_bad_product_codes.py this
    affects only 3/83925 Pillbox rows, a real but rare FDA coding
    convention, not worth special-casing further)."""
    parts = ndc_str.split("-")
    if len(parts) < 2:
        return None
    labeler, product = parts[0], parts[1]
    if not (labeler.isdigit() and product.isdigit()):
        return None
    return f"{int(labeler)}.{int(product)}"


def main() -> None:
    with zipfile.ZipFile(EPILLID_ZIP) as zf:
        with zf.open(EPILLID_LABELS) as f:
            epillid = pd.read_csv(f)

    pillbox = pd.read_csv(PILLBOX_CSV, low_memory=False)

    is_ndc_style = epillid["label"].str.contains("-", na=False)
    ndc_style = epillid[is_ndc_style].copy()
    ndc_style["ndc_extracted"] = ndc_style["label"].str.split("_").str[0]
    ndc_style["ndc_normalized"] = ndc_style["ndc_extracted"].apply(normalize_ndc)
    ndc_style = ndc_style[ndc_style["ndc_normalized"].notna()]  # drop unparseable

    pillbox["ndc_normalized"] = pillbox["product_code"].dropna().astype(str).apply(normalize_ndc)
    pillbox = pillbox[pillbox["ndc_normalized"].notna()]  # drop unparseable
    pillbox_lookup = pillbox.drop_duplicates("ndc_normalized").set_index("ndc_normalized")

    matched = ndc_style["ndc_normalized"].isin(pillbox_lookup.index)
    print(f"Matched: {matched.sum()} / {len(ndc_style)} ({matched.mean():.1%})")

    print("\nSample matches:")
    for _, row in ndc_style[matched].head(8).iterrows():
        pillbox_row = pillbox_lookup.loc[row["ndc_normalized"]]
        print(f"  {row['label']:35s} -> {pillbox_row.get('medicine_name')}")


if __name__ == "__main__":
    main()