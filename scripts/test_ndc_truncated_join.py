"""
Phase 1 - step 0o: re-test the NDC join, truncating ePillID's 11-digit
NDC down to just labeler+product (first 8-9 digits), since Pillbox's
product_code never includes the package-size segment.

We test BOTH an 8-digit and 9-digit truncation, since Pillbox itself has
rows of both lengths (labeler codes are either 4 or 5 digits).
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
    pillbox["ndc_digits"] = pillbox["product_code"].dropna().astype(str).apply(digits_only)
    pillbox_lookup = pillbox.drop_duplicates("ndc_digits").set_index("ndc_digits")

    is_ndc_style = epillid["label"].str.contains("-", na=False)
    ndc_style = epillid[is_ndc_style].copy()
    ndc_style["ndc_extracted"] = ndc_style["label"].str.split("_").str[0]
    ndc_style["ndc_digits_full"] = ndc_style["ndc_extracted"].apply(digits_only)

    # Try both truncation lengths, count matches for each
    for trunc_len in [8, 9]:
        truncated = ndc_style["ndc_digits_full"].str[:trunc_len]
        matched = truncated.isin(pillbox_lookup.index)
        print(f"Truncated to {trunc_len} digits: {matched.sum()} / {len(ndc_style)} matched ({matched.mean():.1%})")

    # Use whichever worked better going forward - show sample real matches
    # at 9 digits (5-digit labeler + 4-digit product, the more common
    # Pillbox format per our length distribution: 24342 rows at 9 digits)
    ndc_style["ndc_9"] = ndc_style["ndc_digits_full"].str[:9]
    matched_9 = ndc_style["ndc_9"].isin(pillbox_lookup.index)
    print(f"\nSample real matches at 9-digit truncation:")
    for _, row in ndc_style[matched_9].head(5).iterrows():
        pillbox_row = pillbox_lookup.loc[row["ndc_9"]]
        print(f"  ePillID: {row['label']}  ->  {pillbox_row.get('medicine_name')}  ({pillbox_row.get('spl_strength')})")


if __name__ == "__main__":
    main()
