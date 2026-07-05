"""
Phase 1 - step 0m: test cross-referencing ePillID pills against Pillbox
metadata by NDC, to recover drug name / ingredients text.

ePillID's `label` column has two known formats (see DEVLOG):
  - NDC-hex style:  "51285-0092-87_BE305F72"  -> NDC is everything before "_"
  - raw hash style: "b79b096bade8ddf..."       -> no NDC recoverable at all

We only attempt the join on the NDC-hex style rows, and use the same
digits-only normalization we proved necessary earlier (NDC segment
lengths vary legitimately, e.g. 4-4-2 vs 5-4-2).
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

    # Identify which ePillID labels are NDC-hex style (contain a "-") vs
    # raw hash style (no "-", just hex characters)
    is_ndc_style = epillid["label"].str.contains("-", na=False)
    print(f"ePillID rows with NDC-hex style label: {is_ndc_style.sum()} / {len(epillid)}")
    print(f"ePillID rows with raw-hash style label: {(~is_ndc_style).sum()} / {len(epillid)}")

    # Extract just the NDC part (before the first "_") from NDC-style rows
    ndc_style = epillid[is_ndc_style].copy()
    ndc_style["ndc_extracted"] = ndc_style["label"].str.split("_").str[0]
    ndc_style["ndc_digits"] = ndc_style["ndc_extracted"].apply(digits_only)

    # Build digit-normalized product_code lookup from Pillbox metadata
    pillbox["ndc_digits"] = pillbox["product_code"].dropna().astype(str).apply(digits_only)
    pillbox_lookup = pillbox.drop_duplicates("ndc_digits").set_index("ndc_digits")

    matched = ndc_style["ndc_digits"].isin(pillbox_lookup.index)
    print(f"\nOf {len(ndc_style)} NDC-style ePillID rows, "
          f"{matched.sum()} ({matched.mean():.1%}) matched a Pillbox row by digit-normalized NDC")

    # Show a few real matches with recovered text
    sample_matches = ndc_style[matched].head(5)
    for _, row in sample_matches.iterrows():
        pillbox_row = pillbox_lookup.loc[row["ndc_digits"]]
        print(f"\n  ePillID label: {row['label']}")
        print(f"  -> medicine_name: {pillbox_row.get('medicine_name')}")
        print(f"  -> spl_strength:  {pillbox_row.get('spl_strength')}")


if __name__ == "__main__":
    main()
