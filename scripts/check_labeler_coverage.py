"""
Phase 1 - step 0p: isolate whether the low match rate is a COVERAGE gap
(our Pillbox CSV just doesn't contain these labelers at all) vs a JOIN
LOGIC problem (the labelers exist but our matching is still wrong).

We check overlap at the labeler-code level only (first 4-5 digits) -
the broadest possible grouping - since if even labelers don't overlap
much, no amount of join-logic fixing will help; we'd need more/different
metadata instead.
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

    # Labeler code = segment before the FIRST dash (e.g. "60505" from
    # "60505-2671-09"), regardless of overall digit count
    epillid_labelers = set(ndc_style["ndc_extracted"].str.split("-").str[0])
    pillbox_labelers = set(pillbox["product_code"].dropna().astype(str).str.split("-").str[0])

    print(f"Unique labeler codes in ePillID:  {len(epillid_labelers)}")
    print(f"Unique labeler codes in Pillbox:   {len(pillbox_labelers)}")

    overlap = epillid_labelers & pillbox_labelers
    print(f"Labelers appearing in BOTH: {len(overlap)} ({len(overlap)/len(epillid_labelers):.1%} of ePillID's labelers)")

    print(f"\nSample ePillID-only labelers (not in Pillbox at all): "
          f"{list(epillid_labelers - pillbox_labelers)[:10]}")


if __name__ == "__main__":
    main()
