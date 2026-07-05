"""
Phase 1 - step 0q: test whether leading zeros are hiding matches.

Hypothesis: some labeler codes are stored with leading zeros in one
dataset but not the other (e.g. "00062" vs "62"). Comparing as INTEGERS
(which naturally drop leading zeros) rather than strings should reveal
if this recovers matches our string-based join missed.
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

    epillid_labelers_str = set(ndc_style["ndc_extracted"].str.split("-").str[0])
    pillbox_labelers_str = set(pillbox["product_code"].dropna().astype(str).str.split("-").str[0])

    # Compare as ints this time - int("00062") == int("62") == 62
    epillid_labelers_int = {int(x) for x in epillid_labelers_str if x.isdigit()}
    pillbox_labelers_int = {int(x) for x in pillbox_labelers_str if x.isdigit()}

    overlap_int = epillid_labelers_int & pillbox_labelers_int
    print(f"Overlap as STRINGS: was 68 (from previous script)")
    print(f"Overlap as INTEGERS (leading zeros ignored): {len(overlap_int)} "
          f"({len(overlap_int)/len(epillid_labelers_int):.1%} of ePillID's labelers)")

    if len(overlap_int) > 68:
        print("\n-> Leading zeros WERE hiding matches. Confirmed real cause.")
    else:
        print("\n-> Leading zeros were NOT the issue. Need another theory.")


if __name__ == "__main__":
    main()
