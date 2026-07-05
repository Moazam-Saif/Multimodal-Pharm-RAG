"""
Phase 1 - step 0l: find the exception to the is_ref -> dr_224 rule.

All paths resolve to real files (100%), so this isn't broken data -
just our assumption "is_ref=True always means dr_224" being slightly
too strict. Let's see what these rows actually are.
"""

import zipfile
import pandas as pd

ZIP_PATH = "data/raw/epillid_data.zip"
CSV_INSIDE_ZIP = "ePillID_data/all_labels.csv"


def main() -> None:
    with zipfile.ZipFile(ZIP_PATH) as zf:
        with zf.open(CSV_INSIDE_ZIP) as f:
            df = pd.read_csv(f)

    ref_rows = df[df["is_ref"] == True]
    not_dr = ref_rows[~ref_rows["image_path"].str.contains("dr_224")]

    print(f"is_ref=True rows NOT in dr_224/: {len(not_dr)} out of {len(ref_rows)}")
    print("\nWhere do they actually point?")
    print(not_dr["image_path"].apply(lambda p: p.split("/")[1]).value_counts())

    print("\nSample rows:")
    print(not_dr[["image_path", "label", "is_ref", "is_front"]].head(5))


if __name__ == "__main__":
    main()
