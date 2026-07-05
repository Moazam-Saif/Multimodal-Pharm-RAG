"""
Phase 1 - step 0j: final verification before we commit to this dataset.

Confirm every image_path in all_labels.csv actually exists as a real
file inside the zip, under the ePillID_data/ prefix.
"""

import zipfile
import pandas as pd

ZIP_PATH = "data/raw/epillid_data.zip"
CSV_INSIDE_ZIP = "ePillID_data/all_labels.csv"


def main() -> None:
    with zipfile.ZipFile(ZIP_PATH) as zf:
        real_files = set(zf.namelist())
        with zf.open(CSV_INSIDE_ZIP) as f:
            df = pd.read_csv(f)

    # image_path in the CSV is relative to "ePillID_data/classification_data/",
    # NOT just "ePillID_data/" - confirmed by directly comparing a CSV value
    # against the real zip listing (see diagnose_path_mismatch.py output).
    full_paths = "ePillID_data/classification_data/" + df["image_path"]

    exists = full_paths.isin(real_files)
    print(f"Rows where image_path resolves to a real file: {exists.sum()} / {len(df)} ({exists.mean():.1%})")

    if not exists.all():
        print("\nSample of non-resolving paths:")
        print(full_paths[~exists].head(5).tolist())

    # Also confirm: is_ref True rows should live under dr_224, False under dc_224
    df["full_path"] = full_paths
    ref_check = df[df["is_ref"] == True]["full_path"].str.contains("dr_224").all()
    consumer_check = df[df["is_ref"] == False]["full_path"].str.contains("dc_224").all()
    print(f"\nAll is_ref=True rows point to dr_224/: {ref_check}")
    print(f"All is_ref=False rows point to dc_224/: {consumer_check}")


if __name__ == "__main__":
    main()