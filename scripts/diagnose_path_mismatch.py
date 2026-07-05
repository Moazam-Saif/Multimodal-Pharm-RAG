"""
Phase 1 - step 0k: diagnose the path mismatch directly.

Rather than guess at the correct prefix, print the raw image_path values
next to the real filenames that we know exist in dc_224/ and dr_224/,
so we can see exactly where they diverge.
"""

import zipfile
import pandas as pd

ZIP_PATH = "data/raw/epillid_data.zip"
CSV_INSIDE_ZIP = "ePillID_data/all_labels.csv"


def main() -> None:
    with zipfile.ZipFile(ZIP_PATH) as zf:
        real_files = zf.namelist()
        with zf.open(CSV_INSIDE_ZIP) as f:
            df = pd.read_csv(f)

    print("=== Raw image_path values, first 5 rows ===")
    print(df["image_path"].head(5).tolist())

    print("\n=== is_ref value counts and their raw image_path samples ===")
    for ref_val in [True, False]:
        sample = df[df["is_ref"] == ref_val]["image_path"].head(3).tolist()
        print(f"is_ref={ref_val}: {sample}")

    print("\n=== Real files actually in dc_224/ (first 5) ===")
    dc_files = [n for n in real_files if "dc_224" in n]
    print(dc_files[:5])

    print("\n=== Real files actually in dr_224/ (first 5) ===")
    dr_files = [n for n in real_files if "dr_224" in n]
    print(dr_files[:5])

    # Direct character-by-character comparison of one CSV path vs one real path
    csv_sample = df["image_path"].iloc[0]
    real_sample = dc_files[0] if dc_files else "N/A"
    print(f"\nCSV says:  {repr(csv_sample)}")
    print(f"Real file: {repr(real_sample)}")


if __name__ == "__main__":
    main()
