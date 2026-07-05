"""
Phase 1 - step 0f: can normalizing NDC formatting recover more matches?

Two hypotheses to test:
(a) Some has_image=True rows have a product_code that matches a real
    filename's NDC prefix, just after stripping to digits-only (removing
    dashes) rather than comparing dashed strings directly.
(b) Some filenames in the zip belong to metadata rows where has_image is
    incorrectly False - i.e. the image exists, but the flag lied.

We check both against the WHOLE metadata table (not just has_image=True
rows) using digit-only NDC comparison, which sidesteps the 4-4-2 vs 5-4-2
segment-length inconsistency entirely.
"""

import re
import zipfile
import pandas as pd

CSV_PATH = "data/raw/pillbox_metadata.csv"
ZIP_PATH = "data/raw/pillbox_images.zip"


def digits_only(s: str) -> str:
    """Strip everything except digits, so '00228-2855-11' -> '00228285511'
    and '0228-2855' -> '02282855' become comparable on their shared core."""
    return re.sub(r"\D", "", s)


def main() -> None:
    df = pd.read_csv(CSV_PATH, low_memory=False)

    with zipfile.ZipFile(ZIP_PATH) as zf:
        real_filenames = zf.namelist()

    # Only look at the NDC-hyphen-style filenames (skip junk/human-readable)
    ndc_style = [n for n in real_filenames if n[0].isdigit() and "-" in n]

    # Build digit-only prefix -> filenames lookup
    prefix_to_files = {}
    for name in ndc_style:
        prefix = digits_only(name.split("_")[0])
        prefix_to_files.setdefault(prefix, []).append(name)

    print(f"Unique digit-normalized NDC prefixes in zip: {len(prefix_to_files)}")
    print(f"Total NDC-style filenames: {len(ndc_style)}")

    # Test against ALL rows, regardless of has_image value
    df["ndc_digits"] = df["product_code"].dropna().astype(str).apply(digits_only)

    matched_mask = df["ndc_digits"].isin(prefix_to_files.keys())
    print(f"\nTotal metadata rows matching a real filename (any has_image value): {matched_mask.sum()}")

    # Of THOSE matches, how many claim has_image == False? (data quality check)
    mismatched_flag = df[matched_mask & (df["has_image"] == False)]
    print(f"Of those matches, rows where has_image is (incorrectly?) False: {len(mismatched_flag)}")

    if len(mismatched_flag) > 0:
        print("\nSample:")
        print(mismatched_flag[["product_code", "has_image", "medicine_name"]].head(5))


if __name__ == "__main__":
    main()
