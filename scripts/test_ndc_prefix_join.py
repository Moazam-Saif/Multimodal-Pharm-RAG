"""
Phase 1 - step 0e: for the plain-digit splimage rows, test a looser match.

New hypothesis: these rows don't have an exact splimage-to-filename match,
but the image filename might still start with their NDC (in dashed form,
e.g. "00002-3228-30"), just with an unpredictable hex suffix we can't
compute - only look up by prefix.
"""

import re
import zipfile
import pandas as pd

CSV_PATH = "data/raw/pillbox_metadata.csv"
ZIP_PATH = "data/raw/pillbox_images.zip"


def ndc_prefix_from_product_code(code: str) -> str:
    """product_code is already dashed, e.g. '00002-3228-30' - use as-is."""
    return code


def main() -> None:
    df = pd.read_csv(CSV_PATH, low_memory=False)
    with_image = df[df["has_image"] == True].copy()

    is_nlmimage = with_image["splimage"].astype(str).str.contains("NLMIMAGE10", na=False)
    plain_rows = with_image[~is_nlmimage]

    with zipfile.ZipFile(ZIP_PATH) as zf:
        real_filenames = zf.namelist()

    # Build a lookup: NDC prefix (before first "_") -> list of real filenames
    # e.g. "00002-3228-30_391E1C80.jpg" -> key "00002-3228-30"
    prefix_to_files = {}
    for name in real_filenames:
        prefix = name.split("_")[0]
        prefix_to_files.setdefault(prefix, []).append(name)

    print(f"Plain-digit rows to test: {len(plain_rows)}")

    codes = plain_rows["product_code"].dropna().astype(str)
    found = 0
    examples = []
    for code in codes:
        if code in prefix_to_files:
            found += 1
            if len(examples) < 5:
                examples.append((code, prefix_to_files[code]))

    print(f"Matched via NDC prefix lookup: {found} / {len(codes)} ({found/len(codes):.1%})")
    print("\nSample matches:")
    for code, files in examples:
        print(f"  {code} -> {files}")


if __name__ == "__main__":
    main()
