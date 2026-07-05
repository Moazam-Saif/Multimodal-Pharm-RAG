"""
Phase 1 - step 0c: test our join hypothesis before committing to it.

Hypothesis: metadata rows with has_image == True have a `splimage` or
`product_code` value that, once reformatted, matches one of the
NDC-prefixed image filenames (e.g. "00002-3228-30_391E1C80.jpg").

We check this against real data instead of assuming.
"""

import zipfile
import pandas as pd

CSV_PATH = "data/raw/pillbox_metadata.csv"
ZIP_PATH = "data/raw/pillbox_images.zip"


def main() -> None:
    # Load full metadata this time - we need has_image filtering across
    # all rows, not just a preview slice.
    df = pd.read_csv(CSV_PATH, low_memory=False)

    print(f"Total metadata rows: {len(df)}")
    with_image = df[df["has_image"] == True]
    print(f"Rows with has_image == True: {len(with_image)}")

    # Look at product_code and splimage together for a handful of
    # image-having rows, side by side.
    print("\n=== product_code vs splimage, first 10 image-having rows ===")
    print(with_image[["product_code", "splimage", "medicine_name"]].head(10))

    # Get real image filenames from the zip, split into the two patterns
    # we spotted by eye.
    with zipfile.ZipFile(ZIP_PATH) as zf:
        names = zf.namelist()

    ndc_style = [n for n in names if n[0].isdigit() and "-" in n]
    print(f"\nImage files matching NDC-hyphen pattern: {len(ndc_style)}")
    print("Sample:", ndc_style[:5])

    # Test hypothesis: does product_code (with dashes) appear as a
    # prefix of any real image filename?
    sample_codes = with_image["product_code"].dropna().astype(str).head(20)
    matches = 0
    for code in sample_codes:
        hit = [n for n in ndc_style if n.startswith(code)]
        if hit:
            matches += 1
    print(f"\nOf 20 sample product_codes, {matches} matched an image filename prefix")


if __name__ == "__main__":
    main()
