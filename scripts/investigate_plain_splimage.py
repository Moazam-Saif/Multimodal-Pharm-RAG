"""
Phase 1 - step 0e: figure out the plain-digit splimage pattern.

866 rows matched cleanly via NLMIMAGE10. The remaining 8,913 use a
plain-digit splimage (e.g. "498840129"). Test whether THAT value,
not product_code, matches the start of an NDC-hyphen style filename
once we strip the hyphens from the filename side instead.
"""

import re
import zipfile
import pandas as pd

CSV_PATH = "data/raw/pillbox_metadata.csv"
ZIP_PATH = "data/raw/pillbox_images.zip"


def main() -> None:
    df = pd.read_csv(CSV_PATH, low_memory=False)
    with_image = df[df["has_image"] == True].copy()

    is_nlmimage = with_image["splimage"].astype(str).str.contains("NLMIMAGE10", na=False)
    plain_rows = with_image[~is_nlmimage]

    with zipfile.ZipFile(ZIP_PATH) as zf:
        real_filenames = zf.namelist()

    # Build a lookup: for each NDC-hyphen style filename, strip
    # everything down to just digits before the underscore, so
    # "00002-3228-30_391E1C80.jpg" -> "00002322830"
    ndc_style = [n for n in real_filenames if re.match(r"^\d", n) and "-" in n]

    digit_to_filename = {}
    for name in ndc_style:
        prefix = name.split("_")[0]          # "00002-3228-30"
        digits_only = prefix.replace("-", "")  # "00002322830"
        digit_to_filename[digits_only] = name

    print(f"Built lookup of {len(digit_to_filename)} digit-normalized filenames")

    # Test: does plain splimage match this digit-only key?
    plain_values = plain_rows["splimage"].dropna().astype(str)
    plain_values = plain_values.apply(lambda s: s.replace(".0", ""))  # pandas float artifact

    matches = plain_values.isin(digit_to_filename.keys())
    print(f"\nOf {len(plain_values)} plain-digit splimage rows, "
          f"{matches.sum()} matched a digit-normalized filename ({matches.mean():.1%})")

    if matches.sum() > 0:
        print("\nSample successful matches (splimage -> real filename):")
        matched_vals = plain_values[matches].head(5)
        for v in matched_vals:
            print(f"  {v} -> {digit_to_filename[v]}")


if __name__ == "__main__":
    main()
