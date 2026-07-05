"""
Phase 1 - step 0d: verify the refined join hypothesis.

New hypothesis: when `splimage` contains "_NLMIMAGE10_", that value
(plus ".jpg") is an EXACT filename match in the zip. The plain-digit
splimage variant (product_code with dashes stripped) does NOT correspond
to files in our zip and should be treated as "no usable image."
"""

import zipfile
import pandas as pd

CSV_PATH = "data/raw/pillbox_metadata.csv"
ZIP_PATH = "data/raw/pillbox_images.zip"


def main() -> None:
    df = pd.read_csv(CSV_PATH, low_memory=False)
    with_image = df[df["has_image"] == True].copy()

    with zipfile.ZipFile(ZIP_PATH) as zf:
        real_filenames = set(zf.namelist())  # set = fast O(1) lookup

    # Split into the two splimage patterns we observed
    is_nlmimage = with_image["splimage"].astype(str).str.contains("NLMIMAGE10", na=False)
    nlm_rows = with_image[is_nlmimage]
    plain_rows = with_image[~is_nlmimage]

    print(f"has_image=True rows with NLMIMAGE10-style splimage: {len(nlm_rows)}")
    print(f"has_image=True rows with plain-digit splimage:      {len(plain_rows)}")

    # Test: does splimage + ".jpg" exactly match a real filename?
    candidate_names = nlm_rows["splimage"].astype(str) + ".jpg"
    exact_matches = candidate_names.isin(real_filenames)
    print(f"\nOf {len(nlm_rows)} NLMIMAGE10-style rows, "
          f"{exact_matches.sum()} have an exact filename match in the zip "
          f"({exact_matches.mean():.1%})")

    # Show a few mismatches, if any, so we can see why
    if not exact_matches.all():
        print("\nSample mismatches:")
        print(candidate_names[~exact_matches].head(5).tolist())


if __name__ == "__main__":
    main()
