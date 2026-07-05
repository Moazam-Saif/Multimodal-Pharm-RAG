"""
Phase 1 - step 0s: before fixing the crash, understand it.

Find every product_code value where a segment isn't purely numeric,
see how common this is and what it looks like, THEN decide how to
handle it (skip, or normalize differently).
"""

import pandas as pd

PILLBOX_CSV = "data/raw/pillbox_metadata.csv"


def main() -> None:
    pillbox = pd.read_csv(PILLBOX_CSV, low_memory=False)
    codes = pillbox["product_code"].dropna().astype(str)

    def has_non_numeric_segment(code: str) -> bool:
        parts = code.split("-")
        return any(not p.isdigit() for p in parts[:2])  # only check labeler+product

    bad_mask = codes.apply(has_non_numeric_segment)
    print(f"Rows with a non-numeric labeler/product segment: {bad_mask.sum()} / {len(codes)} ({bad_mask.mean():.2%})")

    print("\nSample of the actual weird values:")
    print(codes[bad_mask].head(15).tolist())


if __name__ == "__main__":
    main()
