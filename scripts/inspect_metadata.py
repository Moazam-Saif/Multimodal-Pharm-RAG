"""
Phase 1 - step 0: just look at what we actually have.

Before writing any normalization logic, we need to know, with certainty,
the real column names and a sample of real values. Guessing from a
screenshot or the plan doc is how bugs get baked in early.
"""

import pandas as pd

CSV_PATH = "data/raw/pillbox_metadata.csv"


def main() -> None:
    # nrows=20 - we don't need to load all ~84,000 rows into memory
    # just to look at the shape of the data. Read a small slice first.
    df = pd.read_csv(CSV_PATH, nrows=20)

    print("=== Column names (in order) ===")
    for i, col in enumerate(df.columns):
        print(f"{i:2d}  {col}")

    print("\n=== First 3 rows, transposed for readability ===")
    # .T flips rows/columns - with ~50 columns, a normal print wraps
    # illegibly. Transposed, each row becomes a column we can read
    # top to bottom.
    print(df.head(3).T)


if __name__ == "__main__":
    main()