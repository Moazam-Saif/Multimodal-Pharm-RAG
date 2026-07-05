"""
Phase 1 - step 0g: inspect the ePillID dataset zip structure.

Same technique as before - list contents without extracting, so we
understand the real folder/file layout before writing any code.
"""

import zipfile

ZIP_PATH = "data/raw/epillid_data.zip"


def main() -> None:
    with zipfile.ZipFile(ZIP_PATH) as zf:
        names = zf.namelist()
        print(f"Total entries: {len(names)}")

        # Show the top-level folder structure by looking at unique
        # first-path-segments (e.g. "classification_data/" vs "images/")
        top_level = sorted(set(n.split("/")[0] for n in names))
        print("\n=== Top-level entries ===")
        for t in top_level:
            print(t)

        print("\n=== First 20 full paths ===")
        for name in names[:20]:
            print(name)

        # Look specifically for anything that looks like a CSV/label file
        label_files = [n for n in names if n.endswith(".csv")]
        print(f"\n=== CSV files found ({len(label_files)}) ===")
        for f in label_files:
            print(f)


if __name__ == "__main__":
    main()
