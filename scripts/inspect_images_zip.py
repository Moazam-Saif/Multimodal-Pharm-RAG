"""
Phase 1 - step 0b: peek inside the image zip without extracting it.

We need to confirm the actual filename pattern before writing any
join logic against the metadata's `splimage` column.
"""

import zipfile

ZIP_PATH = "data/raw/pillbox_images.zip"


def main() -> None:
    with zipfile.ZipFile(ZIP_PATH) as zf:
        names = zf.namelist()  # list of every file path inside the zip

        print(f"Total entries in zip: {len(names)}")

        print("\n=== First 15 entries ===")
        for name in names[:15]:
            print(name)

        print("\n=== Last 5 entries ===")
        for name in names[-5:]:
            print(name)


if __name__ == "__main__":
    main()
