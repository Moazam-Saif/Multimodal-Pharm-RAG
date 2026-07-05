"""
Phase 1 - step 0i: get the FULL folder structure, not just a sample.

Our first look only showed the first 20 paths (all one folder, by luck
of sort order). We need every distinct subfolder to resolve the
classification_data vs fcn_mix_weight/dc_224 discrepancy we just found.
"""

import zipfile
from collections import Counter

ZIP_PATH = "data/raw/epillid_data.zip"


def main() -> None:
    with zipfile.ZipFile(ZIP_PATH) as zf:
        names = zf.namelist()

    # Get every unique DIRECTORY (not file) path, at every depth level
    dirs = set()
    for name in names:
        parts = name.split("/")
        # Build up each parent directory path, e.g. for "a/b/c.jpg":
        # "a/", "a/b/"
        for i in range(1, len(parts)):
            dirs.add("/".join(parts[:i]) + "/")

    print(f"Total unique directories: {len(dirs)}")
    print("\n=== All directories, sorted ===")
    for d in sorted(dirs):
        print(d)

    # Count how many files sit directly under each of the main image folders
    print("\n=== File counts per folder (top image-holding folders) ===")
    folder_counts = Counter()
    for name in names:
        if name.endswith(".jpg") or name.endswith(".png"):
            folder = "/".join(name.split("/")[:-1]) + "/"
            folder_counts[folder] += 1

    for folder, count in sorted(folder_counts.items(), key=lambda x: -x[1]):
        print(f"{count:6d}  {folder}")


if __name__ == "__main__":
    main()
