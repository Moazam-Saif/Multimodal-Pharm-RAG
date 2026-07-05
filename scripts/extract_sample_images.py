"""
Phase 2 - step 0: pull out sample images so we can actually look at
them before writing any segmentation code.

Extracts a few reference + consumer images (RANDOM, not just the first
pill) so we don't draw conclusions from one lucky/unlucky example.
"""

import zipfile
import random
from pillrag.data import build_pill_dataset

OUTPUT_DIR = "data/samples"
EPILLID_ZIP = "data/raw/epillid_data.zip"
N_SAMPLES = 6


def main() -> None:
    df = build_pill_dataset()
    random.seed(42)  # reproducible - same "random" picks every run

    ref_sample = df[df["is_ref"] == True].sample(N_SAMPLES, random_state=42)
    consumer_sample = df[df["is_ref"] == False].sample(N_SAMPLES, random_state=42)

    with zipfile.ZipFile(EPILLID_ZIP) as zf:
        for i, (_, row) in enumerate(ref_sample.iterrows()):
            with zf.open(row["full_image_path"]) as src, open(f"{OUTPUT_DIR}/ref_{i}.jpg", "wb") as dst:
                dst.write(src.read())
        for i, (_, row) in enumerate(consumer_sample.iterrows()):
            with zf.open(row["full_image_path"]) as src, open(f"{OUTPUT_DIR}/consumer_{i}.jpg", "wb") as dst:
                dst.write(src.read())

    print(f"Saved {N_SAMPLES} reference images (ref_0.jpg .. ref_{N_SAMPLES-1}.jpg)")
    print(f"Saved {N_SAMPLES} consumer images (consumer_0.jpg .. consumer_{N_SAMPLES-1}.jpg)")
    print(f"All in {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()



if __name__ == "__main__":
    main()