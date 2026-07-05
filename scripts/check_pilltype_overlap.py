"""
Phase 1 - step 0u: do we even need the raw-hash-labeled pills?

Since we're running our OWN segmentation (not using ePillID's
pre-segmented output), the real question is: are the pill types
represented by segmented_nih_pills_224/ (hash labels, is_ref=True)
ALREADY also present in fcn_mix_weight/dr_224+dc_224 (NDC labels,
text-joinable)? If so, we can just ignore this folder entirely -
no unique pills are lost.
"""

import zipfile
import pandas as pd

EPILLID_ZIP = "data/raw/epillid_data.zip"
EPILLID_LABELS = "ePillID_data/all_labels.csv"


def main() -> None:
    with zipfile.ZipFile(EPILLID_ZIP) as zf:
        with zf.open(EPILLID_LABELS) as f:
            epillid = pd.read_csv(f)

    is_hash_style = ~epillid["label"].str.contains("-", na=False)

    # pilltype_id is the stable identity column regardless of label format
    hash_pilltypes = set(epillid[is_hash_style]["pilltype_id"])
    ndc_pilltypes = set(epillid[~is_hash_style]["pilltype_id"])

    print(f"Unique pill types in hash-labeled rows:  {len(hash_pilltypes)}")
    print(f"Unique pill types in NDC-labeled rows:    {len(ndc_pilltypes)}")

    overlap = hash_pilltypes & ndc_pilltypes
    only_in_hash = hash_pilltypes - ndc_pilltypes

    print(f"\nPill types in BOTH groups: {len(overlap)}")
    print(f"Pill types ONLY in hash-labeled rows (would be lost if we drop that folder): {len(only_in_hash)}")

    if only_in_hash:
        print(f"\nSample of pill-types we'd lose: {list(only_in_hash)[:5]}")


if __name__ == "__main__":
    main()
