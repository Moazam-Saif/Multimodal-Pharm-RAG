"""
Phase 1: load and normalize the project's dataset.

This module ties together everything we verified via the diagnostic
scripts in scripts/ into one reusable, tested function. See DEVLOG.md
for the full investigation trail explaining every decision below.

Data sources:
    - data/raw/epillid_data.zip       (images + ground truth labels)
    - data/raw/pillbox_metadata.csv   (text metadata, joined by NDC)

Scope (confirmed working end-to-end, see DEVLOG.md "FINAL Phase 1
dataset decision"):
    - Only the 1,000 NDC-labeled pill types from ePillID
      (fcn_mix_weight/dr_224 + dc_224), NOT segmented_nih_pills_224
    - Text metadata recovered for ~96.8% of these via NDC join
"""

import re
import zipfile
from pathlib import Path

import pandas as pd

EPILLID_ZIP = Path("data/raw/epillid_data.zip")
EPILLID_LABELS_INSIDE_ZIP = "ePillID_data/all_labels.csv"
EPILLID_IMAGE_PREFIX_INSIDE_ZIP = "ePillID_data/classification_data/"

PILLBOX_METADATA_CSV = Path("data/raw/pillbox_metadata.csv")

# Pillbox text fields we want to recover per pill - see DEVLOG for why
# these specific fields (medicine_name, spl_strength, spl_ingredients
# are the most useful for Phase 5 text RAG; more can be added later)
PILLBOX_TEXT_FIELDS = ["medicine_name", "spl_strength", "spl_ingredients"]


def _digits_only(s: str) -> str:
    """Strip everything except digits from a string."""
    return re.sub(r"\D", "", s)


def _normalize_ndc(ndc_str: str) -> str | None:
    """Normalize an NDC to a comparable form: labeler+product segments
    only (drops package code, which Pillbox's product_code never has),
    each segment int-converted separately to strip inconsistent leading
    zeros, rejoined with '.' so segments can't bleed into each other.

    Returns None for anything that doesn't parse (e.g. non-numeric
    segments like "0019-N601" - confirmed rare, ~0.00% of Pillbox rows,
    see DEVLOG.md).
    """
    parts = ndc_str.split("-")
    if len(parts) < 2:
        return None
    labeler, product = parts[0], parts[1]
    if not (labeler.isdigit() and product.isdigit()):
        return None
    return f"{int(labeler)}.{int(product)}"


def load_epillid_labels() -> pd.DataFrame:
    """Load and filter all_labels.csv down to our confirmed-working
    scope: NDC-hex-style labels only (the 1,000-pill-type subset),
    with full, verified-resolvable image paths.
    """
    with zipfile.ZipFile(EPILLID_ZIP) as zf:
        with zf.open(EPILLID_LABELS_INSIDE_ZIP) as f:
            df = pd.read_csv(f)

    # NDC-hex style labels contain a "-" (e.g. "51285-0092-87_BE305F72");
    # raw-hash style labels don't. We only keep the former - see
    # DEVLOG.md "FINAL Phase 1 dataset decision" for why.
    is_ndc_style = df["label"].str.contains("-", na=False)
    df = df[is_ndc_style].copy()

    # Confirmed prefix via diagnose_path_mismatch.py - the real files
    # live one level deeper than all_labels.csv's image_path implies.
    df["full_image_path"] = EPILLID_IMAGE_PREFIX_INSIDE_ZIP + df["image_path"]

    # Extract just the NDC portion (everything before "_") from label
    df["ndc_raw"] = df["label"].str.split("_").str[0]
    df["ndc_normalized"] = df["ndc_raw"].apply(_normalize_ndc)

    return df


def load_pillbox_text_lookup() -> pd.DataFrame:
    """Load Pillbox metadata, normalized and de-duplicated to one row
    per unique NDC, ready to join against ePillID's normalized NDCs.
    """
    df = pd.read_csv(PILLBOX_METADATA_CSV, low_memory=False)
    df["ndc_normalized"] = (
        df["product_code"].dropna().astype(str).apply(_normalize_ndc)
    )
    df = df[df["ndc_normalized"].notna()]
    return df.drop_duplicates("ndc_normalized").set_index("ndc_normalized")


def build_pill_dataset() -> pd.DataFrame:
    """The main entry point: produces one clean table, one row per
    ePillID image, with image path + recovered text metadata joined in.

    Rows where text metadata couldn't be recovered (~3.2% - see DEVLOG)
    still have valid images, just null text fields. Callers doing
    text-RAG-specific work should filter on notna() themselves rather
    than have this function silently drop rows - keeping all pills
    with valid images is the more useful default for the image-only
    parts of the pipeline (Phase 2/3).
    """
    epillid = load_epillid_labels()
    pillbox_lookup = load_pillbox_text_lookup()

    merged = epillid.merge(
        pillbox_lookup[PILLBOX_TEXT_FIELDS],
        how="left",
        left_on="ndc_normalized",
        right_index=True,
    )

    return merged[
        [
            "full_image_path",
            "pilltype_id",
            "label",
            "is_ref",
            "is_front",
            *PILLBOX_TEXT_FIELDS,
        ]
    ]


if __name__ == "__main__":
    # Quick manual check when run directly: python -m pillrag.data
    dataset = build_pill_dataset()
    print(f"Total rows: {len(dataset)}")
    print(f"Rows with recovered text metadata: {dataset['medicine_name'].notna().sum()}")
    print(dataset.head())
