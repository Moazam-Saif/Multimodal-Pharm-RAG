"""
Phase 1: load and normalize the project's dataset.

This module ties together everything we verified via the diagnostic
scripts in scripts/ into one reusable, tested function. See DEVLOG.md
for the full investigation trail explaining every decision below.

Data sources:
    - epillid_data.zip       (images + ground truth labels)
    - pillbox_metadata.csv   (text metadata, joined by NDC)

File locations are configurable via environment variables (see
.env.example), defaulting to the local project layout
(data/raw/...). This matters because this module runs in more than
one place with different filesystems - locally (Windows, files live
under the project's data/raw/) and in Colab (files live under a
mounted Google Drive path instead). Originally these paths were
hardcoded, which broke silently the first time this code ran in Colab
- see DEVLOG.md Phase 2 notes for that incident.

Scope (confirmed working end-to-end, see DEVLOG.md "FINAL Phase 1
dataset decision"):
    - Only the 1,000 NDC-labeled pill types from ePillID
      (fcn_mix_weight/dr_224 + dc_224), NOT segmented_nih_pills_224
    - Text metadata recovered for ~96.8% of these via NDC join
"""

import io
import os
import re
import zipfile
from pathlib import Path

import pandas as pd

EPILLID_ZIP = os.environ.get("EPILLID_ZIP_PATH", "data/raw/epillid_data.zip")
EPILLID_LABELS_INSIDE_ZIP = "ePillID_data/all_labels.csv"
EPILLID_IMAGE_PREFIX_INSIDE_ZIP = "ePillID_data/classification_data/"

PILLBOX_METADATA_CSV = os.environ.get(
    "PILLBOX_METADATA_CSV_PATH", "data/raw/pillbox_metadata.csv"
)


def _is_gcs_path(path) -> bool:
    """True if path is a gs:// URI, not a local filesystem path.

    REAL BUG FOUND AND FIXED this session: EPILLID_ZIP and
    PILLBOX_METADATA_CSV were originally wrapped in pathlib.Path(...)
    at module load time. pathlib.Path SILENTLY collapses "gs://" down
    to "gs:/" (double-slash -> single-slash) the moment the string is
    parsed as a path, since Path is fundamentally a local-filesystem
    abstraction with no concept of URI scheme double-slashes. This
    produced a confusing FileNotFoundError for 'gs:/bucket/...' (one
    slash) that looked like a typo in the path, when the REAL bug was
    ever wrapping a gs:// URI in Path() at all. Fixed by keeping these
    as plain strings at module level, and only ever constructing a
    real pathlib.Path for genuinely-local paths, inside
    _open_binary_any below - never for a string that might be a gs://
    URI.
    """
    return str(path).startswith("gs://")


def _open_binary_any(path):
    """Open a file for binary reading, working for BOTH local paths
    and gs:// URIs.

    REAL BUG FOUND AND FIXED this session (see train_metric_learning.py's
    matching helpers and DEVLOG.md for the full GCP migration
    credentials debugging trail this came out of): plain
    zipfile.ZipFile() and pandas' native gs:// handling do NOT reliably
    pick up this project's working Application Default Credentials
    (confirmed: pyarrow's native GCS filesystem raised a real,
    reproduced PermissionError/UNAUTHENTICATED error even though
    google.cloud.storage.Client() and gcsfs.GCSFileSystem() both
    authenticate fine via ADC in the same environment). Fix: for
    gs:// paths, read the ENTIRE file into an in-memory BytesIO via
    gcsfs explicitly, then hand callers that in-memory buffer instead
    of a bare gs:// string - zipfile and pandas both accept a
    file-like object just as happily as a real local path, and this
    sidesteps the broken native-GCS-credential code paths entirely.

    ALSO FIXED this session: path is checked for gs:// BEFORE any
    pathlib.Path() wrapping happens - see _is_gcs_path's docstring for
    why wrapping a gs:// URI in Path() silently corrupts it.

    NOTE: this reads the whole file into memory - fine for
    pillbox_metadata.csv (small) and acceptable for epillid_data.zip
    (read once per build_pill_dataset() call; zipfile.ZipFile needs
    random access into the archive anyway, which BytesIO supports and
    a streaming gcsfs handle would not support as cleanly).
    """
    if _is_gcs_path(path):
        import gcsfs
        fs = gcsfs.GCSFileSystem()
        with fs.open(str(path).replace("gs://", ""), "rb") as f:
            return io.BytesIO(f.read())
    else:
        # Only NOW, for a confirmed-local path, is it safe to
        # construct a real pathlib.Path and open it normally.
        return open(Path(path), "rb")

# Pillbox text fields we want to recover per pill - see DEVLOG for why
# these specific fields (medicine_name, spl_strength, spl_ingredients
# are the most useful for Phase 5 text RAG; more can be added later)
PILLBOX_TEXT_FIELDS = ["medicine_name", "spl_strength", "spl_ingredients"]

# Shape is needed for Phase 2 (choosing a classical-CV fallback technique
# per shape when FastSAM's confidence/fill-ratio filtering can't find a
# usable mask - e.g. Hough Circle detection for round pills). Pillbox has
# TWO shape columns per Phase 1's investigation: pillbox_shape_text is
# only populated when NLM visually verified the shape against a real
# photo and found it differs from the label; spl_shape_text is what the
# manufacturer originally submitted. Established rule (see DEVLOG.md
# Phase 1): prefer pillbox_* when present (visually verified), fall back
# to spl* otherwise.
#
# Color is needed for the same Phase 2 reason - specifically to properly
# test the "low-contrast white-on-white/gray" failure mode by shape,
# rather than assuming a small random sample is representative (see
# DEVLOG.md - a 3-image spot check suggested OVAL might be robust to
# this problem, but that's too small a sample to trust on its own).
# Same pillbox_*-preferred, fallback-to-spl* rule applies.


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
    with zipfile.ZipFile(_open_binary_any(EPILLID_ZIP)) as zf:
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
    df = pd.read_csv(_open_binary_any(PILLBOX_METADATA_CSV), low_memory=False)
    df["ndc_normalized"] = (
        df["product_code"].dropna().astype(str).apply(_normalize_ndc)
    )
    df = df[df["ndc_normalized"].notna()]

    # Resolve shape: prefer pillbox_shape_text (visually verified),
    # fall back to splshape_text (as manufacturer-submitted) - see
    # DEVLOG.md Phase 1 for why this specific fallback direction.
    df["shape"] = df["pillbox_shape_text"].fillna(df["splshape_text"])

    # Same fallback rule for color.
    df["color"] = df["pillbox_color_text"].fillna(df["splcolor_text"])

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
        pillbox_lookup[[*PILLBOX_TEXT_FIELDS, "shape", "color"]],
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
            "shape",
            "color",
        ]
    ]


if __name__ == "__main__":
    # Quick manual check when run directly: python -m pillrag.data
    dataset = build_pill_dataset()
    print(f"Total rows: {len(dataset)}")
    print(f"Rows with recovered text metadata: {dataset['medicine_name'].notna().sum()}")
    print(dataset.head())