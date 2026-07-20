"""
Phase 4: metric-learning fine-tuning of the pill embedding model.

Motivation (see DEVLOG.md's diagnostic trail for the full evidence
chain): search_visual's 0% top-1 accuracy on a 50-image sample was
traced through - query mechanics ruled out, shape-filter logic ruled
out, embedding-pipeline drift between index-build-time and query-time
ruled out (fresh vs. stored embedding for the same image: cosine
similarity 1.000000, bit-identical) - landing on the embedding model
itself: a zero-shot ImageNet-pretrained ResNet-18 was never trained to
distinguish fine-grained pharmaceutical detail (score lines, rim
geometry, subtle indentations) from any pill against any other pill.
Confirmed uniform failure across every shape/color category, not
concentrated in "similar-looking" pills specifically - consistent with
a generally non-discriminative embedding space for this task.

This module trains a NEW embedding model via metric learning
(Supervised Contrastive Loss) so the embedding space is directly
optimized for "same pill close together, different pill far apart",
rather than inheriting whatever notion of similarity ImageNet
classification happened to produce as a side effect.

Real decisions made this session (see DEVLOG.md for full reasoning):
  - Backbone: ResNet-50 (upgrade from the existing ResNet-18 in
    embed.py) - matches the ePillID paper's own baseline architecture.
  - Resolution: 384x384 (upgrade from the existing 224x224).
  - Both of the above were called out in the design report as steps to
    defer until AFTER the training-objective change was validated -
    explicit user decision to do all three at once instead. Known
    consequence: if results improve, we won't cleanly know which
    change (objective/backbone/resolution) drove it.
  - Training data: ONLY the 2,000 reference images (is_ref==True) -
    the same set already indexed in Deep Lake. Consumer images
    (is_ref==False) are the FINAL EVAL set and must stay untouched
    until after training is fully done - using them for
    validation-during-training would leak information into checkpoint
    selection and make the final eval number not truly held-out.
  - Train/val split: 800/200 pill types (not images - every image of
    a given pilltype_id stays in the same split, never leaking one
    angle of a pill into train while its sibling image is in val).
  - Masks: reused directly from Phase 2's manifest parquet files
    (data/masks/manifest_chunk_000.parquet .. _011.parquet), NOT
    re-segmented - matches what the live Deep Lake index was actually
    built from, and avoids re-running FastSAM on every image on every
    epoch. RLE format confirmed via direct decode test this session:
    standard pycocotools COCO RLE (rle_size=[H,W], rle_counts=string),
    decodes cleanly with pycocotools.mask.decode(), no bytes-encoding
    workaround needed.
  - The manifest itself does NOT have an is_ref column - it's joined
    back to build_pill_dataset()'s df on full_image_path to recover
    is_ref (and any other Pillbox metadata needed later, e.g. color).
  - Batch construction: BALANCED sampler, not plain random shuffling.
    SupCon needs real positive pairs (same pilltype_id) within a
    batch to have any "pull together" signal at all - with only ~2
    images per pill type, plain random batches would leave most
    batches with zero true positive pairs, wasting that half of the
    loss. Each batch draws N distinct pill types and includes BOTH of
    that type's images - guarantees every image has a real positive
    in-batch, every batch, every step.
  - Epoch definition: one epoch = every training pill type appears in
    EXACTLY ONE batch (shuffle the 800 training pill types, chop into
    consecutive groups of N). Standard "one full pass" semantics -
    chosen over a fixed-batches-per-epoch/sample-with-replacement
    scheme, since the dataset is small, fixed-size, and every class
    has the same number of images (2) - no real benefit to the more
    complex alternative here.
"""

from __future__ import annotations

import glob
import os
import random
from dataclasses import dataclass

import numpy as np
import pandas as pd
from pycocotools import mask as maskUtils

from pillrag.data import build_pill_dataset

MANIFEST_GLOB = os.environ.get(
    "PILL_MASK_MANIFEST_GLOB",
    "/content/drive/MyDrive/pill-rag/data/masks/manifest_chunk_*.parquet",
)

TRAIN_VAL_SPLIT_SEED = 42
N_VAL_PILLTYPES = 200


def load_reference_manifest() -> pd.DataFrame:
    """Load all 12 manifest chunks, join back to build_pill_dataset()'s
    df to recover is_ref (and other Pillbox metadata), filter to
    is_ref==True only - the 2,000 reference images already indexed in
    Deep Lake.

    Returns a DataFrame with one row per reference image: the
    manifest's own columns (full_image_path, pilltype_id, label,
    shape, method, quality_flag, rle_size, rle_counts) PLUS df's
    columns (is_ref, is_front, medicine_name, spl_strength,
    spl_ingredients, color - shape is already in the manifest too,
    but we keep df's version as the authoritative one since the
    manifest's `shape` was passed in as known_shape at segmentation
    time, sourced from the same df originally anyway).
    """
    chunk_paths = sorted(glob.glob(MANIFEST_GLOB))
    if not chunk_paths:
        raise FileNotFoundError(
            f"No manifest chunks found matching {MANIFEST_GLOB!r} - "
            f"check PILL_MASK_MANIFEST_GLOB env var / Drive mount."
        )

    manifest = pd.concat(
        [pd.read_parquet(p) for p in chunk_paths],
        ignore_index=True,
    )

    dataset_df = build_pill_dataset()

    merged = manifest.merge(
        dataset_df[
            ["full_image_path", "is_ref", "is_front", "medicine_name", "color"]
        ],
        on="full_image_path",
        how="left",
    )

    reference_only = merged[merged["is_ref"] == True].copy()  # noqa: E712

    return reference_only


def decode_rle_mask(rle_size, rle_counts) -> np.ndarray:
    """Decode one manifest row's RLE-encoded mask into a boolean
    (H, W) array. Confirmed working this session via a direct decode
    test on a real manifest row - standard pycocotools COCO RLE,
    counts as a plain string, no bytes-encoding step needed.
    """
    rle = {"size": list(rle_size), "counts": rle_counts}
    decoded = maskUtils.decode(rle)
    return decoded.astype(bool)


@dataclass
class TrainValSplit:
    """Train/val split by pilltype_id, not by image - every image of
    a given pill type stays in the same split.
    """

    train_df: pd.DataFrame
    val_df: pd.DataFrame
    train_pilltype_ids: list[str]
    val_pilltype_ids: list[str]


def make_train_val_split(
    reference_df: pd.DataFrame,
    n_val_pilltypes: int = N_VAL_PILLTYPES,
    seed: int = TRAIN_VAL_SPLIT_SEED,
) -> TrainValSplit:
    """Split the 1,000 reference pilltype_ids into train/val groups,
    holding out n_val_pilltypes for validation-during-training only.

    Consumer images (is_ref==False) are NEVER touched here - they are
    the final eval set, held out entirely until after training
    finishes. This split is ONLY about which reference pilltype_ids
    the training loop gets to see vs. which it doesn't, for the
    purpose of checkpoint selection / monitoring training progress.
    """
    all_pilltype_ids = sorted(reference_df["pilltype_id"].unique())

    rng = random.Random(seed)
    shuffled = all_pilltype_ids.copy()
    rng.shuffle(shuffled)

    val_pilltype_ids = shuffled[:n_val_pilltypes]
    train_pilltype_ids = shuffled[n_val_pilltypes:]

    train_df = reference_df[
        reference_df["pilltype_id"].isin(train_pilltype_ids)
    ].reset_index(drop=True)
    val_df = reference_df[
        reference_df["pilltype_id"].isin(val_pilltype_ids)
    ].reset_index(drop=True)

    return TrainValSplit(
        train_df=train_df,
        val_df=val_df,
        train_pilltype_ids=train_pilltype_ids,
        val_pilltype_ids=val_pilltype_ids,
    )


if __name__ == "__main__":
    # Quick manual check when run directly: python -m pillrag.train_metric_learning
    ref_df = load_reference_manifest()
    print(f"Reference manifest rows (is_ref==True): {len(ref_df)}")
    print(f"Unique pilltype_ids: {ref_df['pilltype_id'].nunique()}")
    print(f"quality_flag breakdown:\n{ref_df['quality_flag'].value_counts()}")

    split = make_train_val_split(ref_df)
    print(f"\nTrain pilltypes: {len(split.train_pilltype_ids)}, "
          f"images: {len(split.train_df)}")
    print(f"Val pilltypes: {len(split.val_pilltype_ids)}, "
          f"images: {len(split.val_df)}")

    # Sanity check: no pilltype_id leakage across splits
    overlap = set(split.train_pilltype_ids) & set(split.val_pilltype_ids)
    print(f"\nTrain/val pilltype_id overlap (should be 0): {len(overlap)}")
