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
  - Data quality check (verified this session, not assumed): confirmed
    via real script output that every one of the 1,000 reference
    pilltype_ids has exactly 2 images, no exceptions, no missing types
    - the balanced sampler's "both images of N pill types per batch"
    design is safe to rely on. Also found 4/2000 reference rows have
    quality_flag=fallback_full_image (degenerate all-True mask, real
    background pixels baked into the embedding per embed.py's own
    documented consequence). Checked whether any pilltype_id has BOTH
    its images as fallback (zero real signal for that class, not just
    one degraded side of a pair) - found exactly one:
    00093-1003-01_B326D9D6. Explicit decision: EXCLUDE this one
    pilltype_id from the training split entirely (see
    EXCLUDED_ZERO_SIGNAL_PILLTYPES below) - training a positive pair
    from two background-heavy full-image crops would teach the model
    "these images are the same class" based on background/lighting
    similarity, not real pill signal, which is actively harmful rather
    than neutral noise. The other 3 fallback rows (each has one real
    `ok` sibling image) are kept as-is - a materially different,
    milder situation with real signal still present on one side of
    the pair. This pilltype_id is NOT removed from the manifest load
    itself (still visible/countable in load_reference_manifest's
    output) - only excluded at the train/val split stage, so it's
    still present for anyone inspecting the raw reference data, and
    still exists in the live Deep Lake index / eval set as before.
"""

from __future__ import annotations

import glob
import os
import random
from dataclasses import dataclass

import numpy as np
import pandas as pd
import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2
from PIL import Image
from pycocotools import mask as maskUtils
from torch.utils.data import Dataset

from pillrag.data import build_pill_dataset
from pillrag.embed import IMAGENET_MEAN, IMAGENET_STD, mask_bounding_box

TRAIN_RESOLUTION = 384

MANIFEST_GLOB = os.environ.get(
    "PILL_MASK_MANIFEST_GLOB",
    "/content/drive/MyDrive/pill-rag/data/masks/manifest_chunk_*.parquet",
)

TRAIN_VAL_SPLIT_SEED = 42
N_VAL_PILLTYPES = 200

# Verified this session (see this module's docstring): the ONLY
# reference pilltype_id where BOTH images are quality_flag=
# fallback_full_image, i.e. zero real pill-crop signal. Excluded from
# the train/val split - NOT from load_reference_manifest's output, and
# NOT from the live Deep Lake index / eval set.
EXCLUDED_ZERO_SIGNAL_PILLTYPES = ["00093-1003-01_B326D9D6"]


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

    EXCLUDED_ZERO_SIGNAL_PILLTYPES (currently just
    00093-1003-01_B326D9D6) are removed before splitting - see this
    module's docstring for why. They end up in NEITHER train_df NOR
    val_df.
    """
    eligible_df = reference_df[
        ~reference_df["pilltype_id"].isin(EXCLUDED_ZERO_SIGNAL_PILLTYPES)
    ]

    all_pilltype_ids = sorted(eligible_df["pilltype_id"].unique())

    rng = random.Random(seed)
    shuffled = all_pilltype_ids.copy()
    rng.shuffle(shuffled)

    val_pilltype_ids = shuffled[:n_val_pilltypes]
    train_pilltype_ids = shuffled[n_val_pilltypes:]

    train_df = eligible_df[
        eligible_df["pilltype_id"].isin(train_pilltype_ids)
    ].reset_index(drop=True)
    val_df = eligible_df[
        eligible_df["pilltype_id"].isin(val_pilltype_ids)
    ].reset_index(drop=True)

    return TrainValSplit(
        train_df=train_df,
        val_df=val_df,
        train_pilltype_ids=train_pilltype_ids,
        val_pilltype_ids=val_pilltype_ids,
    )


def build_train_augmentation() -> A.Compose:
    """The TRAIN-only augmentation pipeline.

    Deliberately NOT a generic augmentation preset - reasoned through
    per-transform against this task's specific failure mode (confusion
    between visually similar pills over SUBTLE features: score lines,
    rim geometry, indentations). See this module's docstring for the
    full safe-vs-risky reasoning. Summary:

    SAFE, used at meaningful strength:
      - Full 360° rotation - pills have no "upright"; teaches genuine
        rotation-invariance without touching fine detail at all.
      - Horizontal + vertical flip - same reasoning.
      - MILD brightness/contrast jitter - real consumer photos vary in
        lighting, but pushed too far this would wash out the surface-
        shading cues that ARE part of the real signal (how light
        catches an indentation/score line).
      - Slight random crop / scale jitter - kept NARROW, since the
        pipeline already crops tightly to the segmentation mask's
        bbox; aggressive jitter risks cropping out real rim/edge
        detail that matters.

    AVOIDED or minimal:
      - Blur - SKIPPED entirely. Score lines and rim geometry are
        fine, thin detail; any meaningful blur risks smoothing away
        exactly the feature being trained to detect.
      - Heavy color/hue jitter - SKIPPED. Color is often a REAL,
        load-bearing distinguishing feature here (Pillbox's own
        metadata tracks color as identifying), not a nuisance
        variable to be jittered away.
      - Random erasing/cutout - SKIPPED. Real risk of blanking the one
        informative region (imprint/score line) in an already-small
        crop, with no guarantee anything equally informative remains.
    """
    return A.Compose([
        A.LongestMaxSize(max_size=TRAIN_RESOLUTION),
        A.PadIfNeeded(
            min_height=TRAIN_RESOLUTION,
            min_width=TRAIN_RESOLUTION,
            border_mode=cv2.BORDER_CONSTANT,
            value=(255, 255, 255),
        ),
        A.Rotate(limit=180, border_mode=cv2.BORDER_CONSTANT, value=(255, 255, 255), p=0.9),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(
            brightness_limit=0.15, contrast_limit=0.15, p=0.5
        ),
        A.RandomResizedCrop(
            size=(TRAIN_RESOLUTION, TRAIN_RESOLUTION),
            scale=(0.85, 1.0),  # narrow - don't crop out real edge/rim detail
            ratio=(0.95, 1.05),
            p=0.5,
        ),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def build_eval_augmentation() -> A.Compose:
    """The VAL/eval pipeline - NO augmentation, just deterministic
    resize + pad + normalize. Used for the held-out validation split
    (and later, for the real consumer-image eval and for building the
    Deep Lake index) so results are reproducible and not inflated/
    deflated by random augmentation.
    """
    return A.Compose([
        A.LongestMaxSize(max_size=TRAIN_RESOLUTION),
        A.PadIfNeeded(
            min_height=TRAIN_RESOLUTION,
            min_width=TRAIN_RESOLUTION,
            border_mode=cv2.BORDER_CONSTANT,
            value=(255, 255, 255),
        ),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


class PillPairDataset(Dataset):
    """A single reference image + its resolved mask, cropped and
    (optionally) augmented, ready for the model.

    Each item is ONE image, not a pre-assembled pair/triplet - pairing
    happens at the BATCH level via the balanced sampler (see
    PillBalancedBatchSampler below), not here. This dataset's only job
    is: given a manifest row, produce (image_tensor, pilltype_id).

    Cropping: reuses embed.py's mask_bounding_box() - SAME tight-bbox-
    crop-no-internal-masking convention already established for
    embed_image(), for consistency between how the live index was
    built and how this training data is prepared. Real background
    pixels inside the bbox are kept as-is, same known consequence
    documented in embed.py (worse for loose/fallback masks - relevant
    here for the 3 remaining fallback_full_image rows in the training
    set, see this module's docstring for why those were kept vs. the
    one fully-excluded pilltype).
    """

    def __init__(self, dataframe: pd.DataFrame, augment: bool):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = (
            build_train_augmentation() if augment else build_eval_augmentation()
        )

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, idx: int):
        row = self.dataframe.iloc[idx]

        image = Image.open(row["full_image_path"]).convert("RGB")
        image_array = np.array(image)

        mask = decode_rle_mask(row["rle_size"], row["rle_counts"])

        top, bottom, left, right = mask_bounding_box(mask)
        cropped = image_array[top : bottom + 1, left : right + 1]

        augmented = self.transform(image=cropped)
        image_tensor = augmented["image"]

        return image_tensor, row["pilltype_id"]
    # Quick manual check when run directly: python -m pillrag.train_metric_learning
    ref_df = load_reference_manifest()
    print(f"Reference manifest rows (is_ref==True): {len(ref_df)}")
    print(f"Unique pilltype_ids: {ref_df['pilltype_id'].nunique()}")
    print(f"quality_flag breakdown:\n{ref_df['quality_flag'].value_counts()}")

    split = make_train_val_split(ref_df)
    print(f"\nExcluded zero-signal pilltypes: {EXCLUDED_ZERO_SIGNAL_PILLTYPES}")
    print(f"Train pilltypes: {len(split.train_pilltype_ids)}, "
          f"images: {len(split.train_df)}")
    print(f"Val pilltypes: {len(split.val_pilltype_ids)}, "
          f"images: {len(split.val_df)}")
    print(f"Total eligible pilltypes (1000 - excluded): "
          f"{len(split.train_pilltype_ids) + len(split.val_pilltype_ids)}")

    # Sanity check: no pilltype_id leakage across splits
    overlap = set(split.train_pilltype_ids) & set(split.val_pilltype_ids)
    print(f"\nTrain/val pilltype_id overlap (should be 0): {len(overlap)}")
