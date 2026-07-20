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
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import albumentations as A
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from albumentations.pytorch import ToTensorV2
from PIL import Image
from pycocotools import mask as maskUtils
from torch.utils.data import Dataset

from pillrag.data import EPILLID_ZIP, build_pill_dataset
from pillrag.embed import IMAGENET_MEAN, IMAGENET_STD, mask_bounding_box

TRAIN_RESOLUTION = 384

# ResNet-50's penultimate-layer (post-global-avg-pool) output size -
# this is what SHIPS as the real embedding after training (same role
# embed.py's EMBED_SIZE=512 plays for ResNet-18 today; ResNet-50's
# is 2048, not 512 - a real, expected consequence of the backbone
# upgrade, not a bug). The projection head below is training-only and
# is discarded at inference time - see PillEmbeddingModel's docstring.
BACKBONE_EMBED_SIZE = 2048

# Standard SupCon projection head output size (Khosla et al. 2020's
# own default, and the ePillID paper's baseline follows the same
# convention) - NOT the shipped embedding size. Only used to compute
# the contrastive loss during training.
PROJECTION_SIZE = 128

# Where extract_reference_images() unpacks the ~2,000 is_ref==True
# training images to real disk (Option A: extract-once, not
# per-item zip access - see DEVLOG.md for why: PyTorch DataLoader
# workers sharing one zipfile.ZipFile handle across processes is a
# known source of subtle corruption, and this is a small, one-time,
# ~2,000-file cost, not the slow full-13,532-image case Phase 1's
# "don't persist derived images" decision was actually about).
#
# Deliberately scoped to REFERENCE images only (not all 5,728) -
# proportionate to what Phase 4 training actually touches right now.
# The eval-set images (3,728 consumer rows) get their own extraction
# step later, when Phase 4 reaches step 9 (the real eval run) - not
# done preemptively here.
EXTRACTED_IMAGE_ROOT = Path(
    os.environ.get(
        "PILL_EXTRACTED_IMAGE_ROOT",
        "/content/drive/MyDrive/pill-rag/data/raw/epillid_extracted",
    )
)

MANIFEST_GLOB = os.environ.get(
    "PILL_MASK_MANIFEST_GLOB",
    "/content/drive/MyDrive/pill-rag/data/masks/manifest_chunk_*.parquet",
)

TRAIN_VAL_SPLIT_SEED = 42
N_VAL_PILLTYPES = 200

# Batch size, expressed in PILL TYPES, not images. With exactly 2
# images/pilltype (verified, see this module's docstring), N=64
# pilltypes/batch = 128 images/batch - a reasonable default SupCon
# batch size, chosen for Colab free-tier T4 memory headroom at
# 384x384 resolution with a ResNet-50 backbone. NOT tuned against a
# real memory profile yet - this is an assumption, not a measured
# constraint. If training hits OOM, the first thing to try is
# lowering this, not the resolution/backbone (those were separate,
# deliberate Phase 4 decisions - see DEVLOG.md).
N_PILLTYPES_PER_BATCH = 64

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


def extract_reference_images(
    reference_df: pd.DataFrame,
    image_root: Path = EXTRACTED_IMAGE_ROOT,
) -> Path:
    """Extract the ~2,000 is_ref==True images this dataframe references
    out of epillid_data.zip and onto real disk, under `image_root`,
    preserving each row's `full_image_path` as the relative path on
    disk (e.g. image_root / "ePillID_data/classification_data/
    fcn_mix_weight/dr_224/xyz.jpg").

    Why this exists (Option A, decided explicitly - see DEVLOG.md):
    `full_image_path` is a path INSIDE epillid_data.zip, not a real
    filesystem path - PillPairDataset.__getitem__ calling
    `Image.open(row["full_image_path"])` directly would fail with
    FileNotFoundError on the very first training step. Extracting once,
    up front, avoids per-item zip access - which would require each
    PyTorch DataLoader worker process to open its own independent
    zipfile.ZipFile handle to be safe (sharing one handle across
    worker processes is a known corruption risk), adding real
    complexity for a data source this small.

    Idempotent / resumable: skips any file that already exists on disk
    at the expected path with a nonzero size, so re-running this after
    a Colab session reset (per this project's "assume full reset"
    rule) only re-extracts what's actually missing, not all ~2,000
    files every time.

    Args:
        reference_df: a DataFrame with a `full_image_path` column
            (zip-relative paths) - typically load_reference_manifest()'s
            output, or a train_df/val_df slice of it. Only the rows
            actually present get extracted; this function doesn't
            assume it's always given the full reference set.
        image_root: local (or Drive-mounted) directory to extract into.
            Defaults to EXTRACTED_IMAGE_ROOT (overridable via the
            PILL_EXTRACTED_IMAGE_ROOT env var, same pattern as
            data.py's EPILLID_ZIP_PATH / MANIFEST_GLOB env vars).

    Returns:
        image_root, for convenient chaining
        (e.g. `root = extract_reference_images(ref_df)`).
    """
    image_root.mkdir(parents=True, exist_ok=True)

    zip_relative_paths = reference_df["full_image_path"].unique().tolist()

    already_present = 0
    newly_extracted = 0

    with zipfile.ZipFile(EPILLID_ZIP) as zf:
        for zip_relative_path in zip_relative_paths:
            dest_path = image_root / zip_relative_path

            if dest_path.exists() and dest_path.stat().st_size > 0:
                already_present += 1
                continue

            dest_path.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(zip_relative_path) as src, open(dest_path, "wb") as dst:
                dst.write(src.read())
            newly_extracted += 1

    print(
        f"extract_reference_images: {newly_extracted} newly extracted, "
        f"{already_present} already present, "
        f"{len(zip_relative_paths)} total unique paths requested, "
        f"root={image_root}"
    )

    return image_root


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

    def __init__(
        self,
        dataframe: pd.DataFrame,
        augment: bool,
        image_root: Path = EXTRACTED_IMAGE_ROOT,
    ):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = (
            build_train_augmentation() if augment else build_eval_augmentation()
        )
        # image_root: the local directory extract_reference_images()
        # already unpacked this dataframe's images into. `full_image_path`
        # itself stays untouched as the zip-relative join key (still
        # needed to match manifest rows back to build_pill_dataset()'s
        # df, per load_reference_manifest()'s own join) - the mapping
        # to a real on-disk path happens ONLY here, at read time, so
        # nothing upstream needs to know local extraction even exists.
        self.image_root = image_root

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, idx: int):
        row = self.dataframe.iloc[idx]

        local_path = self.image_root / row["full_image_path"]
        image = Image.open(local_path).convert("RGB")
        image_array = np.array(image)

        mask = decode_rle_mask(row["rle_size"], row["rle_counts"])

        # Root-cause diagnosis (confirmed via real run, see DEVLOG.md):
        # Phase 2's masks come from FastSAM, which resizes every input
        # to its own internal inference resolution (1024x1024)
        # regardless of the source image's actual size - so
        # rle_size is ALWAYS [1024, 1024], even for these 224x224
        # dr_224/dc_224 source images. The mask's coordinate space
        # does NOT match image_array's shape unless resized first.
        # This is FastSAM's real behavior, not a bug in Phase 2's
        # segmentation - the fix belongs here, at the point where a
        # mask and its image are combined, not upstream.
        #
        # Nearest-neighbor (not bilinear/area) is required: mask is
        # boolean (True=pill/False=background), and any smooth
        # interpolation would produce fractional edge values that are
        # neither True nor False - nearest-neighbor is the only
        # resize mode that preserves a strictly boolean mask.
        if mask.shape != image_array.shape[:2]:
            mask = cv2.resize(
                mask.astype(np.uint8),
                (image_array.shape[1], image_array.shape[0]),  # cv2 wants (W, H)
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)

        top, bottom, left, right = mask_bounding_box(mask)
        cropped = image_array[top : bottom + 1, left : right + 1]

        augmented = self.transform(image=cropped)
        image_tensor = augmented["image"]

        return image_tensor, row["pilltype_id"]


class PillBalancedBatchSampler:
    """Yields batches of DATAFRAME ROW INDICES (not pilltype_ids) into
    a PillPairDataset, guaranteeing every batch contains BOTH images of
    each pill type it draws - the real positive pairs SupCon needs to
    have any "pull together" signal at all (see this module's
    docstring for why plain random shuffling was rejected: with only
    ~2 images/class, most random batches would have zero true positive
    pairs, wasting half the loss).

    Epoch definition (Option 1, decided - see DEVLOG.md/HANDOFF.md):
    one epoch = every pill type in `dataframe` appears in EXACTLY ONE
    batch. Implemented by shuffling the full list of pilltype_ids once
    per epoch, then chopping it into consecutive groups of
    `pilltypes_per_batch` - NOT a fixed-batches-per-epoch or
    sample-with-replacement scheme, since the dataset is small,
    fixed-size, and every class has exactly the same image count (2) -
    no benefit to the more complex alternative here.

    The FINAL batch of an epoch may contain fewer than
    `pilltypes_per_batch` pill types if the total doesn't divide
    evenly - this is expected and left as-is (not padded/dropped),
    since dropping it would mean some pill types never appear in some
    epochs, and padding would require picking which pilltypes get
    repeated, an arbitrary choice not worth adding complexity for.

    Assumes (and asserts) that `dataframe` has been through
    make_train_val_split() and EXCLUDED_ZERO_SIGNAL_PILLTYPES has
    already been removed - it does NOT special-case pilltypes with
    other than exactly 2 images, since that invariant was verified
    project-wide (see this module's docstring's "Data quality check").
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        pilltypes_per_batch: int = N_PILLTYPES_PER_BATCH,
        seed: int = TRAIN_VAL_SPLIT_SEED,
    ):
        self.dataframe = dataframe.reset_index(drop=True)
        self.pilltypes_per_batch = pilltypes_per_batch
        self.rng = random.Random(seed)

        # Map each pilltype_id -> list of its row indices into
        # self.dataframe (should be exactly 2 per pilltype, given the
        # project-wide verified invariant - not enforced here with a
        # hard assert to avoid crashing on a not-yet-excluded fallback
        # pilltype during interactive debugging, but real production
        # training should have already run make_train_val_split()).
        self._pilltype_to_indices: dict[str, list[int]] = (
            self.dataframe.groupby("pilltype_id").indices
        )
        self._pilltype_to_indices = {
            pilltype_id: list(indices)
            for pilltype_id, indices in self._pilltype_to_indices.items()
        }
        self.pilltype_ids = sorted(self._pilltype_to_indices.keys())

    def __iter__(self):
        shuffled_pilltype_ids = self.pilltype_ids.copy()
        self.rng.shuffle(shuffled_pilltype_ids)

        for start in range(0, len(shuffled_pilltype_ids), self.pilltypes_per_batch):
            batch_pilltype_ids = shuffled_pilltype_ids[
                start : start + self.pilltypes_per_batch
            ]

            batch_indices: list[int] = []
            for pilltype_id in batch_pilltype_ids:
                batch_indices.extend(self._pilltype_to_indices[pilltype_id])

            # Shuffle within the batch so the two images of any given
            # pilltype aren't always adjacent - avoids any accidental
            # ordering dependency downstream (e.g. in the loss
            # implementation) relying on positive pairs sitting at
            # fixed relative positions within a batch.
            self.rng.shuffle(batch_indices)

            yield batch_indices

    def __len__(self) -> int:
        return -(-len(self.pilltype_ids) // self.pilltypes_per_batch)  # ceil div


class PillEmbeddingModel(nn.Module):
    """ResNet-50 backbone (ImageNet-pretrained, upgrade from embed.py's
    ResNet-18 - matches the ePillID paper's own baseline architecture,
    see this module's docstring) + a SupCon projection head.

    TRAINING vs. INFERENCE split (standard SupCon convention - Khosla
    et al. 2020): the projection head exists ONLY to compute the
    contrastive loss during training. It is explicitly NOT what gets
    shipped as the final embedding - after training, downstream code
    (the eventual replacement for embed.py's embed_image()) should use
    `self.backbone_embedding(x)` directly (2048-dim, L2-normalized),
    NOT `self.forward(x)`'s projection-head output (128-dim). This
    mirrors embed.py's own docstring precedent of keeping training-
    time and shipped-embedding concerns clearly separated.

    forward() returns the L2-normalized PROJECTION output (128-dim) -
    what SupConLoss consumes. backbone_embedding() returns the
    L2-normalized BACKBONE output (2048-dim) - what the real pill-RAG
    system will eventually use for indexing/search, once this model
    replaces the ResNet-18 in embed.py.
    """

    def __init__(self):
        super().__init__()

        resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        # Strip the final FC classification layer, keep everything up
        # to (and including) the last global-average-pooling layer -
        # same pattern as embed.py's _build_feature_extractor().
        self.backbone = nn.Sequential(*list(resnet50.children())[:-1])

        self.projection_head = nn.Sequential(
            nn.Linear(BACKBONE_EMBED_SIZE, BACKBONE_EMBED_SIZE),
            nn.ReLU(inplace=True),
            nn.Linear(BACKBONE_EMBED_SIZE, PROJECTION_SIZE),
        )

    def backbone_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """The real, shippable embedding - L2-normalized 2048-dim
        ResNet-50 features. Does NOT pass through the projection head.
        This is what embed_image()'s eventual ResNet-50 replacement
        should call, and what gets indexed into Deep Lake / used for
        query-time similarity search - NOT forward()'s output.
        """
        features = self.backbone(x)
        features = torch.flatten(features, 1)
        return F.normalize(features, p=2, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """TRAINING-ONLY path: backbone -> projection head -> L2-norm.
        Returns the 128-dim projection SupConLoss operates on. This is
        NOT the embedding that gets shipped - see backbone_embedding()
        and this class's docstring.
        """
        features = self.backbone(x)
        features = torch.flatten(features, 1)
        projected = self.projection_head(features)
        return F.normalize(projected, p=2, dim=1)


class SupConLoss(nn.Module):
    """Supervised Contrastive Loss (Khosla et al. 2020).

    For each anchor in a batch, pulls its embedding toward all OTHER
    embeddings sharing its label (positives) and pushes it away from
    every embedding with a different label (negatives) - the direct
    "same pill close, different pill far" objective that motivated
    this whole Phase 4 retrain (see this module's top docstring for
    the diagnostic trail establishing why a zero-shot ImageNet
    embedding space wasn't already doing this).

    Standard formulation, not a simplified/approximate variant:
    for each anchor i,
        L_i = -1/|P(i)| * sum_{p in P(i)} log(
            exp(z_i . z_p / temperature)
            / sum_{a in A(i)} exp(z_i . z_a / temperature)
        )
    where P(i) = all OTHER samples in the batch sharing i's label,
    A(i) = all samples in the batch except i itself, and z are the
    L2-normalized projection-head outputs (PillEmbeddingModel.forward's
    output - NOT the raw backbone embedding).

    With this project's BALANCED batch sampler (see
    PillBalancedBatchSampler), every anchor has EXACTLY ONE positive
    in every full batch (its pilltype's other image) - |P(i)|=1 for
    every anchor in a full batch, simplifying the numerator to a
    single term, though the implementation below stays general (does
    not hard-code |P(i)|=1) so it's still correct on the final,
    possibly-short batch of an epoch where sampler.pilltypes_per_batch
    doesn't evenly divide the pilltype count.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings: torch.Tensor, labels: list[str]) -> torch.Tensor:
        """
        Args:
            embeddings: (batch_size, projection_size) L2-normalized
                tensor - PillEmbeddingModel.forward()'s output.
            labels: list of length batch_size, one pilltype_id string
                per embedding, same order. NOT converted to a tensor
                by the caller - this function handles the string ->
                positive-pair-mask conversion internally, since
                pilltype_id is a string, not an int class index, and
                no fixed class-id mapping exists (or should exist -
                the whole point of a metric-learning approach is not
                needing a fixed closed set of classes).

        Returns:
            scalar loss tensor (mean over all anchors that have at
            least one positive in the batch).
        """
        device = embeddings.device
        batch_size = embeddings.shape[0]

        # (batch_size, batch_size) boolean: True where row i and row j
        # share the same pilltype_id (including i==j, removed below).
        labels_array = np.array(labels)
        same_label = torch.tensor(
            labels_array[:, None] == labels_array[None, :], device=device
        )

        self_mask = torch.eye(batch_size, dtype=torch.bool, device=device)
        positive_mask = same_label & ~self_mask  # P(i) per row, excludes self
        negative_mask = ~same_label  # everyone with a DIFFERENT label than i

        # A(i) = every sample except i itself (positives AND negatives,
        # per the SupCon denominator definition above).
        all_except_self_mask = ~self_mask

        similarity = torch.matmul(embeddings, embeddings.T) / self.temperature

        # Numerical stability: subtract the row-wise max before exp(),
        # standard log-sum-exp trick - does not change the result
        # (the max cancels in the ratio) but avoids overflow.
        similarity = similarity - similarity.max(dim=1, keepdim=True).values.detach()
        exp_similarity = torch.exp(similarity)

        denominator = (exp_similarity * all_except_self_mask).sum(dim=1)
        log_prob = similarity - torch.log(denominator.unsqueeze(1) + 1e-12)

        num_positives_per_anchor = positive_mask.sum(dim=1)
        # Anchors with zero positives in this batch (shouldn't happen
        # with the balanced sampler on a full batch, but the FINAL,
        # possibly-short batch of an epoch could in principle produce
        # one if pilltypes_per_batch splits oddly - defensive, not
        # expected in practice) contribute 0 loss rather than NaN from
        # a divide-by-zero, and are excluded from the mean below.
        safe_denominator = num_positives_per_anchor.clamp(min=1)
        mean_log_prob_positive = (
            (positive_mask * log_prob).sum(dim=1) / safe_denominator
        )

        has_positive = num_positives_per_anchor > 0
        loss_per_anchor = -mean_log_prob_positive[has_positive]

        return loss_per_anchor.mean()


if __name__ == "__main__":
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

    # Verify the balanced batch sampler for real - don't assume the
    # logic is right just because it reads correctly, per rule #3.
    train_dataset = PillPairDataset(split.train_df, augment=True)
    sampler = PillBalancedBatchSampler(split.train_df)

    print(f"\nSampler: {len(split.train_pilltype_ids)} train pilltypes, "
          f"{N_PILLTYPES_PER_BATCH} pilltypes/batch "
          f"-> expected {len(sampler)} batches/epoch")

    batches = list(sampler)
    print(f"Actual batches yielded: {len(batches)}")

    total_images_seen = sum(len(b) for b in batches)
    print(f"Total image-indices across all batches: {total_images_seen} "
          f"(should equal train_df length: {len(split.train_df)})")

    # Confirm every full-size batch has EXACTLY 2 rows per pilltype_id
    # (a real positive pair for every image, every batch) - check all
    # but the possibly-short final batch.
    bad_batches = 0
    for batch_indices in batches[:-1]:
        batch_pilltype_ids = split.train_df.loc[batch_indices, "pilltype_id"]
        counts = batch_pilltype_ids.value_counts()
        if not (counts == 2).all():
            bad_batches += 1
    print(f"Non-full batches (excluding final): {bad_batches} (should be 0)")
    print(f"Final batch size: {len(batches[-1])} images "
          f"({len(batches[-1]) // 2} pilltypes)")

    # --- Model + loss sanity check on a REAL batch (not random noise) ---
    # Confirms shapes and a real forward+backward pass work end to end
    # before committing to a full training loop - per rule #3, verify
    # before building further, don't assume architecture code is
    # correct just because it imports cleanly.
    print("\n" + "=" * 60)
    print("STEP 5: PillEmbeddingModel + SupConLoss sanity check")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = PillEmbeddingModel().to(device)

    first_batch_indices = batches[0]
    images = []
    pilltype_ids = []
    for idx in first_batch_indices:
        image_tensor, pilltype_id = train_dataset[idx]
        images.append(image_tensor)
        pilltype_ids.append(pilltype_id)

    image_batch = torch.stack(images).to(device)
    print(f"Real batch: {image_batch.shape[0]} images, "
          f"{len(set(pilltype_ids))} unique pilltypes")

    projections = model(image_batch)
    print(f"Projection output shape: {tuple(projections.shape)} "
          f"(expected: ({image_batch.shape[0]}, {PROJECTION_SIZE}))")
    print(f"Projection L2 norms (should all be ~1.0): "
          f"min={projections.norm(dim=1).min().item():.4f}, "
          f"max={projections.norm(dim=1).max().item():.4f}")

    backbone_embeddings = model.backbone_embedding(image_batch)
    print(f"Backbone embedding shape: {tuple(backbone_embeddings.shape)} "
          f"(expected: ({image_batch.shape[0]}, {BACKBONE_EMBED_SIZE}))")

    criterion = SupConLoss(temperature=0.07)
    loss = criterion(projections, pilltype_ids)
    print(f"SupCon loss (real batch, untrained weights): {loss.item():.4f}")

    loss.backward()
    grad_norms = [
        p.grad.norm().item()
        for p in model.parameters()
        if p.grad is not None
    ]
    print(f"Backward pass: {len(grad_norms)} parameter tensors received "
          f"gradients (should be > 0), "
          f"mean grad norm: {np.mean(grad_norms):.6f}")

    print("\nModel + loss sanity check complete.")
