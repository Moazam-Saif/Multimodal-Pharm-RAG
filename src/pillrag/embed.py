"""
Phase 3: convert segmented pill images into 512-dim ResNet-18 feature
vectors for similarity search.

Scope decision (see DEVLOG.md "REVERSED: Phase 3 index scope" and
HANDOFF.md for full reasoning - verified against the ePillID paper's
own experimental design, not just reasoned independently):
  - The vector INDEX is built from reference images ONLY
    (is_ref==True, 2,000 rows / 1,000 pill types) - this module's
    batch-embed job should only ever be pointed at that subset.
  - Consumer images (is_ref==False) are NEVER embedded into the index.
    They're used as evaluation queries via the SAME embed_image/
    search_visual functions - same code path a real user's upload
    would go through - just not persisted to Deep Lake.

Background-handling decision (explicit user call, not Claude's
default): tight crop to the mask's true bounding box, NO pixel-level
masking within the crop. Real background pixels inside the bbox are
kept as-is. Known, accepted consequence: for masks where the bbox is
very loose (e.g. the 537 fallback_full_image rows, where bbox = the
entire image, or any fused pill+background blob masks from the Phase 2
investigation), this leaves real background fully exposed in the
embedding. This mostly affects EVAL query embeddings for consumer
photos, not the index itself (fallback rows are consumer-only, and
consumer images aren't indexed) - see HANDOFF.md for the full
reasoning chain.

embed_image signature: embed_image(image_path, mask) - takes a raw
image path and an already-resolved boolean mask array, not a
manifest row or pre-loaded pixels. Picked because it's the only
signature that serves BOTH real call sites (the offline batch-embed
job, which decodes RLE from a manifest row to get the mask array
first, and calls this module.py `search_visual`) with no coupling to
manifest schema - see conversation history / DEVLOG.md for the full
comparison of alternatives that were explicitly rejected.
"""

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

# ============================================================
# FastSAM wrapper - factored out here so search_visual (and any
# future caller) uses the EXACT SAME FastSAM call Phase 2's batch run
# used, rather than a fresh, possibly-drifted reimplementation.
#
# Sourced directly from batch_segment_full.py's process_one_image()
# and colab_full_test_setup.py's model instantiation - not guessed.
# Real facts, confirmed by reading both files:
#   - package: ultralytics (from ultralytics import FastSAM)
#   - checkpoint: "FastSAM-x.pt"
#   - call: model(image_path, verbose=False) - no explicit confidence/
#     IoU overrides at inference time; Phase 2 relies on ultralytics'
#     FastSAM defaults and does all real filtering afterward in
#     segment.py's own post-processing (confidence threshold, fill-
#     ratio, etc.) - NOT at the FastSAM call itself.
#   - masks-is-None handling: if results[0].masks is None (FastSAM
#     found zero objects), pass an empty (0,0,0) array through rather
#     than crashing or treating it specially - resolve_pill_mask's own
#     None-handling (including the Hough fallback) is designed to
#     receive this.
#
# `model` itself is NOT instantiated in this module - the caller must
# already have a loaded FastSAM instance (same pattern Phase 2 used:
# model = FastSAM("FastSAM-x.pt"), loaded once per session, not once
# per call - loading weights per-call would be needlessly slow across
# thousands of eval queries).
# ============================================================

def run_fastsam(model, image_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Run FastSAM on a single image, exactly as Phase 2's batch run
    did. Returns (confidences, masks) ready to pass into
    resolve_pill_mask().

    Args:
        model: an already-loaded `ultralytics.FastSAM("FastSAM-x.pt")`
            instance - NOT loaded inside this function, since loading
            weights per-call would be wasteful across many queries.
        image_path: path to a real image file on disk (FastSAM and
            the downstream Hough fallback both need a real path, not
            in-memory bytes).

    Returns:
        confidences: np.ndarray of per-mask confidence scores.
        masks: np.ndarray of shape (N, H, W), or shape (0, 0, 0) if
            FastSAM found zero objects (matches Phase 2's exact
            masks-is-None handling, so resolve_pill_mask sees the same
            input shape it was validated against).
    """
    results = model(image_path, verbose=False)
    confidences = results[0].boxes.conf.cpu().numpy()

    if results[0].masks is not None:
        masks = results[0].masks.data.cpu().numpy()
    else:
        masks = np.zeros((0, 0, 0))

    return confidences, masks

# ImageNet normalization constants - required because ResNet-18 was
# trained on ImageNet-normalized inputs; feeding it raw 0-255 pixel
# values (or even 0-1 floats without this specific mean/std) produces
# a meaningfully different, worse feature vector, since the network's
# learned weights implicitly assume this exact input distribution.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

EMBED_SIZE = 512  # ResNet-18's penultimate-layer output dimensionality


def _build_feature_extractor() -> nn.Module:
    """Load ResNet-18 pretrained on ImageNet, strip the final
    classification layer, keep everything up to (and including) the
    last average-pooling layer. This repurposes a classifier as a
    512-dim feature extractor - the pooled output encodes shape,
    color, texture, and visual pattern information the network
    learned, without committing to any of the 1,000 ImageNet class
    labels it was originally trained to predict.
    """
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model = nn.Sequential(*list(model.children())[:-1])  # drop the FC layer
    model.eval()
    return model


# Module-level singleton - loading ResNet-18's weights is a real cost
# (network download on first use, then real compute to move it onto
# device); building this fresh per embed_image() call would silently
# make batch-embedding all 2,000 reference images far slower than
# necessary. Built once, reused across every call in this process.
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_feature_extractor = _build_feature_extractor().to(_device)

_preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def mask_bounding_box(mask: np.ndarray) -> tuple[int, int, int, int]:
    """Return (top, bottom, left, right) - inclusive pixel-index
    bounds of the True region in a boolean mask.

    Raises ValueError on an all-False mask (nothing to crop to) -
    callers should check this before calling, not rely on this
    function to silently no-op, since a silent no-op here would embed
    a meaningless region for a genuinely maskless case, which should
    never happen downstream of a properly resolved Phase 2 mask (even
    the fallback_full_image case has an all-True mask, never all-False).

    Not reused from segment.py's bounding_box_fill_ratio, which computes
    the same row/col bounds internally but doesn't expose them - this
    is a small, dedicated helper rather than duplicating that internal
    logic inline here.
    """
    row_indices = np.where(mask.any(axis=1))[0]
    col_indices = np.where(mask.any(axis=0))[0]

    if len(row_indices) == 0 or len(col_indices) == 0:
        raise ValueError(
            "mask_bounding_box() called on an all-False mask - nothing "
            "to crop to. This should not happen for a properly resolved "
            "Phase 2 mask (even fallback_full_image masks are all-True)."
        )

    top, bottom = int(row_indices[0]), int(row_indices[-1])
    left, right = int(col_indices[0]), int(col_indices[-1])
    return top, bottom, left, right


def embed_image(image_path: str, mask: np.ndarray) -> np.ndarray:
    """Produce a 512-dim ResNet-18 feature vector for the pill in
    `image_path`, cropped to `mask`'s bounding box.

    Background handling (explicit decision, not a default): tight crop
    to the mask's bounding box ONLY - no pixel-level masking within the
    crop. Real background pixels inside the box are left as-is. See
    this module's docstring for the full reasoning and known
    consequences for loose-bbox masks.

    Args:
        image_path: path to the RAW (unsegmented) source image on disk.
        mask: boolean array, same (H, W) as the image at image_path,
            True where the pill is. This is Phase 2's resolved mask -
            typically `resolve_pill_mask(...).final_mask`, either
            fresh (live query) or RLE-decoded from a manifest row
            (batch job) - this function doesn't care which.

    Returns:
        np.ndarray of shape (512,), dtype float32.
    """
    image = Image.open(image_path).convert("RGB")

    top, bottom, left, right = mask_bounding_box(mask)
    # PIL crop box is (left, top, right, bottom) with right/bottom
    # EXCLUSIVE - mask_bounding_box's bounds are inclusive indices, so
    # +1 on the exclusive ends to include the last True row/col.
    cropped = image.crop((left, top, right + 1, bottom + 1))

    input_tensor = _preprocess(cropped).unsqueeze(0).to(_device)

    with torch.no_grad():
        features = _feature_extractor(input_tensor)

    return features.squeeze().cpu().numpy().astype(np.float32)