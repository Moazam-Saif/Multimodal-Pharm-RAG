"""
Phase 3: live visual search - takes a real user query photo and an
already-loaded FastSAM model, and produces a resolved pill mask ready
for embed_image().

This module is the LIVE QUERY path, as opposed to segment.py's Phase 2
offline batch-processing logic. The two share resolve_pill_mask() and
run_fastsam() (imported here, not reimplemented) so the exact same
FastSAM call and mask-selection logic is used in both places - see
HANDOFF.md's "Quick reference" section for why that consistency
matters (Phase 2's validation only covers this exact pipeline).

known_shape and the offline/online scope note (READ BEFORE CHANGING):
resolve_pill_mask's Hough Circle fallback is normally OFFLINE-ONLY
(segment.py's hough_circle_fallback docstring: using an INFERRED shape
at query time would be circular, since shape is part of what
identification is supposed to determine). That restriction does NOT
apply here: known_shape in this module comes from the USER explicitly
selecting their pill's shape from a dropdown at photo-capture time -
same as a patient telling a pharmacist "it's a round white pill" out
loud. This is a real, independently-known input, not something the
pipeline inferred about itself. Explicit decision (this session):
pass it through to resolve_pill_mask so ROUND queries get the same
Hough fallback protection reference-image indexing already gets. This
formally revises the "don't re-litigate" scope note in HANDOFF.md -
that note was about INFERRED shape, not user-declared shape - and
should be updated there too, not treated as silently overturned.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

from pillrag.embed import run_fastsam
from pillrag.segment import resolve_pill_mask


@dataclass
class QuerySegmentationResult:
    """The outcome of segmenting one live query photo.

    final_mask: boolean array, same (H, W) as the query image. Never
        None - if resolve_pill_mask couldn't find a valid pill mask
        (and the Hough fallback, if applicable, also failed), this
        falls back to an all-True mask covering the whole image,
        rather than leaving callers to handle a None case themselves.
    method: resolve_pill_mask's own method string (e.g. "single",
        "dominant+rescued", "hough_circle_fallback"), or
        "full_image_fallback" if this module's own last-resort
        fallback fired instead.
    degraded: True if final_mask is the full-image fallback, NOT a
        genuine resolved pill mask. Callers (embed_image, then
        search_visual) should treat a match built from a degraded
        mask as lower-confidence - the embedding will include real
        background pixels, per embed.py's own documented consequence
        of tight-bbox-crop-without-internal-masking for loose/full-
        image masks. Never silently mix degraded and non-degraded
        results into one confidence display without flagging this.
    """

    final_mask: np.ndarray
    method: str
    degraded: bool


def segment_query_image(
    image_path: str,
    known_shape: str | None,
    fastsam_model,
) -> QuerySegmentationResult:
    """Segment a single live query photo, for use in search_visual.

    Args:
        image_path: path to the raw query image on disk.
        known_shape: the pill shape the USER selected from a dropdown
            at capture time (e.g. "ROUND", "OVAL", "CAPSULE"), or None
            if the user didn't specify one. This is passed straight
            through to resolve_pill_mask's known_shape parameter - see
            this module's docstring for why that's valid here despite
            the general offline-only scope note on the Hough fallback.
        fastsam_model: an already-loaded `ultralytics.FastSAM(...)`
            instance - NOT loaded inside this function. Same pattern
            as embed.py's run_fastsam: loading weights per-query would
            be wasteful.

    Returns:
        QuerySegmentationResult - final_mask is never None; see the
        dataclass docstring for the full_image_fallback behavior.
    """
    confidences, masks = run_fastsam(fastsam_model, image_path)

    result = resolve_pill_mask(
        confidences,
        masks,
        image_path=image_path,
        known_shape=known_shape,
    )

    if result.final_mask is not None:
        return QuerySegmentationResult(
            final_mask=result.final_mask,
            method=result.method,
            degraded=False,
        )

    # Last-resort fallback: resolve_pill_mask (including its Hough
    # fallback, if known_shape made it eligible) found nothing usable.
    # Embed the full image rather than returning no result at all -
    # explicit user decision this session, accepted as a known lower-
    # confidence path rather than a hard failure.
    with Image.open(image_path) as img:
        width, height = img.size

    full_image_mask = np.ones((height, width), dtype=bool)

    return QuerySegmentationResult(
        final_mask=full_image_mask,
        method="full_image_fallback",
        degraded=True,
    )