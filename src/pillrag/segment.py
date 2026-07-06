"""
Phase 2: FastSAM mask post-processing - selecting the correct pill
mask(s) out of FastSAM's raw "segment everything" output.

See DEVLOG.md "Phase 2: Background Segmentation" for the full
investigation trail behind this logic - several simpler rules were
tried and disproven with real evidence before arriving at this one:
  - "pick the largest mask by area" - failed (largest mask was often a
    background/framing artifact, not the pill)
  - "keep masks above a confidence threshold, then merge" - failed
    (artifact masks can have similar confidence to genuine object masks)
  - "reject masks touching the image edge" - failed (masks don't
    literally touch the outermost pixel even when they clearly span
    almost the full frame)
  - "reject masks with an extreme width:height aspect ratio relative to
    the image frame" - rejected on principle before testing: this
    would misfire on a real up-close user photo, where a genuine pill
    could legitimately fill most of the frame

This module's approach: some photographed pills (e.g. two-tone
capsules) get split by FastSAM into multiple "part" masks whose areas
sum to approximately one "whole" mask's area. We detect that
relationship directly (comparing masks to EACH OTHER, not to fixed
thresholds or the image frame), which stays valid regardless of how
close-up or zoomed a photo is - a property none of the earlier
rejected rules had.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np


DEFAULT_CONFIDENCE_THRESHOLD = 0.6
DEFAULT_AREA_SUM_TOLERANCE = 0.30  # allow 30% difference between a
                                    # candidate "whole" mask's area and
                                    # the sum of its proposed "parts"


@dataclass
class MaskSelectionResult:
    """The outcome of running mask selection on one FastSAM result.

    final_mask: the resolved pill mask (boolean array), or None if no
        confident mask could be identified at all.
    contributing_mask_indices: which of the original raw mask indices
        were used to build final_mask. Useful for debugging/QA - lets
        us re-visualize exactly which raw masks were kept.
    method: which resolution path was taken - "single", "whole_and_parts",
        or "highest_confidence_fallback" - so we can track, across a
        full batch run, how often each case actually occurs.
    """

    final_mask: np.ndarray | None
    contributing_mask_indices: tuple[int, ...]
    method: str


def _mask_areas(masks: np.ndarray) -> dict[int, int]:
    """Pixel area per mask, as ordinary Python ints (NOT numpy uint -
    using numpy's unsigned integer types here caused a silent integer
    overflow bug during initial testing when subtracting areas; see
    DEVLOG.md. Plain Python ints have no such overflow risk)."""
    return {i: int(masks[i].sum()) for i in range(len(masks))}


def _find_whole_and_parts(
    confident_indices: list[int],
    areas: dict[int, int],
    tolerance: float,
) -> tuple[int, tuple[int, int]] | None:
    """Look for a mask whose area approximately equals the SUM of two
    other confident masks' areas. Returns (whole_index, (part_a, part_b))
    for the best such relationship found, or None if none exists.

    Checks EVERY valid pairing (not just the first one tried) and picks
    the CLOSEST match by area difference, rather than stopping at the
    first pairing under tolerance - this avoids a false positive being
    accepted just because it happened to be tested first, which was a
    real weakness in this function's first draft (see DEVLOG.md).
    """
    best_match: tuple[int, tuple[int, int], float] | None = None  # (whole, parts, diff_frac)

    for part_a, part_b in itertools.combinations(confident_indices, 2):
        pair_sum = areas[part_a] + areas[part_b]
        if pair_sum == 0:
            continue  # guard against empty masks, avoid division by zero below

        for candidate_whole in confident_indices:
            if candidate_whole in (part_a, part_b):
                continue

            diff_frac = abs(areas[candidate_whole] - pair_sum) / pair_sum
            if diff_frac > tolerance:
                continue

            if best_match is None or diff_frac < best_match[2]:
                best_match = (candidate_whole, (part_a, part_b), diff_frac)

    if best_match is None:
        return None
    whole_idx, part_indices, _ = best_match
    return whole_idx, part_indices


def select_pill_mask(
    confidences: np.ndarray,
    masks: np.ndarray,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    area_sum_tolerance: float = DEFAULT_AREA_SUM_TOLERANCE,
) -> MaskSelectionResult:
    """Resolve FastSAM's raw multi-mask output down to one final pill
    mask, using confidence filtering plus the whole/parts area
    relationship described in this module's docstring.

    Args:
        confidences: 1D array of per-mask confidence scores, as returned
            by results[0].boxes.conf.cpu().numpy()
        masks: 3D array of per-mask boolean/binary pixel data, as
            returned by results[0].masks.data.cpu().numpy()
        confidence_threshold: masks below this are treated as noise and
            excluded entirely before any further logic runs
        area_sum_tolerance: how much difference to allow between a
            candidate "whole" mask's area and the sum of its proposed
            "parts" areas (see _find_whole_and_parts)

    Returns:
        A MaskSelectionResult. If no mask clears confidence_threshold,
        final_mask is None - callers must handle this (e.g. flag the
        image for manual review rather than silently producing a blank
        or wrong mask).
    """
    if len(confidences) != len(masks):
        raise ValueError(
            f"confidences and masks must have matching length, "
            f"got {len(confidences)} and {len(masks)}"
        )

    confident_indices = [
        i for i, conf in enumerate(confidences) if conf >= confidence_threshold
    ]

    if not confident_indices:
        return MaskSelectionResult(
            final_mask=None, contributing_mask_indices=(), method="none_confident"
        )

    if len(confident_indices) == 1:
        idx = confident_indices[0]
        return MaskSelectionResult(
            final_mask=masks[idx].astype(bool),
            contributing_mask_indices=(idx,),
            method="single",
        )

    areas = _mask_areas(masks)
    whole_and_parts = _find_whole_and_parts(
        confident_indices, areas, area_sum_tolerance
    )

    if whole_and_parts is not None:
        whole_idx, part_indices = whole_and_parts
        # Use the WHOLE mask directly, not a union of the parts - the
        # whole mask is FastSAM's own detection of the complete object
        # and should already be a cleaner, single coherent shape than
        # a union of two separately-detected part masks would be.
        return MaskSelectionResult(
            final_mask=masks[whole_idx].astype(bool),
            contributing_mask_indices=(whole_idx, *part_indices),
            method="whole_and_parts",
        )

    # Fallback: multiple confident masks, but none fit the whole/parts
    # pattern (e.g. genuinely multiple separate objects in frame, or a
    # relationship this module doesn't yet model). Rather than guess,
    # fall back to the single highest-confidence mask - a defensible,
    # explicit default that we can specifically audit later by
    # filtering on method == "highest_confidence_fallback" across a
    # batch run, to see how often this path is actually taken.
    best_idx = max(confident_indices, key=lambda i: confidences[i])
    return MaskSelectionResult(
        final_mask=masks[best_idx].astype(bool),
        contributing_mask_indices=(best_idx,),
        method="highest_confidence_fallback",
    )
