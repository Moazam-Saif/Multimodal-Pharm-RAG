"""
Phase 2: FastSAM mask post-processing - selecting the correct pill
mask(s) out of FastSAM's raw "segment everything" output.

See DEVLOG.md "Phase 2: Background Segmentation" for the full
investigation trail behind this logic. Several rules were tried and
disproven with real evidence before arriving at this one - including
one case (the "whole/parts area-sum" rule alone) that was shipped once
already WITHOUT being re-tested against the exact case it was meant to
fix, and failed identically to the very first naive attempt. That
mistake is recorded in DEVLOG.md as a reminder: every rule in this file
must be re-verified against ALL known test cases before being trusted,
not just reasoned about.

Rejected/superseded approaches, in order tried:
  - "pick the largest mask by area" - failed (largest mask was often a
    background/framing artifact, not the pill)
  - "keep masks above a confidence threshold, then merge via union" -
    failed (artifact masks can have similar confidence to genuine masks)
  - "reject masks touching the image edge" - failed (masks don't
    literally touch the outermost pixel even when they clearly span
    almost the full frame)
  - "reject masks with an extreme width:height aspect ratio relative to
    the image frame" - rejected on principle before testing: would
    misfire on a real up-close user photo, where a genuine pill could
    legitimately fill most of the frame
  - "whole/parts area-sum matching alone" - failed: a coincidental
    arithmetic fit (band artifact's area happened to be close to the
    sum of the two real capsule-half areas) produced the same wrong
    answer as the original naive rule
  - "geometric touching/overlap between candidate masks" - failed: the
    band artifact overlaps the real pill masks heavily, because it's
    CAUSED BY the same region of the image the pill occupies, not
    spatially separate from it

**What actually works** (tested against many distinct real cases - a
two-tone capsule with a band artifact, a clean single-color round
tablet, a consumer photo with a whole-image "blob" artifact, an
orange-oval logo-fragment failure, and multiple banded/printed
capsules): bounding-box fill ratio (a mask's own pixel area divided by
its own bounding box area) to reject rectangular artifacts, a dominance
check to prefer an obviously-complete mask over small unrelated
fragments, a containment check to exclude sub-details (printed
text/digits) from being treated as separate "parts," a rescue step to
recover genuine-but-low-confidence complementary masks, and an adaptive
gap-closing step for residual seams between merged pieces.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import binary_closing


DEFAULT_CONFIDENCE_THRESHOLD = 0.6
DEFAULT_MAX_FILL_RATIO = 0.97
DEFAULT_AREA_SUM_TOLERANCE = 0.30
DEFAULT_DOMINANCE_MULTIPLIER = 2.0
DEFAULT_CONTAINMENT_THRESHOLD = 0.9
DEFAULT_RESCUE_MIN_FILL_RATIO = 0.5
DEFAULT_RESCUE_MAX_OVERLAP_FRAC = 0.15
DEFAULT_GAP_CLOSING_SAFETY_MARGIN = 1.5


@dataclass
class MaskSelectionResult:
    final_mask: np.ndarray | None
    contributing_mask_indices: tuple[int, ...]
    method: str


def bounding_box_fill_ratio(mask: np.ndarray) -> float:
    rows_with_content = mask.any(axis=1)
    cols_with_content = mask.any(axis=0)
    if not rows_with_content.any():
        return 0.0
    row_indices = rows_with_content.nonzero()[0]
    col_indices = cols_with_content.nonzero()[0]
    bbox_height = int(row_indices[-1] - row_indices[0] + 1)
    bbox_width = int(col_indices[-1] - col_indices[0] + 1)
    bbox_area = bbox_height * bbox_width
    return float(mask.sum()) / bbox_area


def _mask_areas(masks: np.ndarray) -> dict[int, int]:
    return {i: int(masks[i].sum()) for i in range(len(masks))}


def _containment_fraction(small_mask: np.ndarray, big_mask: np.ndarray) -> float:
    small_area = small_mask.sum()
    if small_area == 0:
        return 0.0
    overlap = (small_mask & big_mask).sum()
    return float(overlap) / float(small_area)


def _exclude_contained_masks(candidate_indices, masks, containment_threshold):
    kept = []
    for idx in candidate_indices:
        is_contained = False
        for other_idx in candidate_indices:
            if other_idx == idx:
                continue
            if _containment_fraction(masks[idx].astype(bool), masks[other_idx].astype(bool)) >= containment_threshold:
                is_contained = True
                break
        if not is_contained:
            kept.append(idx)
    return kept


def _find_dominant_mask(candidate_indices, areas, multiplier):
    total_area = sum(areas[i] for i in candidate_indices)
    for idx in candidate_indices:
        rest = total_area - areas[idx]
        if rest == 0:
            continue
        if areas[idx] > rest * multiplier:
            return idx
    return None


def _find_whole_and_parts(candidate_indices, areas, tolerance):
    best_match = None
    for part_a, part_b in itertools.combinations(candidate_indices, 2):
        pair_sum = areas[part_a] + areas[part_b]
        if pair_sum == 0:
            continue
        for candidate_whole in candidate_indices:
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
    max_fill_ratio: float = DEFAULT_MAX_FILL_RATIO,
    area_sum_tolerance: float = DEFAULT_AREA_SUM_TOLERANCE,
    dominance_multiplier: float = DEFAULT_DOMINANCE_MULTIPLIER,
    containment_threshold: float = DEFAULT_CONTAINMENT_THRESHOLD,
) -> MaskSelectionResult:
    if len(confidences) != len(masks):
        raise ValueError(
            f"confidences and masks must have matching length, "
            f"got {len(confidences)} and {len(masks)}"
        )

    fill_ratios = {i: bounding_box_fill_ratio(masks[i]) for i in range(len(masks))}

    candidate_indices = [
        i
        for i, conf in enumerate(confidences)
        if conf >= confidence_threshold and fill_ratios[i] <= max_fill_ratio
    ]

    if not candidate_indices:
        return MaskSelectionResult(final_mask=None, contributing_mask_indices=(), method="none_valid")

    if len(candidate_indices) == 1:
        idx = candidate_indices[0]
        return MaskSelectionResult(
            final_mask=masks[idx].astype(bool),
            contributing_mask_indices=(idx,),
            method="single",
        )

    areas = _mask_areas(masks)

    dominant_idx = _find_dominant_mask(candidate_indices, areas, dominance_multiplier)
    if dominant_idx is not None:
        return MaskSelectionResult(
            final_mask=masks[dominant_idx].astype(bool),
            contributing_mask_indices=(dominant_idx,),
            method="dominant",
        )

    non_contained_indices = _exclude_contained_masks(candidate_indices, masks, containment_threshold)
    search_indices = non_contained_indices if len(non_contained_indices) >= 2 else candidate_indices

    whole_and_parts = _find_whole_and_parts(search_indices, areas, area_sum_tolerance)

    if whole_and_parts is not None:
        whole_idx, part_indices = whole_and_parts
        return MaskSelectionResult(
            final_mask=masks[whole_idx].astype(bool),
            contributing_mask_indices=(whole_idx, *part_indices),
            method="whole_and_parts",
        )

    merge_indices = non_contained_indices if len(non_contained_indices) >= 1 else candidate_indices
    combined = np.zeros_like(masks[merge_indices[0]], dtype=bool)
    for idx in merge_indices:
        combined |= masks[idx].astype(bool)
    return MaskSelectionResult(
        final_mask=combined,
        contributing_mask_indices=tuple(merge_indices),
        method="merged_candidates",
    )


def rescue_complementary_mask(
    current_result: MaskSelectionResult,
    confidences: np.ndarray,
    masks: np.ndarray,
    min_fill_ratio: float = DEFAULT_RESCUE_MIN_FILL_RATIO,
    max_fill_ratio: float = DEFAULT_MAX_FILL_RATIO,
    max_overlap_frac: float = DEFAULT_RESCUE_MAX_OVERLAP_FRAC,
) -> MaskSelectionResult:
    if current_result.final_mask is None:
        return current_result

    current_mask = current_result.final_mask
    best_candidate_idx = None
    best_candidate_area = 0

    for i in range(len(masks)):
        if i in current_result.contributing_mask_indices:
            continue
        candidate = masks[i].astype(bool)
        fill_ratio = bounding_box_fill_ratio(candidate)
        if not (min_fill_ratio <= fill_ratio <= max_fill_ratio):
            continue
        candidate_area = int(candidate.sum())
        if candidate_area == 0:
            continue
        overlap = int((candidate & current_mask).sum())
        overlap_frac = overlap / candidate_area
        if overlap_frac > max_overlap_frac:
            continue
        if candidate_area > best_candidate_area:
            best_candidate_idx = i
            best_candidate_area = candidate_area

    if best_candidate_idx is None:
        return current_result

    rescued_mask = current_mask | masks[best_candidate_idx].astype(bool)
    return MaskSelectionResult(
        final_mask=rescued_mask,
        contributing_mask_indices=(*current_result.contributing_mask_indices, best_candidate_idx),
        method=current_result.method + "+rescued",
    )


def _measure_largest_gap(mask: np.ndarray, n_rows_to_check: int = 5) -> int:
    rows_with_content = mask.any(axis=1)
    row_indices = np.where(rows_with_content)[0]
    if len(row_indices) == 0:
        return 0
    center_row = (int(row_indices[0]) + int(row_indices[-1])) // 2
    rows_to_check = range(
        max(0, center_row - n_rows_to_check // 2),
        min(mask.shape[0], center_row + n_rows_to_check // 2 + 1),
    )
    max_gap = 0
    for row in rows_to_check:
        true_indices = np.where(mask[row, :])[0]
        if len(true_indices) < 2:
            continue
        diffs = np.diff(true_indices)
        row_max_gap = int(diffs.max()) - 1 if len(diffs) > 0 else 0
        max_gap = max(max_gap, row_max_gap)
    return max_gap


def close_internal_gaps(mask: np.ndarray, safety_margin: float = DEFAULT_GAP_CLOSING_SAFETY_MARGIN) -> np.ndarray:
    gap_width = _measure_largest_gap(mask)
    if gap_width == 0:
        return mask
    brush_width = int(gap_width * safety_margin)
    return binary_closing(mask, structure=np.ones((5, brush_width)))


def resolve_pill_mask(confidences: np.ndarray, masks: np.ndarray, **kwargs) -> MaskSelectionResult:
    select_kwargs = {
        k: v for k, v in kwargs.items()
        if k in {"confidence_threshold", "max_fill_ratio", "area_sum_tolerance", "dominance_multiplier", "containment_threshold"}
    }
    base_result = select_pill_mask(confidences, masks, **select_kwargs)
    if base_result.final_mask is None:
        return base_result

    rescued_result = rescue_complementary_mask(base_result, confidences, masks)
    closed_mask = close_internal_gaps(rescued_result.final_mask)
    method_changed = not np.array_equal(closed_mask, rescued_result.final_mask)

    return MaskSelectionResult(
        final_mask=closed_mask,
        contributing_mask_indices=rescued_result.contributing_mask_indices,
        method=rescued_result.method + ("+closed" if method_changed else ""),
    )