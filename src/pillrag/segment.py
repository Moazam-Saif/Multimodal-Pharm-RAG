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
DEFAULT_MAX_FILL_RATIO = 0.97  # masks with bbox fill ratio above this
                                # are treated as rectangular artifacts,
                                # not genuine pill shapes - see DEVLOG.md
                                # for the real test data behind this cutoff
DEFAULT_AREA_SUM_TOLERANCE = 0.30  # allow 30% difference between a
                                    # candidate "whole" mask's area and
                                    # the sum of its proposed "parts"
DEFAULT_DOMINANCE_MULTIPLIER = 2.0  # a candidate whose area exceeds
                                     # this multiple of ALL other
                                     # candidates' combined area is
                                     # treated as the whole object
                                     # directly - see DEVLOG.md for the
                                     # real failure case (a logo/text
                                     # fragment coincidentally matching
                                     # the whole/parts area-sum pattern)
                                     # that made this check necessary
DEFAULT_CONTAINMENT_THRESHOLD = 0.9  # a candidate whose area is at
                                      # least this fraction contained
                                      # inside another candidate is
                                      # treated as a sub-detail (e.g.
                                      # printed text sitting ON a pill
                                      # half), not a genuine separate
                                      # "part" - see DEVLOG.md for the
                                      # real failure case this fixes
                                      # (a printed digit fragment
                                      # coincidentally area-matched as
                                      # a "missing part" alongside a
                                      # real pill half)
DEFAULT_MAX_PART_TO_WHOLE_RATIO = 0.85  # a proposed "part" whose area
                                         # exceeds this fraction of its
                                         # proposed "whole" is rejected
                                         # as implausible - a genuine
                                         # part should be meaningfully
                                         # smaller than the whole it's
                                         # part of, not comparable in
                                         # size (or, in one real test
                                         # case, even LARGER than it -
                                         # a logical impossibility a
                                         # pure area-sum check alone
                                         # can't catch). See DEVLOG.md
                                         # "capsule #20" for the real
                                         # failure this fixes: two
                                         # genuinely separate, correctly
                                         # -shaped pill halves whose
                                         # areas coincidentally
                                         # satisfied whole/parts
                                         # arithmetic with each other.
DEFAULT_RESCUE_MIN_FILL_RATIO = 0.5   # a rejected low-confidence mask
                                       # must be at least this "pill-
                                       # shaped" (not a band/blob
                                       # artifact) to be rescued
DEFAULT_RESCUE_MAX_OVERLAP_FRAC = 0.15  # a rescued mask must be at
                                         # least 85% NEW area (not
                                         # substantially overlapping
                                         # what we already have)
DEFAULT_GAP_CLOSING_SAFETY_MARGIN = 1.5  # the gap-closing brush is
                                          # sized to the MEASURED gap
                                          # width times this margin,
                                          # not a fixed guess - see
                                          # DEVLOG.md: a brush smaller
                                          # than the real gap provably
                                          # fails to close it
DEFAULT_SUSPICIOUS_AREA_FRACTION = 0.5  # if a resolved mask's area is
                                         # under this fraction of the
                                         # single largest raw mask
                                         # across ALL masks (any
                                         # confidence), the result is
                                         # considered untrustworthy and
                                         # a lower-threshold retry is
                                         # attempted. See DEVLOG.md
                                         # "capsule #20" - confirmed
                                         # this correctly flags a real
                                         # failure (final area 5,780 vs
                                         # largest-any-mask 322,038)
                                         # while NOT flagging a genuine
                                         # correct result (final area
                                         # exactly equal to largest-any
                                         # -mask, both 620,460)
DEFAULT_RETRY_CONFIDENCE_THRESHOLD = 0.3  # confidence threshold used
                                           # for the suspicious-result
                                           # retry - lower than the
                                           # normal default, specifically
                                           # to surface real pill pieces
                                           # that FastSAM scored
                                           # inexplicably low (see
                                           # DEVLOG.md "capsule #20" -
                                           # both genuine pill halves
                                           # were below the normal 0.6
                                           # threshold, at 0.45 and 0.40)


@dataclass
class MaskSelectionResult:
    """The outcome of running mask selection on one FastSAM result.

    final_mask: the resolved pill mask (boolean array), or None if no
        valid mask could be identified at all.
    contributing_mask_indices: which of the original raw mask indices
        were used to build final_mask. Useful for debugging/QA - lets
        us re-visualize exactly which raw masks were kept.
    method: which resolution path was taken - "single", "dominant",
        "whole_and_parts", "merged_candidates", or "none_valid", each
        optionally suffixed with "+rescued" and/or "+closed" if those
        additional steps changed the result - so we can track, across
        a full batch run, how often each case actually occurs.
    """

    final_mask: np.ndarray | None
    contributing_mask_indices: tuple[int, ...]
    method: str


def bounding_box_fill_ratio(mask: np.ndarray) -> float:
    """What fraction of a mask's own bounding box does it actually
    fill? A near-perfect rectangle (e.g. a band/frame artifact) is
    close to 1.0. A rounded pill shape - even an irregular one like
    half a two-tone capsule - is meaningfully lower, since its bounding
    box has empty corners. Verified against multiple real test cases
    in DEVLOG.md before being trusted as a filtering signal.
    """
    rows_with_content = mask.any(axis=1)
    cols_with_content = mask.any(axis=0)

    if not rows_with_content.any():
        return 0.0  # empty mask, guard against downstream errors

    row_indices = rows_with_content.nonzero()[0]
    col_indices = cols_with_content.nonzero()[0]

    bbox_height = int(row_indices[-1] - row_indices[0] + 1)
    bbox_width = int(col_indices[-1] - col_indices[0] + 1)
    bbox_area = bbox_height * bbox_width

    return float(mask.sum()) / bbox_area


def _mask_areas(masks: np.ndarray) -> dict[int, int]:
    """Pixel area per mask, as ordinary Python ints (NOT numpy uint -
    using numpy's unsigned integer types here caused a silent integer
    overflow bug during initial testing when subtracting areas; see
    DEVLOG.md. Plain Python ints have no such overflow risk)."""
    return {i: int(masks[i].sum()) for i in range(len(masks))}


def _containment_fraction(small_mask: np.ndarray, big_mask: np.ndarray) -> float:
    """What fraction of small_mask's own area lies inside big_mask?
    A genuine sub-detail (e.g. printed text sitting on a pill's
    surface) should be almost entirely contained in the pill piece it
    sits on; an unrelated, genuinely separate part of the pill should
    not be. Verified in real testing (see DEVLOG.md): a spurious
    "part" mask that broke whole/parts matching was found to be 100%
    contained inside one of the genuine pill-half masks, while the two
    genuine halves themselves barely overlap each other at all
    (~0.1%) - a clean, reliable separation.
    """
    small_area = small_mask.sum()
    if small_area == 0:
        return 0.0
    overlap = (small_mask & big_mask).sum()
    return float(overlap) / float(small_area)


def _exclude_contained_masks(
    candidate_indices: list[int],
    masks: np.ndarray,
    containment_threshold: float,
) -> list[int]:
    """Remove candidates that are substantially contained inside
    another candidate - these are sub-details (e.g. printed text) on
    a real pill piece, not genuine separate parts, and including them
    in the whole/parts search invites spurious arithmetic matches (see
    DEVLOG.md for the real failure case this fixes).
    """
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


def _find_dominant_mask(
    candidate_indices: list[int],
    areas: dict[int, int],
    multiplier: float,
) -> int | None:
    """Check whether one candidate's area dramatically exceeds the
    combined area of every other candidate (by more than `multiplier`
    times). If so, that mask is almost certainly the complete object on
    its own, and should be used directly rather than risking the
    whole/parts area-sum search below - which has no way to distinguish
    a genuine part-of-object relationship from a coincidental one among
    small, unrelated fragments (confirmed to happen in real testing:
    see DEVLOG.md - a dominant pill mask was ignored in favor of three
    small logo/text fragments whose areas coincidentally summed
    correctly against each other).

    Returns the dominant mask's index, or None if no single candidate
    is dramatically larger than the rest (e.g. the two-tone capsule
    case, where two roughly-equal-sized halves should NOT trigger this -
    verified: neither half exceeds even 1.5x the other in testing).
    """
    total_area = sum(areas[i] for i in candidate_indices)

    for idx in candidate_indices:
        rest = total_area - areas[idx]
        if rest == 0:
            continue  # only one candidate total - handled separately by caller
        if areas[idx] > rest * multiplier:
            return idx

    return None


def _find_whole_and_parts(
    candidate_indices: list[int],
    areas: dict[int, int],
    tolerance: float,
    max_part_to_whole_ratio: float = DEFAULT_MAX_PART_TO_WHOLE_RATIO,
) -> tuple[int, tuple[int, int]] | None:
    """Look for a mask whose area approximately equals the SUM of two
    other candidate masks' areas. Returns (whole_index, (part_a, part_b))
    for the best (closest-matching) such relationship found, or None if
    none exists.

    IMPORTANT: candidate_indices must already be filtered down to masks
    that passed confidence, fill-ratio, AND containment checks before
    this function runs - it has no way to distinguish a genuine
    whole/parts relationship from a coincidental one on its own (proven
    the hard way multiple times - see DEVLOG.md - a band artifact, a
    printed-text fragment, and two genuinely separate correctly-shaped
    pill halves whose sizes happened to satisfy the arithmetic with
    each other).

    Also rejects matches where either proposed "part" exceeds
    max_part_to_whole_ratio of its proposed "whole" - a genuine part
    should be meaningfully smaller than the whole it belongs to, not
    comparable in size (or, as found in real testing, even larger than
    it - a case pure area-sum arithmetic cannot catch on its own).

    Checks EVERY valid pairing (not just the first one tried) and picks
    the CLOSEST match by area difference, rather than stopping at the
    first pairing under tolerance.
    """
    best_match: tuple[int, tuple[int, int], float] | None = None  # (whole, parts, diff_frac)

    for part_a, part_b in itertools.combinations(candidate_indices, 2):
        pair_sum = areas[part_a] + areas[part_b]
        if pair_sum == 0:
            continue  # guard against empty masks, avoid division by zero below

        for candidate_whole in candidate_indices:
            if candidate_whole in (part_a, part_b):
                continue

            diff_frac = abs(areas[candidate_whole] - pair_sum) / pair_sum
            if diff_frac > tolerance:
                continue

            whole_area = areas[candidate_whole]
            larger_part_area = max(areas[part_a], areas[part_b])
            if whole_area == 0 or larger_part_area / whole_area > max_part_to_whole_ratio:
                continue  # implausible: a "part" nearly as big (or bigger) than its "whole"

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
    """Resolve FastSAM's raw multi-mask output down to one final pill
    mask.

    Pipeline, in order:
      1. Reject masks below confidence_threshold (noise/low-confidence
         duplicates)
      2. Reject masks above max_fill_ratio (near-rectangular artifacts)
      3. Among what remains: if exactly one mask survives, use it
         directly. If multiple survive, check whether one is DOMINANT
         - if so, use it directly, skipping the riskier whole/parts
         search below entirely
      4. Exclude any remaining candidate that is substantially
         CONTAINED inside another candidate (a sub-detail like printed
         text sitting on a pill piece, not a genuine separate part -
         see DEVLOG.md)
      5. Look for a whole/parts area-sum relationship among what's left
      6. If no whole/parts relationship is found, merge (union) all
         remaining candidates together as the final mask

    See DEVLOG.md "Phase 2: Background Segmentation" for the full
    real-evidence trail behind every one of these steps.

    Args:
        confidences: 1D array of per-mask confidence scores, as returned
            by results[0].boxes.conf.cpu().numpy()
        masks: 3D array of per-mask boolean/binary pixel data, as
            returned by results[0].masks.data.cpu().numpy()
        confidence_threshold: masks below this are excluded before any
            further logic runs
        max_fill_ratio: masks with bounding-box fill ratio above this
            are excluded as likely rectangular artifacts
        area_sum_tolerance: how much difference to allow between a
            candidate "whole" mask's area and the sum of its proposed
            "parts" areas
        dominance_multiplier: how many times larger than the combined
            area of all other candidates a single candidate must be to
            be treated as the whole object directly
        containment_threshold: how much of a candidate's own area must
            lie inside another candidate for it to be excluded as a
            sub-detail rather than a genuine separate part

    Returns:
        A MaskSelectionResult. final_mask is None if no mask survives
        both the confidence and fill-ratio filters - callers must
        handle this case explicitly (e.g. flag the image for manual
        review) rather than assume a mask always comes back.
    """
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
        return MaskSelectionResult(
            final_mask=None, contributing_mask_indices=(), method="none_valid"
        )

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

    # Exclude sub-details (e.g. printed text/digits) from the whole/parts
    # search - see DEVLOG.md for the real failure this fixes.
    non_contained_indices = _exclude_contained_masks(candidate_indices, masks, containment_threshold)
    search_indices = non_contained_indices if len(non_contained_indices) >= 2 else candidate_indices

    whole_and_parts = _find_whole_and_parts(
        search_indices, areas, area_sum_tolerance
    )

    if whole_and_parts is not None:
        whole_idx, part_indices = whole_and_parts
        return MaskSelectionResult(
            final_mask=masks[whole_idx].astype(bool),
            contributing_mask_indices=(whole_idx, *part_indices),
            method="whole_and_parts",
        )

    # No whole/parts relationship found. Merge (union) all remaining
    # valid candidates together - see DEVLOG.md for why this is
    # necessary (FastSAM does not always produce a separate "whole
    # object" mask alongside its parts).
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
    """Check whether a REJECTED (low-confidence) mask genuinely
    completes the current result - i.e. is pill-shaped and largely
    NEW area, not a duplicate of what we already have. If found, merge
    it in.

    This exists because FastSAM sometimes correctly detects the right
    shape for a genuine missing piece of a pill, but scores its own
    confidence very low for reasons unrelated to correctness (verified
    in real testing - see DEVLOG.md: a plain, texture-free pill section
    was detected with the exact right shape, but scored only ~0.3
    confidence, consistently, regardless of color/grayscale input).
    Rather than lower the global confidence threshold (which would
    reintroduce genuine noise), this specifically looks for a
    low-confidence mask that plausibly completes an otherwise-
    incomplete result.
    """
    if current_result.final_mask is None:
        return current_result

    current_mask = current_result.final_mask

    best_candidate_idx = None
    best_candidate_area = 0

    for i in range(len(masks)):
        if i in current_result.contributing_mask_indices:
            continue  # already used

        candidate = masks[i].astype(bool)
        fill_ratio = bounding_box_fill_ratio(candidate)

        if not (min_fill_ratio <= fill_ratio <= max_fill_ratio):
            continue  # not pill-shaped enough to be a genuine piece

        candidate_area = int(candidate.sum())
        if candidate_area == 0:
            continue

        overlap = int((candidate & current_mask).sum())
        overlap_frac = overlap / candidate_area
        if overlap_frac > max_overlap_frac:
            continue  # mostly duplicates existing area, not a new piece

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
    """Scan several horizontal rows through the mask's own vertical
    center, measure the widest gap (a run of background sandwiched
    between foreground pixels) found in any of them. Checking multiple
    rows, not just one, is more robust to an unlucky single-row
    measurement (e.g. a row that happens to cross a narrow feature).
    """
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


def close_internal_gaps(
    mask: np.ndarray,
    safety_margin: float = DEFAULT_GAP_CLOSING_SAFETY_MARGIN,
) -> np.ndarray:
    """Measure the mask's own largest internal gap directly, then
    close it using a brush sized to that measurement (plus a safety
    margin) - rather than a fixed, arbitrary brush size.

    This exists because merging two separately-detected pieces of a
    pill (e.g. left and right halves of a capsule) can leave a thin
    seam where a real physical feature (e.g. a printed band) sat
    between them but was excluded as an artifact. Verified in testing
    (see DEVLOG.md): a fixed, too-small brush provably fails to bridge
    a real gap (a 25px brush could not close an 81px gap), so sizing
    the brush to the ACTUAL measured gap is the principled approach,
    not a guess.
    """
    gap_width = _measure_largest_gap(mask)
    if gap_width == 0:
        return mask

    brush_width = int(gap_width * safety_margin)
    return binary_closing(mask, structure=np.ones((5, brush_width)))


def _is_suspiciously_small(
    final_area: int,
    all_masks: np.ndarray,
    min_fraction: float = DEFAULT_SUSPICIOUS_AREA_FRACTION,
    max_fill_ratio: float = DEFAULT_MAX_FILL_RATIO,
) -> bool:
    """Check whether a resolved mask's area is suspiciously small
    compared to the single largest PILL-SHAPED mask across ALL raw
    masks (regardless of confidence). The largest genuinely pill-
    shaped raw mask - even if low-confidence - is a reasonable proxy
    for "how big the real pill probably is."

    IMPORTANT: the reference mask must itself pass the fill-ratio
    check (not be a band/blob artifact) - confirmed necessary in real
    testing (see DEVLOG.md "capsule_0_WHITE regression"): comparing
    against the literal largest mask, with no shape filtering, can
    pick a BAND ARTIFACT as the reference (fill_ratio ~0.99), making a
    legitimately correct, recoverable result look artificially small
    and triggering an unnecessary, harmful retry. Excluding
    artifact-shaped masks from the reference calculation fixes this.
    """
    pill_shaped_areas = [
        int(m.sum()) for m in all_masks if bounding_box_fill_ratio(m) <= max_fill_ratio
    ]
    if not pill_shaped_areas:
        return False  # no pill-shaped mask exists at all to compare against

    largest_pill_shaped = max(pill_shaped_areas)
    if largest_pill_shaped == 0:
        return False
    return final_area < largest_pill_shaped * min_fraction


def resolve_pill_mask(
    confidences: np.ndarray,
    masks: np.ndarray,
    retry_confidence_threshold: float = DEFAULT_RETRY_CONFIDENCE_THRESHOLD,
    suspicious_area_fraction: float = DEFAULT_SUSPICIOUS_AREA_FRACTION,
    **kwargs,
) -> MaskSelectionResult:
    """The full, real pipeline: select -> (retry if suspicious) ->
    rescue -> close gaps.

    This is the function calling code should actually use end-to-end;
    select_pill_mask/rescue_complementary_mask/close_internal_gaps are
    exposed individually mainly for testing and debugging.

    The retry step exists because a normal-threshold result can
    sometimes be built entirely from small, high-confidence noise
    fragments when the GENUINE pill pieces were scored inexplicably
    low by FastSAM (see DEVLOG.md "capsule #20" - both real pill
    halves were below the default 0.6 threshold). If the initial
    result's area looks suspiciously small compared to the largest
    raw mask overall, the ENTIRE selection pipeline (not just a mask
    grab) is retried at a lower confidence threshold - all existing
    protections (fill-ratio, dominance, containment, whole/parts
    plausibility) still apply, just against a wider candidate pool.
    The retry result is only accepted if it's actually larger/more
    complete than the original suspicious result.

    Any keyword arguments are passed through to select_pill_mask (e.g.
    confidence_threshold, max_fill_ratio) where applicable by name.
    """
    select_kwargs = {
        k: v
        for k, v in kwargs.items()
        if k
        in {
            "confidence_threshold",
            "max_fill_ratio",
            "area_sum_tolerance",
            "dominance_multiplier",
            "containment_threshold",
        }
    }
    base_result = select_pill_mask(confidences, masks, **select_kwargs)

    if base_result.final_mask is not None:
        final_area = int(base_result.final_mask.sum())
        if _is_suspiciously_small(final_area, masks, suspicious_area_fraction):
            retry_kwargs = dict(select_kwargs)
            retry_kwargs["confidence_threshold"] = retry_confidence_threshold
            retry_result = select_pill_mask(confidences, masks, **retry_kwargs)

            if retry_result.final_mask is not None:
                retry_area = int(retry_result.final_mask.sum())
                if retry_area > final_area:
                    base_result = MaskSelectionResult(
                        final_mask=retry_result.final_mask,
                        contributing_mask_indices=retry_result.contributing_mask_indices,
                        method=retry_result.method + "+retried",
                    )

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
