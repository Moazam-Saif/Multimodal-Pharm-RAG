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

**What actually works** (tested against 3 distinct real cases - a
two-tone capsule with a band artifact, a clean single-color round
tablet, and a consumer photo with a whole-image "blob" artifact):
bounding-box fill ratio (a mask's own pixel area divided by its own
bounding box area). Band/blob artifacts are near-perfect rectangles
(fill ratio ~0.995-1.0); genuine pill shapes, even irregular ones like
a two-tone capsule half, are meaningfully lower (~0.79-0.93 across all
3 real test cases). This signal, unlike the rejected ones above, is
independent of image framing/zoom AND independent of a mask's
relationship to other masks - it only depends on the mask's own shape.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np


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


@dataclass
class MaskSelectionResult:
    """The outcome of running mask selection on one FastSAM result.

    final_mask: the resolved pill mask (boolean array), or None if no
        valid mask could be identified at all.
    contributing_mask_indices: which of the original raw mask indices
        were used to build final_mask. Useful for debugging/QA - lets
        us re-visualize exactly which raw masks were kept.
    method: which resolution path was taken - "single", "dominant",
        "whole_and_parts", "merged_candidates", or "none_valid" - so we
        can track, across a full batch run, how often each case
        actually occurs.
    """

    final_mask: np.ndarray | None
    contributing_mask_indices: tuple[int, ...]
    method: str


def bounding_box_fill_ratio(mask: np.ndarray) -> float:
    """What fraction of a mask's own bounding box does it actually
    fill? A near-perfect rectangle (e.g. a band/frame artifact) is
    close to 1.0. A rounded pill shape - even an irregular one like
    half a two-tone capsule - is meaningfully lower, since its bounding
    box has empty corners. Verified against 3 distinct real test cases
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
) -> tuple[int, tuple[int, int]] | None:
    """Look for a mask whose area approximately equals the SUM of two
    other candidate masks' areas. Returns (whole_index, (part_a, part_b))
    for the best (closest-matching) such relationship found, or None if
    none exists.

    IMPORTANT: candidate_indices must already be filtered down to masks
    that passed BOTH confidence and fill-ratio checks before this
    function runs - this function has no way to distinguish a genuine
    whole/parts relationship from a coincidental one on its own (this
    was proven the hard way - see DEVLOG.md - when a band artifact's
    area coincidentally summed close to two real mask areas). Filtering
    artifacts out beforehand is what makes this safe to use.

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
) -> MaskSelectionResult:
    """Resolve FastSAM's raw multi-mask output down to one final pill
    mask.

    Pipeline, in order:
      1. Reject masks below confidence_threshold (noise/low-confidence
         duplicates - e.g. the 0.29-confidence fragment seen in testing)
      2. Reject masks above max_fill_ratio (near-rectangular artifacts -
         e.g. band artifacts and whole-image "blob" detections, BOTH
         confirmed to reach fill ratios of 0.995-1.0 in real testing,
         vs. 0.79-0.93 for genuine pill shapes)
      3. Among what remains: if exactly one mask survives, use it
         directly. If multiple survive, check whether one is DOMINANT
         (its area exceeds dominance_multiplier times the combined area
         of all others) - if so, use it directly, skipping the riskier
         whole/parts search below entirely. This step exists because a
         real test case (see DEVLOG.md) showed the whole/parts search
         can find a coincidental match among small, unrelated fragments
         (e.g. logo/imprint text) even when an obviously-correct,
         dominant mask (the actual pill) was sitting right there.
      4. If no single mask is dominant, look for a whole/parts area-sum
         relationship (handles pills that get split into multiple
         same-object masks, e.g. a two-tone capsule's two roughly-equal
         halves, verified in testing to correctly NOT trigger the
         dominance check in step 3, since neither half is dramatically
         larger than the other)
      5. If no whole/parts relationship is found among multiple
         survivors, merge (union) all of them together as the final
         mask - confirmed necessary in testing, since FastSAM does not
         always produce a separate "whole object" mask alongside its
         parts (see DEVLOG.md)

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
            be treated as the whole object directly, skipping the
            whole/parts search

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

    whole_and_parts = _find_whole_and_parts(
        candidate_indices, areas, area_sum_tolerance
    )

    if whole_and_parts is not None:
        whole_idx, part_indices = whole_and_parts
        return MaskSelectionResult(
            final_mask=masks[whole_idx].astype(bool),
            contributing_mask_indices=(whole_idx, *part_indices),
            method="whole_and_parts",
        )

    # No whole/parts relationship found. This does NOT necessarily mean
    # one of the candidates is spurious - it may simply mean FastSAM
    # never produced a separate "whole object" mask at all (confirmed
    # to happen in real testing - see DEVLOG.md: a two-tone capsule
    # split into exactly two half-masks with no third "whole" mask
    # among the valid candidates, once the coincidental band artifact
    # was correctly filtered out). The correct final mask in this
    # situation is the UNION of all valid candidates - verified visually
    # to reconstruct the complete pill shape correctly in that test case.
    #
    # Note: by this point candidate_indices always has length > 1 (the
    # length == 1 case returned earlier above), so this always runs -
    # there is no remaining "single leftover candidate" case to handle
    # separately.
    combined = np.zeros_like(masks[candidate_indices[0]], dtype=bool)
    for idx in candidate_indices:
        combined |= masks[idx].astype(bool)
    return MaskSelectionResult(
        final_mask=combined,
        contributing_mask_indices=tuple(candidate_indices),
        method="merged_candidates",
    )