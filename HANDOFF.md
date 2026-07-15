# HANDOFF.md

Quick-orientation companion to `DEVLOG.md`. Read this FIRST - it gets
you to a working state and tells you exactly what's proven vs. what's
still open. `DEVLOG.md` has the full reasoning/evidence trail behind
every decision here; come back to it when you need to understand *why*
something works the way it does, or before changing anything that
looks questionable at first glance.

## How this project expects to be worked on

(Full version: `DEVLOG.md`'s "Working instructions" section at the top.
Summary:)

1. **Teach, don't just build** - this is a learning project. Explain
   concepts before using them, explain code as you write it.
2. **Never silently pick between real options** - ask, with tradeoffs
   laid out, when there's a genuine fork.
3. **Verify, don't assume** - test claims (URLs, joins, API behavior)
   before asserting them as fact. Don't trust a tool "succeeding" as
   proof something is genuinely correct.
4. **Diagnose with real evidence before theorizing** - print actual
   values, don't guess-and-check blindly.
5. **Update DEVLOG.md as you go**, including dead ends and mistakes,
   not just what worked.
6. **Proportionate effort** - not every anomaly needs a deep dive.
7. **The user runs all code themselves** on their own machine/Colab -
   give exact content and commands, wait for real output before
   concluding anything.
8. **Re-test broadly after any fix**, not just against the one case
   that motivated it. This project was burned once already by shipping
   a fix that was never re-tested and failed identically to the
   original bug (see DEVLOG.md "MISTAKE" entries) - don't repeat that.

## Where things actually stand right now

**Phase 1 (data): DONE.** 5,728 images, 1,000 pill types, 96.8% text
metadata coverage. Entry point: `pillrag.data.build_pill_dataset()`.

## Phase 3 (embedding/vector search): SCOPE DECIDED, not yet built

**Vector index: reference images ONLY** (`is_ref==True`, 2,000 rows /
1,000 pill types) - this is the searchable catalogue.
**Consumer images (`is_ref==False`, 3,728 rows): NOT indexed** - used
purely as evaluation queries with known ground truth
(`pilltype_id`/`label`). This REVERSES the earlier "index all 5,728
rows" framing from Phase 2 planning, which was correct for Phase 2
(segmentation needs every image regardless of downstream use) but does
NOT carry over to Phase 3's index scope - two genuinely different
questions that got conflated once, caught and corrected by the user.

Verified against the ePillID paper itself (not just reasoned
independently) - confirms this reference-as-gallery/consumer-as-query
split matches the dataset creators' own experimental design. Also
verified full consumer<->reference pilltype_id coverage in OUR actual
5,728-row subset (960/960 consumer pilltype_ids have a matching
reference - zero gaps), since the paper notes the FULL ePillID dataset
has consumer images for only ~960 of ~4,902 total pill types - this
gap needed to be checked against our filtered subset specifically, not
assumed to not apply. See DEVLOG.md "REVERSED: Phase 3 index scope".

**Practical effect on the outstanding SAM3 fallback work**: the 537
`fallback_full_image` cases are no longer an index-integrity risk
(consumer images were never going into the index) - they're now an
eval-accuracy risk instead (bad segmentation -> bad embedding -> that
eval query may fail to retrieve its own correct reference). Still
worth fixing via the planned SAM3 re-segmentation, just for a
different reason.

**`embed_image` signature**: still being decided - real fork
(image_path+mask vs image_array+mask vs manifest_row), leaning image_
path+mask since it serves both the batch-embed job AND the live
`search_visual` query path with the same function, no manifest-schema
coupling. NOT yet confirmed by user - don't build against it yet.

**Background handling inside crop**: DECIDED - tight bbox crop to the
mask's true bounding box, NO pixel-level masking within it (i.e. real
background pixels inside the bbox are kept, not zeroed/grayed/blurred).
Known consequence, not an oversight: this leaves the 537
`fallback_full_image` rows (bbox = whole image) and any fused-blob
masks fully exposed to background noise in their embeddings until the
SAM3 follow-up replaces those masks - but those rows aren't going into
the index anyway per the above, so this mainly affects EVAL query
embeddings for consumer-photo fallback cases, not the index itself.

**Phase 2 (segmentation): mask-selection logic is DONE and VALIDATED.**
Entry point: `pillrag.segment.resolve_pill_mask(confidences, masks,
image_path=None, known_shape=None, **kwargs)`.

Validated against:
- 9 specific known-hard cases (two-tone capsule + band artifact, clean
  round tablet, consumer photo + blob artifact, capsules with printed
  text/logos causing various failure modes, two genuine pill-halves
  that coincidentally satisfied whole/parts arithmetic)
- A wide, genuinely random 30-image CAPSULE sample
- Wide random samples for ROUND and OVAL (done earlier, see DEVLOG.md)

**UPDATE: the real batch run is DONE.** All 5,728 images processed,
1,688s (~28 min), 0.295s/image. `quality_flag`: ok=5,191 (90.6%),
fallback_full_image=537 (9.4%), error=0. Output: 12 chunk files in
`data/masks/manifest_chunk_000.parquet` through `_011.parquet`
(Parquet, not JSON - picked without asking first, flagged after the
fact, easy to switch if it matters). Script: `batch_segment_full.py`.

**NOW UNDER INVESTIGATION: why is the fallback rate 9.4%, not the ~1%
the 100-image timing test predicted?** Root-caused most of the way,
but genuinely NOT finished - see DEVLOG.md's "Fallback investigation"
section for the full trail, including two wrong/partial hypotheses
along the way. Current state:

- It's overwhelmingly a consumer-photo problem, not shape/color:
  `is_ref==False` gets 14.3% fallback rate vs `is_ref==True`'s 0.2% -
  confirmed as the real driver, not a confound, by holding shape+color
  fixed and still seeing the same gap.
- Of the 533 consumer-photo fallbacks, 76% (410) are cases where
  FastSAM found a good candidate but our OWN `select_pill_mask` fill-
  ratio filter (`max_fill_ratio=0.97`) rejected it - traced this to
  100% certainty by re-checking all 410 directly, not inferred.
- WHY those candidates have fill ratio ~0.97-1.0: ~19.5% (80/410) is a
  confirmed black-letterbox-padding artifact tricking the filter.
  **The remaining ~330: RESOLVED.** Claude's earlier (a)/(b) split
  (some genuinely correct CAPSULE/OVAL masks vs. separate background-
  sliver artifacts) was INCOMPLETE - user was right to push back.
  Rendered ALL masks FastSAM produced per image (not just the single
  rejected candidate) and found the real dominant mechanism: **FastSAM
  frequently fuses the pill together with a chunk of adjacent
  background into one blob** on low-contrast consumer photos - and
  often a correct, tighter, unfused pill mask exists at a LOWER rank
  in the same output. This confirms the user's original theory from
  early in the investigation. Full trail: DEVLOG.md "RESOLVED: (a)/(b)
  dispute settled via all-masks render".

**DECIDED FIX APPROACH: move to SAM 3 (semantic-prior segmenter) for
the fallback population, NOT a smarter FastSAM-candidate-picker.**
Explicitly considered and rejected trying to pick the better
lower-ranked FastSAM mask heuristically - no defensible rule exists
that separates "correct unfused pill" from "artifact" using
fill-ratio/shape alone (the signals point opposite directions
depending on pill shape - see DEVLOG.md for the full reasoning). SAM 3
sidesteps this by being asked for "the pill" directly via text prompt
instead of producing ambiguous unlabeled candidates to choose between.

**SAM 3 status**: publicly available (released Nov 2025), integrated
into `ultralytics` (matches this project's existing FastSAM tooling),
licensed for commercial+research use (Meta's SAM License, no blocker
for this project). **Access is gated** - requires requesting approval
via `https://huggingface.co/facebook/sam3` (not yet done / in
progress). Compute is heavier than FastSAM (~840M params, GPU-bound)
but should fit Colab's free-tier T4 at our scale - needs its own
timing test once access is granted, same discipline as the original
100-image FastSAM test.

**FORWARD PLAN (explicit user call) - Phase 3 is NOT blocked on this:**
1. Proceed with Phase 3 development NOW using the current manifest
   as-is - the 537 `fallback_full_image` rows stay exactly as they
   are for the time being.
2. SAM3 access request is in progress in parallel.
3. Once granted: re-segment ONLY the 537 fallback images with SAM3,
   replacing their whole-image fallback masks with real ones.
4. Re-index just those images' embeddings in Phase 3 - a targeted
   update to a known subset, not a full pipeline re-run.

**NOT yet done / open right now:**

- [ ] **Request SAM3 access** (`https://huggingface.co/facebook/sam3`)
      - blocking step for the fallback re-segmentation follow-up, but
        NOT blocking for Phase 3 itself. See DEVLOG.md for full
        access-request steps.
- [ ] **SAM3 re-segmentation of the 537 fallback images** - waiting on
      access above. Do a small timing test first before committing to
      the full 537, same as was done for FastSAM's 100-image test.


**Hough Circle fallback: considered adequately validated, closed for now.**
Re-confirmed working on the original "TV" tablet case. Searched broadly
for more genuine low-contrast failures (90 ROUND images across 3
independent samples, including a targeted 40-image WHITE-specific
search) and found zero - concluding the fallback is a rare-case safety
net, not a common path, and accepting this without further root-cause
digging into the TV tablet case (explicit user call). No further Hough
validation work owed unless a new failure surfaces during the batch run.

- [ ] **Dominance-check bug (idx=9, `00093-7305-65_7C2F3E59`) is
      STILL UNFIXED.** Two candidate fixes tested with real data and
      REJECTED - see DEVLOG.md for full numbers:
      - Largest-connected-component filtering: doesn't work, the
        artifact is one large connected shape, not fragmented specks
      - Circularity as a final-result gate: no clean threshold found;
        8/30 presumed-correct fresh results overlap the known artifact's
        score, and those 8 were never visually confirmed either way
      **Explicitly open, unresolved question**: is this bug rare (a
      visual check of a DIFFERENT 19-candidate false-alarm batch
      suggested so) or more common than known (the circularity overlap
      suggests otherwise)? These two investigations point in different
      directions and have not been reconciled.
      **DECISION MADE**: proceed to the batch run anyway, accepting this
      as a known, unquantified risk (explicit user call - see DEVLOG.md).
- [ ] No batch run across all 5,728 images has been attempted yet.
      All four planning questions below are now resolved; the
      ~100-image timing test is the immediate next step.

## Batch-run design decisions (all four open questions now resolved)

See DEVLOG.md's "Batch-run planning" entries for full reasoning. Summary:

- **Output**: mask only (RLE-encoded), NOT a background-blanked image -
  reaffirms the existing "don't persist segmented images" rule. Phase 3
  applies the mask to the original image in memory at embedding time.
- **Bundling**: chunked manifests, ~500 images per file - balances I/O
  speed against crash-resilience/resumability.
- **`None`/failure handling**: flag for review, but still fall back to
  the WHOLE raw image as an all-True mask so Phase 3 always has
  something to embed on. This fallback MUST be marked distinctly in the
  manifest (e.g. `quality_flag` or `method="fallback_full_image"`) - never
  silently indistinguishable from a genuine result.
- **`known_shape` wiring**: YES, wired through per-image from
  `df["shape"]`, enabling the Hough fallback during this batch. Confirmed
  within `hough_circle_fallback`'s documented offline-indexing-only scope.
- **GPU budgeting**: time a ~100-image batch first, extrapolate, report
  back before committing to the full 5,728-image run.

**Known accepted risk**: the idx=9 dominance-check bug (border/corner
artifact winning the dominance check on some ROUND images) remains
UNFIXED - proceeding with the batch run anyway per explicit user decision.

## Immediate next step

The fallback root-cause investigation is DONE (fusion mechanism
confirmed, see "Where things actually stand" above). Two parallel
threads now:

1. **Continue Phase 3 development** using the current manifest as-is
   - the 537 `fallback_full_image` rows are a known, accepted,
     temporary state, not a blocker.
2. **Request SAM3 access** at `https://huggingface.co/facebook/sam3`
   (if not already requested) - once granted, come back to re-segment
   the 537 fallback images and re-index them. See DEVLOG.md's "DECIDED
   forward plan" for the full sequence.

## Quick reference - what NOT to re-litigate

These were each investigated properly and decided; don't redo the
investigation, just note the decision (see DEVLOG.md for full reasoning
if you want it):

- Dataset: ePillID's `fcn_mix_weight/dr_224+dc_224` folders only (NOT
  `segmented_nih_pills_224` - different, disjoint, deferred pill set)
- Text metadata: joined via NDC, prefer `pillbox_*` fields over `spl*`
- `select_pill_mask()`'s pipeline order: confidence filter -> fill-ratio
  filter -> dominance check -> containment filter -> whole/parts search
  (with plausibility check) -> merge fallback
- `resolve_pill_mask()` wraps that with: suspicious-result retry at a
  lower threshold, rescue of complementary low-confidence masks,
  adaptive gap-closing, and the opt-in Hough Circle fallback
- OVAL doesn't need a dedicated low-contrast fallback in THIS dataset
  (dataset-specific finding, not a general CV claim)
- CAPSULE needed containment filtering + whole/parts plausibility
  checking + rescue - all built and validated
- The 3,902 extra ePillID pills with hash-style labels (no recoverable
  NDC/text metadata) are deferred, both from text RAG AND visual search
  (inconsistent preprocessing history risk - see DEVLOG.md)
- Colab session resets: ALWAYS assume full reset after reload OR
  restart - re-mount Drive, reinstall packages, re-set env vars,
  re-import everything. Use `notebooks/colab_full_test_setup.py`
  (two-cell version with working `os.kill`-based auto-restart) rather
  than doing this manually each time.
- numpy/scipy: do NOT try to fix Colab import errors via `pyproject.toml`
  version pins - this was tried repeatedly and never worked. The real
  fix is always: install, then restart the runtime, then re-import.

## Key files

- `DEVLOG.md` - full history, all reasoning, all mistakes, all evidence
- `HANDOFF.md` - this file
- `README.md` - setup instructions for a human developer
- `src/pillrag/data.py` - Phase 1, `build_pill_dataset()`
- `src/pillrag/segment.py` - Phase 2, `resolve_pill_mask()`
- `notebooks/colab_full_test_setup.py` - two-cell Colab setup with
  auto-restart + all 9 known test cases pre-loaded, for fast
  re-verification
- `scripts/README.md` - index of every one-off diagnostic script from
  Phase 1's data investigation
