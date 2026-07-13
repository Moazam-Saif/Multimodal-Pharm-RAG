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

**NOT yet done / open right now:**

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

**Run the ~100-image timing batch** (not yet written) to estimate
full-batch runtime before committing to the real 5,728-image run.

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
