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
   proof something is genuinely correct. When docs and the actual
   runtime disagree (e.g. a library's real error message), trust the
   runtime.
4. **Diagnose with real evidence before theorizing** - print actual
   values, don't guess-and-check blindly. For visual/segmentation
   debugging specifically: look at actual images sooner rather than
   leaning on aggregate statistics alone - a 100%-confirmation-rate
   result can still be built on an incomplete picture of *why*.
5. **Update DEVLOG.md as you go**, including dead ends and mistakes,
   not just what worked.
6. **Proportionate effort** - not every anomaly needs a deep dive.
7. **The user runs all code themselves** on their own machine/Colab -
   give exact content and commands, wait for real output before
   concluding anything.
8. **Re-test broadly after any fix**, not just against the one case
   that motivated it.
9. **Never let a secret (API token, key) sit in a script, chat
   message, or committed file.** Read from environment variables only;
   if one is ever exposed, revoke and replace it before continuing.

## Where things actually stand right now

**Phase 1 (data): DONE.** 5,728 images, 1,000 pill types, 96.8% text
metadata coverage. Entry point: `pillrag.data.build_pill_dataset()`.

**Phase 2 (segmentation): DONE, with one known unfixed bug and one
tracked follow-up.** Entry point: `pillrag.segment.resolve_pill_mask
(confidences, masks, image_path=None, known_shape=None, **kwargs)`.

- Mask-selection logic validated against 9 known hard cases + wide
  random samples per shape (CAPSULE/ROUND/OVAL) - see DEVLOG.md if you
  want the details, not needed to move forward.
- **Real batch run complete**: all 5,728 images processed in ~28 min.
  `quality_flag`: ok=5,191 (90.6%), fallback_full_image=537 (9.4%),
  error=0. Output: `data/masks/manifest_chunk_000.parquet` through
  `_011.parquet` (RLE-encoded masks, Parquet format).
- **Fallback investigation (why 9.4%, not the predicted ~1%) is
  CLOSED.** Root cause confirmed: on low-contrast consumer photos,
  FastSAM frequently fuses the pill together with adjacent background
  into one blob, which then fails the fill-ratio filter - and often a
  correct, unfused mask exists at a lower rank in the same output.
  Considered and explicitly REJECTED trying to heuristically pick the
  better lower-ranked candidate (no defensible rule separates "correct
  unfused pill" from "artifact" using fill-ratio/shape alone). Decided
  fix: SAM 3 (semantic-prior segmenter), not a smarter picker - see
  "SAM3 follow-up" below.
- **Known unfixed bug, accepted risk**: dominance-check bug
  (idx=9, `00093-7305-65_7C2F3E59`) where a border/corner artifact can
  win the dominance check on some ROUND images. Two candidate fixes
  tried and rejected (see DEVLOG.md). Whether it's rare or common is
  UNRECONCILED - two investigations point opposite ways. Explicit
  decision: proceed anyway, accepted as unquantified risk.

## Phase 3 (embedding + vector search): index is LIVE, query side not built

**Scope decision**: vector index = reference images ONLY
(`is_ref==True`, 2,000 rows / 1,000 pill types). Consumer images
(`is_ref==False`, 3,728 rows) are NOT indexed - they're evaluation
queries with known ground truth (`pilltype_id`/`label`). Verified
against the ePillID paper's own reference-gallery/consumer-query
experimental design, and against full consumer<->reference coverage
in this project's actual subset (960/960 consumer pilltype_ids have a
matching reference - zero gaps). See DEVLOG.md "REVERSED: Phase 3
index scope" for the full trail (this reversed an earlier "index all
5,728 rows" framing that was correct for Phase 2 but not Phase 3).

**`embed_image(image_path, mask)` - DONE, built and verified.**
Located in `pillrag/embed.py`. Crops to the mask's true bounding box
(via `mask_bounding_box()`), NO pixel-level masking within the crop
(explicit user decision - real background pixels inside the bbox are
kept as-is). ResNet-18 feature extractor (ImageNet-pretrained, FC
layer stripped), ImageNet normalization, 512-dim float32 output.
Verified: correct shape/dtype, different images produce meaningfully
different embeddings, same image is deterministic across calls.

**Batch-embed of the 2,000 reference images - DONE.**
`batch_embed_reference.py`. Combined embed+metadata approach (user's
explicit call over a separate-then-join alternative). Output:
`data/embeddings/reference_embeddings.parquet` (full_image_path,
embedding[512], pilltype_id, label, drug_name, color, shape). 2000/2000
embedded, 0 skipped, 36s total.

**Deep Lake vector store - DONE, LIVE, VERIFIED.**
`al://saifmoazam2/pillrag-reference-embeddings` (Deep Lake v4.x native
API). `upload_to_deeplake.py`. 2000/2000 rows committed, re-opened
fresh and verified (row count + spot check on real data). Schema:
full_image_path (text), embedding (embedding(512), has a clustered
similarity-search index), pilltype_id/label/drug_name/color/shape
(text). Auth via `ACTIVELOOP_TOKEN` env var, set by the user each
session (Colab Secrets recommended) - NEVER hardcode a token in any
script.

**NOT yet built:**
- [ ] **`search_visual`**: takes a raw image, runs `resolve_pill_mask`
      + `embed_image`, queries the Deep Lake dataset for nearest
      neighbors, returns matched pilltype_id/label/drug_name. This is
      the actual next thing to build.
- [ ] **Eval script**: run all 3,728 consumer images through
      `search_visual`, check retrieval accuracy against known
      `pilltype_id`. The real end-to-end test of Phase 2+3.

## SAM3 follow-up (tracked, not blocking)

Decided fix for the 537 `fallback_full_image` cases (see Phase 2
section above) - re-segment them with SAM 3, a semantic-prior model
prompted with a pill-related text concept, instead of trying to
out-guess FastSAM's ambiguous candidates. SAM 3 is publicly available,
integrated into `ultralytics`, commercially licensed - NOT a "wait for
it" situation, just gated access.

**This does NOT block Phase 3** - the 537 fallback rows aren't in the
index anyway (consumer-only), so this is now an eval-accuracy
improvement, not an index-integrity fix.

- [ ] Request SAM3 access at `https://huggingface.co/facebook/sam3`
      (login, agree to SAM License, submit request form) - if not
      already done.
- [ ] Once granted: time a small batch first (same discipline as the
      original 100-image FastSAM test), then re-segment the 537
      fallback images, then re-index just those images' embeddings.

## Immediate next step

**Build `search_visual`** (segment -> embed -> query Deep Lake ->
return match), then the eval script that runs all 3,728 consumer
images through it and reports real retrieval accuracy. That's the
actual end-to-end proof this system works.

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
- Hough Circle fallback: considered adequately validated, closed - a
  rare-case safety net, not a common path (searched broadly, found
  zero genuine additional failures across 90 ROUND images)
- OVAL doesn't need a dedicated low-contrast fallback in THIS dataset
  (dataset-specific finding, not a general CV claim)
- CAPSULE needed containment filtering + whole/parts plausibility
  checking + rescue - all built and validated
- The 3,902 extra ePillID pills with hash-style labels (no recoverable
  NDC/text metadata) are deferred, both from text RAG AND visual search
- Phase 3 index scope: reference-only (see Phase 3 section above) -
  do NOT re-open this, it's verified two independent ways
- `embed_image` signature (image_path+mask, not image_array+mask or
  manifest_row) - decided, built, verified working. Don't re-litigate.
- Background handling in embed_image: tight bbox crop, no internal
  pixel masking - explicit user decision, known consequence for loose
  masks documented in embed.py's docstring
- Colab session resets: ALWAYS assume full reset after reload OR
  restart - re-mount Drive, reinstall packages, re-set env vars,
  re-import everything, rebuild df/model/ZIP_PATH. Use
  `notebooks/colab_full_test_setup.py` for the base pipeline, or
  `setup_deeplake.py` if Deep Lake work is also needed this session
  (it does the full base rebuild too, not just Deep Lake auth - an
  earlier version of this file only did the Deep Lake piece and caused
  a confusing FileNotFoundError, see DEVLOG.md).
- numpy/scipy: do NOT try to fix Colab import errors via `pyproject.toml`
  version pins - this was tried repeatedly and never worked. The real
  fix is always: install, then restart the runtime, then re-import.
- Deep Lake API: use `types.Embedding(size=..., dtype="float32")`, NOT
  `dimensions=`/`dtype=types.Float32()` - the latter matched some doc
  pages but not the actually-installed deeplake 4.6.5. If a future
  deeplake upgrade changes this again, trust the runtime TypeError's
  own reported signature over any doc page.

## Key files

- `DEVLOG.md` - full history, all reasoning, all mistakes, all evidence
- `HANDOFF.md` - this file
- `README.md` - setup instructions for a human developer
- `src/pillrag/data.py` - Phase 1, `build_pill_dataset()`
- `src/pillrag/segment.py` - Phase 2, `resolve_pill_mask()`
- `src/pillrag/embed.py` - Phase 3, `embed_image()`, `mask_bounding_box()`
- `notebooks/colab_full_test_setup.py` - two-cell Colab setup for the
  base pipeline (data/model/9 known test cases)
- `notebooks/setup_deeplake.py` - two-cell Colab setup for Deep Lake
  work specifically (includes the full base rebuild too)
- `batch_segment_full.py` - Phase 2's real 5,728-image batch run
- `batch_embed_reference.py` - Phase 3's 2,000-reference-image batch embed
- `upload_to_deeplake.py` - Phase 3's Deep Lake upload
- `scripts/README.md` - index of every one-off diagnostic script from
  Phase 1's data investigation
