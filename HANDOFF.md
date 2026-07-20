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

## Phase 3 (embedding + vector search): index LIVE, search_visual WORKS MECHANICALLY but has 0% accuracy - root cause diagnosed, fix in progress (Phase 4)

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

**`src/pillrag/visual_search.py` - FULLY BUILT AND MECHANICALLY VERIFIED:**
- [x] `segment_query_image()` - VERIFIED (round_clean test case, exact
      match to known-good resolve_pill_mask result).
- [x] `embed_query_image()` - VERIFIED (round_clean test case, exact
      match to expected embedding shape/dtype).
- [x] `search_visual()` - VERIFIED MECHANICALLY WORKING: ran a real
      50-image timing test against the live Deep Lake dataset. TQL
      query composes correctly (WHERE + ORDER BY COSINE_SIMILARITY in
      one query), the `similarity` column alias works, DatasetView row
      access works, shape filtering works correctly (spot-checked).
      **BUT: 0% top-1 and 0% top-5 accuracy across all 50 real
      queries** - uniform failure across every shape AND every color
      category (not concentrated in similar-looking pills). This is a
      REAL, DIAGNOSED problem, not a bug in this code - see below.

**ROOT CAUSE, confirmed via a real diagnostic trail (full evidence
chain in DEVLOG.md - query mechanics, shape filter, and embedding-
pipeline drift were each individually ruled out with real evidence
before landing here): the embedding model itself is not
discriminative enough for this task.** embed.py's ResNet-18 is a
zero-shot ImageNet-pretrained feature extractor - it was never trained
to distinguish fine-grained pharmaceutical detail (score lines, rim
geometry, subtle indentations) between one pill and another. It
produces a real, non-random notion of visual similarity (results
aren't garbage - a true match's cosine similarity was 0.93, in the
same 0.93-0.96 band as the wrong matches) but nowhere NEAR
discriminative enough to reliably tell pills apart. Confirmed this
dataset is the ePillID paper's own published benchmark - their best
approach was a metric-learning model TRAINED specifically on pill
data, not a generic pretrained classifier.

**Also observed, NOT yet investigated**: every single query during
the 50-image test printed a repeated
`WARNING:deeplake.storage.s3:...INVALID_ACCESS_KEY_ID` warning. Real
queries still returned correct-shaped results despite this, so it may
be cosmetic - but this has NOT been root-caused or connected to (or
ruled out as connected to) the accuracy problem. Don't assume it's
harmless without checking.

**Eval script (all 3,728 consumer images): NOT YET BUILT, and
DELIBERATELY BLOCKED until Phase 4's retrained model exists.** Running
the full eval now, against the current ResNet-18 embeddings, would
just be a slower, more expensive way of confirming the same 0%
accuracy already seen on the 50-image sample - not a useful next step.
The consumer images (is_ref==False, 3,728 rows) must stay COMPLETELY
UNTOUCHED until Phase 4 training is fully done - see Phase 4 section
below for why (they're reserved as the one real, held-out eval set).

## Phase 4 (metric-learning fine-tuning): IN PROGRESS - stage 1 of training script written, NOT YET RUN

**Goal**: replace the current zero-shot ResNet-18 embedding model with
one trained via metric learning (Supervised Contrastive Loss), so the
embedding space is directly optimized for "same pill close together,
different pill far apart" - see DEVLOG.md's diagnostic trail for the
full evidence chain that led here.

**Scope, as explicitly decided this session:**
  - Metric learning (SupCon loss) + Grad-CAM verification afterward.
  - Region-based embeddings explicitly PARKED - decide whether it's
    needed based on what Grad-CAM shows AFTER training, not built
    preemptively alongside it.
  - Backbone: ResNet-50 (up from ResNet-18) - matches the ePillID
    paper's own baseline.
  - Resolution: 384x384 (up from 224x224).
  - **NOTE**: both the backbone and resolution changes were called out
    in the user's own uploaded design report as steps to defer until
    AFTER the training-objective change was validated in isolation -
    explicit user decision to do all three at once instead. Known,
    accepted consequence: if results improve, we won't cleanly know
    which of the three changes actually drove it.

**Data scope decisions (important - don't re-litigate without reason):**
  - Train ONLY on the 2,000 reference images (is_ref==True) - the same
    set already indexed in Deep Lake.
  - Consumer images (is_ref==False, 3,728 rows) are the FINAL EVAL
    set and must NOT be touched during training OR used for
    validation-during-training (that would leak info into checkpoint
    selection and make the eventual real accuracy number not truly
    held-out).
  - Validation-during-training uses a HELD-OUT SPLIT OF REFERENCE
    PILL TYPES instead: 800/200 pilltype-level split (not image-level -
    both images of a pill type always stay in the same split).
  - Masks: reused directly from Phase 2's existing manifest parquet
    files (data/masks/manifest_chunk_000..011.parquet) via RLE decode -
    NOT re-segmented. Confirmed via real decode test this session:
    standard pycocotools COCO RLE (rle_size=[H,W], rle_counts=string
    column), decodes cleanly, no bytes-encoding workaround needed.
  - Data quality, VERIFIED via a real script (not assumed): all 1,000
    reference pilltype_ids have exactly 2 images (no exceptions). 4/2000
    rows have quality_flag=fallback_full_image; of those, exactly ONE
    pilltype_id (00093-1003-01_B326D9D6) has BOTH images as fallback
    (zero real pill-crop signal) - this one is EXCLUDED from the
    train/val split (see EXCLUDED_ZERO_SIGNAL_PILLTYPES in
    train_metric_learning.py). The other 3 fallback rows each have a
    real `ok` sibling image and are kept as-is.
  - Batch construction: a BALANCED sampler, not plain random shuffling
    - each batch draws N distinct pill types and includes BOTH images
    of each, guaranteeing every image has a real positive pair in-
    batch (SupCon needs this; with only ~2 images/class, random
    batches would mostly have zero true positive pairs, wasting the
    "pull together" half of the loss). One epoch = every training
    pill type appears in exactly one batch (shuffle pilltype_ids,
    chop into groups of N) - not a fixed-batches/sample-with-
    replacement scheme.
  - Augmentation, deliberately NOT a generic preset (reasoning in
    DEVLOG.md) - SAFE: full 360° rotation, h/v flip, MILD brightness/
    contrast/crop jitter. AVOIDED or minimal: blur (would smear score
    lines/rim detail), heavy color jitter (color is a real
    distinguishing feature here, not noise), aggressive cutout (risks
    blanking the one informative region in a small crop).

**`src/pillrag/train_metric_learning.py` - STAGE 1 ONLY, NOT YET RUN.**
Contains `load_reference_manifest()`, `decode_rle_mask()`,
`make_train_val_split()` - data loading and the train/val split only.
Does NOT yet contain: the Dataset class (augmentation + crop + resize),
the balanced batch sampler, the model (ResNet-50 + projection head),
the SupCon loss, or the training loop itself. **Run this first and
confirm the expected counts (999 eligible pilltypes, 799 train / 200
val, 0 overlap) before building anything further on top of it.**

**Immediate next steps, in order:**
1. Run train_metric_learning.py's data-loading stage, confirm counts.
2. Build the Dataset class (RLE decode -> bbox crop via
   mask_bounding_box -> augmentation -> resize to 384x384).
3. Build the balanced batch sampler.
4. Build the model (ResNet-50 + SupCon projection head) and the
   two-phase training loop (freeze backbone + train head first, then
   unfreeze layer4 at a lower LR - standard technique to avoid
   catastrophic forgetting of ResNet's pretrained features).
5. Implement SupCon loss, train, save checkpoints + loss curves.
6. Run Grad-CAM on the trained model to verify what it's actually
   attending to.
7. ONLY THEN decide whether region-based embeddings are needed, based
   on real Grad-CAM evidence (diffuse attention -> yes; already
   localized on rim/imprint -> probably not).
8. Re-embed the 2,000 reference images with the new model, rebuild
   the Deep Lake index (embedding dimension may differ from 512 -
   this will likely need a NEW dataset, not an in-place patch).
9. ONLY THEN run the real, one-time eval on all 3,728 consumer images
   (untouched until now) for the honest final accuracy number.

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

See "Phase 4" section above for the current, real next steps. (This
section previously said "build search_visual" - that's done; the
project has moved to fixing the accuracy problem search_visual
revealed.)

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
- **REVISED, not silently overturned**: `known_shape` at query time.
  The Hough Circle fallback's offline-only scope note (still true for
  an INFERRED shape - see segment.py's "IMPORTANT SCOPE NOTE") does
  NOT apply to a USER-DECLARED shape. The product requires the user to
  always select shape from a dropdown before taking a photo - this is
  a real, independently-known input (like telling a pharmacist "it's a
  round white pill"), not the pipeline inferring shape about itself.
  `known_shape` is REQUIRED (not `Optional`) everywhere in
  visual_search.py. Shape filtering in `search_visual` is a HARD
  filter (`shape == known_shape OR shape == ''`), not a soft boost -
  see DEVLOG.md for the full reasoning trail on why a soft boost was
  considered and rejected.
- `embed_image` signature (image_path+mask, not image_array+mask or
  manifest_row) - decided, built, verified working. Don't re-litigate.
- Background handling in embed_image: tight bbox crop, no internal
  pixel masking - explicit user decision, known consequence for loose
  masks documented in embed.py's docstring
- **Phase 3's ResNet-18 embeddings give 0% top-1/top-5 accuracy** -
  confirmed via a real, multi-step diagnostic trail (query mechanics,
  shape filter, embedding-pipeline drift ALL individually ruled out
  with real evidence) - root cause is the embedding model itself, not
  a bug in visual_search.py. Don't re-investigate query/filter/drift
  again without new evidence - see DEVLOG.md's full trail if you want
  to double-check the reasoning.
- **Phase 4 scope**: metric learning (SupCon) + Grad-CAM verification
  ONLY. Region-based embeddings explicitly PARKED - decide AFTER
  Grad-CAM results, don't build preemptively.
- **Phase 4 data scope**: train ONLY on the 2,000 reference images.
  Consumer images (is_ref==False) are the FINAL EVAL set - NEVER use
  them for training OR validation-during-training, even though it
  might seem convenient. This is intentional, not an oversight -
  don't "helpfully" add them to a validation loop.
- **Phase 4 backbone/resolution decision diverges from the design
  report's own recommended priority order** (report says isolate the
  training-objective change first, defer backbone/resolution) - user
  explicitly chose to change all three at once. This was a deliberate,
  informed tradeoff, not something to "fix" by reverting to the
  report's order without asking.
- **00093-1003-01_B326D9D6 is excluded from Phase 4 training** (both
  its reference images are quality_flag=fallback_full_image, zero real
  signal) - but NOT removed from the manifest load, live Deep Lake
  index, or eventual eval set. Don't re-add it to training without
  re-checking its mask quality first.
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
- `src/pillrag/embed.py` - Phase 3, `embed_image()`, `mask_bounding_box()`, `run_fastsam()`
- `src/pillrag/visual_search.py` - Phase 3, live query path:
  `segment_query_image()` (verified), `embed_query_image()`
  (verified), `search_visual()` (verified mechanically working, but
  see Phase 3/Phase 4 sections above - 0% accuracy, root cause is the
  embedding model, fix is Phase 4)
- `src/pillrag/train_metric_learning.py` - Phase 4, metric-learning
  training script. STAGE 1 ONLY so far (data loading + train/val
  split): `load_reference_manifest()`, `decode_rle_mask()`,
  `make_train_val_split()`. NOT yet run - run and confirm expected
  counts before building the Dataset class/sampler/model/loss/loop on
  top of it.
- `Pill_Retrieval_Design_Report.docx` - user-provided design report
  that independently arrived at the same root-cause diagnosis
  (generic ImageNet embeddings lack fine-grained discriminative
  power) and proposed the metric-learning + Grad-CAM direction. Read
  via `pandoc -t markdown` (viewing as raw text shows zip binary, not
  content - it's a real docx, needs pandoc/docx extraction). Reference/
  inspiration document, NOT followed as a literal spec - some of its
  recommended ordering (isolate training objective before changing
  backbone/resolution) was explicitly overridden by user decision, see
  Phase 4 section above.
- `notebooks/colab_full_test_setup.py` - two-cell Colab setup for the
  base pipeline (data/model/9 known test cases)
- `notebooks/setup_deeplake.py` - two-cell Colab setup for Deep Lake
  work specifically (includes the full base rebuild too)
- `batch_segment_full.py` - Phase 2's real 5,728-image batch run
- `batch_embed_reference.py` - Phase 3's 2,000-reference-image batch embed
- `upload_to_deeplake.py` - Phase 3's Deep Lake upload
- `scripts/README.md` - index of every one-off diagnostic script from
  Phase 1's data investigation
