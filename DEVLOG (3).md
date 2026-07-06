# Development Log

Running record of what we've built, what we decided, why, and what's still
open. Updated as we go — newest entries at the bottom.

## Working instructions (read this first — for any AI or human continuing this project)

This is a learning + portfolio project. The person building it is learning
CV/ML tooling (OpenCV, HuggingFace, PyTorch, segmentation, embeddings, RAG)
from scratch, alongside actually building the thing. Follow these rules:

1. **Teach, don't just build.** Explain every new concept before writing
   code that uses it. Explain every file and every non-trivial line, not
   just what it does but why it's written that way.
2. **Never silently pick between real implementation options.** If there's
   a genuine fork (dataset choice, library choice, architecture choice,
   normalization rule), stop and ask, with the real tradeoffs laid out
   plainly — don't just pick the "best" one and move on.
3. **Verify, don't assume.** This project has been burned repeatedly by
   assuming a source was reachable, a format was consistent, or a join
   would work, without checking. When a claim can be tested (does this
   URL actually resolve for a real user? does this join actually match?),
   test it before asserting it as fact. If a fetch/tool "succeeds," that
   is not proof something is genuinely live/correct for the user — say so
   plainly rather than overclaiming (this happened once already with a
   dead NLM domain — don't repeat it).
4. **When something breaks or returns a surprising result (0% match, a
   crash, a mismatch), diagnose it with real evidence** (print raw values,
   compare side by side, check `repr()` for hidden characters) **before
   theorizing.** Don't guess-and-check blindly; look first.
5. **Update this file (DEVLOG.md) as you go** — not just what was
   decided, but the investigation trail that led there (what was tried,
   what failed, why). This file is the project's memory across sessions;
   assume the next session starts with no other context.
6. **Proportionate effort.** Not every anomaly deserves a deep-dive - e.g.
   a data quality issue affecting 0.00% of rows was correctly just
   skipped rather than specially handled. Use judgment; note it either way.
7. **The person runs all code themselves** (Windows machine, separate from
   wherever this AI/assistant is reasoning) - so provide exact file
   content and exact commands to run, and wait for real output before
   concluding anything about whether something worked.
8. Currently working from a plan document covering 8 phases (data
   collection → segmentation → embedding → imprint extraction → text RAG
   → API → frontend → evaluation). Check current phase status in the
   "Open / next steps" section at the bottom of this file before assuming
   what's done.

---

## Environment

- **OS**: Windows (native), project lives at `D:\Multimodal-Pharm-RAG`
- **GPU**: none locally — heavy compute (batch segmentation, batch embedding)
  will run on Google Colab (free GPU tier); everything else (data prep, API,
  frontend) runs locally
- **Python**: 3.13.14
- **Package manager**: `venv` (built into Python) — chosen over conda/uv for
  simplicity, since it's the most common option in tutorials/docs we'll be
  referencing
- **Package layout**: src-layout (`src/pillrag/`) — forces an editable
  install (`pip install -e .`) so `import pillrag` always resolves through
  a real install, never an accidental local-file shadow

## Project structure decision

```
pill-rag/
├── notebooks/        # Colab-run GPU batch jobs (segmentation, embedding)
├── src/pillrag/       # installable package - core pipeline logic
├── scripts/           # local one-off scripts (no GPU needed)
├── api/                # FastAPI backend (Phase 6)
├── frontend/           # React or Gradio - format TBD in Phase 7
├── data/
│   ├── raw/                        # source downloads, gitignored
│   └── samples/segmented_preview/  # small QA sample, NOT full dataset
├── tests/
├── pyproject.toml
├── .env.example
├── .gitignore
```

**Decision: don't persist the full set of segmented images.** Segmentation
is deterministic (same input + same FastSAM weights = same output), so it's
a reproducible derived artifact, not a source of truth. Persisting only the
raw images + final embeddings (in Deep Lake, cloud-hosted) avoids slow
Colab↔Drive I/O on thousands of small files. Keep a ~50-image sample under
`data/samples/segmented_preview/` for manual quality inspection only.

## Environment setup (done)

1. `python -m venv venv` — created isolated environment
2. `.\venv\Scripts\Activate.ps1` — activated (had to run
   `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`
   first, PowerShell blocks scripts by default)
3. `pip install -e .` — editable install of the `pillrag` package, verified
   via `import pillrag; print(pillrag.__file__)` resolving into `src/`

**Dependency policy**: nothing gets `pip install`ed loose. Every dependency
is added to `pyproject.toml`'s `dependencies` list first, so that file stays
an accurate, readable history of what the project actually uses and when it
was introduced.

- `pandas>=2.0` — added for metadata inspection/normalization (pulled in
  `numpy`, `python-dateutil`, `six`, `tzdata` as transitive deps)

## Phase 1: Data Collection (in progress)

### Metadata

- Source: NIH Pillbox archived data (project retired Jan 28/29, 2021, no
  longer updated, but NLM explicitly keeps it available for research use)
- Downloaded from:
  `https://datadiscovery.nlm.nih.gov/api/views/crzr-uvwg/rows.csv?accessType=DOWNLOAD`
- Saved as `data/raw/pillbox_metadata.csv` (~56MB)
- 55 columns, confirmed via `scripts/inspect_metadata.py`

**Key columns identified:**
| Column | Meaning |
|---|---|
| `has_image` | **Must filter on this before anything else** — many rows have no photo |
| `splimage` | Likely links to the actual image filename — to be confirmed against the zip listing |
| `file_name` | Points to an `.xml` SPL document, NOT the image |
| `splimprint` / `pillbox_imprint` | Pill engraving text |
| `splshape_text` / `pillbox_shape_text` | Pill shape |
| `splcolor_text` / `pillbox_color_text` | Pill color |

**Resolved: the `spl*` vs `pillbox_*` duplicate column pattern.**
Confirmed via the official Pillbox Engine data dictionary
(github.com/HHS/pillbox_docs/wiki/Pillbox-Engine-data-dictionary):
- `spl*` = as originally submitted by the manufacturer to FDA
- `pillbox_*` = only populated when NLM had a photo AND found the true
  value differs from the label — i.e. a human-verified correction

**Normalization rule decided**: prefer `pillbox_*` when present (it's been
checked against the real photo), fall back to `spl*` otherwise. Not a
guess — directly justified by the data dictionary's description of the
correction process.

**Other findings from research (not yet acted on in code):**
- Multi-label duplication is expected and legitimate: the same physical
  pill can appear under dozens of NDCs/rows (example given in NLM's own
  docs: a "I2"-imprinted 200mg ibuprofen appears under ~80 different
  labels). Image-to-metadata-row is not 1:1 — don't assume it.
- The dataset has known, undocumented errors; no federal agency audits the
  physical-characteristics data. Worth a caveat in Phase 8 evaluation
  write-up: a "wrong" model prediction might reflect a labeling error, not
  a retrieval failure.

### Images

- Source: `https://ftp.nlm.nih.gov/projects/pillbox/pillbox_production_images_full_202008.zip`
  (NLM's official final image library, ~1GB, no login required)
- Downloaded, saved as `data/raw/pillbox_images.zip` (not yet extracted)
- Wrote `scripts/inspect_images_zip.py` to list the zip's contents via
  Python's `zipfile` module (reads the archive's table of contents only —
  doesn't decompress anything, safe/fast even on a 1GB file)

### Image↔metadata join investigation (completed — led to a pivot, see below)

Ran a sequence of scripts to establish how metadata rows link to real image
files in `pillbox_images.zip` (8,693 files total). Findings, in order:

1. **`inspect_images_zip.py`** — the zip mixes three filename patterns:
   - NDC-hyphen style: `00002-3228-30_391E1C80.jpg` (2,858 files)
   - Human-readable, no ID: `0.1_mg_Clonidine_HCl_Tablet.jpg`
   - Junk/placeholder: `no_product_image.jpg`, `tablets.jpg`, etc.

2. **`test_join_hypothesis.py`** — tested `product_code` (dashed NDC) as a
   direct filename prefix. 0/20 matched. `splimage` turned out to have two
   distinct formats: plain digits (`498840129`) vs an `_NLMIMAGE10_`-tagged
   form (`42794-0019-02_NLMIMAGE10_BC40DE26`).

3. **`verify_nlmimage_join.py`** — tested `splimage + ".jpg"` as an exact
   filename match, but only for the `_NLMIMAGE10_` rows.
   **Result: 866/866 exact matches (100%).** This is our one solid,
   confirmed join key.
   Plain-digit `splimage` rows: 8,913 of the 9,779 `has_image=True` rows —
   none matched this way.

4. **`test_ndc_prefix_join.py`** — tested whether the 8,913 plain-digit rows
   could match via `product_code` as a loose filename prefix (ignoring the
   unpredictable hex suffix). **Result: 4/8,913 (0.0%).** Effectively zero.

5. **`test_digit_normalized_join.py`** — tested whether NDC segment-length
   formatting (4-4-2 vs 5-4-2 vs 5-3-2 are all valid per FDA spec) was
   hiding matches, by stripping all dashes and comparing digit-only.
   Tested against ALL rows (not just has_image=True), and also checked
   whether any of the ~1,914 unmatched zip files belonged to rows
   incorrectly flagged `has_image=False`.
   **Result: 4 matches total (noise). 0 has_image=False mismatches.**
   Ruled out "it's just a formatting problem."

**Root-cause research (web search):**
- DailyMed's own site confirms the **RxImage API ceased operation Dec 31,
  2021**, with API-sourced pill images removed from DailyMed Oct 31, 2021.
  This explains the plain-digit `splimage` rows: they point to images that
  lived on a now-permanently-dead API, not in any static file we can get.
  This is a real, permanent data gap — not a bug in our logic.
- NDC segment-length formats legitimately vary (4-4-2, 5-3-2, 5-4-1 per
  FDA/drugs.com) — confirmed real, but not the cause of our mismatch here
  (ruled out in step 5 above).

**Conclusion at this point:** only ~866 of ~84,000 Pillbox metadata rows
have a confirmed, real, joinable image in our zip. The other ~1,914
NDC-style images in the zip have no matching row in this CSV at all
(likely a different data-export snapshot than the zip) — decided (per
user) to look for a better dataset rather than chase this further.

### Pivot: NLM PIR (dead domain) → ePillID dataset

**NLM PIR dead end**: `pir.nlm.nih.gov` returns `DNS_PROBE_FINISHED_NXDOMAIN`
— the whole subdomain is gone, not just one broken page. (Note: an earlier
`web_fetch` on this same URL appeared to succeed and was reported as
"confirmed live" — that was wrong; `web_fetch` can apparently return cached/
archived content without that being obvious, so a successful fetch is not
proof a site is currently reachable by a real browser. Lesson: verify
external URLs via the user's own browser before calling them confirmed.)

No working mirror or NLM-hosted replacement found via search. Pivoted to
searching for third-party re-hosts instead.

**Found: ePillID** (github.com/usuyama/ePillID-benchmark) — a CVPR 2020
workshop benchmark (Usuyama et al., Microsoft Research), built directly
from the same underlying NLM Pillbox/RxIMAGE source we'd already been
using. Actively maintained GitHub repo, MIT-ish license, real GitHub
Release (not a fragile external link):
- Release: `ePillID_data_v1.0`, tag date Sep 8 2020, asset
  `ePillID_data.zip`, **153 MB** (confirmed directly from the GitHub
  Releases page, not just search text)
- 13,532 images, 4,902 unique pill types (confirmed against our own
  download — matches the paper's reported class counts)
- Includes BOTH reference (studio) and consumer (real-world) images,
  with an explicit `is_ref` boolean column — no guessing required
- Released "for research purposes only" per the repo README — noted,
  fine for our stated learning/portfolio use

Downloaded, saved as `data/raw/epillid_data.zip`.

### ePillID structure investigation

1. **`inspect_epillid_zip.py`** — top-level: single folder `ePillID_data/`.
   Found `all_labels.csv` (our ground truth) plus 5-fold CSV splits under
   `folds/` (pre-made train/test splits for cross-validation - not needed
   for our use case, we're not training a classifier from scratch).

2. **`inspect_epillid_labels.py`** — read `all_labels.csv` directly out of
   the zip via `zf.open()` (no extraction needed). Columns: `images`,
   `pilltype_id`, `label_code_id`, `prod_code_id`, `is_ref`, `is_front`,
   `is_new`, `image_path`, `label`. 13,532 rows, `label`/`pilltype_id`
   both have 4,902 unique values, most pills have exactly 7 images.

3. **`full_epillid_structure.py`** — mapped the complete folder tree.
   Three image folders exist, not one:
   - `classification_data/segmented_nih_pills_224/` — 25,996 files
   - `classification_data/fcn_mix_weight/dc_224/` — 5,000 files (consumer)
   - `classification_data/fcn_mix_weight/dr_224/` — 2,001 files (reference)

4. **`verify_epillid_labels.py`** (first attempt) — tested whether
   `all_labels.csv`'s `image_path` column resolves to real files by
   prepending `"ePillID_data/"`. **Result: 0/13532 (0%).** A real signal,
   not noise — worth diagnosing rather than dismissing.

5. **`diagnose_path_mismatch.py`** — used `repr()` to compare a raw CSV
   `image_path` value against a real filename character-by-character.
   **Root cause found**: real files live under an extra `classification_data/`
   path segment we weren't including. Simple prefix bug, not a data problem.
   Fixed: correct prefix is `"ePillID_data/classification_data/"`.

6. **`verify_epillid_labels.py`** (re-run after fix) — **100% of image_paths
   now resolve (13532/13532).** One remaining oddity: not all `is_ref=True`
   rows point to `dr_224/` (only 2,000 of 9,804 do).

7. **`find_dr_exceptions.py`** — resolved that oddity. The other 7,804
   `is_ref=True` rows point to `segmented_nih_pills_224/` — the folder we'd
   earlier (wrongly) guessed might be an unrelated/superset dataset. It's
   not separate at all — `all_labels.csv` is one single, consistent ground
   truth covering all three image folders. Two things to remember going
   forward:
   - `label` is NOT a consistent format across the whole file: NDC-hex
     style (e.g. `51285-0092-87_BE305F72`) for `fcn_mix_weight/` rows, but
     a raw hash string (e.g. `b79b096bade8ddf...`) for
     `segmented_nih_pills_224/` rows. Safe to use as a grouping key (same
     value = same pill) either way, just can't assume it's human-readable.
   - Filenames in `segmented_nih_pills_224/` use a shorter NDC segment
     format (e.g. `0002-3270`) than seen elsewhere — consistent with the
     already-confirmed fact that NDC segment lengths legitimately vary.

### Phase 1: COMPLETE

Built `src/pillrag/data.py` - the real, reusable module consolidating
everything verified above:
- `load_epillid_labels()` - loads + filters all_labels.csv to the
  NDC-hex-style rows, resolves full image paths
- `load_pillbox_text_lookup()` - loads Pillbox metadata, normalized NDC
  index, deduplicated
- `build_pill_dataset()` - the main entry point, joins the two, returns
  one clean table (image path, pill id, is_ref, is_front, medicine_name,
  spl_strength, spl_ingredients)

Verified via `python -m pillrag.data`:
- **5,728 total rows** (matches all prior diagnostic scripts exactly)
- **5,544 rows with recovered text metadata (96.8%)** (matches
  corrected_ndc_join.py exactly)

Added `pandas>=2.0` to pyproject.toml as a real project dependency (was
already added locally when first inspecting the metadata CSV).

## Phase 2: Background Segmentation

### Visual inspection of real images (before any code)

Extracted and looked at 12 real sample images (6 reference + 6 consumer,
random seed 42 for reproducibility) via `scripts/extract_sample_images.py`
to check our assumptions before building anything.

**Finding: background style does NOT cleanly split along reference vs
consumer lines**, contrary to initial assumption. Observed across the
sample:
- Reference images: mix of flat gray, flat pale gray-green, solid black,
  and black-bar-top-bottom-with-gray-middle patterns
- Consumer images: mostly similarly clean (solid black, black bars), but
  2 of 6 showed genuine real-world surfaces (tan textured table, reddish
  woven fabric) - these are the only samples that looked like true
  "messy real-world" photos

**Conclusion**: most images are already fairly controlled/clean, but a
real minority have genuine varied backgrounds - segmentation is still
necessary (a naive "crop out black pixels" shortcut would fail on the
gray-background and real-surface images), and still valuable to learn
properly since real end-user query photos (the actual eventual use case)
will look like the messy consumer examples, not the clean references.

### Tooling decision: Ultralytics FastSAM

Chose **Ultralytics's FastSAM integration** over the original research
repo. Reasoning: official, well-documented, actively maintained API
(`from ultralytics import FastSAM`), consistent with the broader
Ultralytics/YOLO ecosystem (transferable skill), vs. the original
FastSAM repo which is closer to unmaintained research code with rougher
setup.

Confirmed via docs.ultralytics.com/models/fast-sam (checked live, not
from training memory - see working-instructions rule #3):
- API: `model = FastSAM("FastSAM-x.pt")`, then `model(image_path)` for
  "segment everything" mode, or `model.predict(path, bboxes=...)` for
  prompted segmentation
- FastSAM decouples into 2 stages: all-instance segmentation (via
  YOLOv8-seg backbone) + prompt-guided selection - for our simple
  one-pill-per-image case, "everything" mode + picking the largest mask
  should suffice, no bbox/point prompts needed
- **Known gotcha** (from an Ultralytics GitHub discussion): image
  dimensions not a multiple of 32 can cause a rare runtime error. Our
  ePillID images are 224x224, and 224/32=7 exactly - safe for THIS
  dataset, but worth remembering if processing differently-sized images
  later (e.g. a real user's uploaded query photo won't be a clean 224x224)

**Weight size decision: FastSAM-x** (larger/more accurate) over
FastSAM-s (smaller/faster). Reasoning: our task doesn't need real-time
speed (one-time batch job, not live inference), correctness matters more
here, and switching to -s later is a one-line change if -x proves too
slow on Colab's free-tier GPU.

### Testing across pill types - confidence threshold pattern found

Tested FastSAM on 2 more real samples for contrast against the two-tone
capsule case:

**Single-color round tablet** (`ref` sample, `63304-0579-01...`):
- **1 object detected**, confidence 0.8796, area 620,460 px
- Visualized: clean, correctly-bounded circular mask, no artifacts
- Supports the theory that the capsule's 4-object split was caused by
  its two-tone coloring specifically, not a general FastSAM problem

**Consumer photo on messy fabric background** (`dc_224/4274.jpg`,
`00555-9016-58...`):
- **2 objects detected**
- Mask 0: clean, correct circular pill outline, confidence **0.9630**
- Mask 1: nearly the ENTIRE image (1,016,000 / 1,048,576 px) - FastSAM
  treating the whole frame as one catch-all region - confidence only
  **0.4816**
- Notably, postprocessing time dropped to 32.6ms here vs. >1000ms in
  both earlier tests - postprocessing time appears to scale with
  internal mask complexity, not a fixed per-image cost (worth
  remembering when estimating full batch runtime)

**Pattern confirmed across all 3 real tests so far**: the genuine pill
mask is consistently HIGH confidence (0.88-0.96); artifact/catch-all
masks (image-wide blob, horizontal band) are consistently LOW confidence
(0.29-0.48). The one complication: the two-tone capsule produces TWO
separate high-confidence masks (both need to be kept and merged), not
just one.

**Proposed selection rule** (TESTED, FAILED - see below):
1. Filter out any mask below a confidence threshold (candidate: ~0.6-0.7,
   comfortably separates real detections from artifacts in all 3 tests)
2. Merge all remaining masks together (union) into one final pill mask -
   handles both the "one clean mask" case and the "split into
   same-object parts" case without needing to distinguish them

### Confidence-only rule tested and FAILED - band artifact has mid-range confidence too

Implemented `get_pill_mask()` (confidence threshold 0.6 + union merge) and
tested against the capsule case specifically (our hardest known case,
with the band artifact). **Result: failed.** The band artifact (Mask 1,
confidence 0.8969) survived the 0.6 threshold right alongside the two
genuine capsule halves (0.9352, 0.8963) - only the truly-spurious
duplicate mask (0.2931) got correctly filtered. Since we merge via union
(`.max(axis=0)`), the band swallowed the whole result into one giant
rectangle again - same failure as our very first "pick largest mask"
attempt, just reached a different way.

**Real lesson**: confidence score alone cannot distinguish "genuine
object part" from "band/frame artifact" - both can score similarly high.
Need an additional, shape-based signal specifically to catch the band
artifact. Hypothesis to test next: the band artifact spans the full
image width edge-to-edge with a straight border, which real pill shapes
(even irregular ones) shouldn't do - candidate rule: reject any mask
that touches both the left AND right edges of the image across a wide
vertical span.

### Area-sum rule (user's idea) tested directly in notebook - worked, but with a caveat

User proposed: among confident masks, if two have similar area and a
third is close to DOUBLE that, the two are likely parts of one whole
object. Rejected the earlier edge-touching/aspect-ratio ideas first on
principle (would misfire on a real up-close user photo - framing-
dependent signals are fragile). Tested area-sum idea directly against
real numbers in the notebook: correctly found mask 1 (334,289 px) ≈
mask 0 + mask 2 (139,640 + 147,517 = 287,157, ~16% difference, within a
30% tolerance) → correctly identified 0+2 as parts, 1 as whole.

**However**: this success came from a naive first-draft loop that
returns on the FIRST valid pairing found, without checking whether OTHER
pairings might also satisfy the same tolerance (a real weakness, plus a
numpy uint64 overflow bug on a failed comparison earlier in the same
loop - both noted for the "harden into real module" pass).

### MISTAKE: shipped "hardened" segment.py without re-testing - caught by user

Wrote `src/pillrag/segment.py`'s `select_pill_mask()` to fix the two
known bugs above (overflow, first-match-wins) and presented it as done.
**Did not re-verify it against the capsule case before doing so** - a
real lapse, caught by the user, not by me. Tested in Colab: it
reproduced the EXACT same failure as the ORIGINAL naive rule, selecting
Mask 1 (the band artifact) as "whole" with masks 0+2 as its "parts" -
identical wrong answer, differently arrived at.

**Root cause**: fixing the overflow and first-match-wins bugs never
addressed the actual underlying flaw - the area-sum arithmetic itself is
ambiguous. Band area (334,289) vs the real halves' sum (287,157) is
~16% apart, comfortably inside the 30% tolerance - these exact numbers
were already sitting in this devlog from the original notebook test,
and should have been checked against the new code before calling it
fixed.

**Real fix needed, not yet implemented**: area-sum matching alone is
insufficient. Need a genuine geometric relationship check - do the
proposed "parts" masks actually spatially touch/overlap each other, the
way two halves of one physical object necessarily would? A coincidental
band artifact has no such spatial relationship to the real capsule
halves, even though its area happens to arithmetically fit.

### Geometric touching/overlap idea tested - ALSO failed

Tested directly in notebook: does the band artifact (mask 1) actually
touch/overlap the real capsule halves (masks 0, 2), the way two
unrelated unrelated regions shouldn't, but the way something spatially
connected to them would?

**Result: yes, heavily** - mask 1 overlaps mask 0 by 139,640 px (= ALL of
mask 0's area) and mask 2 by 147,517 px (= ALL of mask 2's area). This
makes sense once understood: the band artifact isn't a random unrelated
region elsewhere in the image - it's caused by the image's own light/
dark horizontal composition, which is centered exactly where the real
pill sits. So it necessarily overlaps the real pill masks heavily; this
signal cannot separate "artifact caused by the pill's own location" from
"genuine part of the pill." Ruled out.

**Next hypothesis, not yet tested**: look at the mask's own shape
REGULARITY rather than its size or position - a rectangle-like band
should have very straight edges and corners; a pill (even an elongated
capsule) should have smoothly curved ends. Candidate approach: compare
each mask's actual pixel area against the area of its own convex hull
(the tightest convex/bulging shape containing all its pixels) or its own
bounding-box area - a near-perfect rectangle should have very little
"wasted space" in its bounding box, while a pill shape (rounded ends)
should have visibly more.

### Bounding-box fill ratio - tested, WORKS across all 3 known cases

Defined `bounding_box_fill_ratio(mask)` = mask's own pixel area / its own
bounding box area. Rectangle-like artifacts should be close to 1.0;
rounded pill shapes should be meaningfully lower. Tested against all
three real cases we have evidence for, not just the case it was
designed around:

| Case | Real pill mask(s) | Artifact mask(s) |
|---|---|---|
| Capsule (band artifact) | Mask 0: 0.9275, Mask 2: 0.9105 (halves) | Mask 1 (band): **0.9950**, Mask 3 (noise): 0.4569 |
| Round tablet (clean, single mask) | Mask 0: **0.8079** (close to circle's theoretical π/4 ≈ 0.785) | none present |
| Consumer photo (whole-image blob artifact) | Mask 0: **0.7900** | Mask 1 (blob): **1.0000** |

**Clean separation confirmed across all 3 cases**: genuine pill shapes
consistently fall in ~0.79-0.93; band/blob artifacts consistently sit at
~0.995-1.0. A threshold around 0.97 should reliably separate them.
Interesting note: Mask 3 (the low-confidence noise fragment in the
capsule case) has a LOW fill ratio (0.4569) too - suggesting fill ratio
might independently help catch some noise even without confidence
filtering, though confidence filtering should still run first as the
primary noise filter.

**Decision: rebuild `select_pill_mask()` using confidence filtering +
fill-ratio filtering (reject near-rectangular masks) BEFORE the
whole/parts area-sum matching runs.** This should let the whole/parts
logic run only against genuinely pill-shaped candidates, avoiding the
mistake from the previous attempt where the band artifact was allowed
into the whole/parts comparison at all.

### Fill-ratio filter tested against all 3 cases together - partial success, new gap found

Implemented the confidence+fill-ratio filter and tested against all 3
real cases in one pass, BEFORE touching the real module file further
(applying the lesson from the earlier mistake).

**Round tablet: correct** - single candidate survives (Mask 0), used
directly.
**Consumer photo: correct** - band/blob artifact (Mask 1) correctly
excluded by fill ratio, single genuine candidate (Mask 0) survives, used
directly.
**Capsule: still wrong, but differently wrong than before** - fill ratio
correctly excluded the band artifact (Mask 1) from candidates this time.
But this left only masks 0 and 2 (the two real halves) as candidates -
no third "whole capsule" mask remains for `_find_whole_and_parts` to
match against, since the only mask that had previously looked like "the
whole" WAS the band artifact we just (correctly) excluded. Result:
fell through to `highest_confidence_fallback`, returning ONLY mask 0
(just the left half of the capsule) - not fixed, just failing in a new
way.

**Real insight**: the assumption that FastSAM always provides a
separate, correctly-shaped "whole object" mask alongside its "parts" was
wrong for this image - here, the only mask resembling "the whole" WAS
the coincidental band artifact. The genuinely correct fix: when multiple
valid (post-filter) candidate masks remain and none of them individually
represents the complete pill, MERGE them together directly (union) as
the final mask, rather than searching for a pre-existing whole mask that
may not exist. Not yet implemented/tested - see next entry.

### Merge-fallback tested - ALL 3 cases now correct

Added: when multiple valid (post-filter) candidates remain and no
whole/parts relationship is found among them, merge (union) all of them
together as the final mask, rather than falling back to a single
highest-confidence one (which would have thrown away real pill area,
as seen in the previous entry's capsule failure).

Rewrote `select_pill_mask()` in `src/pillrag/segment.py` with this
change, AND - applying the lesson from the earlier mistake directly -
tested the exact updated function against all 3 known cases together,
in the same Colab session, before touching the real module file again
or considering it done:

| Case | Method used | Result |
|---|---|---|
| Capsule | `merged_candidates`, indices (0, 2) | Correct - complete capsule shape, band excluded |
| Round tablet | `single`, index (0,) | Correct - only real mask, no artifacts present |
| Consumer (fabric bg) | `single`, index (0,) | Correct - blob artifact excluded |

All three visually confirmed as clean, correctly-shaped, complete pill
silhouettes with backgrounds properly excluded.

**Also cleaned up a real code-quality issue found while making this
fix**: the previous version's `highest_confidence_fallback` branch
became unreachable dead code once the length==1 case was already
handled earlier in the function (by the time that branch could run,
candidate_indices always has length > 1). Removed it rather than leave
unreachable code in place; `MaskSelectionResult.method` docstring
updated to list only the 4 actually-reachable values: "single",
"whole_and_parts", "merged_candidates", "none_valid".

**Current confidence level**: this logic is verified against 3 distinct
real cases (two-tone capsule + band artifact, clean single-color round
tablet, consumer photo + whole-image blob artifact). Not yet tested
against: irregular/scored tablets, oval tablets, multiple genuinely
overlapping objects in one frame, or a broader random sample across the
full dataset. Should run against a wider sample before batch-processing
all 5,728 images, to catch any case this logic doesn't yet handle.

### Colab environment setup (done)

- New Colab notebook, Runtime → Change runtime type → **T4 GPU** (free
  tier - confirmed via `!nvidia-smi`: Tesla T4, 15,360 MiB, 0% used)
- Uploaded `epillid_data.zip` via `google.colab.files.upload()` (153MB,
  direct browser upload - fine for a single-session experiment; will
  reconsider Google Drive mounting if we need persistence across
  multiple Colab sessions later)
- `!unzip -q epillid_data.zip -d data` - confirmed same top-level
  structure as found locally (`ePillID_data/`)
- `!pip install ultralytics` - clean install (ultralytics 8.4.87). Colab
  already had PyTorch pre-installed with CUDA 12.8 support built in - no
  manual GPU/driver setup needed, unlike a fresh local machine
- `FastSAM("FastSAM-x.pt")` - auto-downloaded weights (138MB) from
  Ultralytics' GitHub releases on first use, ~3 seconds

### First real FastSAM test - important finding: naive mask selection fails

Ran FastSAM on one known reference image (the blue/white capsule sample
we'd already visually inspected locally, back in the very first sample
extraction). Result: **4 objects detected**, not 1, as expected from a
single-pill photo:

| Mask | Area (px) | Confidence | What it actually is (confirmed via isolated visualization) |
|---|---|---|---|
| 0 | 139,640 | 0.9352 | Left half of the capsule (rounded-left, correct shape) |
| 1 | 334,289 | 0.8969 | **NOT the pill** - a full-width horizontal band artifact, likely from the black-bars-top-and-bottom image composition (see Phase 2 visual inspection notes above) |
| 2 | 147,517 | 0.8963 | Right half of the capsule (rounded-right, correct shape, mirrors Mask 0) |
| 3 | 23,058 | 0.2931 | Low-confidence, likely a redundant/lower-quality duplicate of Mask 0's region |

**Root cause of the split**: the capsule's two halves are different, high-
contrast colors (blue vs white). FastSAM appears to be segmenting each
color region as its own object rather than recognizing "one capsule" as
a whole - understandable, since FastSAM has no pill-specific training,
it's a general-purpose segmenter.

**Initial hypothesis rejected with evidence, not assumed**: "just pick
the largest mask by area" seemed like a clean rule (tested via
`.sum(dim=(1,2)).argmax()`), but visualizing that specific mask in
isolation revealed it was the band artifact (Mask 1), not the pill -
it's large simply because it's a big rectangle, not because it's the
correct/complete object. This disproves area-alone as a selection
criterion.

**Correct understanding**: for this image, the real pill = Mask 0 +
Mask 2 combined (the two genuine half-capsule shapes). Mask 1 (band) and
Mask 3 (low-confidence duplicate) are both noise to reject.

**Not yet solved**: a general, reliable rule for picking/combining the
correct mask(s) across all 5,728 images, most of which won't be
two-colored capsules (many are round, single-colored tablets - simpler
shapes that may not split this way at all). Need to test against a
wider variety of sample images before deciding a general rule such as
"reject any mask touching all 4 image edges" + "reject low confidence" +
"merge remaining masks."

### Colab persistence problems (encountered and resolved)

While retrying the FastSAM mask experiments on more sample images, hit a
real, informative sequence of Colab-specific issues:

1. **Session reload wiped everything** - uploaded files (`epillid_data.zip`,
   later `pillbox_metadata.csv`) disappeared after a routine reload.
   Confirmed this is expected Colab behavior: session disk is fully
   ephemeral, separate from Google Drive.

2. **Decision: mount Google Drive** for permanent file storage across
   sessions, rather than re-uploading via `files.upload()` every time.
   Files uploaded once to `MyDrive/pill-rag/data/raw/` (both
   `epillid_data.zip` and `pillbox_metadata.csv`), confirmed present via
   `!ls -la` (matches expected sizes: ~153MB, ~56MB).

3. **Real bug this exposed in our own code**: `pillrag/data.py` had
   hardcoded paths (`data/raw/epillid_data.zip` etc.), which broke the
   moment we tried pointing at a different location (Drive's mounted
   path). **Fixed properly** rather than patched around: paths are now
   read from environment variables (`EPILLID_ZIP_PATH`,
   `PILLBOX_METADATA_CSV_PATH` - added to `.env.example`), defaulting to
   the original local project layout if unset. Locally, nothing changes
   (no env vars set = same default behavior as before). In Colab, set
   both env vars to the Drive-mounted path BEFORE importing
   `pillrag.data` (env vars are read once, at import time - order
   matters, confirmed by getting this wrong once mid-session).
   Committed and pushed to GitHub so Colab's `pip install git+...` picks
   up the fix.

4. **Session RESTART (different from reload) wiped installed packages**
   too, not just uploaded files/variables - had to `!pip install
   ultralytics` and reinstall `pillrag` from GitHub again after
   restarting to clear a `--force-reinstall` package-conflict warning.
   Google Drive mount survived this restart without needing
   re-authorization.

**Practical lesson for future Colab sessions**: after ANY session
reset (reload OR restart), expect to re-run, in order: (1) mount Drive,
(2) reinstall packages (`ultralytics`, `pillrag` from GitHub), (3) set
the `EPILLID_ZIP_PATH`/`PILLBOX_METADATA_CSV_PATH` env vars, (4) then
import/use `pillrag`. Uploaded files in Drive and the repo on GitHub are
the only things that reliably persist.

**Verified fix works end-to-end**: `python -m pillrag.data`-equivalent
call in Colab now returns the same numbers as locally - 5728 total rows,
5544 with recovered text metadata.

### Wide random sample (15 images) tested against the verified selection logic

Tested the pushed, verified `select_pill_mask()` against a genuinely
random sample of 15 images (seed 123, different from the earlier
hand-picked test cases) to check for failure modes beyond our 3 known
cases before committing to a full batch run.

**Result: 9 "single", 3 "merged_candidates", 3 "none_valid".** The 3
`none_valid` cases were investigated individually (not assumed) -
visualized each one's original photo + all raw masks.

**Finding: all 3 failures are white/off-white pills on a white/light-
gray background** - a real, known, actively-researched computer vision
limitation (confirmed via web search - multiple papers describe this
exact "low visual contrast between foreground and background" problem
as a documented weakness of the SAM/FastSAM model family in general, not
specific to our setup or fixable by threshold-tuning). One 2026 paper
found via search is actively researching exactly this gap.

**Classical CV fallback investigated as a second-attempt technique**
(since FastSAM structurally cannot help here, regardless of our
post-processing logic):
- Tried Otsu's thresholding first - found a clear, confident boundary,
  but it traced the pill's DROP SHADOW, not the pill itself (visually
  confirmed - the detected region was a crescent shape matching the
  shadow under the pill's curved edge, not the pill's circular outline)
- Tried Canny edge detection instead - successfully traced most of the
  pill's actual rim, even though contrast is low (edge detection
  responds to rate-of-change, not absolute brightness difference,
  which suits subtle shading transitions better than thresholding does)
- Tried Hough Circle Transform (`cv2.HoughCircles`) on the same image -
  **successfully found the pill's true circular boundary**, visually
  confirmed near-perfect against the real pill edge, correctly
  excluding the shadow. This works even from a partial/noisy edge
  trace, since Hough Circles is specifically designed to find circles
  from incomplete boundary evidence via a voting mechanism.

**Decision**: use Hough Circle detection as a fallback specifically for
ROUND pills when FastSAM's primary logic returns `none_valid`. Oval/
capsule shapes will need a different classical technique (e.g. ellipse
fitting) - not yet tested.

### IMPORTANT SCOPE NOTE: shape-based routing only applies to offline indexing, not live queries

User raised a sharp, important point worth recording clearly, since
it's easy to forget later: using Pillbox's known `shape` field
(round/oval/capsule) to pick a classical-CV fallback technique is
legitimate ONLY while building our reference index (Phase 2/3 batch
processing), where we already know each pill's identity and shape from
metadata beforehand.

**This does NOT apply to real end-user query photos** (the actual
product use case, Phase 6/7+) - a user photographing an unknown pill
means we don't know its shape either, since determining the pill's
identity (which implies its shape) IS the point of the query. Using
shape to help segment a query photo would be circular reasoning.

**Decision for later (Phase 6/7, not now)**: for live query photos,
either (a) try multiple shape-agnostic classical fallbacks and pick
whichever produces the most confident/sensible result, or (b) simply
ask the user what shape their pill is via a UI prompt - a common,
reasonable pattern in real pill-identifier apps, and useful for
narrowing RAG retrieval candidates too, not just segmentation. Not
decided yet, revisit when we reach the API/frontend phases.

### `pillrag.data` extended with shape field (for offline routing only)

Added `shape` column to `build_pill_dataset()`'s output, using the same
prefer-`pillbox_*`-fallback-to-`spl*` rule established in Phase 1 for
the shape-text column pair (`pillbox_shape_text` populated only when
NLM visually verified against a real photo; `splshape_text` is the
manufacturer-submitted original). Implemented via pandas' `.fillna()` -
`df["pillbox_shape_text"].fillna(df["splshape_text"])`.

### Open / next steps

- [ ] Check the real shape distribution across our 1,000 pill types
      (round vs oval vs capsule vs other) to know how much of the
      low-contrast problem Hough Circles alone can address
- [ ] Investigate an ellipse-fitting classical technique for oval/
      capsule-shaped low-contrast pills (Hough Circles only handles
      round shapes)
- [ ] Wire the Hough Circle (and eventual ellipse) fallback into
      `segment.py` properly, triggered when `select_pill_mask()` returns
      `none_valid` AND the pill's known shape (offline indexing only)
      indicates which classical technique to try
- [ ] Re-test the full pipeline (FastSAM primary + classical fallback)
      against the same 3 `none_valid` wide-sample cases, plus a fresh
      wider sample, before batch-processing all 5,728 images
- [ ] Revisit MEDISEG dataset for validating segmentation quality against
      real ground-truth masks (deferred from Phase 1)

### Decision: text metadata source for ePillID pills

**Problem**: ePillID gives us images + pill identity (NDC-based IDs), but
no drug name, ingredients, or description text — needed for Phase 5 text
RAG.

**Decision**: cross-reference ePillID's NDCs back against
`data/raw/pillbox_metadata.csv` (already downloaded, Phase 1 step 0) to
recover `medicine_name`, `spl_ingredients`, `spl_strength`, etc. This
redeems the earlier Pillbox metadata work — we're no longer using Pillbox
for *images* (that's ePillID's job now), but it's still the right source
for *text* metadata, joined by NDC instead of by image filename.

**Not yet done**: need to extract the NDC portion out of ePillID's
`pilltype_id`/`label` values (format varies - NDC-hex for most rows, raw
hash for `segmented_nih_pills_224` rows, per the investigation above) and
match it against `product_code` in the Pillbox CSV. Given NDC segment-
length formatting issues already found earlier (4-4-2 vs 5-4-2 etc.),
this join needs the same digits-only normalization approach we used
before, not a direct string match.

### NDC join investigation (ePillID -> Pillbox text metadata)

Same investigate-before-assuming discipline as the image join earlier.
Sequence of scripts and findings:

1. **`test_epillid_pillbox_text_join.py`** — first attempt, digits-only
   normalization (same approach that worked for the image join).
   **Result: 0/5728 (0%).** ePillID's `label` column is NDC-hex style for
   5,728 rows (contains `-`), raw hash for the other 7,804 (no NDC at
   all, can't be joined this way).

2. **`diagnose_ndc_join.py`** — printed raw digit strings + length
   distributions side by side. **Root cause #1 found**: ePillID NDCs are
   consistently 11 digits (full labeler-product-package format); Pillbox's
   `product_code` is only 8 or 9 digits (labeler-product only, package
   segment absent). Direct comparison could never match - different
   segment counts entirely, not just different digit-string values.

3. **`test_ndc_truncated_join.py`** — truncated ePillID's NDC to 8/9
   digits to drop the package segment. **Result: only 3.2% (186/5728) at
   best.** Better than 0%, but far too low to be the full explanation.

4. **`check_labeler_coverage.py`** — isolated whether this was a coverage
   gap (Pillbox CSV genuinely lacks these labelers) vs a join-logic bug,
   by checking labeler-code-only overlap (the broadest possible grouping).
   **Result: 59.6% string-overlap** - real signal that most labelers ARE
   present, meaning the join logic itself was still the problem, not
   dataset coverage.

5. **`test_leading_zero_theory.py`** — tested comparing labeler codes as
   Python integers instead of strings (`int("00062") == int("62")`).
   **Root cause #2 found and confirmed: overlap jumped from 59.6% to
   98.2%.** Labeler codes are stored inconsistently zero-padded across
   the two datasets.

6. **`corrected_ndc_join.py`** — combined both fixes: split NDC into
   labeler + product segments, `int()`-convert EACH segment separately
   (correct per-segment zero-stripping), rejoin with a `.` separator
   (critical - concatenating stripped segments directly without a
   separator can produce false collisions, e.g. labeler="6"+product="25"
   and labeler="62"+product="5" would both become "625" as plain digits).
   Crashed on first run: `ValueError` on `product_code` value `'0019-N601'`
   - a non-numeric product segment.

7. **`inspect_bad_product_codes.py`** — checked how common this was before
   deciding how to handle it (proportionate effort principle). **Result:
   3 / 83,925 rows (0.00%)** - all `N`-prefixed, likely a distinct FDA
   coding convention for a specific product type. Rare enough to just
   skip safely rather than build special-case handling.

8. **`corrected_ndc_join.py`** (patched) — `normalize_ndc` now returns
   `None` for unparseable values instead of crashing; both dataframes
   filter out `None` results with `.notna()` BEFORE the join (important:
   without this, multiple `None` keys would falsely "match" each other).
   **Final result: 5544 / 5728 matched (96.8%).** Sample matches
   manually verified sane (real drug names, plausible NDC pairings).

### Investigation: recovering the hash-labeled 3,902 pills (unresolved, deferred)

Tried to recover text metadata for the 7,804 raw-hash-labeled rows
(3,902 unique pill types, confirmed via `check_pilltype_overlap.py` to be
ENTIRELY disjoint from the 1,000 NDC-labeled pill types - zero overlap,
not redundant).

- Confirmed via the ePillID paper itself (arxiv 2005.14288): these come
  from the original NIH Pillbox dataset directly, pre-segmented and
  hash-labeled by the authors (likely for anonymization) - NOT from the
  NIH PIR challenge like the 1,000 NDC-labeled pills are.
- Searched the ePillID GitHub repo for the actual hash construction
  (which field, which algorithm) - not documented anywhere in the README,
  paper, or visible source files. Genuine dead end for reverse-engineering
  the hash directly.
- Considered (but did not pursue, per user - low expected payoff for
  effort): (a) checking if our own confirmed 866 Pillbox images happen to
  overlap by NDC with this group - would only catch a lucky few, doesn't
  solve the general problem; (b) content-based image matching (comparing
  actual pixels between ePillID's segmented output and our raw Pillbox
  photos) - technically possible but is really Phase 3-level work
  (embedding + similarity search) being pulled forward just to answer a
  Phase 1 metadata question - disproportionate.

**Decision: deferred/excluded.** These 3,902 pill types stay out of scope
for now. Not a dead loss for the project overall - see MEDISEG below,
which ended up being a better fit for a different part of the plan
entirely.

### Detour: searched for a linked raw+metadata alternative (MEDISEG)

Per user request, searched broadly for other pill datasets with
unsegmented raw images and linked metadata, to see if a better single
source existed. Found **MEDISEG** (arXiv 2603.10825, published Mar 2026 -
genuinely current work, hosted on Figshare, CC BY 4.0 license):
- 8,262 images, 32 distinct pill types, TWO subsets ("3-Pills" controlled,
  "32-Pills" realistic/cluttered)
- Real multi-pill scenes: individual pills through cluttered dosette boxes
  - genuinely raw/unsegmented, with occlusion and clutter
- Ships with COCO-format instance segmentation masks (real ground truth,
  not pre-segmented output) - this is different from anything we have:
  a way to check our OWN FastSAM segmentation against verified correct
  masks
- `metadata.csv` links to real drug info (brand name, strength,
  ingredients, manufacturer) via Hong Kong drug registration numbers
  (HK-xxxxx) - a different regulatory system than US NDC, so this data
  CANNOT be cross-referenced against our Pillbox/ePillID data at all;
  it's a fully separate, disjoint collection (confirmed with user - not
  "more of the same" pills, a different set entirely)

**Decision: MEDISEG is not an addition to our main pill index** (disjoint
pill identities, no way to merge). **Deferred for later**: revisit in
Phase 2 specifically, since it's the only dataset we have with genuine
raw multi-pill photos + real segmentation ground truth to validate our
own segmentation pipeline against - not needed for Phase 1.

### FINAL Phase 1 dataset decision

Going forward, the project's real data foundation is:
- **Images**: ePillID's `fcn_mix_weight/dr_224/` (2,001 reference) +
  `dc_224/` (5,000 consumer) - the NDC-labeled subset only, NOT
  `segmented_nih_pills_224/`
- **Identity/ground truth**: `all_labels.csv`, filtered to the NDC-hex-
  style `label` rows (5,728 of 13,532 total rows) - covers 1,000 unique
  pill types
- **Text metadata**: recovered via the corrected NDC join against
  `data/raw/pillbox_metadata.csv` - 96.8% coverage (5,544 / 5,728 rows)
  confirmed working
- **Out of scope for now**: the other 3,902 ePillID pill types (no
  recoverable text metadata) and MEDISEG (disjoint identity system,
  earmarked for Phase 2 segmentation validation instead)

**Also decided: the deferred 3,902 pills stay out of the VISUAL search
index too, not just text RAG.** Reasoning: those images come
pre-segmented by ePillID's own (undocumented) pipeline, while our 1,000
NDC-labeled pills will go through our own FastSAM segmentation in Phase
2. Mixing images with two different preprocessing histories in one
embedding index risks subtly biasing similarity search toward
"how the image was processed" rather than purely "what pill it shows" -
a hard-to-detect quality issue, not a hard crash. Since we lack raw
(unsegmented) source images for these 3,902 anyway (their originals are
unrecoverable from pillbox_images.zip, see NDC join investigation above),
we can't reprocess them consistently even if we wanted to. Fully deferred,
not just for text.