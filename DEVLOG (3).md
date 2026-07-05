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

### Open / next steps

- [ ] Decide: use LLM (Gemini) to enrich the recovered but terse Pillbox
      text fields (medicine_name, spl_strength, spl_ingredients) for
      better Phase 5 text RAG quality? (still open from original plan)
- [ ] Revisit MEDISEG in Phase 2, specifically for validating our own
      FastSAM segmentation output against its real ground-truth masks
- [ ] Begin Phase 2: FastSAM background segmentation

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




