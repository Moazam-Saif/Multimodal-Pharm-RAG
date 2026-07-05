# scripts/ index

Every script here was written to answer ONE specific question, with real
evidence, before writing permanent code. This is the map from question to
script to answer. Full narrative/reasoning for all of these lives in
`DEVLOG.md` - this file is just the fast lookup table.

Scripts are one-off investigations - safe to leave as-is, not meant to be
imported or reused. Permanent, reusable logic lives in `src/pillrag/`.

## Phase 1: Data investigation

| Script | Question | Answer |
|---|---|---|
| `inspect_metadata.py` | What columns does the Pillbox metadata CSV actually have? | 55 columns; found `has_image`, `splimage`, `spl*`/`pillbox_*` pairs |
| `inspect_images_zip.py` | What's actually inside the Pillbox image zip? | 3 mixed filename patterns; 8,693 files total |
| `test_join_hypothesis.py` | Does `product_code` match image filenames directly? | No (0/20) - `splimage` has 2 sub-formats instead |
| `verify_nlmimage_join.py` | Does the `_NLMIMAGE10_`-style `splimage` match exactly? | Yes, 100% (866/866) - our one solid Pillbox image join |
| `test_ndc_prefix_join.py` | Can the other 8,913 plain-digit rows match via NDC prefix? | No, ~0% (4/8913) |
| `test_digit_normalized_join.py` | Is that a digit-formatting issue? | No - ruled out, genuinely missing (dead RxImage API) |
| `inspect_epillid_zip.py` | What's inside the ePillID zip (the pivot dataset)? | 1 top folder, 3 image subfolders, `all_labels.csv` found |
| `inspect_epillid_labels.py` | What does `all_labels.csv` actually contain? | 13,532 rows, 4,902 unique pill types, `is_ref`/`is_front` flags |
| `full_epillid_structure.py` | What's the full folder tree + file counts? | 3 image folders: `segmented_nih_pills_224` (25,996), `dc_224` (5,000), `dr_224` (2,001) |
| `verify_epillid_labels.py` | Do `all_labels.csv` image paths resolve to real files? | First try: 0% (wrong path prefix). After fix: 100% |
| `diagnose_path_mismatch.py` | Why did the path resolution fail? | Missing `classification_data/` path segment - simple prefix bug |
| `find_dr_exceptions.py` | Why don't all `is_ref=True` rows point to `dr_224/`? | They correctly point to `segmented_nih_pills_224/` instead - a second, legitimate reference-image source |
| `test_epillid_pillbox_text_join.py` | Can we recover drug names for ePillID pills via Pillbox? | First try: 0% - NDC digit-count mismatch |
| `diagnose_ndc_join.py` | Why 0%? | ePillID NDCs are 11 digits; Pillbox's are 8-9 (missing package segment) |
| `test_ndc_truncated_join.py` | Does truncating to 8/9 digits fix it? | Only partially - 3.2% - still too low |
| `check_labeler_coverage.py` | Is this a coverage gap or a logic bug? | Logic bug - 59.6% of labelers ARE present as strings |
| `test_leading_zero_theory.py` | Are inconsistent leading zeros hiding matches? | Yes - confirmed, 98.2% overlap once compared as integers |
| `corrected_ndc_join.py` | Full fix: does per-segment int-normalization work? | Yes - 96.8% (5544/5728) recovered, crashed once on non-numeric NDC, fixed |
| `inspect_bad_product_codes.py` | How common are non-numeric NDC segments? | Rare - 3/83,925 rows (0.00%) - safe to skip |
| `examine_hash_rows.py` | What are the 7,804 "raw hash" ePillID rows? | Pre-segmented images from a separate source (original Pillbox), hash-labeled |
| `check_pilltype_overlap.py` | Do the hash-labeled pills overlap with our 1,000 NDC pills? | No - 3,902 entirely distinct pill types, zero overlap |
| `test_hash_hypothesis.py` | Can we reverse the hash to recover their identity? | Not attempted after further research showed the hash construction is undocumented (see DEVLOG) |

## Not yet built

Phase 2+ scripts/notebooks will be indexed here as they're created.
