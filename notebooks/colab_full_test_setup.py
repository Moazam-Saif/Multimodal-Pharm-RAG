"""
Phase 2: Full Colab test environment setup - TWO CELLS, run in order,
to get back to a fully working state, including every known test case
from our FastSAM mask-selection investigation, ready to use
immediately.

WHY TWO CELLS: installing a new package version and then continuing to
use it in the SAME process doesn't work reliably in Colab - the
already-running Python process keeps whatever was loaded in memory
before the install (see DEVLOG.md's confirmed root-cause finding: an
official Colab GitHub issue explains Colab pre-loads numpy via
matplotlib before our own install can matter). A restart is required
between installing and using. There is NO official, documented
google.colab API for a programmatic restart (a feature request for one,
googlecolab/colabtools#5204, remains open/unresolved as of this
writing - an earlier version of this file wrongly assumed it existed
and shipped, causing an AttributeError; see DEVLOG.md). This uses the
actually-working approach instead: killing the kernel process directly
via `os.kill(os.getpid(), 9)`, which Colab automatically reconnects
after. Either way, the restart unavoidably ends execution of the
current cell, so cell 2 must be run separately, after reconnection.

USAGE:
  1. Paste CELL 1 below, run it. It installs everything and restarts
     the runtime automatically - you'll see the runtime disconnect and
     reconnect on its own, no dialog click needed.
  2. Once reconnected, paste CELL 2, run it. This builds the model,
     dataset, and all 9 known test cases.

After CELL 2 completes, you'll have:
  - model            - loaded FastSAM instance
  - df                - the full pill dataset (5728 rows)
  - ZIP_PATH          - path to the ePillID data zip
  - TEST_CASES        - dict of {name: (confidences, masks, image_path)}
                        for all 9 known test cases, already run
                        through the model

Usage after setup:
    result = resolve_pill_mask(*TEST_CASES["capsule_20_RED"][:2])
    print(result.method, result.final_mask.sum())

NOTE: if you're on a SECOND Google account (shared-folder Drive access,
used to get a fresh GPU quota - see DEVLOG.md), edit the
EPILLID_ZIP_PATH / PILLBOX_METADATA_CSV_PATH lines in CELL 2 to your
account's actual shared-folder path first, found via:
    !find /content/drive -iname "epillid_data.zip"
"""

# ============================================================
# CELL 1 - run this first, then wait for the runtime to restart
# ============================================================

from google.colab import drive
drive.mount('/content/drive')

get_ipython().system('pip install -q ultralytics')
get_ipython().system('pip install -q --upgrade --force-reinstall git+https://github.com/Moazam-Saif/Multimodal-Pharm-RAG.git')

print("Install complete. Restarting runtime now - this is expected, "
      "the tab will show 'reconnecting' briefly. Once reconnected, "
      "run CELL 2 below.")

# NOTE: there is no official, documented google.colab API for this
# (a feature request for one - googlecolab/colabtools#5204 - remains
# open/unresolved, and an earlier version of this file wrongly assumed
# it existed and shipped - see DEVLOG.md for that mistake). The
# actually-working approach, confirmed across multiple real Colab
# user reports, is to kill the kernel process directly - Colab
# auto-reconnects to a fresh one afterward.
import os
os.kill(os.getpid(), 9)


# ============================================================
# CELL 2 - run this AFTER the runtime has restarted and reconnected
# ============================================================

import os
os.environ["EPILLID_ZIP_PATH"] = "/content/drive/MyDrive/pill-rag/data/raw/epillid_data.zip"
os.environ["PILLBOX_METADATA_CSV_PATH"] = "/content/drive/MyDrive/pill-rag/data/raw/pillbox_metadata.csv"

import zipfile
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

from ultralytics import FastSAM
from pillrag.data import build_pill_dataset
from pillrag.segment import (
    select_pill_mask,
    resolve_pill_mask,
    rescue_complementary_mask,
    close_internal_gaps,
    bounding_box_fill_ratio,
    MaskSelectionResult,
)

model = FastSAM("FastSAM-x.pt")
df = build_pill_dataset()
ZIP_PATH = "/content/drive/MyDrive/pill-rag/data/raw/epillid_data.zip"

print(f"Setup complete. Dataset: {len(df)} rows, "
      f"{df['medicine_name'].notna().sum()} with text metadata.")
print("Expected: 5728 rows, 5544 with text metadata.")

_capsule_all = df[df["shape"] == "CAPSULE"]
_capsule_pure_white = _capsule_all[
    (_capsule_all["color"] == "WHITE") & (_capsule_all["is_ref"] == True)
]
_white_capsule_sample = _capsule_pure_white.sample(15, random_state=42)
_other_colors = ["ORANGE", "BLUE", "GREEN", "PINK", "BROWN", "RED", "YELLOW", "PURPLE"]
_other_color_examples = []
for _color in _other_colors:
    _candidates = _capsule_all[
        (_capsule_all["color"] == _color) & (_capsule_all["is_ref"] == True)
    ]
    if len(_candidates) > 0:
        _other_color_examples.append(_candidates.sample(1, random_state=42))
_other_color_sample = pd.concat(_other_color_examples)
_combined_capsule_sample = pd.concat([_white_capsule_sample, _other_color_sample])

_oval_pills = df[(df["shape"] == "OVAL") & (df["is_ref"] == True)]
_oval_sample = _oval_pills.sample(20, random_state=7)

TEST_CASE_PATHS = {
    "capsule_bandtest": "ePillID_data/classification_data/fcn_mix_weight/dr_224/00002-3228-30_PART_1_OF_1_CHAL10_SB_391E1C80.jpg",
    "round_clean": "ePillID_data/classification_data/fcn_mix_weight/dr_224/63304-0579-01_PART_1_OF_1_CHAL10_SB_5D26AEB5.jpg",
    "consumer_blob": "ePillID_data/classification_data/fcn_mix_weight/dc_224/4274.jpg",
    "capsule_0_WHITE": _combined_capsule_sample.iloc[0]["full_image_path"],
    "capsule_1_WHITE": _combined_capsule_sample.iloc[1]["full_image_path"],
    "capsule_2_WHITE": _combined_capsule_sample.iloc[2]["full_image_path"],
    "capsule_15_ORANGE": _combined_capsule_sample.iloc[15]["full_image_path"],
    "capsule_20_RED": _combined_capsule_sample.iloc[20]["full_image_path"],
    "orange_oval_logo": _oval_sample.iloc[18]["full_image_path"],
}

TEST_CASES = {}
with zipfile.ZipFile(ZIP_PATH) as zf:
    for _name, _path in TEST_CASE_PATHS.items():
        _local_path = f"testcase_{_name}.jpg"
        with zf.open(_path) as src, open(_local_path, "wb") as dst:
            dst.write(src.read())

        _results = model(_local_path, verbose=False)
        _confidences = _results[0].boxes.conf.cpu().numpy()
        _masks = _results[0].masks.data.cpu().numpy()

        TEST_CASES[_name] = (_confidences, _masks, _local_path)

print(f"\nBuilt {len(TEST_CASES)} test cases: {list(TEST_CASES.keys())}")
print("\nUsage: result = resolve_pill_mask(*TEST_CASES['capsule_20_RED'][:2])")
