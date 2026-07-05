# Pill RAG

Multimodal RAG pipeline for pharmaceutical pill identification from a
photo. Combines computer vision (background segmentation + visual
embeddings) with text-based RAG (hybrid retrieval + reranking) to
identify a pill and retrieve relevant drug information.

**Status**: Phase 1 (data) complete. Phase 2 (segmentation) in progress.
See `DEVLOG.md` for the full development history and reasoning behind
every decision - **read that before making changes**, especially before
re-deciding anything that looks questionable at first glance; there's
usually a documented reason.

## Project structure

```
pill-rag/
├── notebooks/        # Colab-run GPU batch jobs (segmentation, embedding)
├── src/pillrag/      # installable package - core, reusable pipeline logic
├── scripts/          # one-off local investigation scripts (see scripts/README.md)
├── api/               # FastAPI backend (not yet built)
├── frontend/          # React or Gradio, TBD (not yet built)
├── data/
│   ├── raw/                        # source downloads, gitignored
│   └── samples/segmented_preview/  # small QA sample only
├── tests/
├── DEVLOG.md          # full development history, decisions, and reasoning
├── pyproject.toml
```

## Setup (Windows, no local GPU)

This project assumes: no local GPU (heavy compute runs on Google Colab),
Windows native (not WSL), Python 3.11+.

1. **Clone and enter the repo**
   ```powershell
   git clone <repo-url>
   cd pill-rag
   ```

2. **Create and activate a virtual environment**
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```
   If PowerShell blocks the activation script:
   ```powershell
   Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
   ```

3. **Install the package in editable mode**
   ```powershell
   pip install -e .
   ```
   This installs everything listed in `pyproject.toml`'s `dependencies`,
   and makes `import pillrag` resolve to `src/pillrag/` (src-layout - see
   DEVLOG.md for why this layout was chosen).

4. **Download the raw data** (not tracked in git - see `.gitignore`)

   | File | Source | Save as |
   |---|---|---|
   | Pillbox metadata CSV (~56MB) | `https://datadiscovery.nlm.nih.gov/api/views/crzr-uvwg/rows.csv?accessType=DOWNLOAD` | `data/raw/pillbox_metadata.csv` |
   | ePillID dataset (~153MB) | [GitHub Release](https://github.com/usuyama/ePillID-benchmark/releases/tag/ePillID_data_v1.0), asset `ePillID_data.zip` | `data/raw/epillid_data.zip` |

   (We also downloaded the original NIH Pillbox image zip early on, but
   it's **not used** in the final pipeline - see DEVLOG.md's "Pivot"
   section for why. No need to download it unless you want to retrace
   that investigation yourself.)

5. **Verify the setup**
   ```powershell
   python -m pillrag.data
   ```
   Expected output: `Total rows: 5728`, `Rows with recovered text
   metadata: 5544`. If these numbers differ, something about the source
   data has changed since this was written - check DEVLOG.md's join
   logic against the current data before assuming your setup is broken.

## Working with Colab (Phase 2+)

Heavy compute (batch segmentation, batch embedding) runs on Google Colab
notebooks under `notebooks/`, not locally. See DEVLOG.md for the
reasoning on this split. Notebook-specific setup instructions will be
added here once Phase 2 notebooks exist.

## For anyone (or any AI) picking this project up

Read `DEVLOG.md` in full before writing new code. It contains:
- Every dataset investigated, including dead ends, and why each was
  rejected or adopted
- Every bug found in the data-joining logic, how it was diagnosed (with
  evidence, not guesses), and how it was fixed
- The exact current scope decision and what's explicitly deferred
- A working-instructions prompt (at the top of DEVLOG.md) describing how
  this project expects to be worked on - the same approach used to build
  everything so far
