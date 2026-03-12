# weebsorter

A Windows desktop application that classifies images into top-level categories
using **CLIP** (openai/clip-vit-large-patch14), sub-tags anime art and cosplay
using **WD-ViT-Tagger-v3** (SmilingWolf, via ONNX Runtime), and clusters real
faces by identity using **InsightFace buffalo** ONNX models.

---

## Requirements

| Requirement | Notes |
|-------------|-------|
| **uv** | Package manager — the *only* tool you need pre-installed |
| **Python 3.11** (system) | Only needed if your uv-managed Python lacks tkinter (see below) |
| ~3 GB free disk space | CLIP (~1.7 GB) + WD tagger (~400 MB) + buffalo face model (~200–500 MB), cached in `./models/` |

### Install uv (if you haven't already)

```powershell
# In PowerShell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

---

## Installation

```bat
cd path\to\anime-tagger

:: Create .venv, install all dependencies, generate uv.lock
uv sync
```

That's it. `uv sync` creates `.venv\` inside the project folder and resolves
every dependency to a reproducible `uv.lock`.

---

## Running

### Option A — double-click the launcher

```
run.bat
```

### Option B — command line

```bat
cd path\to\anime-tagger
uv run anime-tagger
```

### What happens on launch

1. Two native Windows folder-picker dialogs appear **before** Gradio starts.
   - Dialog 1: select your **input** folder (the images to classify).
   - Dialog 2: select your **output** folder (where copies will be saved).
   - Cancelling either dialog exits the application cleanly.
2. Gradio opens at `http://127.0.0.1:7860` in your browser automatically.
3. Select a face model (Fast `buffalo_s` or Accurate `buffalo_l`), then click **▶ Start Processing**.
   Model weights download on first run (progress shown in the terminal).
4. A live progress counter updates every 500 ms — Gradio stays fully
   responsive even during long batches.
5. On completion a summary panel appears and `classification_log.csv` is
   saved to the output folder root.

---

## Output folder structure

All files in the output folder are **copies** of the originals.
The input folder is never modified, moved, or deleted.

```
output/
├── faces/
│   └── person_{N}/              ← real people, clustered by face identity
├── cosplay/
│   └── {character_or_series}/   ← cosplay photos, sub-tagged by WD Tagger
├── landscapes/
├── architecture/
├── food/
├── screenshots/
├── abstract/
├── other/
├── anime/
│   ├── {character_or_series}/   ← named from highest-confidence series tag (≥ 0.20),
│   │                               falling back to highest-confidence character tag (≥ 0.70)
│   └── unknown/                 ← anime images where neither series nor character was detected
├── classification_log.csv
└── face_cluster_log.csv
```

### Filename collision handling

If a destination file already exists the incoming file is saved as
`stem_1.ext`, `stem_2.ext`, etc. — nothing is ever overwritten.

---

## Classification pipeline

### Stage 1 — CLIP (all images)

- Model: `openai/clip-vit-large-patch14`
- Method: zero-shot with multiple rich text prompts per class, averaged in
  embedding space for higher accuracy.
- Labels: `anime_art`, `cosplay`, `real_face`, `landscape`, `architecture`, `food`,
  `screenshot`, `abstract`, `other`.

### Stage 2 — WD-ViT-Tagger-v3 (anime art and cosplay)

- Model: `SmilingWolf/wd-vit-tagger-v3` (ONNX Runtime — DirectML GPU on Windows, CPU fallback)
- Triggered when Stage 1 returns `anime_art` or `cosplay`.
- Extracts: character tags (≥ 0.70), series/copyright tags (≥ 0.15),
  rating, and top-10 general tags (≥ 0.35).
- **Folder naming priority:**
  1. Highest-confidence series/copyright tag (≥ 0.20)
  2. Highest-confidence character tag (≥ 0.70) — used when no series is detected
  3. `unknown/` / `unknown_character/` — when neither clears the threshold
- **Training data cutoff: February 2024.** Characters and series from anime
  that premiered after that date will not be recognised.

### Stage 3 — Face identity clustering (real faces and cosplay)

- Models: InsightFace **buffalo_s** (~200 MB, fast) or **buffalo_l** (~500 MB, accurate)
  — downloaded automatically as ONNX files, no C++ build tools required.
- **Detection:** SCRFD face detector locates faces and 5-point landmarks.
- **Recognition:** ArcFace encodes each face as a 512-dim embedding.
- **Clustering:** DBSCAN groups embeddings by cosine distance (eps = 0.40 default)
  so each cluster = one identity.
- `faces/` images are reorganised into `person_1/`, `person_2/`, … subfolders.
- Images where no face is detected move to `unknown_face/`.
- Results are logged to `face_cluster_log.csv`.

> **Tuning tip:** Use the **eps slider** in the UI and click **🔄 Re-cluster Faces** to re-run clustering instantly without reprocessing all images. Lower eps = stricter (more folders, less merging). Higher eps = more lenient (fewer folders, more merging). Default 0.40 works well for diverse collections.

---

## CSV logs

### `classification_log.csv`

| Column | Description |
|--------|-------------|
| `original_filename` | File name only |
| `original_path` | Full path to the source file |
| `destination_path` | Full path to the copied file |
| `top_level_label` | CLIP label (or SKIPPED / ERROR) |
| `clip_confidence` | Softmax probability from CLIP |
| `detected_characters` | Semicolon-separated character tags |
| `detected_series` | Semicolon-separated series/copyright tags |
| `wd_rating` | Rating tag from WD Tagger |
| `wd_top10_tags` | Top-10 general tags as `tag:score` pairs |

### `face_cluster_log.csv`

| Column | Description |
|--------|-------------|
| `filename` | File name |
| `path` | Full path to the file |
| `face_detected` | TRUE / FALSE |
| `faces_in_image` | Number of faces found |
| `cluster_id` | Integer cluster ID (-1 = noise/unknown) |
| `destination` | Final path after clustering |

---

## Model cache location

All model weights download to `./models/` inside the project folder.
They are **never** written to `C:\Users\..\.cache\huggingface\`.

---

## GPU acceleration

The project ships with GPU acceleration enabled out of the box — no CUDA
Toolkit or cuDNN installation required.

| Component | GPU backend | Requirement |
|-----------|-------------|-------------|
| **CLIP** (Stage 1) | PyTorch CUDA 12.4 | NVIDIA GPU + up-to-date drivers |
| **WD Tagger** (Stage 2) | ONNX Runtime DirectML | Any DirectML-capable GPU (NVIDIA, AMD, Intel) on Windows 10/11 |
| **Face models** (Stage 3) | ONNX Runtime DirectML | Any DirectML-capable GPU (NVIDIA, AMD, Intel) on Windows 10/11 |

On a typical NVIDIA RTX GPU this gives roughly **3× the throughput** of CPU-only.

### If torch reverts to the CPU build

`uv sync` can occasionally revert PyTorch to the CPU wheel if the lockfile is
regenerated without the CUDA index. Verify which build is active:

```powershell
.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

If the version shows `+cpu` or `cuda.is_available()` returns `False`, force the
CUDA wheel back in:

```powershell
uv pip install torch --index-url https://download.pytorch.org/whl/cu124 --reinstall
```

This targets CUDA 12.4, which is compatible with any NVIDIA driver that reports
`CUDA Version: 12.x` or higher in `nvidia-smi`.

---

## tkinter note

If you see:

```
ERROR: tkinter is not available in this Python environment.
```

Your uv-managed Python does not bundle tkinter.  Run with a system Python
that was installed with tkinter (the default option when installing from
[python.org](https://www.python.org/downloads/windows/)):

```bat
uv run --python python anime-tagger
```

---

## Supported image formats

JPEG, PNG, WEBP, BMP, GIF (first frame only).
Corrupted or unreadable files are skipped and logged — they never crash the batch.
