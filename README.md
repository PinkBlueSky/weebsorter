# weebsorter

A Windows desktop application that classifies images into top-level categories
using **CLIP** (openai/clip-vit-large-patch14) and then sub-tags anime art using
**WD-ViT-Tagger-v3** (SmilingWolf, via ONNX Runtime).

---

## Requirements

| Requirement | Notes |
|-------------|-------|
| **uv** | Package manager — the *only* tool you need pre-installed |
| **Python 3.11** (system) | Only needed if your uv-managed Python lacks tkinter (see below) |
| ~2 GB free disk space | CLIP model (~1.7 GB) + WD tagger (~400 MB), cached in `./models/` |

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
3. Click **▶ Start Processing**.
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
└── classification_log.csv
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
- Labels: `anime_art`, `real_face`, `landscape`, `architecture`, `food`,
  `screenshot`, `abstract`, `other`.

### Stage 2 — WD-ViT-Tagger-v3 (anime only)

- Model: `SmilingWolf/wd-vit-tagger-v3` (ONNX Runtime, CPU by default)
- Triggered only when Stage 1 returns `anime_art`.
- Extracts: character tags (≥ 0.70), series/copyright tags (≥ 0.15),
  rating, and top-10 general tags (≥ 0.35).
- **Folder naming priority:**
  1. Highest-confidence series/copyright tag (≥ 0.20)
  2. Highest-confidence character tag (≥ 0.70) — used when no series is detected
  3. `unknown/` — when neither series nor character clears the threshold
- **Training data cutoff: February 2024.** Characters and series from anime
  that premiered after that date will not be recognised.

---

## CSV log columns

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

---

## Model cache location

All HuggingFace model weights download to `./models/` inside the project
folder (`HF_HOME` is set to this path in code before any import).
They are **never** written to `C:\Users\..\.cache\huggingface\`.

---

## GPU acceleration

> **Requirements:** an NVIDIA GPU with up-to-date drivers.
> AMD and Intel GPUs are not supported by this path.

By default the project installs the **CPU-only** PyTorch wheel (~200 MB).
Switching to GPU cuts processing time significantly on large batches.

### Step 1 — find your CUDA version

Open a terminal and run:

```bat
nvidia-smi
```

Look for `CUDA Version: XX.X` in the top-right of the output.
Common values are `12.1`, `12.4`, `11.8`.

### Step 2 — update pyproject.toml

Edit `pyproject.toml` and change `torch-backend`:

```toml
[tool.uv]
# CPU (default):
torch-backend = "cpu"

# CUDA 12.1:
torch-backend = "cu121"

# CUDA 12.4:
torch-backend = "cu124"

# CUDA 11.8:
torch-backend = "cu118"
```

Use the value that matches your CUDA version from Step 1.

### Step 3 — swap onnxruntime for the GPU build

In the `[project]` `dependencies` list, replace:

```toml
"onnxruntime>=1.17.0",
```

with:

```toml
"onnxruntime-gpu>=1.17.0",
```

### Step 4 — reinstall

```bat
uv sync
```

`uv sync` will download the CUDA-enabled PyTorch (~2.5 GB) and
`onnxruntime-gpu`. The WD Tagger will then automatically use
`CUDAExecutionProvider` and CLIP will run on the GPU via PyTorch.

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
