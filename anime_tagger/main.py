"""
anime_tagger/main.py — Entry point for Anime Tagger.

Execution order:
  1. Set HF_HOME to ./models (must happen before ANY huggingface import).
  2. Verify tkinter is available; print a clear error and exit if not.
  3. Open two sequential native Windows folder-picker dialogs.
  4. Create the output folder if needed.
  5. Launch the Gradio application.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# ── 1. Set HF_HOME BEFORE any huggingface / transformers import ──────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.environ["HF_HOME"] = str(_PROJECT_ROOT / "models")

# Suppress tokenizer parallelism warning that fires when threads are used.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# ── 2. Tkinter availability check ────────────────────────────────────────────
try:
    import tkinter as tk
    from tkinter import filedialog as _fd

    _TKINTER_OK = True
except ImportError:
    _TKINTER_OK = False


def _assert_tkinter() -> None:
    """Exit with a helpful message when tkinter is missing."""
    if not _TKINTER_OK:
        print(
            "\n"
            "ERROR: tkinter is not available in this Python environment.\n"
            "\n"
            "anime-tagger requires tkinter for the folder-selection dialogs.\n"
            "The uv-managed Python inside .venv does not bundle tkinter.\n"
            "\n"
            "Fix — run with a system Python that includes tkinter:\n"
            "  1. Download Python 3.11+ from https://www.python.org/downloads/windows/\n"
            "  2. During setup, tick  'tcl/tk and IDLE'  (checked by default).\n"
            "  3. Then run:  uv run --python python  anime-tagger\n"
            "     (uv will use the system interpreter that has tkinter.)\n",
            file=sys.stderr,
        )
        sys.exit(1)


# ── 3. Folder-picker dialogs ─────────────────────────────────────────────────

def _pick_folders() -> tuple[Path, Path]:
    """
    Show two sequential native Windows folder-picker dialogs.
    Returns (input_folder, output_folder) as resolved Path objects.
    Exits the process if the user cancels either dialog.
    """
    root = tk.Tk()
    root.withdraw()
    # Bring the dialog to the foreground on Windows.
    root.wm_attributes("-topmost", True)

    print("Opening dialog — please select the INPUT folder (images to classify)…")
    input_str = _fd.askdirectory(
        parent=root,
        title="Anime Tagger — Select INPUT folder (images to classify)",
    )
    if not input_str:
        print("No input folder selected. Exiting.")
        root.destroy()
        sys.exit(0)

    input_folder = Path(input_str).resolve()
    print(f"  Input  folder : {input_folder}")

    print("Opening dialog — please select the OUTPUT folder (where copies will go)…")
    output_str = _fd.askdirectory(
        parent=root,
        title="Anime Tagger — Select OUTPUT folder (classified copies go here)",
    )
    if not output_str:
        print("No output folder selected. Exiting.")
        root.destroy()
        sys.exit(0)

    output_folder = Path(output_str).resolve()
    print(f"  Output folder : {output_folder}")

    root.destroy()
    return input_folder, output_folder


# ── 4 & 5. Main entry point ──────────────────────────────────────────────────

def main() -> None:
    _assert_tkinter()

    input_folder, output_folder = _pick_folders()

    # Create the output folder if it does not exist.
    # SAFETY: copy only, never modify source — this only creates the output dir.
    output_folder.mkdir(parents=True, exist_ok=True)

    # Deferred import so HF_HOME is already set when transformers/gradio load.
    from anime_tagger.app import build_app  # noqa: PLC0415

    demo = build_app(input_folder, output_folder)
    demo.launch(
        server_name="127.0.0.1",
        inbrowser=True,
        share=False,
        ssr_mode=False,  # Disable SSR to avoid Gradio 6.x httpx client lifecycle errors
    )


if __name__ == "__main__":
    main()
