"""
anime_tagger/utils/image_utils.py — Image discovery and loading utilities.

ABSOLUTE SAFETY RULE: This module only reads files.
It never deletes, moves, renames, or modifies any file on the filesystem.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PIL import Image, UnidentifiedImageError

# ── Supported extensions ───────────────────────────────────────────────────────
_SUPPORTED_SUFFIXES: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}
)


def get_image_files(folder: Path) -> list[Path]:
    """
    Return a sorted list of all supported image files found recursively
    under *folder*.  Only files with a supported extension are included.
    """
    files: list[Path] = []
    for path in sorted(folder.rglob("*")):
        if path.is_file() and path.suffix.lower() in _SUPPORTED_SUFFIXES:
            files.append(path)
    return files


def load_image_safe(path: Path) -> Optional[Image.Image]:
    """
    Open an image file and return it as an RGB PIL Image.

    Handles:
      • JPEG, PNG, WEBP, BMP — opened normally.
      • GIF — first frame extracted.
      • Images with an alpha channel — composited onto a white background.
      • Corrupted or unreadable files — returns None (caller logs and skips).

    This function never modifies, moves, or deletes the source file.
    """
    try:
        img = Image.open(path)
        img.load()  # Force full decode; raises on corrupt files

        # ── GIF: extract first frame ──────────────────────────────────────────
        if getattr(img, "format", None) == "GIF" or getattr(img, "is_animated", False):
            img.seek(0)
            img = img.copy()

        # ── Palette / transparency handling ───────────────────────────────────
        if img.mode in ("P", "PA"):
            img = img.convert("RGBA")

        if img.mode == "RGBA":
            background = Image.new("RGB", img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[3])  # alpha as mask
            return background

        if img.mode != "RGB":
            img = img.convert("RGB")

        return img

    except (UnidentifiedImageError, OSError, SyntaxError, EOFError):
        return None
    except Exception:  # noqa: BLE001 — catch any other decode errors
        return None
