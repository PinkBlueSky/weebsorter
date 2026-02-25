"""
anime_tagger/utils/file_ops.py — Safe file operations.

ABSOLUTE SAFETY RULE: This module only ever copies files.
It never deletes, moves, renames, or modifies any file in the input folder
or anywhere else on the filesystem except the designated output folder.
Every file operation in this module is a copy.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

# ── Windows-illegal filename characters ───────────────────────────────────────
# Covers the characters forbidden in Windows filenames/folder names,
# plus control characters 0x00–0x1F.
_ILLEGAL_CHARS_RE = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
# Reserved Windows names (case-insensitive)
_RESERVED_NAMES = frozenset({
    "CON", "PRN", "AUX", "NUL",
    "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8", "COM9",
    "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9",
})
_MAX_FOLDER_NAME_LEN = 100  # Conservative limit; NTFS allows 255 per component


def sanitize_folder_name(name: str) -> str:
    """
    Return a Windows-safe version of *name* for use as a folder name.

    - Strips illegal characters (replaced with '_').
    - Trims leading/trailing dots and spaces.
    - Replaces Windows reserved names.
    - Truncates to _MAX_FOLDER_NAME_LEN characters.
    - Falls back to "unknown" if the result is empty.
    """
    sanitized = _ILLEGAL_CHARS_RE.sub("_", name)
    sanitized = sanitized.strip(". ")
    if sanitized.upper() in _RESERVED_NAMES:
        sanitized = f"_{sanitized}_"
    sanitized = sanitized[:_MAX_FOLDER_NAME_LEN].rstrip(". ")
    return sanitized or "unknown"


def safe_copy(src: Path, dst_dir: Path) -> Path:
    """
    Copy *src* into *dst_dir*, creating *dst_dir* if necessary.

    Filename collisions are resolved by appending _1, _2, … to the stem
    rather than overwriting the existing file.

    Returns the destination path.

    SAFETY: copy only, never modify source
    """
    # SAFETY: copy only, never modify source — create destination directory
    dst_dir.mkdir(parents=True, exist_ok=True)

    dst = dst_dir / src.name
    if dst.exists():
        stem   = src.stem
        suffix = src.suffix
        counter = 1
        while dst.exists():
            dst = dst_dir / f"{stem}_{counter}{suffix}"
            counter += 1

    shutil.copy2(src, dst)  # SAFETY: copy only, never modify source
    return dst
