"""
anime_tagger/classifiers/wd_tagger.py — Stage 2: anime sub-categorisation.

Uses SmilingWolf/wd-vit-tagger-v3 (ONNX Runtime) to extract:
  • character tags   (Danbooru category 4)
  • series / copyright tags  (category 3)
  • rating           (category 9)
  • top-10 general tags (category 0)

Triggered only when Stage 1 (CLIP) returns "anime_art".
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from PIL import Image

if TYPE_CHECKING:
    pass

# ── Model repository ──────────────────────────────────────────────────────────
_REPO_ID   = "SmilingWolf/wd-vit-tagger-v3"
_MODEL_FILE = "model.onnx"
_TAGS_FILE  = "selected_tags.csv"

# ── Danbooru category constants ───────────────────────────────────────────────
_CAT_GENERAL   = 0
_CAT_COPYRIGHT = 3   # series / franchise
_CAT_CHARACTER = 4
_CAT_RATING    = 9

# ── Score thresholds ──────────────────────────────────────────────────────────
_THRESH_GENERAL    = 0.35
_THRESH_CHARACTER  = 0.70
_THRESH_SERIES     = 0.50
_THRESH_SERIES_BEST = 0.50   # minimum confidence to name the output subfolder

# ── Input image size expected by the model ────────────────────────────────────
_IMAGE_SIZE = 448


class WDResult(TypedDict):
    rating: str                        # e.g. "rating:general"
    characters: dict[str, float]       # {tag_name: confidence, …}
    series: dict[str, float]           # {series_name: confidence, …}
    best_series: str | None            # highest-confidence series tag, or None
    top10_tags: list[tuple[str, float]] # top-10 general tags sorted by confidence


class WDTagger:
    """ONNX-backed WD-ViT-Tagger-v3 wrapper."""

    def __init__(self) -> None:
        self._session: ort.InferenceSession | None = None
        self._tags: list[str] = []
        self._categories: list[int] = []
        # Index sets per category
        self._rating_idx: list[int] = []
        self._general_idx: list[int] = []
        self._character_idx: list[int] = []
        self._copyright_idx: list[int] = []

    # ── Loading ───────────────────────────────────────────────────────────────

    def load(self) -> None:
        """Download (if needed) and load the ONNX model + tags CSV."""
        if self._session is not None:
            return  # Already loaded

        model_path = hf_hub_download(repo_id=_REPO_ID, filename=_MODEL_FILE)
        tags_path  = hf_hub_download(repo_id=_REPO_ID, filename=_TAGS_FILE)

        # Prefer GPU if CUDAExecutionProvider is available
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if "CUDAExecutionProvider" in ort.get_available_providers()
            else ["CPUExecutionProvider"]
        )
        self._session = ort.InferenceSession(model_path, providers=providers)

        self._load_tags(tags_path)

    def _load_tags(self, tags_path: str) -> None:
        with open(tags_path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)

        if not rows:
            raise ValueError(f"Empty tags CSV: {tags_path}")

        required = {"name", "category"}
        if not required.issubset(rows[0].keys()):
            raise ValueError(
                f"selected_tags.csv is missing columns. "
                f"Expected {required}, got {set(rows[0].keys())}"
            )

        self._tags = [r["name"] for r in rows]
        self._categories = [int(r["category"]) for r in rows]

        self._rating_idx    = [i for i, c in enumerate(self._categories) if c == _CAT_RATING]
        self._general_idx   = [i for i, c in enumerate(self._categories) if c == _CAT_GENERAL]
        self._character_idx = [i for i, c in enumerate(self._categories) if c == _CAT_CHARACTER]
        self._copyright_idx = [i for i, c in enumerate(self._categories) if c == _CAT_COPYRIGHT]

    # ── Preprocessing ─────────────────────────────────────────────────────────

    @staticmethod
    def _preprocess(img: Image.Image) -> np.ndarray:
        """
        Pad to square (white background), resize to 448×448,
        return float32 NHWC array with values in [0, 255].
        """
        if img.mode != "RGB":
            img = img.convert("RGB")

        w, h = img.size
        side = max(w, h)
        padded = Image.new("RGB", (side, side), (255, 255, 255))
        padded.paste(img, ((side - w) // 2, (side - h) // 2))
        padded = padded.resize((_IMAGE_SIZE, _IMAGE_SIZE), Image.BICUBIC)

        arr = np.asarray(padded, dtype=np.float32)   # [H, W, 3]
        arr = np.expand_dims(arr, axis=0)             # [1, H, W, 3]
        return arr

    # ── Inference ─────────────────────────────────────────────────────────────

    def tag(self, pil_image: Image.Image) -> WDResult:
        """
        Run WD-ViT-Tagger-v3 inference on a PIL image.

        Returns a WDResult dict with rated, character, series, and general tags.
        """
        assert self._session is not None, "Call load() before tag()"

        arr = self._preprocess(pil_image)
        input_name = self._session.get_inputs()[0].name
        preds: np.ndarray = self._session.run(None, {input_name: arr})[0][0]

        # ── Ratings ───────────────────────────────────────────────────────────
        rating = "rating:general"  # default
        if self._rating_idx:
            best_r = max(self._rating_idx, key=lambda i: preds[i])
            rating = self._tags[best_r]

        # ── Character tags ────────────────────────────────────────────────────
        characters: dict[str, float] = {
            self._tags[i]: float(preds[i])
            for i in self._character_idx
            if preds[i] >= _THRESH_CHARACTER
        }

        # ── Series / copyright tags ───────────────────────────────────────────
        series: dict[str, float] = {
            self._tags[i]: float(preds[i])
            for i in self._copyright_idx
            if preds[i] >= _THRESH_SERIES
        }
        best_series: str | None = None
        if series:
            best_series_name = max(series, key=series.get)  # type: ignore[arg-type]
            if series[best_series_name] >= _THRESH_SERIES_BEST:
                best_series = best_series_name

        # ── Top-10 general tags ───────────────────────────────────────────────
        general_scores = [
            (self._tags[i], float(preds[i]))
            for i in self._general_idx
            if preds[i] >= _THRESH_GENERAL
        ]
        top10 = sorted(general_scores, key=lambda x: x[1], reverse=True)[:10]

        return WDResult(
            rating=rating,
            characters=characters,
            series=series,
            best_series=best_series,
            top10_tags=top10,
        )
