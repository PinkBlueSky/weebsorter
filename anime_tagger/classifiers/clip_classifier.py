"""
anime_tagger/classifiers/clip_classifier.py — Stage 1 classification.

Uses openai/clip-vit-large-patch14 via HuggingFace transformers to classify
each image zero-shot into one of eight top-level categories.

The classifier pre-computes per-class text embeddings by averaging over
multiple rich template prompts, which improves accuracy over single prompts.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from transformers import CLIPImageProcessor, CLIPModel, CLIPTokenizer

if TYPE_CHECKING:
    from PIL.Image import Image

# ── Model identifier ──────────────────────────────────────────────────────────
_CLIP_MODEL_ID = "openai/clip-vit-large-patch14"

# ── Label → rich text prompts ─────────────────────────────────────────────────
# Multiple prompts per class are averaged in embedding space, following the
# "ensembling" technique described in the original CLIP paper (Radford 2021).
_LABEL_PROMPTS: dict[str, list[str]] = {
    "anime_art": [
        "a 2D drawn anime illustration or manga artwork, not a real photograph",
        "a hand-drawn or digitally painted Japanese anime style illustration",
        "an anime or manga image with cel-shading or flat digital art colouring",
        "2D anime artwork such as a key visual, fan art, or official illustration",
        "a digital illustration of an anime character with stylised proportions and flat vibrant colours, clearly not a real photo",
        "drawn anime art in the style of Naruto or Sword Art Online — a 2D illustration, not a photograph",
    ],
    "real_face": [
        "a real photograph of a human face or person",
        "a photo of a real person, including cosplay, anime costumes, and anime-style makeup",
        "a photograph or selfie of a real human being, not an illustration or drawing",
        "a portrait or candid photo of a real person, even if they are wearing wigs, costumes, or fictional outfits",
    ],
    "landscape": [
        "an outdoor landscape photograph showing nature scenery",
        "a scenic view of mountains, forests, fields, rivers, or sky",
        "nature photography showing hills, coastlines, or rural countryside",
        "a wide outdoor scene with natural terrain and environment",
        "a panoramic view of a natural landscape with no people",
    ],
    "architecture": [
        "a photograph of a building, bridge, or architectural structure",
        "urban cityscape or street photography showing buildings and infrastructure",
        "interior or exterior architectural photography of a constructed space",
        "a photo of a house, skyscraper, temple, museum, or other man-made structure",
    ],
    "food": [
        "a photograph of food, a meal, or a dish served on a plate",
        "culinary photography of prepared food or drinks",
        "a restaurant dish, home-cooked meal, or food plating close-up",
        "close-up photography of food items, snacks, or beverages",
    ],
    "screenshot": [
        "a screenshot of a computer screen, website, or software application",
        "a screen capture showing a desktop, GUI, or user interface with text",
        "a mobile phone screenshot or web browser screenshot showing digital content",
        "a screenshot containing text menus, icons, or interface elements",
    ],
    "abstract": [
        "abstract digital art with no recognisable real-world objects",
        "an abstract geometric pattern, fractal, or non-representational design",
        "a wallpaper with abstract shapes, gradients, or flowing artistic patterns",
        "non-figurative artwork made of colours and shapes without a concrete subject",
    ],
    "other": [
        "a generic photograph or image that does not fit specific categories",
        "a miscellaneous image of objects, products, documents, or mixed content",
        "an unclassified image showing something that is not anime, food, or a landscape",
    ],
}


class CLIPClassifier:
    """Zero-shot image classifier using CLIP (openai/clip-vit-large-patch14)."""

    def __init__(self) -> None:
        self._model: CLIPModel | None = None
        # Use separate image/text processors to avoid processor_config.json
        # lookup introduced in transformers 5.x — works with both 4.x and 5.x.
        self._image_processor: CLIPImageProcessor | None = None
        self._tokenizer: CLIPTokenizer | None = None
        self._class_embeddings: torch.Tensor | None = None
        self._labels: list[str] = list(_LABEL_PROMPTS.keys())
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

    def load(self) -> None:
        """Download (if needed) and load the CLIP model and processors."""
        if self._model is not None and self._class_embeddings is not None:
            return  # Already loaded

        try:
            self._image_processor = CLIPImageProcessor.from_pretrained(_CLIP_MODEL_ID)
            self._tokenizer = CLIPTokenizer.from_pretrained(_CLIP_MODEL_ID)
            self._model = CLIPModel.from_pretrained(_CLIP_MODEL_ID).to(self._device)
            self._model.eval()

            # Pre-compute averaged per-class text embeddings
            self._class_embeddings = self._build_class_embeddings()
        except Exception:
            # Reset so the next load() call retries from scratch
            self._model = None
            self._image_processor = None
            self._tokenizer = None
            self._class_embeddings = None
            raise

    def _build_class_embeddings(self) -> torch.Tensor:
        """
        For each label, encode all template prompts, L2-normalise each,
        average them, and re-normalise to get a single class embedding.
        Returns a tensor of shape [num_classes, embed_dim].
        """
        assert self._model is not None
        assert self._tokenizer is not None

        class_vecs: list[torch.Tensor] = []

        with torch.no_grad():
            for label in self._labels:
                prompts = _LABEL_PROMPTS[label]
                inputs = self._tokenizer(
                    text=prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                ).to(self._device)
                text_out = self._model.text_model(**inputs)
                text_feats = self._model.text_projection(text_out.pooler_output)
                # L2-normalise each prompt embedding
                text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
                # Average over prompts for this class, then re-normalise
                class_vec = text_feats.mean(dim=0)
                class_vec = class_vec / class_vec.norm()
                class_vecs.append(class_vec)

        return torch.stack(class_vecs)  # [num_classes, embed_dim]

    def classify(self, pil_image: "Image") -> tuple[str, float]:
        """
        Classify a PIL Image into one of the eight top-level labels.

        Returns:
            (label, confidence)  where confidence is the softmax probability.
        """
        assert self._model is not None, "Call load() before classify()"
        assert self._image_processor is not None
        assert self._class_embeddings is not None

        with torch.no_grad():
            inputs = self._image_processor(
                images=pil_image,
                return_tensors="pt",
            ).to(self._device)
            vision_out = self._model.vision_model(**inputs)
            image_feat = self._model.visual_projection(vision_out.pooler_output)
            image_feat = image_feat / image_feat.norm(dim=-1, keepdim=True)

            # Cosine similarity → softmax probabilities
            logits = (image_feat @ self._class_embeddings.T) * 100.0  # temperature scaling
            probs = logits.softmax(dim=-1)[0]

        best_idx = int(probs.argmax().item())
        label = self._labels[best_idx]
        confidence = float(probs[best_idx].item())
        return label, confidence
