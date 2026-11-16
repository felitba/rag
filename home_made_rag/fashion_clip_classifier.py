from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageOps
from transformers import CLIPModel, CLIPProcessor

from home_made_rag.config import CONFIG


class FashionClipCategoryClassifier:
    """Zero-shot classifier for the catalog's existing garment categories."""

    def __init__(self, labels: Sequence[str] | None = None) -> None:
        settings = CONFIG["category_classifier"]
        self.model_name = settings["model_name"]
        self.labels = sorted(set(labels or settings.get("labels", [])))
        if not self.labels:
            raise ValueError("No hay categorías en el catálogo para clasificar la imagen")
        self.device = self._pick_device(CONFIG["embeddings"]["device"])
        self._model: CLIPModel | None = None
        self._processor: CLIPProcessor | None = None

    def classify(self, image_path: str | Path) -> dict[str, Any]:
        model, processor = self._get_model()
        with Image.open(image_path) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
        inputs = processor(images=[image], return_tensors="pt").to(self.device)
        prompts = [self._prompt(label) for label in self.labels]
        text_inputs = processor(
            text=prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)

        with torch.inference_mode():
            image_features = self._features(model.get_image_features(**inputs))
            text_features = self._features(model.get_text_features(**text_inputs))
            image_features = torch.nn.functional.normalize(image_features, dim=-1)
            text_features = torch.nn.functional.normalize(text_features, dim=-1)
            similarities = (image_features @ text_features.T)[0].cpu().numpy()

        order = np.argsort(similarities)[::-1]
        alternatives = [
            {"label": self.labels[index], "similarity": round(float(similarities[index]), 4)}
            for index in order[: min(3, len(order))]
        ]
        margin = float(similarities[order[0]] - similarities[order[1]]) if len(order) > 1 else 0.0
        return {
            "label": alternatives[0]["label"],
            "similarity": alternatives[0]["similarity"],
            "margin": round(margin, 4),
            "alternatives": alternatives[1:],
        }

    def _get_model(self) -> tuple[CLIPModel, CLIPProcessor]:
        if self._model is None:
            self._model = CLIPModel.from_pretrained(self.model_name).to(self.device)
            self._model.eval()
            self._processor = CLIPProcessor.from_pretrained(self.model_name)
        if self._processor is None:
            raise RuntimeError("FashionCLIP processor was not initialized")
        return self._model, self._processor

    @staticmethod
    def _features(output: Any) -> torch.Tensor:
        if isinstance(output, torch.Tensor):
            return output
        features = getattr(output, "pooler_output", None)
        if isinstance(features, torch.Tensor):
            return features
        raise TypeError("FashionCLIP devolvió un output sin tensor de features")

    @staticmethod
    def _prompt(label: str) -> str:
        descriptions = {
            "tops": "a top, shirt, or blouse",
            "bottoms": "pants, shorts, or a skirt",
            "all-body": "a dress or jumpsuit",
            "outerwear": "a coat or jacket",
            "jewellery": "a piece of jewelry",
            "shoes": "a pair of shoes",
            "bags": "a bag or handbag",
            "sunglasses": "a pair of sunglasses",
            "hats": "a hat",
            "scarves": "a scarf",
            "accessories": "a fashion accessory",
        }
        return f"a fashion product photo of {descriptions.get(label, label)}"

    @staticmethod
    def _pick_device(requested: str) -> str:
        if requested != "auto":
            return requested
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
