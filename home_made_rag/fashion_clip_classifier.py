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
    """Zero-shot classifier for the catalog's garment categories."""

    CATEGORY_PROMPTS = {
        "tops": [
            "a fashion catalog photo of a crewneck sweatshirt, sweater, hoodie, pullover, long-sleeve shirt, blouse, t-shirt, or knit top",
            "a person wearing a sweatshirt, sweater, hoodie, or pullover on their upper body",
            "a close-up product photo of an upper-body garment such as a sweatshirt or sweater",
        ],
        "bottoms": [
            "a fashion catalog photo of pants, jeans, shorts, a skirt, or trousers",
            "a person wearing pants, jeans, shorts, or a skirt on their lower body",
            "a close-up product photo of trousers or another lower-body garment",
        ],
        "all-body": [
            "a fashion catalog photo of a dress, jumpsuit, or romper",
            "a person wearing a one-piece dress or jumpsuit",
            "a close-up product photo of a full-body garment such as a dress",
        ],
        "outerwear": [
            "a fashion catalog photo of a coat, jacket, blazer, or parka",
            "a person wearing a coat or jacket over their clothes",
            "a close-up product photo of outerwear such as a coat or jacket",
        ],
        "jewellery": [
            "a fashion catalog photo of jewelry such as a necklace, earrings, bracelet, or ring",
            "a person wearing jewelry such as earrings, a necklace, or a bracelet",
            "a close-up product photo of a piece of jewelry",
        ],
        "shoes": [
            "a fashion catalog photo of footwear such as sneakers, boots, heels, or sandals",
            "a person wearing a pair of shoes on their feet",
            "a close-up product photo of a shoe or boot",
        ],
        "bags": [
            "a fashion catalog photo of a handbag, backpack, tote, or purse",
            "a person carrying a handbag or backpack",
            "a close-up product photo of a fashion bag",
        ],
        "sunglasses": [
            "a fashion catalog photo of sunglasses or eyewear",
            "a person wearing sunglasses",
            "a close-up product photo of a pair of sunglasses",
        ],
        "hats": [
            "a fashion catalog photo of a hat, cap, or beanie",
            "a person wearing a hat or cap on their head",
            "a close-up product photo of a fashion hat",
        ],
        "scarves": [
            "a fashion catalog photo of a scarf or neck wrap",
            "a person wearing a scarf around their neck",
            "a close-up product photo of a scarf",
        ],
        "accessories": [
            "a small fashion accessory such as a belt, watch, jewelry, hat, scarf, or sunglasses",
            "a close-up product photo of an accessory, not a shirt, sweater, pants, or dress",
            "a person wearing a fashion accessory such as a belt, watch, scarf, or jewelry",
        ],
    }

    def __init__(self, labels: Sequence[str] | None = None) -> None:
        settings = CONFIG["category_classifier"]
        self.model_name = settings["model_name"]
        self.labels = sorted(set(labels or settings.get("labels", [])))
        if not self.labels:
            raise ValueError("No hay categorías en el catálogo para clasificar la imagen")
        self.device = self._pick_device(CONFIG["embeddings"]["device"])
        self._model: CLIPModel | None = None
        self._processor: CLIPProcessor | None = None

    def classify(self, image_path: Image.Image | str | Path) -> dict[str, Any]:
        model, processor = self._get_model()
        if isinstance(image_path, Image.Image):
            image = ImageOps.exif_transpose(image_path).convert("RGB")
        else:
            with Image.open(image_path) as opened:
                image = ImageOps.exif_transpose(opened).convert("RGB")
        inputs = processor(images=self._image_views(image), return_tensors="pt").to(self.device)
        prompts_by_label = [self._prompts(label) for label in self.labels]
        prompts = [prompt for group in prompts_by_label for prompt in group]
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
            # Average the full frame and, for lifestyle/landscape photos, a tighter
            # center crop so the wearer and garment count more than the scenery.
            image_features = torch.nn.functional.normalize(
                image_features.mean(dim=0, keepdim=True), dim=-1
            )
            text_features = torch.nn.functional.normalize(text_features, dim=-1)
            prototypes = []
            offset = 0
            for prompt_group in prompts_by_label:
                next_offset = offset + len(prompt_group)
                prototype = text_features[offset:next_offset].mean(dim=0, keepdim=True)
                prototypes.append(torch.nn.functional.normalize(prototype, dim=-1))
                offset = next_offset
            category_features = torch.cat(prototypes, dim=0)
            similarities = (image_features @ category_features.T)[0].cpu().numpy()

        order = np.argsort(similarities)[::-1]
        ranked_categories = [
            {"label": self.labels[index], "similarity": round(float(similarities[index]), 4)}
            for index in order[: min(3, len(order))]
        ]
        margin = float(similarities[order[0]] - similarities[order[1]]) if len(order) > 1 else 0.0
        return {
            "label": ranked_categories[0]["label"],
            "similarity": ranked_categories[0]["similarity"],
            "margin": round(margin, 4),
            "alternatives": ranked_categories[1:],
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

    def _prompts(self, label: str) -> list[str]:
        normalized = str(label).strip().lower().replace("_", "-")
        descriptions = self.CATEGORY_PROMPTS.get(normalized)
        if descriptions:
            return descriptions
        category = str(label).strip().replace("_", " ").replace("-", " ")
        return [
            f"a fashion catalog photo of an item in the {category} category",
            f"a person wearing a {category} fashion item",
            f"a close-up fashion product photo in the {category} category",
        ]

    @staticmethod
    def _image_views(image: Image.Image) -> list[Image.Image]:
        views = [image]
        width, height = image.size
        if width / max(height, 1) >= 1.25:
            left = round(width * 0.20)
            right = round(width * 0.80)
            top = round(height * 0.05)
            bottom = round(height * 0.95)
            if right > left and bottom > top:
                views.append(image.crop((left, top, right, bottom)))
        return views

    @staticmethod
    def _pick_device(requested: str) -> str:
        if requested != "auto":
            return requested
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
