from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageOps
from transformers import CLIPModel, CLIPProcessor

from home_made_rag.catalog import Catalog
from home_made_rag.config import CONFIG


class FashionClipReranker:
    """Reranks retrieved garments with FashionCLIP image/text embeddings."""

    def __init__(self) -> None:
        settings = CONFIG["reranker"]
        self.batch_size = int(settings["batch_size"])
        self.model_name = settings["model_name"]
        self.device = self._pick_device(CONFIG["embeddings"]["device"])
        self._model: CLIPModel | None = None
        self._processor: CLIPProcessor | None = None

    def rerank(
        self,
        matches: Sequence[dict[str, Any]],
        catalog: Catalog,
        *,
        image_path: str | Path | None = None,
        text: str | None = None,
    ) -> list[dict[str, Any]]:
        if (image_path is None) == (text is None):
            raise ValueError("FashionCLIP necesita una imagen o un texto, pero no ambos")
        if not matches:
            return []

        model, processor = self._get_model()
        paths = [str(catalog.image_path(match["item_id"])) for match in matches]
        candidate_embeddings = self._encode_images(model, processor, paths)

        if image_path is not None:
            query_embeddings = self._encode_images(model, processor, [str(image_path)])
        else:
            query_embeddings = self._encode_text(model, processor, [text])

        scores = candidate_embeddings @ query_embeddings[0]
        reranked = []
        for match, score in zip(matches, scores):
            reranked_match = dict(match)
            reranked_match["rerank_score"] = float(score)
            reranked.append(reranked_match)

        reranked.sort(key=lambda match: match["rerank_score"], reverse=True)
        return reranked

    def _get_model(self) -> tuple[CLIPModel, CLIPProcessor]:
        if self._model is None:
            self._model = CLIPModel.from_pretrained(self.model_name).to(self.device)
            self._model.eval()
            self._processor = CLIPProcessor.from_pretrained(self.model_name)
        if self._processor is None:
            raise RuntimeError("FashionCLIP processor was not initialized")
        return self._model, self._processor

    def _encode_images(
        self,
        model: CLIPModel,
        processor: CLIPProcessor,
        paths: Sequence[str],
    ) -> np.ndarray:
        images = []
        for path in paths:
            with Image.open(path) as opened_image:
                images.append(ImageOps.exif_transpose(opened_image).convert("RGB"))

        embeddings = []
        for start in range(0, len(images), self.batch_size):
            inputs = processor(
                images=images[start : start + self.batch_size],
                return_tensors="pt",
            ).to(self.device)
            with torch.inference_mode():
                features = self._feature_tensor(model.get_image_features(**inputs))
                embeddings.append(features.cpu().numpy())
        return self._normalize(np.concatenate(embeddings, axis=0))

    def _encode_text(
        self,
        model: CLIPModel,
        processor: CLIPProcessor,
        texts: Sequence[str | None],
    ) -> np.ndarray:
        if any(text is None for text in texts):
            raise ValueError("El texto de consulta no puede ser None")
        embeddings = []
        for start in range(0, len(texts), self.batch_size):
            inputs = processor(
                text=list(texts[start : start + self.batch_size]),
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(self.device)
            with torch.inference_mode():
                features = self._feature_tensor(model.get_text_features(**inputs))
                embeddings.append(features.cpu().numpy())
        return self._normalize(np.concatenate(embeddings, axis=0))

    @staticmethod
    def _feature_tensor(output: Any) -> torch.Tensor:
        if isinstance(output, torch.Tensor):
            return output
        features = getattr(output, "pooler_output", None)
        if isinstance(features, torch.Tensor):
            return features
        raise TypeError("FashionCLIP devolvio un output sin tensor de features")

    @staticmethod
    def _normalize(embeddings: np.ndarray) -> np.ndarray:
        if embeddings.ndim != 2 or embeddings.shape[0] == 0:
            raise ValueError("FashionCLIP devolvio embeddings con una forma invalida")
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / np.clip(norms, 1e-12, None)

    @staticmethod
    def _pick_device(requested: str) -> str:
        if requested != "auto":
            return requested
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
