from __future__ import annotations

from pathlib import Path
from collections.abc import Iterable

import clip
import torch
from PIL import Image, ImageOps


class ClipEmbedding:
    """Loads OpenAI CLIP once and creates normalized image and text embeddings."""

    def __init__(self, model_name: str = "ViT-B/32") -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load(
            model_name,
            device=self.device,
        )
        self.model.eval()

    def embed_image(self, image_path: str | Path) -> list[float]:
        return self.embed_images([image_path])[0]

    def embed_images(self, paths: Iterable[str | Path]) -> list[list[float]]:
        """Embed one batch of images, preserving input order."""
        tensors = []
        
        for image_path in paths:
            path = Path(image_path)
            
            if not path.is_file():
                raise FileNotFoundError(f"No existe la imagen: {path}")

            with Image.open(path) as opened_image:
                image = ImageOps.exif_transpose(opened_image).convert("RGB")
                tensors.append(self.preprocess(image))

        if not tensors:
            return []

        batch = torch.stack(tensors).to(self.device)

        with torch.inference_mode():
            features = self.model.encode_image(batch)
            features = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)

        return features.cpu().float().tolist()

    def embed_text(self, text: str) -> list[float]:
        tokens = clip.tokenize([text], truncate=True).to(self.device)

        with torch.inference_mode():
            features = self.model.encode_text(tokens)
            features = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)

        return features.cpu().float().tolist()[0]
