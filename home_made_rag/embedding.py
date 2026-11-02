from __future__ import annotations

from pathlib import Path
from collections.abc import Iterable

import clip
import torch
from PIL import Image, ImageOps
from home_made_rag.config import CONFIG


class ClipEmbedding:
    """Loads OpenAI CLIP once and creates normalized image and text embeddings."""

    def __init__(self, model_name: str | None = None, device: str | None=None) -> None:
        settings = CONFIG["embeddings"]
        self.device = self._pick_device(device or settings["device"])
        self.model, self.preprocess = clip.load(
            model_name or settings["model_name"],
            device=self.device,
        )
        self.model.float()
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
    
    @staticmethod
    def _pick_device(requested: str) -> str:
        if requested != "auto":
            return requested
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
