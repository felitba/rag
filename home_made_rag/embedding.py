from __future__ import annotations

from pathlib import Path

import clip
import torch
from PIL import Image, ImageOps


class ClipEmbedding:
    """Loads OpenAI CLIP once and creates normalized image embeddings."""

    def __init__(self, model_name: str = "ViT-B/32") -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load(
            model_name,
            device=self.device,
        )
        self.model.eval()

    # TODO: change this...?
    def embed_image(self, image_path: str | Path) -> list[float]:
        path = Path(image_path)
        if not path.is_file():
            raise FileNotFoundError(f"No existe la imagen: {path}")

        with Image.open(path) as opened_image:
            image = ImageOps.exif_transpose(opened_image).convert("RGB")
            tensor = self.preprocess(image).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            features = self.model.encode_image(tensor)
            features = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)

        return features[0].detach().cpu().float().tolist()
