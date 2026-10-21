from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


def _path_from_env(name: str, default: str) -> Path:
    value = Path(os.getenv(name, default)).expanduser()
    return value if value.is_absolute() else (PROJECT_ROOT / value).resolve()


CHROMA_PATH = _path_from_env("CHROMA_PATH", "storage/chroma")
DATA_PATH = PROJECT_ROOT / "app" / "data"
CATALOG_PATH = PROJECT_ROOT / "catalog"
GARMENTS_DIR = CATALOG_PATH / "garments"
UPLOADS_DIR = PROJECT_ROOT / "storage" / "uploads"

COLLECTION_NAME = os.getenv("COLLECTION_NAME", "fashion_garments")
CLIP_MODEL = os.getenv("CLIP_MODEL", "ViT-B/32")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

TOP_K_GARMENTS = int(os.getenv("TOP_K_GARMENTS", "30"))
MAX_OUTFITS = int(os.getenv("MAX_OUTFITS", "5"))
MAX_IMAGES_PER_OUTFIT_FOR_LLM = int(
    os.getenv("MAX_IMAGES_PER_OUTFIT_FOR_LLM", "2")
)
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "10"))
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
MAX_IMAGE_PIXELS = 25_000_000

# Ensure required local directories exist.
CHROMA_PATH.mkdir(parents=True, exist_ok=True)
GARMENTS_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
