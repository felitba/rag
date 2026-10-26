
import json
from pathlib import Path

import chromadb

PROJECT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_DIR / "config.json"

with CONFIG_PATH.open("r", encoding="utf-8") as file:
    CONFIG = json.load(file)

def resolve_path(key: str) -> Path:
    """Resolve a configured path relative to the project root."""
    path = Path(CONFIG["dataset"][key])

    if not path.is_absolute():
        path = PROJECT_DIR / path

    return path


DATA_DIR = resolve_path("local_data_dir")
IMAGES_DIR = resolve_path("images_dir")
ITEMS_FILE = resolve_path("items_file")
OUTFITS_FILE = resolve_path("outfits_file")

CHROMA_DIR = PROJECT_DIR / CONFIG["vector_store"]["persist_directory"]

print("Project:", CONFIG["project"]["name"])
print("Images:", IMAGES_DIR)
print("ChromaDB:", CHROMA_DIR)
