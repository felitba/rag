
import json
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent # .../rag/home_made_rag
ROOT_DIR = PACKAGE_DIR.parent                 # .../rag
CONFIG_PATH = PACKAGE_DIR / "config.json"

with CONFIG_PATH.open("r", encoding="utf-8") as file:
    CONFIG = json.load(file)

def resolve_path(relative: str) -> Path:
    """Turn a path from config.json into an absolute path under the repo root."""
    path = Path(relative)

    if not path.is_absolute():
        path = ROOT_DIR / path

    return path


DATA_DIR = resolve_path(CONFIG["dataset"]["local_data_dir"])
IMAGES_DIR = resolve_path(CONFIG["dataset"]["images_dir"])
ITEMS_FILE = resolve_path(CONFIG["dataset"]["items_file"])
OUTFITS_FILE = resolve_path(CONFIG["dataset"]["outfits_file"])
CHROMA_DIR = resolve_path(CONFIG["vector_store"]["persist_directory"])
LOG_FILE = resolve_path(CONFIG["project"]["log_file"])
