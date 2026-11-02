import argparse
import json
import sys
from io import BytesIO
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import hf_hub_download
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from home_made_rag.config import (
    CONFIG,
    IMAGES_DIR,
    ITEMS_FILE,
    OUTFITS_FILE,
    ROOT_DIR,
)

def download_json(filename):
    path = hf_hub_download(
        repo_id=CONFIG["dataset"]["huggingface_id"],
        filename=filename,
        repo_type="dataset"
    )
    
    with open(path, encoding="utf-8") as file:
        return json.load(file)
    
def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare a subset of Polyvore outfits."
    )
    parser.add_argument(
        "--n",
        type=int,
        default=200,
        help="Number of source outfits to inspect (default: 200)."
    )
    args = parser.parse_args()
    
    if args.n < 1:
        parser.error("--n must be greater than zero")
    
    return args

def main():
    args = parse_args()
    settings = CONFIG["dataset"]
    
    source_outfits = download_json(
        f"{settings['configuration']}/{settings['split']}.json"
    )[:args.n]
    metadata = download_json("polyvore_item_metadata.json")
    
    outfits = {
        str(outfit["set_id"]): [
            str(item["item_id"]) for item in outfit["items"]
        ]
        for outfit in source_outfits
    }
    
    needed_ids = {
        item_id
        for item_ids in outfits.values()
        for item_id in item_ids
    }
    
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    ITEMS_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTFITS_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    dataset = load_dataset(
        settings["huggingface_id"],
        settings["configuration"],
        split=settings["split"],
        streaming=True,
    )
    
    items = {}
    pending_ids = set(needed_ids)
    
    for row in dataset:
        item_id = str(row["item_id"])
        
        if item_id not in pending_ids:
            continue
        
        pending_ids.remove(item_id)
        
        # reject unexpected IDs before using them as filenames
        if not item_id.isdecimal():
            print(f"Skipping invalid item ID: {item_id!r}")
            continue
        
        image_path = IMAGES_DIR / f"{item_id}.jpg"
        
        try:
            item_metadata = metadata[item_id]
            image_data = row["image"]
            
            if image_data["bytes"] is not None:
                image_source = BytesIO(image_data["bytes"])
            else:
                image_source = image_data["path"]
                
            with Image.open(image_source) as image:
                image.convert("RGB").save(image_path, format="JPEG")
            
            items[item_id] = {
                "item_id": item_id,
                "image_path": image_path.relative_to(ROOT_DIR).as_posix(),
                "category": item_metadata["semantic_category"],
                "title": item_metadata["title"],
                "description": item_metadata["description"],
                "available": True,
                "price": None,
                
            }
        except (OSError, ValueError, KeyError, TypeError) as error:
            image_path.unlink(missing_ok=True)
            print(f"Skipping item {item_id}: {error}")
        
        if not pending_ids:
            break
        
    # keep an outfit when every referenced item was saved
    valid_outfits = {
        outfit_id: item_ids
        for outfit_id, item_ids in outfits.items()
        if item_ids and all(item_id in items for item_id in item_ids)
    }
    
    used_ids = {
        item_id
        for item_ids in valid_outfits.values()
        for item_id in item_ids
    }
    
    # remove items saved during this run whose outfits were dropped
    for item_id in items.keys() - used_ids:
        (IMAGES_DIR / f"{item_id}.jpg").unlink(missing_ok=True)
        
    valid_items = [
        item
        for item_id, item in items.items()
        if item_id in used_ids
    ]
    
    with ITEMS_FILE.open("w", encoding="utf-8") as file:
        for item in valid_items:
            file.write(json.dumps(item, ensure_ascii=False) + "\n")
        
    with OUTFITS_FILE.open("w", encoding="utf-8") as file:
        json.dump(valid_outfits, file, ensure_ascii=False, indent=2)
    
    print(f"Write {len(valid_outfits)} outfits.")
    print(f"Write {len(valid_items)} items.")
    print(f"Dropped {len(outfits) - len(valid_outfits)} outfits.")
    
if __name__ == "__main__":
    main()
