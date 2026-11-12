import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from home_made_rag.catalog import Catalog
from home_made_rag.config import CONFIG
from home_made_rag.expansion import QueryExpander


def main():
    parser = argparse.ArgumentParser(
        description="Run the vision-based query expansion without touching ChromaDB.",
    )
    parser.add_argument("--image", type=Path, help="Path to a local garment image.")
    parser.add_argument("--item-id", help="Catalog item ID to use as the image.")
    parser.add_argument("--text", help="Optional text description from the user.")
    args = parser.parse_args()

    catalog = Catalog()
    image_path = args.image

    if args.item_id:
        image_path = catalog.image_path(args.item_id)

    if image_path is None and not (args.text or "").strip():
        parser.error("Provide --image, --item-id or --text.")

    settings = CONFIG["query_expansion"]
    expander = QueryExpander(
        model_name=settings["model_name"],
        temperature=settings["temperature"],
        num_complements=settings["num_complements"],
        image_detail=settings["image_detail"],
        image_max_side=settings["image_max_side"],
        max_output_tokens=settings.get("max_output_tokens", 800),
    )

    categories = sorted({
        item.get("category")
        for item in catalog.items.values()
        if item.get("category")
    })

    result = expander.expand(
        image_path=image_path,
        text=args.text,
        catalog_categories=categories,
    )

    print(json.dumps(result.model_dump(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
