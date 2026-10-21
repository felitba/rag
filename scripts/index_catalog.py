from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import CHROMA_PATH, CLIP_MODEL, COLLECTION_NAME  # noqa: E402
from app.services.catalog_service import CatalogService  # noqa: E402
from app.services.chroma_service import ChromaService  # noqa: E402
from app.services.clip_service import ClipService  # noqa: E402


def main() -> None:
    catalog = CatalogService(ROOT)

    # Validate all references/images before resetting the existing index.
    catalog.validate_image_files()

    outfits_by_garment: dict[str, list[str]] = defaultdict(list)
    for outfit_id, garment_ids in catalog.outfits.items():
        for garment_id in garment_ids:
            if outfit_id not in outfits_by_garment[garment_id]:
                outfits_by_garment[garment_id].append(outfit_id)

    print(f"Cargando CLIP ({CLIP_MODEL})...")
    clip_service = ClipService(model_name=CLIP_MODEL)

    print("Reconstruyendo colección ChromaDB...")
    chroma_service = ChromaService(
        persist_directory=CHROMA_PATH,
        collection_name=COLLECTION_NAME,
        clip_service=clip_service,
        reset_collection=True,
    )

    for index, garment in enumerate(catalog.garments_list, start=1):
        image_path = catalog.resolve_image_path(garment["image_path"])
        outfit_ids = outfits_by_garment.get(garment["garment_id"], [])

        chroma_service.index_garment(
            garment=garment,
            image_path=image_path,
            outfit_ids=outfit_ids,
        )
        print(
            f"[{index}/{len(catalog.garments_list)}] "
            f"Indexada {garment['garment_id']} -> {outfit_ids}"
        )

    print(
        f"Indexación terminada. Colección '{COLLECTION_NAME}': "
        f"{chroma_service.count()} prendas."
    )


if __name__ == "__main__":
    main()
