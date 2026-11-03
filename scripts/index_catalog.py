import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from home_made_rag.catalog import Catalog
from home_made_rag.chroma_db import ChromaDB
from home_made_rag.config import CONFIG
from home_made_rag.embedding import ClipEmbedding

def main():
    # load and validate catalog
    catalog = Catalog()
    catalog.validate_images()
    item_ids = list(catalog.items)
    print(f"Catalog OK: {len(item_ids)} items, {len(catalog.outfits)} outfits")
    
    # load CLIP
    clip_model = ClipEmbedding()
    print(f"CLIP loaded on {clip_model.device}")
    
    chroma = ChromaDB()
    chroma.reset()
    
    batch_size = CONFIG["embeddings"]["batch_size"]
    
    for start in range(0, len(item_ids), batch_size):
        batch_ids = item_ids[start:start + batch_size]
        paths = [catalog.image_path(item_id) for item_id in batch_ids]
        
        chroma.add_batch(
            ids=batch_ids,
            embeddings=clip_model.embed_images(paths),
            metadatas=[catalog.metadata(item_id) for item_id in batch_ids],
            documents=[catalog.document(item_id) for item_id in batch_ids],
        )
        print(f"[{start + len(batch_ids)}/{len(item_ids)}] indexed")
        
    total = chroma.count()
    if total != len(item_ids):
        raise SystemExit(f"Count mismatch: Chroma has {total}, catalog has {len(item_ids)}")
    
    print(f"Done. Collection has {total} items")

if __name__ == "__main__":
    main()