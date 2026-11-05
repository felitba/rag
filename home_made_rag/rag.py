import json

from home_made_rag.config import CONFIG

class RAG:
    def __init__(self, embedding, llm, chroma, catalog):
        self.embedding = embedding
        self.llm = llm
        self.chroma = chroma
        self.catalog = catalog

    def recommend(self, image_path=None, text=None):
        if image_path is not None:
            query_embedding = self.embedding.embed_image(image_path)
        elif text:
            query_embedding = self.embedding.embed_text(text)
        else:
            raise ValueError("Se necesita una imagen o un texto para recomendar")

        matches = self._search(query_embedding, CONFIG["vector_store"]["top_k"])
        return self._collect_outfits(matches, CONFIG["retrieval"]["max_outfits"])

    def explain(self, image_path, outfits):
        return self.llm.explain(image_path, outfits)
    
    def _search(self, query_embedding, top_k):
        result = self.chroma.retrieve(query_embedding, top_k=top_k)
        return [
            {
                "item_id": item_id,
                "distance": distance,
                "outfit_ids": json.loads(metadata["outfit_ids_json"]),
            }
            for item_id, distance, metadata in zip(
                result["ids"][0],
                result["distances"][0],
                result["metadatas"][0],
            )
        ]
        
    def _collect_outfits(self, matches, max_outfits):
        if CONFIG["retrieval"]["exclude_query_item"]:
            matches = [match for match in matches if match["distance"] > 1e-4]

        found = {}
        for match in matches:
            for outfit_id in match["outfit_ids"]:
                if outfit_id not in found:
                    found[outfit_id] = {
                        "outfit_id": outfit_id,
                        "score": round(1 - match["distance"], 6),
                        "matched_garment_ids": [],
                    }
                if match["item_id"] not in found[outfit_id]["matched_garment_ids"]:
                    found[outfit_id]["matched_garment_ids"].append(match["item_id"])

        outfits = list(found.values())[:max_outfits]

        for outfit in outfits:
            outfit["items"] = [
                self._item_view(item_id)
                for item_id in self.catalog.outfits[outfit["outfit_id"]]
            ]

        return outfits
    
    def _item_view(self, item_id):
        item = self.catalog.items[item_id]
        return {
            "garment_id": item_id,
            "category": item["category"],
            "title": item.get("title") or None,
            "color": None,
            "style": None,
            "price": item.get("price"),
            "image_url": self.catalog.image_url(item_id),
            "_image_abs_path": str(self.catalog.image_path(item_id)),
        }

