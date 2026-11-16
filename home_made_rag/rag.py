import json
import logging

from home_made_rag.config import CONFIG

logger = logging.getLogger(__name__)

class RAG:
    def __init__(self, embedding, llm, chroma, catalog, reranker=None, generator=None):
        self.embedding = embedding
        self.llm = llm
        self.chroma = chroma
        self.catalog = catalog
        self.reranker = reranker
        self.generator = generator

    def recommend(self, image_path=None, text=None, max_outfits=None):
        if image_path is not None:
            query_embedding = self.embedding.embed_image(image_path)
        elif text:
            query_embedding = self.embedding.embed_text(text)
        else:
            raise ValueError("Se necesita una imagen o un texto para recomendar")

        matches = self._search(query_embedding, CONFIG["vector_store"]["top_k"])
        if self.reranker is not None:
            matches = self.reranker.rerank(
                matches,
                self.catalog,
                image_path=image_path,
                text=text,
            )
        return self._collect_outfits(
            matches,
            max_outfits if max_outfits is not None else CONFIG["retrieval"]["max_outfits"],
        )

    def recommend_with_generation(self, image_path=None, text=None):
        """Run retrieval once, then generate catalog-grounded suggestions when available."""
        settings = CONFIG.get("generator", {})
        enabled = bool(settings.get("enabled", False))
        has_generator = self.generator is not None
        candidate_limit = (
            settings.get("top_n_outfits", CONFIG["retrieval"]["max_outfits"])
            if enabled and has_generator else None
        )
        retrieved = self.recommend(image_path=image_path, text=text, max_outfits=candidate_limit)

        if not enabled:
            return retrieved, "Outfits recuperados del catálogo.", "disabled"
        if not has_generator:
            return retrieved, "Outfits recuperados del catálogo; generación no disponible.", "unavailable"
        if not retrieved:
            return retrieved, "No encontramos suficientes outfits del catálogo para generar sugerencias.", "insufficient_candidates"

        try:
            generated = self.generator.generate(retrieved, prompt=text)
            if not generated.get("outfits"):
                explanation = generated.get("explanation") or "El catálogo recuperado no ofrece suficientes prendas para crear un conjunto nuevo."
                return [], explanation, "insufficient_candidates"
            recommendations = self._resolve_generated(generated, retrieved)
            if not recommendations:
                raise ValueError("El generador no devolvió combinaciones válidas")
            explanation = generated.get("explanation") or "Sugerencias nuevas basadas en prendas de los outfits recuperados."
            return recommendations, explanation, "generated"
        except Exception:
            logger.exception("Outfit generation failed; returning retrieved recommendations")
            return retrieved, "La generación no estuvo disponible; mostramos los outfits recuperados del catálogo.", "fallback"

    def _resolve_generated(self, generated, candidates):
        candidate_outfits = {outfit["outfit_id"]: outfit for outfit in candidates}
        candidate_items = {
            item["garment_id"]: item
            for outfit in candidates
            for item in outfit["items"]
        }
        source_item_ids = {
            outfit_id: {item["garment_id"] for item in outfit["items"]}
            for outfit_id, outfit in candidate_outfits.items()
        }
        resolved = []
        for draft in generated.get("outfits", [])[: self.generator.max_generated_outfits]:
            garment_ids = draft.get("garment_ids", [])
            source_ids = draft.get("source_outfit_ids", [])
            if (
                not isinstance(garment_ids, list)
                or len(garment_ids) < 2
                or len(set(garment_ids)) != len(garment_ids)
                or not isinstance(source_ids, list)
                or not source_ids
                or any(item_id not in candidate_items for item_id in garment_ids)
                or any(outfit_id not in candidate_outfits for outfit_id in source_ids)
            ):
                logger.warning("Discarding generated outfit with invalid catalog references")
                continue

            cited_item_ids = set().union(*(source_item_ids[outfit_id] for outfit_id in source_ids))
            selected = set(garment_ids)
            if not selected.issubset(cited_item_ids):
                logger.warning("Discarding generated outfit whose garments are outside cited sources")
                continue
            if any(selected == source_item_ids[outfit_id] for outfit_id in candidate_outfits):
                logger.info("Discarding generated outfit that duplicates a retrieved outfit")
                continue

            resolved.append({
                "outfit_id": f"generated_{len(resolved) + 1:02d}",
                "name": draft.get("name") or f"Look {len(resolved) + 1}",
                "description": draft.get("description") or "",
                "explanation": draft.get("explanation") or "",
                "style_tags": draft.get("style_tags", []),
                "source_outfits": source_ids,
                "generated": True,
                "score": None,
                "matched_garment_ids": [],
                "items": [candidate_items[item_id] for item_id in garment_ids],
            })
        return resolved

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
                    score = match.get("rerank_score", 1 - match["distance"])
                    found[outfit_id] = {
                        "outfit_id": outfit_id,
                        "score": round(score, 6),
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
            "color": item.get("color"),
            "style": item.get("style"),
            "price": item.get("price"),
            "image_url": self.catalog.image_url(item_id),
            "_image_abs_path": str(self.catalog.image_path(item_id)),
            "_catalog_metadata": {
                key: item.get(key)
                for key in ("category", "title", "description", "color", "style")
                if item.get(key) not in (None, "")
            },
        }
