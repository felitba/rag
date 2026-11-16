import json
import logging

from home_made_rag.config import CONFIG
from home_made_rag.color_metadata import estimate_garment_color
from home_made_rag.hybrid_reranker import HybridGarmentReranker

logger = logging.getLogger(__name__)

class RAG:
    def __init__(
        self,
        embedding,
        chroma,
        catalog,
        generator=None,
        attribute_classifier=None,
        hybrid_reranker=None,
        monitor=None,
        image_preprocessor=None,
    ):
        self.embedding = embedding
        self.chroma = chroma
        self.catalog = catalog
        self.generator = generator
        self.attribute_classifier = attribute_classifier
        self.hybrid_reranker = hybrid_reranker or HybridGarmentReranker()
        self.monitor = monitor
        self.image_preprocessor = image_preprocessor

    def classify_query_image(self, image_path):
        category = None
        if self.attribute_classifier:
            try:
                category = self.attribute_classifier.classify(image_path)
            except Exception:
                logger.exception("Input garment category classification failed")
        return {
            "category": category,
            "color": estimate_garment_color(image_path),
        }

    def compare_retrieved_metadata(self, outfits, query_attributes):
        """Compare detected query attributes with metadata for retrieved garments."""
        if not query_attributes:
            return []

        category = (query_attributes.get("category") or {}).get("label")
        color = (query_attributes.get("color") or {}).get("color")
        garment_ids = list(dict.fromkeys(
            garment_id
            for outfit in outfits
            for garment_id in outfit.get("matched_garment_ids", [])
        ))
        comparisons = []
        for garment_id in garment_ids:
            item = self.catalog.items.get(garment_id, {})
            item_category = item.get("category")
            item_color = item.get("color")
            category_match = (
                self._normalize_category(item_category) == self._normalize_category(category)
                if item_category and category else None
            )
            color_match = (
                str(item_color).strip().lower() == str(color).strip().lower()
                if item_color and color else None
            )
            if category_match is False or color_match is False:
                alignment = "mismatch"
            elif category_match is True and color_match is True:
                alignment = "match"
            elif category_match is True:
                alignment = "category_only"
            else:
                alignment = "unknown"
            comparisons.append({
                "garment_id": garment_id,
                "title": item.get("title") or None,
                "category": item_category,
                "category_match": category_match,
                "query_color": color,
                "metadata_color": item_color,
                "color_match": color_match,
                "alignment": alignment,
            })
        return comparisons

    @staticmethod
    def _normalize_category(value):
        if not value:
            return None
        normalized = str(value).strip().lower()
        aliases = {
            "jewelry": "jewellery",
            "clothes": "clothing",
            "shirt": "tops",
            "blouse": "tops",
            "pants": "bottoms",
            "trousers": "bottoms",
            "coat": "outerwear",
            "jacket": "outerwear",
            "handbag": "bags",
            "sandal": "shoes",
        }
        return aliases.get(normalized, normalized)

    def recommend(
        self,
        image_path=None,
        text=None,
        max_outfits=None,
        query_attributes=None,
        return_query_attributes=False,
        request_id=None,
    ):
        if image_path is not None:
            query_image = (
                self.image_preprocessor.process(image_path)
                if self.image_preprocessor is not None
                else image_path
            )
            query_embedding = self.embedding.embed_image(query_image)
        elif text:
            query_embedding = self.embedding.embed_text(text)
        else:
            raise ValueError("Se necesita una imagen o un texto para recomendar")

        matches = self._search(query_embedding, CONFIG["vector_store"]["top_k"])
        if self.monitor:
            try:
                self.monitor.record_candidates(request_id, "retrieved", matches, self.catalog)
            except Exception:
                logger.exception("Could not record pre-reranking candidates")
        if image_path is not None and query_attributes is None:
            query_attributes = self.classify_query_image(query_image)
        matches = self.hybrid_reranker.rerank(matches, query_attributes)
        if self.monitor:
            try:
                self.monitor.record_candidates(request_id, "reranked", matches, self.catalog)
            except Exception:
                logger.exception("Could not record post-reranking candidates")
        outfits = self._collect_outfits(
            matches,
            max_outfits if max_outfits is not None else CONFIG["retrieval"]["max_outfits"],
        )
        if return_query_attributes:
            return outfits, query_attributes
        return outfits

    def recommend_with_generation(self, image_path=None, text=None, query_attributes=None, request_id=None):
        """Run retrieval once, then generate catalog-grounded suggestions when available."""
        settings = CONFIG.get("generator", {})
        enabled = bool(settings.get("enabled", False))
        has_generator = self.generator is not None
        candidate_limit = (
            settings.get("top_n_outfits", CONFIG["retrieval"]["max_outfits"])
            if enabled and has_generator else None
        )
        retrieved, query_attributes = self.recommend(
            image_path=image_path,
            text=text,
            max_outfits=candidate_limit,
            query_attributes=query_attributes,
            return_query_attributes=True,
            request_id=request_id,
        )

        if not enabled:
            return retrieved, "Outfits recuperados del catálogo.", "disabled", retrieved, query_attributes
        if not has_generator:
            return retrieved, "Outfits recuperados del catálogo; generación no disponible.", "unavailable", retrieved, query_attributes
        if not retrieved:
            return retrieved, "No encontramos suficientes outfits del catálogo para generar sugerencias.", "insufficient_candidates", retrieved, query_attributes

        try:
            generated = self.generator.generate(retrieved, prompt=text, request_id=request_id)
            if not generated.get("outfits"):
                explanation = generated.get("explanation") or "El catálogo recuperado no ofrece suficientes prendas para crear un conjunto nuevo."
                return [], explanation, "insufficient_candidates", retrieved, query_attributes
            recommendations = self._resolve_generated(generated, retrieved)
            if not recommendations:
                raise ValueError("El generador no devolvió combinaciones válidas")
            explanation = generated.get("explanation") or "Sugerencias nuevas basadas en prendas de los outfits recuperados."
            return recommendations, explanation, "generated", retrieved, query_attributes
        except Exception:
            logger.exception("Outfit generation failed; returning retrieved recommendations")
            return retrieved, "La generación no estuvo disponible; mostramos los outfits recuperados del catálogo.", "fallback", retrieved, query_attributes

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

    def _search(self, query_embedding, top_k):
        result = self.chroma.retrieve(query_embedding, top_k=top_k)
        return [
            {
                "item_id": item_id,
                "distance": distance,
                "category": metadata.get("category"),
                "color": metadata.get("color"),
                "color_confidence": metadata.get("color_confidence"),
                "color_r": metadata.get("color_r"),
                "color_g": metadata.get("color_g"),
                "color_b": metadata.get("color_b"),
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
                for key in (
                    "category", "categories", "catgeories",
                    "title", "description", "related", "url_name", "color", "style",
                )
                if item.get(key) not in (None, "")
            },
        }
