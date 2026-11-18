import json
import logging
import numpy as np


from home_made_rag.config import CONFIG
from home_made_rag.color_metadata import estimate_garment_color
from home_made_rag.context_precision import evaluate_context_precision
from home_made_rag.hybrid_reranker import HybridGarmentReranker

logger = logging.getLogger(__name__)

OUT_OF_SCOPE_MESSAGE = (
    "Solo puedo ayudarte a armar outfits con las prendas del catálogo. "
    "Contame qué prenda, estilo u ocasión tenés en mente."
)


class OutOfScopeRequest(Exception):
    """The user's text is not a request about clothing."""


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
        rewriter=None,
    ):
        self.embedding = embedding
        self.chroma = chroma
        self.catalog = catalog
        self.generator = generator
        self.attribute_classifier = attribute_classifier
        self.hybrid_reranker = hybrid_reranker or HybridGarmentReranker()
        self.monitor = monitor
        self.image_preprocessor = image_preprocessor
        self.rewriter = rewriter
        self.categories = sorted({item["category"] for item in catalog.items.values()})


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
        context_precision_result=None,
    ):
        if image_path is not None:
            query_image = (
                self.image_preprocessor.process(image_path)
                if self.image_preprocessor is not None
                else image_path
            )
            if self.monitor and request_id:
                try:
                    self.monitor.record_query_images(request_id, image_path, query_image)
                except Exception:
                    logger.exception("Could not save query images for monitoring")
            query_embedding = self.embedding.embed_image(query_image)
            matches = self._search(query_embedding, CONFIG["vector_store"]["top_k"])
            if query_attributes is None:
                query_attributes = self.classify_query_image(query_image)
        elif not text:
            raise ValueError("Se necesita una imagen o un texto para recomendar")
        
        text_queries, rewriting_trace = [], None
        if text:
            owned_category = ((query_attributes or {}).get("category") or {}).get("label")
            text_queries, rewriting_trace = self._rewrite(
                text, owned_category, request_id, has_photo=image_path is not None,
            )
            if image_path is None:
                if not text_queries:
                    self._record_rewriting(request_id, rewriting_trace, text_queries, [])
                    raise OutOfScopeRequest(text)
                matches = self._search_text(text_queries)

        retrieved_precision = evaluate_context_precision(matches, query_attributes, self.catalog)
        matches = retrieved_precision["annotated_matches"]
        if self.monitor:
            try:
                self.monitor.record_candidates(request_id, "retrieved", matches, self.catalog)
            except Exception:
                logger.exception("Could not record pre-reranking candidates")
        matches = self.hybrid_reranker.rerank(matches, query_attributes)
        reranked_precision = evaluate_context_precision(matches, query_attributes, self.catalog)
        matches = reranked_precision["annotated_matches"]
        context_metrics = {
            "retrieved": retrieved_precision["score"],
            "reranked": reranked_precision["score"],
            "available": retrieved_precision["available"],
            "k": retrieved_precision["k"],
            "relevant_count": retrieved_precision["relevant_count"],
            "criteria": retrieved_precision["criteria"],
            "color_similarity_threshold": retrieved_precision["color_similarity_threshold"],
        }
        if context_precision_result is not None:
            context_precision_result.update(context_metrics)
        if self.monitor:
            try:
                self.monitor.set_context_precision(request_id, context_metrics)
            except Exception:
                logger.exception("Could not record context precision")
        if self.monitor:
            try:
                self.monitor.record_candidates(request_id, "reranked", matches, self.catalog)
            except Exception:
                logger.exception("Could not record post-reranking candidates")
        
        limit = max_outfits if max_outfits is not None else CONFIG["retrieval"]["max_outfits"]
        if text_queries:
            candidates = self._collect_outfits(matches, None)
            outfits = self._rank_by_text(candidates, text_queries, has_photo=image_path is not None)[:limit]
        else:
            outfits = self._collect_outfits(matches, limit)
        if rewriting_trace is not None:
            self._record_rewriting(request_id, rewriting_trace, text_queries, outfits)
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
        context_precision = {}
        try:
            retrieved, query_attributes = self.recommend(
                image_path=image_path,
                text=text,
                max_outfits=candidate_limit,
                query_attributes=query_attributes,
                return_query_attributes=True,
                request_id=request_id,
                context_precision_result=context_precision,
            )
        except OutOfScopeRequest:
            return [], OUT_OF_SCOPE_MESSAGE, "out_of_scope", [], None, context_precision
        self._record_generator_outfits(request_id, "generator_input", retrieved)

        if not enabled:
            self._record_generator_outfits(request_id, "generator_output", retrieved)
            return retrieved, "Outfits recuperados del catálogo.", "disabled", retrieved, query_attributes, context_precision
        if not has_generator:
            self._record_generator_outfits(request_id, "generator_output", retrieved)
            return retrieved, "Outfits recuperados del catálogo; generación no disponible.", "unavailable", retrieved, query_attributes, context_precision
        if not retrieved:
            self._record_generator_outfits(request_id, "generator_output", retrieved)
            return retrieved, "No encontramos suficientes outfits del catálogo para generar sugerencias.", "insufficient_candidates", retrieved, query_attributes, context_precision

        try:
            resolved_recommendations = []

            def validate_generated_output(generated):
                if not generated.get("outfits"):
                    raise ValueError("El proveedor no propuso outfits utilizables")
                resolved = self._resolve_generated(generated, retrieved)
                if not resolved:
                    raise ValueError("El proveedor no devolvió combinaciones válidas")
                resolved_recommendations.extend(resolved)

            generated = self.generator.generate(
                retrieved,
                prompt=text,
                request_id=request_id,
                validate_output=validate_generated_output,
            )
            recommendations = resolved_recommendations
            self._record_generator_outfits(request_id, "generator_output", recommendations)
            explanation = generated.get("explanation") or "Sugerencias nuevas basadas en prendas de los outfits recuperados."
            return recommendations, explanation, "generated", retrieved, query_attributes, context_precision
        except Exception:
            logger.exception("Outfit generation failed; returning retrieved recommendations")
            self._record_generator_outfits(request_id, "generator_output", retrieved)
            return retrieved, "La generación no estuvo disponible; mostramos los outfits recuperados del catálogo.", "fallback", retrieved, query_attributes, context_precision

    def _record_generator_outfits(self, request_id, stage, outfits):
        if not self.monitor:
            return
        try:
            self.monitor.record_generator_outfits(request_id, stage, outfits, self.catalog)
        except Exception:
            logger.exception("Could not record %s outfits for monitoring", stage)

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
            rejection_reason = None
            # if not isinstance(garment_ids, list):
            #     rejection_reason = "garment_ids_not_list"
            # elif len(garment_ids) < 2:
            #     rejection_reason = "fewer_than_two_garments"
            # elif len(set(garment_ids)) != len(garment_ids):
            #     rejection_reason = "duplicate_garment_within_look"
            # elif not isinstance(source_ids, list):
            #     rejection_reason = "source_outfit_ids_not_list"
            # elif not source_ids:
            #     rejection_reason = "missing_source_outfits"
            # elif len(set(source_ids)) != len(source_ids):
            #     rejection_reason = "duplicate_source_outfit_id"
            # elif any(item_id not in candidate_items for item_id in garment_ids):
            #     rejection_reason = "garment_id_outside_candidates"
            # elif any(outfit_id not in candidate_outfits for outfit_id in source_ids):
            #     rejection_reason = "source_outfit_id_outside_candidates"
            # if rejection_reason:
            #     logger.warning("Discarding generated outfit reason=%s", rejection_reason)
            #     continue

            cited_item_ids = set().union(*(source_item_ids[outfit_id] for outfit_id in source_ids))
            selected = set(garment_ids)
            matching_existing_outfits = [
                outfit_id
                for outfit_id, items in source_item_ids.items()
                if selected == items
            ]
            is_existing_outfit = bool(matching_existing_outfits)
            if not selected.issubset(cited_item_ids):
                logger.warning(
                    "Discarding generated outfit reason=garment_not_in_cited_source_outfit"
                )
                continue
            # if not is_existing_outfit and len(set(source_ids)) < 2:
            #     logger.warning(
            #         "Discarding generated outfit reason=fewer_than_two_source_outfits_for_new_combination"
            #     )
            #     continue
            contributing_sources = []
            for outfit_id in source_ids:
                other_source_items = set().union(*(
                    source_item_ids[other_id]
                    for other_id in source_ids
                    if other_id != outfit_id
                ))
                exclusive_selected_items = selected.intersection(
                    source_item_ids[outfit_id] - other_source_items
                )
                if exclusive_selected_items:
                    contributing_sources.append(outfit_id)
            if not contributing_sources:
                logger.warning(
                    "Discarding generated outfit reason=no_contributing_source_outfits"
                )
                continue
            # if not is_existing_outfit and len(contributing_sources) < 2:
            #     logger.warning(
            #         "Discarding generated outfit that does not combine garments from at least two source outfits"
            #     )
            #     continue
            # Garments may recur across different generated looks; only duplicates
            # within this one look are rejected above.
            if is_existing_outfit:
                resolved_sources = [oid for oid in matching_existing_outfits if oid in source_ids]
                if not resolved_sources:
                    resolved_sources = matching_existing_outfits[:1]
                source_outfit_id = resolved_sources[0]
                source_outfit = candidate_outfits.get(source_outfit_id, {})
                resolved.append({
                    "outfit_id": source_outfit_id,
                    "name": draft.get("name") or source_outfit.get("name") or source_outfit_id.replace("_", " "),
                    "description": draft.get("description") or source_outfit.get("description") or "",
                    "explanation": draft.get("explanation") or "",
                    "style_tags": draft.get("style_tags", source_outfit.get("style_tags", [])),
                    "source_outfits": resolved_sources,
                    "generated": False,
                    "is_existing_outfit": True,
                    "score": source_outfit.get("score"),
                    "matched_garment_ids": source_outfit.get("matched_garment_ids", []),
                    "items": [candidate_items[item_id] for item_id in garment_ids],
                })
            else:
                generated_count = sum(1 for item in resolved if item.get("generated"))
                resolved.append({
                    "outfit_id": f"generated_{generated_count + 1:02d}",
                    "name": draft.get("name") or f"Look {len(resolved) + 1}",
                    "description": draft.get("description") or "",
                    "explanation": draft.get("explanation") or "",
                    "style_tags": draft.get("style_tags", []),
                    "source_outfits": contributing_sources,
                    "generated": True,
                    "is_existing_outfit": False,
                    "score": None,
                    "matched_garment_ids": [],
                    "items": [candidate_items[item_id] for item_id in garment_ids],
                })
        return resolved
    
    def _rewrite(self, text, owned_category, request_id, has_photo):
        """Pre-retrieval: turn the user's text into embedded English garment queries."""
        queries, source, details = None, "original_text", {}
        if self.rewriter is not None:
            try:
                queries = self.rewriter.rewrite(
                    text, self.categories, owned_category=owned_category, request_id=request_id,
                    has_photo=has_photo, details=details,
                )
                source = "rewritten"
            except Exception as error:
                details["error"] = f"{type(error).__name__}: {error}"
                logger.exception("Query rewriting failed; searching with the original text")
        if queries is None:
            queries = [{"text": text, "category": None}]
        for query in queries:
            query["embedding"] = self.embedding.embed_text(query["text"])
        trace = {
            "original": text,
            "source": source,
            "in_scope": bool(queries),
            "has_photo": has_photo,
            "owned_category": owned_category,
            **details,
        }
        return queries, trace

    def _garment_label(self, garment_id):
        item = self.catalog.items[garment_id]
        return {
            "garment_id": garment_id,
            "category": item["category"],
            "title": item.get("title") or item.get("url_name") or None,
            "image_url": self.catalog.image_url(garment_id),
        }

    def _record_rewriting(self, request_id, trace, text_queries, outfits):
        """Save every step of the text path so the monitor can show it."""
        if not self.monitor or not request_id:
            return
        settings = CONFIG.get("query_rewriting", {})
        trace = dict(trace)
        trace["queries"] = [
            {key: value for key, value in query.items() if key != "embedding"}
            for query in text_queries
        ]
        trace["photo_weight"] = settings.get("photo_weight", 0.5)
        trace["text_weight"] = settings.get("text_weight", 0.5)
        trace["ranking"] = [
            {
                "outfit_id": outfit["outfit_id"],
                "score": outfit.get("score"),
                "text_score": outfit.get("text_score"),
                "photo_score": outfit.get("photo_score"),
                "matches": [
                    {**match, **(self._garment_label(match["garment_id"]) if match["garment_id"] else {})}
                    for match in outfit.get("text_matches", [])
                ],
            }
            for outfit in outfits
        ]
        try:
            self.monitor.set_query_rewriting(request_id, trace)
        except Exception:
            logger.exception("Could not record the query rewriting trace")

    def _search_text(self, text_queries):
        """One CLIP text search per rewritten query; merged, best distance per garment."""
        use_filter = CONFIG.get("query_rewriting", {}).get("category_filter", True)
        best = {}
        for query in text_queries:
            where = {"category": query["category"]} if use_filter and query["category"] else None
            found = self._search(query["embedding"], CONFIG["vector_store"]["top_k"], where=where)
            query["category_filter"] = bool(where)
            query["match_count"] = len(found)
            query["top_matches"] = [
                {**self._garment_label(match["item_id"]), "similarity": round(1 - match["distance"], 4)}
                for match in found[:3]
            ]
            for match in found:
                known = best.get(match["item_id"])
                if known is None or match["distance"] < known["distance"]:
                    best[match["item_id"]] = match
        return sorted(best.values(), key=lambda match: match["distance"])

    def _rank_by_text(self, outfits, text_queries, has_photo):
        """Score each candidate outfit by how well its garments cover the text queries."""
        if not outfits:
            return outfits
        garment_ids = {item["garment_id"] for outfit in outfits for item in outfit["items"]}
        vectors = self.chroma.get_embeddings(sorted(garment_ids))

        for outfit in outfits:
            similarities = []
            outfit["text_matches"] = []
            for query in text_queries:
                best_id, best_similarity, seen = None, 0.0, []
                for item in outfit["items"]:
                    if query["category"] and item["category"] != query["category"]:
                        continue
                    similarity = float(np.dot(query["embedding"], vectors[item["garment_id"]]))
                    seen.append(similarity)
                    if similarity > best_similarity:
                        best_id, best_similarity = item["garment_id"], similarity
                if query["category"] or not seen:
                    similarities.append(best_similarity)
                    rule = "mejor prenda de la categoría"
                else:
                    similarities.append(sum(seen) / len(seen))
                    rule = "promedio de todas las prendas"
                outfit["text_matches"].append({
                    "query": query["text"],
                    "garment_id": best_id,
                    "best_similarity": round(best_similarity, 4),
                    "score": round(similarities[-1], 4),
                    "rule": rule,
                })
                if best_id and best_id not in outfit["matched_garment_ids"]:
                    outfit["matched_garment_ids"].append(best_id)
            outfit["text_score"] = round(sum(similarities) / len(similarities), 6)

        if has_photo:
            settings = CONFIG.get("query_rewriting", {})
            photo = self._min_max([outfit["score"] for outfit in outfits])
            words = self._min_max([outfit["text_score"] for outfit in outfits])
            for outfit, photo_score, text_score in zip(outfits, photo, words):
                outfit["photo_score"] = outfit["score"]
                outfit["score"] = round(
                    settings.get("photo_weight", 0.5) * photo_score
                    + settings.get("text_weight", 0.5) * text_score,
                    6,
                )
        else:
            for outfit in outfits:
                outfit["score"] = outfit["text_score"]
        return sorted(outfits, key=lambda outfit: outfit["score"], reverse=True)

    @staticmethod
    def _min_max(values):
        low, high = min(values), max(values)
        if high - low < 1e-9:
            return [1.0 for _ in values]
        return [(value - low) / (high - low) for value in values]


    def _search(self, query_embedding, top_k,where=None):
        result = self.chroma.retrieve(query_embedding, top_k=top_k, where=where)
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
