import json
import logging
import time
from contextlib import contextmanager

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


@contextmanager
def timed(timings, stage):
    """Add the milliseconds spent inside the `with` block to timings[stage]."""
    started = time.perf_counter()
    try:
        yield
    finally:
        if timings is not None:
            elapsed_ms = (time.perf_counter() - started) * 1000
            timings[stage] = round(timings.get(stage, 0.0) + elapsed_ms, 1)


# Categories where a look can hold one garment only.
SINGLE_GARMENT_SLOTS = ("shoes", "bottoms", "all-body", "outerwear", "bags", "hats", "sunglasses", "scarves")


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
        compatibility_scorer=None,
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
        self.compatibility_scorer = compatibility_scorer
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
        timings=None,
        trace=None,
    ):
        if image_path is not None:
            with timed(timings, "preprocessing"):
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
            with timed(timings, "embedding"):
                query_embedding = self.embedding.embed_image(query_image)
            with timed(timings, "retrieval"):
                matches = self._search(query_embedding, CONFIG["vector_store"]["top_k"])
            if query_attributes is None:
                with timed(timings, "attributes"):
                    query_attributes = self.classify_query_image(query_image)
        elif not text:
            raise ValueError("Se necesita una imagen o un texto para recomendar")
        
        text_queries, rewriting_trace = [], None
        if text:
            owned_category = ((query_attributes or {}).get("category") or {}).get("label")
            with timed(timings, "rewriting"):
                text_queries, rewriting_trace = self._rewrite(
                    text, owned_category, request_id, has_photo=image_path is not None,
                )
            if image_path is None:
                if not text_queries:
                    self._record_rewriting(request_id, rewriting_trace, text_queries, [])
                    raise OutOfScopeRequest(text)
                with timed(timings, "retrieval"):
                    matches = self._search_text(text_queries)

        retrieved_precision = evaluate_context_precision(matches, query_attributes, self.catalog)
        matches = retrieved_precision["annotated_matches"]
        if self.monitor:
            try:
                self.monitor.record_candidates(request_id, "retrieved", matches, self.catalog)
            except Exception:
                logger.exception("Could not record pre-reranking candidates")
        with timed(timings, "rerank"):
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
        with timed(timings, "outfits"):
            if text_queries:
                candidates = self._collect_outfits(matches, None)
                outfits = self._rank_by_text(candidates, text_queries, has_photo=image_path is not None)[:limit]
            else:
                candidates = self._collect_outfits(matches, None)
                outfits = candidates[:limit]
        if trace is not None:
            trace["outfit_hop"] = self._outfit_hop_trace(
                candidates, outfits, limit,
                has_text=bool(text_queries), has_photo=image_path is not None,
            )
        if rewriting_trace is not None:
            self._record_rewriting(request_id, rewriting_trace, text_queries, outfits)
        if return_query_attributes:
            return outfits, query_attributes
        return outfits

    def recommend_with_generation(self, image_path=None, text=None, query_attributes=None, request_id=None):
        """Run the whole pipeline and save how long each stage took, even if it fails."""
        timings = {}
        # Everything the monitor needs to explain this request: the settings it ran
        # with, how garments became outfits, and what the generator proposed.
        trace = {"config": self._config_snapshot()}
        try:
            return self._recommend_with_generation(image_path, text, query_attributes, request_id, timings, trace)
        finally:
            if self.monitor and request_id:
                try:
                    self.monitor.set_stage_timings(request_id, timings)
                    self.monitor.set_pipeline_trace(request_id, trace)
                except Exception:
                    logger.exception("Could not record stage timings or pipeline trace")

    def _config_snapshot(self):
        """The settings that shape a result, saved per request so runs can be compared."""
        generator = CONFIG.get("generator", {})
        rewriting = CONFIG.get("query_rewriting", {})
        reranker = self.hybrid_reranker
        return {
            "image_preprocessing": bool(self.image_preprocessor is not None and getattr(self.image_preprocessor, "enabled", True)),
            "category_classifier": self.attribute_classifier is not None,
            "query_rewriting": self.rewriter is not None,
            "category_filter": bool(rewriting.get("category_filter", True)),
            "photo_weight": rewriting.get("photo_weight", 0.5),
            "text_weight": rewriting.get("text_weight", 0.5),
            "top_k": CONFIG["vector_store"]["top_k"],
            "max_outfits": CONFIG["retrieval"]["max_outfits"],
            "rerank_weights": dict(reranker.weights),
            "color_tone_weight": reranker.color_tone_weight,
            "color_coverage_weight": reranker.color_coverage_weight,
            "color_delta_e_scale": reranker.color_delta_e_scale,
            "context_precision_color_threshold": CONFIG.get("context_precision", {}).get("color_similarity_threshold"),
            "generator": bool(generator.get("enabled", False)) and self.generator is not None,
            "generator_model": getattr(self.generator, "model", None) or generator.get("model"),
            "top_n_outfits": generator.get("top_n_outfits"),
            "max_generated_outfits": generator.get("max_generated_outfits"),
            "generator_uses_images": bool(generator.get("use_images", False)),
            "type_aware_compatibility": (
                self.compatibility_scorer.status()
                if self.compatibility_scorer is not None
                else {"enabled": False, "available": False, "reason": "not configured"}
            ),
        }

    @staticmethod
    def _outfit_hop_trace(candidates, kept, limit, has_text, has_photo):
        """How retrieved garments turned into the outfits passed on."""
        if has_text and has_photo:
            rule = "photo_and_text"
        elif has_text:
            rule = "text_coverage"
        else:
            rule = "first_match"
        kept_ids = {outfit["outfit_id"] for outfit in kept}
        return {
            "rule": rule,
            "outfits_found": len(candidates),
            "outfits_kept": len(kept),
            "limit": limit,
            "outfits": [
                {
                    "outfit_id": outfit["outfit_id"],
                    "kept": outfit["outfit_id"] in kept_ids,
                    "score": outfit.get("score"),
                    "photo_score": outfit.get("photo_score"),
                    "text_score": outfit.get("text_score"),
                    "garment_count": len(outfit.get("items", [])),
                    "matches": outfit.get("match_details", []),
                }
                # Kept outfits first, in their final order; then the best ones left out.
                for outfit in [*kept, *sorted(
                    (o for o in candidates if o["outfit_id"] not in kept_ids),
                    key=lambda o: o.get("score") or 0, reverse=True,
                )[:10]]
            ],
        }

    def _recommend_with_generation(self, image_path, text, query_attributes, request_id, timings, trace):
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
                timings=timings,
                trace=trace,
            )
        except OutOfScopeRequest:
            return [], OUT_OF_SCOPE_MESSAGE, "out_of_scope", [], None, context_precision
        self._evaluate_outfits(retrieved, trace, timings, "generator_input")
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
            attempts = trace.setdefault("generation", {"attempts": []})["attempts"]

            def validate_generated_output(generated):
                # Attach the verdicts to the provider attempt that produced these drafts.
                drafts = []
                if attempts:
                    attempts[-1]["drafts"] = drafts
                if not generated.get("outfits"):
                    raise ValueError("El proveedor no propuso outfits utilizables")
                resolved = self._resolve_generated(generated, retrieved, report=drafts)
                if not resolved:
                    raise ValueError("El proveedor no devolvió combinaciones válidas")
                resolved_recommendations.extend(resolved)

            with timed(timings, "generation"):
                generated = self.generator.generate(
                    retrieved,
                    prompt=text,
                    request_id=request_id,
                    validate_output=validate_generated_output,
                    attempt_log=attempts,
                )
            recommendations = resolved_recommendations
            self._evaluate_outfits(recommendations, trace, timings, "generator_output")
            self._record_generator_outfits(request_id, "generator_output", recommendations)
            explanation = generated.get("explanation") or "Sugerencias nuevas basadas en prendas de los outfits recuperados."
            return recommendations, explanation, "generated", retrieved, query_attributes, context_precision
        except Exception:
            logger.exception("Outfit generation failed; returning retrieved recommendations")
            self._evaluate_outfits(retrieved, trace, timings, "generator_output")
            self._record_generator_outfits(request_id, "generator_output", retrieved)
            return retrieved, "La generación no estuvo disponible; mostramos los outfits recuperados del catálogo.", "fallback", retrieved, query_attributes, context_precision

    def _record_generator_outfits(self, request_id, stage, outfits):
        if not self.monitor:
            return
        try:
            self.monitor.record_generator_outfits(request_id, stage, outfits, self.catalog)
        except Exception:
            logger.exception("Could not record %s outfits for monitoring", stage)

    def _evaluate_outfits(self, outfits, trace, timings, stage):
        """Attach type-aware scores to each outfit and persist a compact trace."""
        if self.compatibility_scorer is None:
            return
        try:
            with timed(timings, "compatibility"):
                reports = self.compatibility_scorer.evaluate_outfits(outfits, self.catalog)
            for outfit, report in zip(outfits, reports):
                outfit["type_aware_compatibility"] = report
            compatibility_trace = trace.setdefault("compatibility", {
                "model": self.compatibility_scorer.status(),
                "stages": {},
            })
            compatibility_trace["stages"][stage] = [
                {
                    "outfit_id": outfit.get("outfit_id"),
                    "status": report.get("status"),
                    "compatibility_score": report.get("compatibility_score"),
                    "compatibility_similarity": report.get("compatibility_similarity"),
                    "mean_distance": report.get("mean_distance"),
                    "pair_count": report.get("pair_count", 0),
                }
                for outfit, report in zip(outfits, reports)
            ]
        except Exception as error:
            logger.exception("Could not evaluate outfits with type-aware compatibility")
            report = {
                "status": "error",
                "model": getattr(self.compatibility_scorer, "model_label", "Type-aware embeddings"),
                "reason": f"{type(error).__name__}: {error}",
                "pair_count": 0,
            }
            for outfit in outfits:
                outfit["type_aware_compatibility"] = report.copy()
            trace.setdefault("compatibility", {}).update({
                "model": self.compatibility_scorer.status(),
                "error": report["reason"],
            })

    def _resolve_generated(self, generated, candidates, report=None):
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
        # Garments that retrieval matched to the query, in any retrieved outfit.
        query_matched_ids = {
            garment_id
            for outfit in candidates
            for garment_id in outfit.get("matched_garment_ids", [])
        }
        resolved = []
        verdicts = []
        for draft in generated.get("outfits", []):
            garment_ids = draft.get("garment_ids", [])
            source_ids = draft.get("source_outfit_ids", [])
            # One line per draft for the monitor: what the model proposed and the verdict.
            verdict = {
                "name": draft.get("name"),
                "garment_ids": garment_ids if isinstance(garment_ids, list) else [],
                "cited_source_outfits": source_ids if isinstance(source_ids, list) else [],
                "accepted": False,
                "reason": None,
            }
            if report is not None:
                report.append(verdict)
            if isinstance(garment_ids, list) and isinstance(source_ids, list):
                # The sources follow from the garments, so they are computed, not trusted:
                # keep the model's valid citations (in order, no repeats) and add the
                # retrieved outfit of any garment they leave uncovered.
                cited = [oid for oid in dict.fromkeys(source_ids) if oid in candidate_outfits]
                covered = set().union(*(source_item_ids[oid] for oid in cited))
                for item_id in garment_ids:
                    if item_id in covered:
                        continue
                    for oid, items in source_item_ids.items():
                        if item_id in items:
                            cited.append(oid)
                            covered |= items
                            break
                source_ids = cited
            rejection_reason = None
            if not isinstance(garment_ids, list):
                rejection_reason = "garment_ids_not_list"
            elif len(garment_ids) < 2:
                rejection_reason = "fewer_than_two_garments"
            elif len(set(garment_ids)) != len(garment_ids):
                rejection_reason = "duplicate_garment_within_look"
            elif not isinstance(source_ids, list):
                rejection_reason = "source_outfit_ids_not_list"
            elif not source_ids:
                rejection_reason = "missing_source_outfits"
            elif len(set(source_ids)) != len(source_ids):
                rejection_reason = "duplicate_source_outfit_id"
            elif any(item_id not in candidate_items for item_id in garment_ids):
                rejection_reason = "garment_id_outside_candidates"
            elif any(outfit_id not in candidate_outfits for outfit_id in source_ids):
                rejection_reason = "source_outfit_id_outside_candidates"
            if rejection_reason:
                logger.warning("Discarding generated outfit reason=%s", rejection_reason)
                verdict["reason"] = rejection_reason
                continue

            cited_item_ids = set().union(*(source_item_ids[outfit_id] for outfit_id in source_ids))
            selected = set(garment_ids)
            matching_existing_outfits = [
                outfit_id
                for outfit_id, items in source_item_ids.items()
                if selected <= items  # the whole outfit or a trimmed version of it
            ]
            is_existing_outfit = bool(matching_existing_outfits)
            if not selected.issubset(cited_item_ids):
                logger.warning(
                    "Discarding generated outfit reason=garment_not_in_cited_source_outfit"
                )
                verdict["reason"] = "garment_not_in_cited_source_outfit"
                continue
            if not is_existing_outfit and len(set(source_ids)) < 2:
                logger.warning(
                    "Discarding generated outfit reason=fewer_than_two_source_outfits_for_new_combination"
                )
                verdict["reason"] = "fewer_than_two_source_outfits_for_new_combination"
                continue
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
                verdict["reason"] = "no_contributing_source_outfits"
                continue
            if not is_existing_outfit and len(contributing_sources) < 2:
                logger.warning(
                    "Discarding generated outfit that does not combine garments from at least two source outfits"
                )
                verdict["reason"] = "fewer_than_two_contributing_source_outfits"
                continue
            if is_existing_outfit:
                resolved_sources = [oid for oid in matching_existing_outfits if oid in source_ids]
                if not resolved_sources:
                    resolved_sources = matching_existing_outfits[:1]
            else:
                resolved_sources = contributing_sources

            # What changed against the real outfit is a fact, so it is computed here
            # instead of taken from the model's own account (changes_from_source).
            omitted_items = []
            if not is_existing_outfit:
                look_kind = "mix"
            elif selected == source_item_ids[resolved_sources[0]]:
                look_kind = "existing"
            else:
                look_kind = "trimmed"
                omitted_items = [
                    {
                        "garment_id": item["garment_id"],
                        "category": item.get("category"),
                        "color": item.get("color"),
                        "image_url": item.get("image_url"),
                    }
                    for item in candidate_outfits[resolved_sources[0]]["items"]
                    if item["garment_id"] not in selected
                ]

            if look_kind == "mix":
                # A mix is the model's own judgement, so its composition is checked:
                # one garment per body slot (jewellery, accessories and tops may repeat).
                slots = [candidate_items[item_id].get("category") for item_id in garment_ids]
                repeated = sorted(
                    slot for slot in SINGLE_GARMENT_SLOTS if slots.count(slot) > 1
                )
                if repeated:
                    logger.warning("Discarding generated outfit reason=repeated_body_slot slots=%s", repeated)
                    verdict["reason"] = "repeated_body_slot"
                    continue

            verdict.update(accepted=True, look_kind=look_kind, source_outfits=resolved_sources)
            verdicts.append(verdict)
            # Garments may recur across different generated looks; only duplicates
            # within this one look are rejected above.
            resolved.append({
                "outfit_id": f"generated_{len(resolved) + 1:02d}",
                "name": draft.get("name") or f"Look {len(resolved) + 1}",
                "description": draft.get("description") or "",
                "explanation": draft.get("explanation") or "",
                "changes_from_source": draft.get("changes_from_source") or "",
                "look_kind": look_kind,
                "omitted_items": omitted_items,
                "style_tags": draft.get("style_tags", []),
                "source_outfits": resolved_sources,
                "generated": True,
                "score": None,
                "matched_garment_ids": [item_id for item_id in garment_ids if item_id in query_matched_ids],
                "items": [candidate_items[item_id] for item_id in garment_ids],
            })

        # More looks are requested than shown. New combinations go first, then complete
        # real outfits; an outfit with garments removed only fills a slot left empty.
        priority = {"mix": 0, "existing": 1, "trimmed": 2}
        order = sorted(range(len(resolved)), key=lambda index: priority[resolved[index]["look_kind"]])
        limit = self.generator.max_generated_outfits
        for index in order[limit:]:
            verdicts[index].update(accepted=False, reason="lower_priority_than_shown_looks")
        shown = [resolved[index] for index in order[:limit]]
        for position, look in enumerate(shown, start=1):
            look["outfit_id"] = f"generated_{position:02d}"
        return shown
    
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

        # Position of each garment in the reranked list, before any filtering.
        ranks = {match["item_id"]: rank for rank, match in enumerate(matches, start=1)}
        found = {}
        for match in matches:
            for outfit_id in match["outfit_ids"]:
                score = match.get("rerank_score", 1 - match["distance"])
                if outfit_id not in found:
                    found[outfit_id] = {
                        "outfit_id": outfit_id,
                        "score": round(score, 6),
                        "matched_garment_ids": [],
                        "match_details": [],
                    }
                if match["item_id"] not in found[outfit_id]["matched_garment_ids"]:
                    found[outfit_id]["matched_garment_ids"].append(match["item_id"])
                    found[outfit_id]["match_details"].append({
                        "garment_id": match["item_id"],
                        "rank": ranks[match["item_id"]],
                        "score": round(score, 6),
                        "cosine_similarity": round(1 - match["distance"], 6),
                        "category": match.get("category"),
                        "image_url": self.catalog.image_url(match["item_id"]),
                    })

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
