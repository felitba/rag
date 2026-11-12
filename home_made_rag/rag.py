from __future__ import annotations

import json
import logging
import math

from home_made_rag.config import CONFIG


logger = logging.getLogger(__name__)


class RAG:
    def __init__(self, embedding, chroma, catalog, expander=None, llm=None):
        self.embedding = embedding
        self.chroma = chroma
        self.catalog = catalog
        self.expander = expander
        self.llm = llm

    def recommend(self, image_path=None, text=None):
        text = (text or "").strip()

        if image_path is None and not text:
            raise ValueError("Se necesita una imagen o un texto para recomendar")

        expansion = self._expand(image_path, text)

        if expansion is None:
            return self._recommend_baseline(image_path, text)

        query_embedding = self._query_embedding(image_path, text, expansion)
        hypotheses = self._embed_hypotheses(expansion)

        if not hypotheses:
            logger.warning(
                "[RAG] expansion returned no usable hypotheses; using baseline retrieval"
            )
            return self._recommend_baseline(image_path, text, expansion=expansion)

        candidate_pool_k = int(
            CONFIG["retrieval"].get(
                "candidate_pool_k",
                CONFIG["vector_store"]["top_k"],
            )
        )
        matches = self._search(query_embedding, candidate_pool_k)
        candidates = self._candidate_outfits(matches)

        logger.info(
            "[RAG] expansion_hypotheses=%d candidate_pool_k=%d candidate_outfits=%d",
            len(hypotheses),
            candidate_pool_k,
            len(candidates),
        )

        if not candidates:
            logger.warning("[RAG] no candidate outfits found after retrieval")
            return {
                "outfits": [],
                "expansion": self._expansion_payload(expansion),
                "explanation": self._generic_message(),
                "explanation_source": "generic",
                "hypothesis_coverage": [],
            }

        outfits, coverage = self._rerank_outfits(
            candidates,
            query_embedding,
            hypotheses,
            int(CONFIG["retrieval"]["max_outfits"]),
        )
        explanation, explanation_source = self._select_explanation(expansion, coverage)

        logger.info(
            "[RAG] selected_outfits=%d covered_hypotheses=%d/%d explanation_source=%s",
            len(outfits),
            sum(1 for item in coverage if item["covered"]),
            len(coverage),
            explanation_source,
        )

        return {
            "outfits": outfits,
            "expansion": self._expansion_payload(expansion),
            "explanation": explanation,
            "explanation_source": explanation_source,
            "hypothesis_coverage": coverage,
        }

    def explain(self, image_path, outfits):
        if self.llm is None:
            raise RuntimeError("El explicador multimodal no está configurado")
        return self.llm.explain(image_path, outfits)

    def _expand(self, image_path, text):
        settings = CONFIG.get("query_expansion") or {}

        if self.expander is None or not settings.get("enabled", True):
            return None

        categories = sorted({
            item.get("category")
            for item in self.catalog.items.values()
            if item.get("category")
        })

        try:
            return self.expander.expand(
                image_path=image_path,
                text=text or None,
                catalog_categories=categories,
            )
        except Exception as exc:
            logger.warning(
                "[RAG] query expansion failed; using baseline retrieval: %s",
                exc,
            )
            return None

    def _recommend_baseline(self, image_path, text, expansion=None):
        if image_path is not None:
            query_embedding = self.embedding.embed_image(image_path)
        else:
            query_embedding = self.embedding.embed_text(text)

        matches = self._search(query_embedding, CONFIG["vector_store"]["top_k"])
        outfits = self._collect_outfits(matches, CONFIG["retrieval"]["max_outfits"])

        return {
            "outfits": outfits,
            "expansion": self._expansion_payload(expansion),
            "explanation": self._generic_message(),
            "explanation_source": "generic",
            "hypothesis_coverage": [],
        }

    def _query_embedding(self, image_path, text, expansion):
        if image_path is not None:
            return self.embedding.embed_image(image_path)

        anchor = getattr(getattr(expansion, "observed", None), "anchor_description_en", None)
        if anchor and anchor.strip():
            return self.embedding.embed_text(anchor.strip())

        return self.embedding.embed_text(text)

    def _embed_hypotheses(self, expansion):
        hypotheses = [
            garment
            for garment in expansion.complementary
            if garment.description_en and garment.description_en.strip()
        ]

        if not hypotheses:
            return []

        embeddings = self.embedding.embed_texts([
            garment.description_en.strip()
            for garment in hypotheses
        ])
        return list(zip(hypotheses, embeddings))

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

    def _candidate_outfits(self, matches):
        if CONFIG["retrieval"].get("exclude_query_item", True):
            matches = [match for match in matches if match["distance"] > 1e-4]

        found = {}

        for match in matches:
            for outfit_id in match["outfit_ids"]:
                entry = found.get(outfit_id)
                if entry is None:
                    entry = {
                        "outfit_id": outfit_id,
                        "visual_matches": [],
                    }
                    found[outfit_id] = entry
                entry["visual_matches"].append(match)

        return list(found.values())

    def _collect_outfits(self, matches, max_outfits):
        candidates = self._candidate_outfits(matches)
        outfits = []

        for candidate in candidates[:max_outfits]:
            matched_ids = []
            for match in candidate["visual_matches"]:
                if match["item_id"] not in matched_ids:
                    matched_ids.append(match["item_id"])

            outfits.append({
                "outfit_id": candidate["outfit_id"],
                "score": round(1 - candidate["visual_matches"][0]["distance"], 6),
                "matched_garment_ids": matched_ids,
                "items": [
                    self._item_view(item_id)
                    for item_id in self.catalog.outfits[candidate["outfit_id"]]
                ],
            })

        return outfits

    def _rerank_outfits(self, candidates, query_embedding, hypotheses, max_outfits):
        item_ids = set()
        for candidate in candidates:
            item_ids.update(self.catalog.outfits[candidate["outfit_id"]])

        item_vectors = self.chroma.get_embeddings(item_ids)
        scored = []

        for candidate in candidates:
            outfit_id = candidate["outfit_id"]
            vectors = [
                (item_id, item_vectors[item_id])
                for item_id in self.catalog.outfits[outfit_id]
                if item_id in item_vectors
            ]

            if not vectors:
                logger.warning(
                    "[RAG] skipping outfit %s because it has no embeddings",
                    outfit_id,
                )
                continue

            visual_score = max(
                self._cosine(query_embedding, vector)
                for _, vector in vectors
            )
            hypothesis_similarities = [
                max(
                    self._cosine(hypothesis_embedding, vector)
                    for _, vector in vectors
                )
                for _, hypothesis_embedding in hypotheses
            ]
            hypothesis_score = sum(hypothesis_similarities) / len(hypothesis_similarities)

            best_visual_id = max(
                vectors,
                key=lambda pair: self._cosine(query_embedding, pair[1]),
            )[0]
            matched_ids = [best_visual_id]

            for _, hypothesis_embedding in hypotheses:
                best_id = max(
                    vectors,
                    key=lambda pair: self._cosine(hypothesis_embedding, pair[1]),
                )[0]
                if best_id not in matched_ids:
                    matched_ids.append(best_id)

            candidate["visual_score"] = visual_score
            candidate["hypothesis_score"] = hypothesis_score
            candidate["hypothesis_similarities"] = hypothesis_similarities
            candidate["matched_garment_ids"] = matched_ids
            scored.append(candidate)

        if not scored:
            return [], []

        visual_norms = self._min_max([candidate["visual_score"] for candidate in scored])
        hypothesis_norms = self._min_max([
            candidate["hypothesis_score"]
            for candidate in scored
        ])
        visual_weight = float(CONFIG["retrieval"].get("rerank_visual_weight", 0.35))
        hypothesis_weight = float(
            CONFIG["retrieval"].get("rerank_hypothesis_weight", 0.65)
        )

        for candidate, visual_norm, hypothesis_norm in zip(
            scored,
            visual_norms,
            hypothesis_norms,
        ):
            candidate["visual_norm"] = visual_norm
            candidate["hypothesis_norm"] = hypothesis_norm
            candidate["score"] = (
                visual_weight * visual_norm
                + hypothesis_weight * hypothesis_norm
            )

        scored.sort(
            key=lambda candidate: (
                candidate["score"],
                candidate["hypothesis_score"],
                candidate["visual_score"],
            ),
            reverse=True,
        )
        selected = scored[:max_outfits]
        coverage = self._hypothesis_coverage(selected, hypotheses, item_vectors)

        outfits = [
            {
                "outfit_id": candidate["outfit_id"],
                "score": round(candidate["score"], 6),
                "visual_score": round(candidate["visual_score"], 6),
                "hypothesis_score": round(candidate["hypothesis_score"], 6),
                "matched_garment_ids": candidate["matched_garment_ids"],
                "items": [
                    self._item_view(item_id)
                    for item_id in self.catalog.outfits[candidate["outfit_id"]]
                ],
            }
            for candidate in selected
        ]

        return outfits, coverage

    def _hypothesis_coverage(self, outfits, hypotheses, item_vectors):
        threshold = float(
            CONFIG.get("explanation", {}).get("min_hypothesis_similarity", 0.24)
        )
        selected_item_ids = set()
        for outfit in outfits:
            selected_item_ids.update(self.catalog.outfits[outfit["outfit_id"]])

        pairs = []
        for hypothesis_index, (_, hypothesis_embedding) in enumerate(hypotheses):
            for item_id in selected_item_ids:
                vector = item_vectors.get(item_id)
                if vector is None:
                    continue
                pairs.append((
                    self._cosine(hypothesis_embedding, vector),
                    hypothesis_index,
                    item_id,
                ))

        pairs.sort(key=lambda pair: pair[0], reverse=True)

        covered = {}
        used_items = set()

        for similarity, hypothesis_index, item_id in pairs:
            if hypothesis_index in covered or item_id in used_items:
                continue
            if similarity < threshold:
                continue
            covered[hypothesis_index] = (similarity, item_id)
            used_items.add(item_id)

        coverage = []

        for hypothesis_index, (garment, hypothesis_embedding) in enumerate(hypotheses):
            if hypothesis_index in covered:
                similarity, item_id = covered[hypothesis_index]
                coverage.append({
                    "hypothesis_index": hypothesis_index,
                    "category": garment.category,
                    "description_en": garment.description_en,
                    "garment_id": item_id,
                    "similarity": round(similarity, 6),
                    "covered": True,
                })
                continue

            best_similarity = max(
                (
                    self._cosine(hypothesis_embedding, item_vectors[item_id])
                    for item_id in selected_item_ids
                    if item_id in item_vectors
                ),
                default=0.0,
            )
            coverage.append({
                "hypothesis_index": hypothesis_index,
                "category": garment.category,
                "description_en": garment.description_en,
                "garment_id": None,
                "similarity": round(best_similarity, 6),
                "covered": False,
            })

        return coverage

    def _select_explanation(self, expansion, coverage):
        min_covered = int(
            CONFIG.get("explanation", {}).get("min_covered_hypotheses", 2)
        )
        covered_count = sum(1 for item in coverage if item["covered"])
        summary = (
            getattr(expansion, "stylist_summary_es", "") or ""
        ).strip() if expansion else ""

        if summary and covered_count >= min_covered:
            return summary, "expansion"

        return self._generic_message(), "generic"

    @staticmethod
    def _generic_message():
        return CONFIG.get("explanation", {}).get(
            "generic_message_es",
            "Estos son los looks del catálogo más cercanos a tu prenda. "
            "No pudimos confirmar que incluyan todos los complementos sugeridos, "
            "así que tomalos como punto de partida.",
        )

    @staticmethod
    def _expansion_payload(expansion):
        if expansion is None:
            return None
        if hasattr(expansion, "model_dump"):
            return expansion.model_dump()
        return expansion

    @staticmethod
    def _min_max(values):
        if not values:
            return []

        low = min(values)
        high = max(values)

        if high - low < 1e-12:
            return [0.5] * len(values)

        return [(value - low) / (high - low) for value in values]

    @staticmethod
    def _cosine(left, right):
        dot = sum(a * b for a, b in zip(left, right))
        left_norm = math.sqrt(sum(a * a for a in left))
        right_norm = math.sqrt(sum(b * b for b in right))
        return float(dot / max(left_norm * right_norm, 1e-12))

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
