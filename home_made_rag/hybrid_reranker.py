from __future__ import annotations

from typing import Any

from home_made_rag.config import CONFIG
from home_made_rag.color_metadata import combined_color_similarity


class HybridGarmentReranker:
    """Rerank first-stage CLIP hits with cosine, color, and category scores."""

    def __init__(self, weights: dict[str, float] | None = None) -> None:
        settings = CONFIG.get("hybrid_reranker", {})
        self.weights = weights or settings.get("weights", {})
        color_settings = settings.get("color_similarity", {})
        self.color_delta_e_scale = color_settings.get("delta_e_scale", 40.0)
        self.color_tone_weight = float(color_settings.get("tone_weight", 0.75))
        self.color_coverage_weight = float(color_settings.get("coverage_weight", 0.25))
        expected = {"cosine_similarity", "color", "category"}
        if set(self.weights) != expected:
            raise ValueError(f"hybrid_reranker weights must contain {sorted(expected)}")
        if any(value < 0 for value in self.weights.values()) or sum(self.weights.values()) <= 0:
            raise ValueError("hybrid_reranker weights must be non-negative and sum to > 0")
        if (
            self.color_tone_weight < 0
            or self.color_coverage_weight < 0
            or self.color_tone_weight + self.color_coverage_weight <= 0
        ):
            raise ValueError("color similarity weights must be non-negative and sum to > 0")

    def rerank(
        self,
        matches: list[dict[str, Any]],
        query_attributes: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        query_color = (query_attributes or {}).get("color") or {}
        query_category = ((query_attributes or {}).get("category") or {}).get("label")
        reranked = []
        for match in matches:
            cosine_similarity = max(0.0, min(1.0, 1.0 - float(match["distance"])))
            candidate_color = {
                "color": match.get("color"),
                "color_r": match.get("color_r"),
                "color_g": match.get("color_g"),
                "color_b": match.get("color_b"),
                "color_confidence": match.get("color_confidence"),
            }
            color_parts = combined_color_similarity(
                query_color,
                candidate_color,
                delta_e_scale=self.color_delta_e_scale,
                tone_weight=self.color_tone_weight,
                coverage_weight=self.color_coverage_weight,
            )
            components: dict[str, float | None] = {
                "cosine_similarity": cosine_similarity,
                "color": color_parts["score"],
                "category": self._exact_match(query_category, match.get("category")),
            }
            available_weight = sum(
                self.weights[key] for key, value in components.items() if value is not None
            )
            score = sum(
                self.weights[key] * value
                for key, value in components.items()
                if value is not None
            ) / available_weight
            reranked_match = dict(match)
            reranked_match["rerank_score"] = round(score, 6)
            reranked_match["rerank_components"] = {
                **components,
                "color_tone_similarity": color_parts["tone"],
                "color_coverage_similarity": color_parts["coverage"],
            }
            reranked.append(reranked_match)

        reranked.sort(key=lambda match: match["rerank_score"], reverse=True)
        return reranked

    @staticmethod
    def _exact_match(query_value: str | None, item_value: str | None) -> float | None:
        if not query_value or not item_value:
            return None
        return 1.0 if query_value.strip().lower() == item_value.strip().lower() else 0.0
