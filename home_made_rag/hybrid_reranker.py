from __future__ import annotations

from typing import Any

from home_made_rag.config import CONFIG


class HybridGarmentReranker:
    """Rerank first-stage CLIP hits with cosine, color, and category scores."""

    def __init__(self, weights: dict[str, float] | None = None) -> None:
        settings = CONFIG.get("hybrid_reranker", {})
        self.weights = weights or settings.get("weights", {})
        expected = {"cosine_similarity", "color", "category"}
        if set(self.weights) != expected:
            raise ValueError(f"hybrid_reranker weights must contain {sorted(expected)}")
        if any(value < 0 for value in self.weights.values()) or sum(self.weights.values()) <= 0:
            raise ValueError("hybrid_reranker weights must be non-negative and sum to > 0")

    def rerank(
        self,
        matches: list[dict[str, Any]],
        query_attributes: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        query_color = ((query_attributes or {}).get("color") or {}).get("color")
        query_category = ((query_attributes or {}).get("category") or {}).get("label")
        reranked = []
        for match in matches:
            cosine_similarity = max(0.0, min(1.0, 1.0 - float(match["distance"])))
            components: dict[str, float | None] = {
                "cosine_similarity": cosine_similarity,
                "color": self._exact_match(query_color, match.get("color")),
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
            reranked_match["rerank_components"] = components
            reranked.append(reranked_match)

        reranked.sort(key=lambda match: match["rerank_score"], reverse=True)
        return reranked

    @staticmethod
    def _exact_match(query_value: str | None, item_value: str | None) -> float | None:
        if not query_value or not item_value:
            return None
        return 1.0 if query_value.strip().lower() == item_value.strip().lower() else 0.0
