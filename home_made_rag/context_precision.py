from __future__ import annotations

from typing import Any

from home_made_rag.color_metadata import combined_color_similarity
from home_made_rag.config import CONFIG


def evaluate_context_precision(
    matches: list[dict[str, Any]],
    query_attributes: dict[str, Any] | None,
    catalog,
) -> dict[str, Any]:
    """Compute AP@k using query/category/color metadata as proxy relevance labels."""
    settings = CONFIG.get("context_precision", {})
    color_threshold = float(settings.get("color_similarity_threshold", 0.65))
    color_settings = CONFIG.get("hybrid_reranker", {}).get("color_similarity", {})
    query = query_attributes or {}
    query_category = query.get("category") or {}
    category_label = query_category.get("label")
    query_color = query.get("color") or {}
    query_color_score = combined_color_similarity(
        query_color,
        query_color,
        delta_e_scale=color_settings.get("delta_e_scale", 40.0),
        tone_weight=color_settings.get("tone_weight", 0.75),
        coverage_weight=color_settings.get("coverage_weight", 0.25),
    )["score"]

    criteria = []
    if category_label:
        criteria.append("category")
    if query_color_score is not None:
        criteria.append("color")

    annotated_matches = []
    relevant_count = 0
    precision_sum = 0.0
    for rank, match in enumerate(matches, start=1):
        item = catalog.items.get(str(match.get("item_id")), {})
        item_category = match.get("category") or item.get("category")
        candidate_color = {
            "color": match.get("color") or item.get("color"),
            "color_r": match.get("color_r") if match.get("color_r") is not None else item.get("color_r"),
            "color_g": match.get("color_g") if match.get("color_g") is not None else item.get("color_g"),
            "color_b": match.get("color_b") if match.get("color_b") is not None else item.get("color_b"),
            "color_confidence": match.get("color_confidence") if match.get("color_confidence") is not None else item.get("color_confidence"),
        }
        candidate_color_score = combined_color_similarity(
            query_color,
            candidate_color,
            delta_e_scale=color_settings.get("delta_e_scale", 40.0),
            tone_weight=color_settings.get("tone_weight", 0.75),
            coverage_weight=color_settings.get("coverage_weight", 0.25),
        )["score"]
        category_match = (
            str(category_label).strip().lower() == str(item_category).strip().lower()
            if category_label and item_category
            else False if category_label
            else None
        )
        color_match = (
            candidate_color_score >= color_threshold
            if query_color_score is not None and candidate_color_score is not None
            else False if query_color_score is not None
            else None
        )
        available_checks = []
        if category_label:
            available_checks.append(bool(category_match))
        if query_color_score is not None:
            available_checks.append(bool(color_match))
        relevant = all(available_checks) if available_checks else None
        if relevant:
            relevant_count += 1
            precision_sum += relevant_count / rank

        annotated = dict(match)
        annotated["context_relevance"] = {
            "relevant": relevant,
            "category_match": category_match,
            "color_similarity": candidate_color_score,
            "color_match": color_match,
        }
        annotated_matches.append(annotated)

    available = bool(criteria)
    score = (
        round(precision_sum / relevant_count, 6) if relevant_count else 0.0
    ) if available else None
    return {
        "score": score,
        "available": available,
        "k": len(matches),
        "relevant_count": relevant_count if available else None,
        "criteria": criteria,
        "color_similarity_threshold": color_threshold,
        "annotated_matches": annotated_matches,
    }
