from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from app.services.catalog_service import CatalogService
from chroma_db import ChromaService
from embedding import ClipEmbedding


class OutfitService:
    def __init__(
        self,
        catalog: CatalogService,
        clip_service: ClipEmbedding,
        chroma_service: ChromaService,
    ) -> None:
        self.catalog = catalog
        self.clip_service = clip_service
        self.chroma_service = chroma_service

    def recommend(
        self,
        image_path: Path,
        top_k_garments: int = 30,
        max_outfits: int = 5,
    ) -> list[dict[str, Any]]:
        matches = self.chroma_service.search_similar(
            query_image_path=image_path,
            top_k=top_k_garments,
        )
        if not matches:
            return []

        outfit_scores: dict[str, float] = defaultdict(float)
        matched_garments: dict[str, list[str]] = defaultdict(list)

        # A rank-based score is a transparent baseline. It is not a probability.
        for rank, match in enumerate(matches):
            garment_id = match["garment_id"]
            weight = 1.0 / (rank + 1)

            for outfit_id in set(match["outfit_ids"]):
                if outfit_id not in self.catalog.outfits:
                    continue
                outfit_scores[outfit_id] += weight
                matched_garments[outfit_id].append(garment_id)

        ranked_outfit_ids = sorted(
            outfit_scores,
            key=lambda outfit_id: outfit_scores[outfit_id],
            reverse=True,
        )

        recommendations: list[dict[str, Any]] = []
        for outfit_id in ranked_outfit_ids:
            if len(recommendations) >= max_outfits:
                break

            garment_ids = self.catalog.outfits[outfit_id]
            garment_records = [self.catalog.garments[g_id] for g_id in garment_ids]

            # Keep curated looks complete: don't recommend a set with an
            # unavailable component. Change this policy if partial outfits are OK.
            if any(not garment.get("available", True) for garment in garment_records):
                continue

            items: list[dict[str, Any]] = []
            for garment in garment_records:
                image_path_item = self.catalog.resolve_image_path(garment["image_path"])
                item = {
                    **garment,
                    "image_url": self.catalog.image_url(garment["image_path"]),
                    "_image_abs_path": str(image_path_item),
                }
                items.append(item)

            if not items:
                continue

            recommendations.append({
                "outfit_id": outfit_id,
                "score": round(outfit_scores[outfit_id], 6),
                "matched_garment_ids": list(
                    dict.fromkeys(matched_garments[outfit_id])
                ),
                "items": items,
            })

        return recommendations
