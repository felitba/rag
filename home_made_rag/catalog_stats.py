"""Describe the catalog the assistant searches, so the UI can show its coverage and gaps.

Everything here is counted from the local catalog files; nothing is hard-coded
about Polyvore, so the same page works for any store's catalog.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

from home_made_rag.config import CONFIG

# Fields a garment record may carry, with the label the UI shows for each.
METADATA_FIELDS = [
    ("title", "Título"),
    ("description", "Descripción"),
    ("categories", "Categorías de la tienda"),
    ("related", "Etiquetas relacionadas"),
    ("url_name", "Nombre en la URL"),
    ("color", "Color"),
    ("price", "Precio"),
]

# Whole-word matches in the garment's own text. "women" must be checked first:
# the men pattern would otherwise also match inside "women".
WOMEN_PATTERN = re.compile(r"\bwom[ae]n('s|s)?\b")
MEN_PATTERN = re.compile(r"\bm[ae]n('s|s)?\b")


def _has_value(value: Any) -> bool:
    return value not in (None, "", [], {})


def _share(count: int, total: int) -> float:
    return round(count / total, 4) if total else 0.0


def _ranked(counter: Counter, total: int) -> list[dict[str, Any]]:
    return [
        {"value": value, "count": count, "share": _share(count, total)}
        for value, count in counter.most_common()
    ]


def _text(item: dict[str, Any]) -> str:
    parts = [item.get("title") or "", item.get("description") or ""]
    parts.extend(str(value) for value in (item.get("categories") or []))
    return " ".join(parts).lower()


def _sample(catalog, item: dict[str, Any]) -> dict[str, Any]:
    """One garment exactly as the catalog stores it, minus fields that only matter internally."""
    hidden = {"image_path", "color_r", "color_g", "color_b"}
    return {
        "garment_id": item["item_id"],
        "image_url": catalog.image_url(item["item_id"]),
        "record": {key: value for key, value in item.items() if key not in hidden},
    }


def build_catalog_info(catalog, indexed_ids: list[str] | None = None) -> dict[str, Any]:
    """Store identity plus statistics of the garments the assistant can retrieve."""
    store = CONFIG.get("store", {})
    dataset = CONFIG.get("dataset", {})

    # Only indexed garments can be found by a search, so they are what the
    # statistics describe. Without an index, fall back to the whole catalog file.
    indexed = [catalog.items[item_id] for item_id in (indexed_ids or []) if item_id in catalog.items]
    items = indexed or list(catalog.items.values())
    total = len(items)
    searchable_ids = {item["item_id"] for item in items}

    outfit_sizes = Counter()
    reachable_outfits = 0
    for garment_ids in catalog.outfits.values():
        if any(garment_id in searchable_ids for garment_id in garment_ids):
            reachable_outfits += 1
            outfit_sizes[len(garment_ids)] += 1

    with_text = [text for text in (_text(item) for item in items) if text.strip()]
    women = sum(1 for text in with_text if WOMEN_PATTERN.search(text))
    men = sum(1 for text in with_text if MEN_PATTERN.search(text))

    # Two real records: the most complete one and one with no text at all.
    by_id = sorted(items, key=lambda item: item["item_id"])
    rich = next(
        (item for item in by_id if all(_has_value(item.get(key)) for key in ("title", "description", "related"))),
        None,
    )
    sparse = next((item for item in by_id if not _has_value(item.get("title"))), None)

    return {
        "store": {
            "name": store.get("name") or "el catálogo",
            "url": store.get("url") or None,
            "about": store.get("about") or None,
            "source": dataset.get("huggingface_id"),
        },
        "garments": total,
        "garments_in_catalog_file": len(catalog.items),
        "outfits": reachable_outfits,
        "outfits_in_catalog_file": len(catalog.outfits),
        "categories": _ranked(Counter(item.get("category") or "sin categoría" for item in items), total),
        "colors": _ranked(Counter(item.get("color") or "sin color" for item in items), total),
        "color_sources": _ranked(Counter(item.get("color_source") or "sin dato" for item in items), total),
        "metadata_coverage": [
            {
                "field": field,
                "label": label,
                "count": (count := sum(1 for item in items if _has_value(item.get(field)))),
                "share": _share(count, total),
            }
            for field, label in METADATA_FIELDS
        ],
        "outfit_sizes": [
            {"size": size, "count": count} for size, count in sorted(outfit_sizes.items())
        ],
        "audience": {
            "garments_with_text": len(with_text),
            "mention_women": women,
            "mention_men": men,
        },
        "samples": [_sample(catalog, item) for item in (rich, sparse) if item is not None],
        "photo_retention_days": (
            CONFIG.get("monitoring", {}).get("retention_days", 30)
            if CONFIG.get("monitoring", {}).get("enabled", True) else 0
        ),
    }
