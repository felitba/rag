"""Paired web-image ablation: CLIP top-k with and without background removal.

Images and source/license metadata are saved under data/background_ablation.
The query category comes from the Commons search bucket; query color is estimated
once from the original image and held fixed for both retrieval variants. This
keeps the context-precision labels constant while testing only the query image
preprocessing used to create the CLIP vector.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from urllib.parse import quote

import numpy as np
import requests
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from home_made_rag.catalog import Catalog
from home_made_rag.chroma_db import ChromaDB
from home_made_rag.color_metadata import estimate_garment_color
from home_made_rag.config import CONFIG
from home_made_rag.context_precision import evaluate_context_precision
from home_made_rag.embedding import ClipEmbedding
from home_made_rag.hybrid_reranker import HybridGarmentReranker
from home_made_rag.image_preprocessing import ForegroundPreprocessor

OUT = ROOT / "data" / "background_ablation"
MANIFEST = OUT / "queries.json"

# Query bucket -> Commons search phrase. Each bucket contributes up to four
# distinct, freely licensed Commons images with a known coarse garment category.
SEARCHES = [
    ("tops", "filetype:bitmap shirt"),
    ("outerwear", "filetype:bitmap jacket"),
    ("all-body", "filetype:bitmap dress"),
    ("shoes", "filetype:bitmap shoes"),
    ("bags", "filetype:bitmap handbag"),
]

TITLE_TERMS = {
    "tops": ("shirt", "blouse", "sweater", "t-shirt", "tee shirt"),
    "outerwear": ("jacket", "coat", "parka", "windbreaker"),
    "all-body": ("dress", "gown", "jumpsuit", "romper"),
    "shoes": ("shoe", "shoes", "sneaker", "boot", "sandal", "footwear"),
    "bags": ("handbag", "purse", "tote bag", "shoulder bag", "bag "),
}
TITLE_EXCLUSIONS = (
    "outfit", "fashion show", "people", "street", "store", "shop", "museum",
    "display", "collection", "catalog", "catalogue", "poster", "illustration",
    "drawing", "painting", "logo", "brand", "advertisement", "advertising",
    "fashion week", "wardrobe", "traditional", "costume", "uniform", "elderly",
    "worn with", "man,", "men's", "men in", "woman with", "at work", "workshop",
    "repair", "detail", "pose ", "file:a shot of", "marathon", "dress-up",
    "selling shoes", "broken", "wedding shoes", "bounce", "the purse",
    "carl buergerniss", "artistic", "circular knitted", "cat hairs",
    "counterfeit", "florence earl", "pop art",
    "man in", "boy wearing", "girl wearing", "lost grey shoe", "purse, coin",
    "louella long", "am 15987-6", "20e dp137228",
)


def download_commons(count: int, force: bool = False) -> list[dict]:
    OUT.mkdir(parents=True, exist_ok=True)
    if MANIFEST.exists() and not force:
        rows = json.loads(MANIFEST.read_text(encoding="utf-8"))
        if len(rows) >= count and all((OUT / row["filename"]).is_file() for row in rows[:count]):
            return rows[:count]

    headers = {"User-Agent": "FashionRAG-BackgroundAblation/1.0 (academic evaluation)"}
    rows: list[dict] = []
    seen: set[str] = set()
    per_bucket = max(2, (count + len(SEARCHES) - 1) // len(SEARCHES))
    for category, phrase in SEARCHES:
        params = {
            "action": "query", "generator": "search", "gsrsearch": phrase,
            "gsrnamespace": 6, "gsrlimit": max(50, per_bucket * 15),
            "prop": "imageinfo", "iiprop": "url|size|extmetadata", "iiurlwidth": 768,
            "format": "json", "origin": "*",
        }
        response = requests.get("https://commons.wikimedia.org/w/api.php", params=params,
                                headers=headers, timeout=30)
        response.raise_for_status()
        pages = response.json().get("query", {}).get("pages", {})
        added = 0
        # Preserve Commons relevance order, then reject titles that signal a
        # collage, event, shop, or unrelated image rather than a garment photo.
        for page in pages.values():
            title = page.get("title", "")
            normalized_title = title.lower().replace("_", " ")
            if not any(term in normalized_title for term in TITLE_TERMS[category]):
                continue
            if any(term in normalized_title for term in TITLE_EXCLUSIONS):
                continue
            image_info = (page.get("imageinfo") or [{}])[0]
            url = image_info.get("thumburl") or image_info.get("url")
            if not url or url in seen:
                continue
            if min(image_info.get("width", 0), image_info.get("height", 0)) < 320:
                continue
            ratio = image_info.get("width", 1) / max(1, image_info.get("height", 1))
            if not 0.45 <= ratio <= 2.1:
                continue
            if not url.lower().split("?")[0].endswith((".jpg", ".jpeg", ".png", ".webp")):
                continue
            metadata = image_info.get("extmetadata", {})
            filename = f"query_{len(rows) + 1:02d}.jpg"
            try:
                image_response = requests.get(url, headers=headers, timeout=45)
                image_response.raise_for_status()
                image_response.headers.get("content-type", "").lower()
                image_path = OUT / filename
                image_path.write_bytes(image_response.content)
                with Image.open(image_path) as image:
                    image.verify()
                # Normalize downloaded formats to RGB JPEG for a predictable evaluation.
                with Image.open(image_path) as image:
                    image.convert("RGB").save(image_path, "JPEG", quality=92)
            except Exception:
                (OUT / filename).unlink(missing_ok=True)
                continue
            rows.append({
                "filename": filename,
                "category": category,
                "commons_title": title,
                "source_page": "https://commons.wikimedia.org/wiki/" + quote(title.replace(" ", "_"), safe=":/()"),
                "image_url": url,
                "license": (metadata.get("LicenseShortName", {}).get("value") or "unknown"),
                "artist": (metadata.get("Artist", {}).get("value") or "unknown"),
            })
            seen.add(url)
            added += 1
            if len(rows) >= count or added >= per_bucket:
                break
        if len(rows) >= count:
            break

    if len(rows) < count:
        raise RuntimeError(f"Only downloaded {len(rows)} of {count} requested Commons images")
    rows = rows[:count]
    for index, row in enumerate(rows, 1):
        # Keep filenames contiguous if a bucket produced more than the requested total.
        old_path = OUT / row["filename"]
        new_name = f"query_{index:02d}.jpg"
        new_path = OUT / new_name
        if old_path != new_path:
            old_path.replace(new_path)
            row["filename"] = new_name
    MANIFEST.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    return rows


def download_random_extra(count: int, seed: int = 42) -> Path:
    """Append a seeded, category-balanced sample without replacing the first 20."""
    if count < len(SEARCHES) or count % len(SEARCHES):
        raise ValueError(f"--download-extra must be a positive multiple of {len(SEARCHES)}")
    base_rows = json.loads(MANIFEST.read_text(encoding="utf-8"))
    existing_titles = {row["commons_title"] for row in base_rows}
    existing_urls = {row["image_url"].split("?")[0] for row in base_rows}
    per_bucket = count // len(SEARCHES)
    headers = {"User-Agent": "FashionRAG-BackgroundAblation/1.0 (academic evaluation)"}
    selected: list[tuple[str, list[dict]]] = []

    # Build larger eligible pools, then shuffle locally with a fixed seed.
    for category, phrase in SEARCHES:
        params = {
            "action": "query", "generator": "search", "gsrsearch": phrase,
            "gsrnamespace": 6, "gsrlimit": 100,
            "prop": "imageinfo", "iiprop": "url|size|extmetadata", "iiurlwidth": 768,
            "format": "json", "origin": "*",
        }
        response = requests.get("https://commons.wikimedia.org/w/api.php", params=params,
                                headers=headers, timeout=30)
        response.raise_for_status()
        candidates = []
        for page in response.json().get("query", {}).get("pages", {}).values():
            title = page.get("title", "")
            normalized_title = title.lower().replace("_", " ")
            if title in existing_titles or not any(term in normalized_title for term in TITLE_TERMS[category]):
                continue
            if any(term in normalized_title for term in TITLE_EXCLUSIONS):
                continue
            image_info = (page.get("imageinfo") or [{}])[0]
            url = image_info.get("thumburl") or image_info.get("url")
            if not url or url.split("?")[0] in existing_urls:
                continue
            if min(image_info.get("width", 0), image_info.get("height", 0)) < 320:
                continue
            ratio = image_info.get("width", 1) / max(1, image_info.get("height", 1))
            if not 0.45 <= ratio <= 2.1:
                continue
            if not url.lower().split("?")[0].endswith((".jpg", ".jpeg", ".png", ".webp")):
                continue
            candidates.append({"title": title, "url": url, "image_info": image_info})
        selected.append((category, candidates))

    rng = random.Random(seed)
    new_rows = []
    for category, candidates in selected:
        rng.shuffle(candidates)
        downloaded = 0
        for candidate in candidates:
            if downloaded >= per_bucket:
                break
            index = len(base_rows) + len(new_rows) + 1
            filename = f"query_{index:02d}.jpg"
            image_path = OUT / filename
            try:
                image_response = requests.get(candidate["url"], headers=headers, timeout=45)
                image_response.raise_for_status()
                image_path.write_bytes(image_response.content)
                with Image.open(image_path) as image:
                    image.verify()
                with Image.open(image_path) as image:
                    image.convert("RGB").save(image_path, "JPEG", quality=92)
            except Exception:
                image_path.unlink(missing_ok=True)
                continue
            image_info = candidate["image_info"]
            metadata = image_info.get("extmetadata", {})
            title = candidate["title"]
            url = candidate["url"]
            new_rows.append({
                "filename": filename,
                "category": category,
                "commons_title": title,
                "source_page": "https://commons.wikimedia.org/wiki/" + quote(title.replace(" ", "_"), safe=":/()"),
                "image_url": url,
                "license": (metadata.get("LicenseShortName", {}).get("value") or "unknown"),
                "artist": (metadata.get("Artist", {}).get("value") or "unknown"),
            })
            existing_titles.add(title)
            existing_urls.add(url.split("?")[0])
            downloaded += 1
        if downloaded < per_bucket:
            raise RuntimeError(f"Only found {downloaded}/{per_bucket} extra images for {category}")

    all_rows = base_rows + new_rows
    manifest_path = OUT / f"queries_{len(all_rows)}.json"
    manifest_path.write_text(json.dumps(all_rows, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest_path


def _matches(result: dict) -> list[dict]:
    return [
        {"item_id": item_id, "distance": float(distance), **metadata}
        for item_id, distance, metadata in zip(
            result["ids"][0], result["distances"][0], result["metadatas"][0]
        )
    ]


def mean_ci(values: list[float], seed: int = 42, rounds: int = 10000) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    sample = np.asarray(values, dtype=float)
    boot = np.mean(rng.choice(sample, size=(rounds, len(sample)), replace=True), axis=1)
    return float(sample.mean()), float(np.quantile(boot, .025)), float(np.quantile(boot, .975))


def evaluate(
    count: int,
    top_k: int,
    *,
    save_baseline: bool = True,
    manifest_path: Path = MANIFEST,
    seed: int = 42,
) -> Path:
    if not manifest_path.exists():
        raise FileNotFoundError(f"No query manifest at {manifest_path}; run with --download first")
    queries = json.loads(manifest_path.read_text(encoding="utf-8"))[:count]
    catalog = Catalog()
    store = ChromaDB()
    clip = ClipEmbedding()
    settings = CONFIG["image_preprocessing"]
    processor = ForegroundPreprocessor(
        enabled=True,
        model_name=settings["model_name"],
        max_side=settings["max_side"],
        background_color=settings["background_color"],
        padding_ratio=settings["padding_ratio"],
    )
    reranker = HybridGarmentReranker()
    rows = []
    for index, query in enumerate(queries, 1):
        path = OUT / query["filename"]
        original = Image.open(path).convert("RGB")
        processed = processor.process(original)
        # Fixed labels across variants: query category is the Commons search bucket;
        # color is estimated from the same unprocessed photo in both conditions.
        attributes = {
            "category": {"label": query["category"], "uncertain": False},
            "color": estimate_garment_color(original),
        }
        before = _matches(store.retrieve(clip.embed_image(original), top_k=top_k))
        after = _matches(store.retrieve(clip.embed_image(processed), top_k=top_k))
        reranked_before = reranker.rerank(before, attributes)
        reranked_after = reranker.rerank(after, attributes)
        score_before = evaluate_context_precision(before, attributes, catalog)
        score_after = evaluate_context_precision(after, attributes, catalog)
        score_reranked_before = evaluate_context_precision(reranked_before, attributes, catalog)
        score_reranked_after = evaluate_context_precision(reranked_after, attributes, catalog)
        rows.append({
            **query,
            "query_color": attributes["color"],
            "context_precision_without_background_removal": score_before["score"],
            "context_precision_with_background_removal": score_after["score"],
            "context_precision_without_background_removal_after_reranking": score_reranked_before["score"],
            "context_precision_with_background_removal_after_reranking": score_reranked_after["score"],
            "relevant_without": score_before["relevant_count"],
            "relevant_with": score_after["relevant_count"],
            "top_k": top_k,
            "retrieved_without": [m["item_id"] for m in before],
            "retrieved_with": [m["item_id"] for m in after],
            "reranked_without": [m["item_id"] for m in reranked_before],
            "reranked_with": [m["item_id"] for m in reranked_after],
        })
        print(f"[{index:02d}/{len(queries)}] {query['category']:10s} AP@{top_k} "
              f"raw {score_before['score']:.3f}->{score_reranked_before['score']:.3f} | "
              f"background removed {score_after['score']:.3f}->{score_reranked_after['score']:.3f}", flush=True)

    before = [r["context_precision_without_background_removal"] for r in rows]
    after = [r["context_precision_with_background_removal"] for r in rows]
    delta = [b - a for a, b in zip(before, after)]
    report = {
        "experiment": "naive CLIP RAG: background processor ablation",
        "retrieval": "CLIP image embedding + cosine top-k; no reranker, query rewriting, or generator",
        "top_k": top_k,
        "n_queries": len(rows),
        "query_labels": "Commons search-bucket category and color estimated once from original query image; held fixed in both arms",
        "context_precision_note": "Proxy AP@k from catalog category and color metadata; not human relevance judgments",
        "mean_context_precision_without_background_removal": mean_ci(before),
        "mean_context_precision_with_background_removal": mean_ci(after),
        "mean_paired_delta_with_minus_without": mean_ci(delta),
        "queries": rows,
    }
    report_path = OUT / "background_ablation_results.json"
    if save_baseline:
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        plot(report, OUT / "background_ablation.png")
    reranking_report = build_reranking_report(rows, top_k, seed=seed, manifest_path=manifest_path)
    suffix = "" if len(rows) == 20 else f"_{len(rows)}"
    reranking_report_path = OUT / f"background_reranking_ablation{suffix}_results.json"
    reranking_report_path.write_text(
        json.dumps(reranking_report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    plot_reranking_ablation(reranking_report, OUT / f"background_reranking_ablation{suffix}.png")
    plot_combined_reranking_effect(reranking_report, OUT / f"background_reranking_comparison{suffix}.png")
    return report_path


def build_reranking_report(rows: list[dict], top_k: int, *, seed: int, manifest_path: Path) -> dict:
    values = {
        "without_background_before_reranking": [
            row["context_precision_without_background_removal"] for row in rows
        ],
        "with_background_before_reranking": [
            row["context_precision_with_background_removal"] for row in rows
        ],
        "without_background_after_reranking": [
            row["context_precision_without_background_removal_after_reranking"] for row in rows
        ],
        "with_background_after_reranking": [
            row["context_precision_with_background_removal_after_reranking"] for row in rows
        ],
    }
    effects = {
        "background_effect_before_reranking": [
            after - before for before, after in zip(
                values["without_background_before_reranking"],
                values["with_background_before_reranking"],
            )
        ],
        "background_effect_after_reranking": [
            after - before for before, after in zip(
                values["without_background_after_reranking"],
                values["with_background_after_reranking"],
            )
        ],
        "reranking_effect_without_background_removal": [
            after - before for before, after in zip(
                values["without_background_before_reranking"],
                values["without_background_after_reranking"],
            )
        ],
        "reranking_effect_with_background_removal": [
            after - before for before, after in zip(
                values["with_background_before_reranking"],
                values["with_background_after_reranking"],
            )
        ],
    }
    return {
        "experiment": "paired 2x2 ablation: background removal x hybrid reranking",
        "retrieval": "CLIP image embedding + cosine top-k; reranking uses configured hybrid weights",
        "reranker_weights": CONFIG.get("hybrid_reranker", {}).get("weights", {}),
        "top_k": top_k,
        "n_queries": len(rows),
        "seed": seed,
        "manifest": str(manifest_path),
        "query_labels": "Commons search-bucket category and color estimated once from original query image; held fixed across all four conditions",
        "context_precision_note": "Proxy AP@k from catalog category and color metadata; not human relevance judgments",
        "conditions": {name: mean_ci(scores) for name, scores in values.items()},
        "paired_effects": {name: mean_ci(scores) for name, scores in effects.items()},
        "queries": rows,
    }


def plot_reranking_ablation(report: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = report["queries"]
    panels = [
        (
            "Before reranking",
            "without_background_before_reranking",
            "with_background_before_reranking",
            "context_precision_without_background_removal",
            "context_precision_with_background_removal",
        ),
        (
            "After reranking",
            "without_background_after_reranking",
            "with_background_after_reranking",
            "context_precision_without_background_removal_after_reranking",
            "context_precision_with_background_removal_after_reranking",
        ),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    colors = ["#8da0cb", "#66c2a5"]
    for ax, (title, no_bg_key, bg_key, no_bg_row_key, bg_row_key) in zip(axes, panels):
        no_bg = np.asarray([row[no_bg_row_key] for row in rows])
        with_bg = np.asarray([row[bg_row_key] for row in rows])
        for left, right in zip(no_bg, with_bg):
            ax.plot([0, 1], [left, right], color="#718096", alpha=.4,
                    linewidth=.8, marker="o", markersize=2.5)
        for x, key, color in ((0, no_bg_key, colors[0]), (1, bg_key, colors[1])):
            mean, low, high = report["conditions"][key]
            ax.bar(x, mean, color=color, width=.55, zorder=2)
            ax.errorbar(x, mean, yerr=[[mean - low], [high - mean]], fmt="none",
                        ecolor="#263238", capsize=4, linewidth=1.4, zorder=3)
        ax.set_xticks([0, 1], ["Without background\nremoval", "With background\nremoval"])
        ax.set_ylim(0, 1)
        ax.set_title(title)
        ax.grid(axis="y", alpha=.25)
        ax.set_ylabel(f"Context precision (AP@{report['top_k']})")
    fig.suptitle("Effect of background removal before and after hybrid reranking", fontsize=14)
    fig.text(.5, .01,
             "Bars: mean across 20 queries with 95% bootstrap CI; lines: paired query scores. "
             "Fixed labels and same CLIP top-k candidate pool per condition.",
             ha="center", fontsize=8.5, color="#4b5563")
    fig.tight_layout(rect=(0, .07, 1, .93))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_combined_reranking_effect(report: dict, path: Path) -> None:
    """One chart: before/after reranking on x, background condition as series."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = ["Before reranking", "After reranking"]
    series = [
        ("Without background removal", "#8da0cb", [
            "without_background_before_reranking", "without_background_after_reranking",
        ]),
        ("With background removal", "#66c2a5", [
            "with_background_before_reranking", "with_background_after_reranking",
        ]),
    ]
    x = np.arange(2)
    width = .34
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for offset, (label, color, keys) in enumerate(series):
        positions = x + (offset - .5) * width
        means = [report["conditions"][key][0] for key in keys]
        low = [report["conditions"][key][0] - report["conditions"][key][1] for key in keys]
        high = [report["conditions"][key][2] - report["conditions"][key][0] for key in keys]
        ax.bar(positions, means, width=width, color=color, label=label, zorder=2)
        ax.errorbar(positions, means, yerr=[low, high], fmt="none", ecolor="#263238",
                    capsize=4, linewidth=1.4, zorder=3)
    ax.set_xticks(x, groups)
    ax.set_ylim(0, 1)
    ax.set_ylabel(f"Context precision (AP@{report['top_k']})")
    ax.set_title("Reranking effect with and without background removal")
    ax.grid(axis="y", alpha=.25)
    ax.legend(frameon=False)
    fig.text(.5, .015,
             f"Mean over {report['n_queries']} paired queries; bars show 95% bootstrap CI. "
             "Same candidate set before/after ranking within each background condition.",
             ha="center", fontsize=8.5, color="#4b5563")
    fig.tight_layout(rect=(0, .06, 1, .94))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot(report: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = report["queries"]
    before = np.asarray([r["context_precision_without_background_removal"] for r in rows])
    after = np.asarray([r["context_precision_with_background_removal"] for r in rows])
    mean_b, lo_b, hi_b = report["mean_context_precision_without_background_removal"]
    mean_a, lo_a, hi_a = report["mean_context_precision_with_background_removal"]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [1, 1.65]})
    ax.bar([0, 1], [mean_b, mean_a], color=["#8da0cb", "#66c2a5"], width=.62)
    ax.errorbar([0, 1], [mean_b, mean_a],
                yerr=[[mean_b - lo_b, mean_a - lo_a], [hi_b - mean_b, hi_a - mean_a]],
                fmt="none", ecolor="#263238", capsize=5, linewidth=1.5)
    ax.set_xticks([0, 1], ["Without\nbackground removal", "With\nbackground removal"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Context precision (AP@30)")
    ax.set_title("Mean score with 95% bootstrap CI")
    ax.grid(axis="y", alpha=.25)
    for i, (left, right) in enumerate(zip(before, after)):
        bx.plot([0, 1], [left, right], color="#718096", alpha=.5, linewidth=.9, marker="o", markersize=3)
    bx.set_xticks([0, 1], ["Without processor", "With processor"])
    bx.set_ylim(0, 1)
    bx.set_ylabel("Context precision (AP@30)")
    bx.set_title("Paired query scores (each line = one web image)")
    bx.grid(axis="y", alpha=.25)
    fig.suptitle("Naive CLIP RAG: effect of image background removal", fontsize=14)
    fig.text(.5, .01, "Top-30 cosine retrieval; fixed query category/color labels; proxy relevance from catalog metadata",
             ha="center", fontsize=9, color="#4b5563")
    fig.tight_layout(rect=(0, .05, 1, .94))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="Download query images from Wikimedia Commons")
    parser.add_argument("--force-download", action="store_true", help="Replace the existing Commons sample")
    parser.add_argument("--download-extra", type=int, default=0, help="Append a seeded random, category-balanced sample")
    parser.add_argument("--download-only", action="store_true", help="Download and record images without evaluating")
    parser.add_argument("--plot-only", action="store_true", help="Regenerate the reranking plot from saved results")
    parser.add_argument("--reranking-only", action="store_true", help="Run the 2x2 study without replacing the original ablation outputs")
    parser.add_argument("--n", type=int, default=20, help="Number of external query images")
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--manifest", type=Path, help="Query manifest to evaluate")
    args = parser.parse_args()
    if args.n < 1 or args.top_k < 1:
        parser.error("--n and --top-k must be positive")
    manifest_path = args.manifest or MANIFEST
    if args.download:
        rows = download_commons(args.n, args.force_download)
        print(f"Downloaded {len(rows)} query images to {OUT}")
        print(f"Source/license manifest: {MANIFEST}")
        manifest_path = MANIFEST
    if args.download_extra:
        manifest_path = download_random_extra(args.download_extra, args.seed)
        print(f"Added {args.download_extra} random images; extended manifest: {manifest_path}")
    if args.download_only:
        return
    suffix = "" if args.n == 20 else f"_{args.n}"
    if args.plot_only:
        saved_report = OUT / f"background_reranking_ablation{suffix}_results.json"
        reranking_report = json.loads(saved_report.read_text(encoding="utf-8"))
        plot_reranking_ablation(reranking_report, OUT / f"background_reranking_ablation{suffix}.png")
        plot_combined_reranking_effect(reranking_report, OUT / f"background_reranking_comparison{suffix}.png")
        print(f"Plot: {OUT / f'background_reranking_comparison{suffix}.png'}")
        return
    result = evaluate(
        args.n, args.top_k, save_baseline=not args.reranking_only,
        manifest_path=manifest_path, seed=args.seed,
    )
    if not args.reranking_only:
        report = json.loads(result.read_text(encoding="utf-8"))
        print("\nMean AP@k (95% bootstrap CI)")
        print("without:", report["mean_context_precision_without_background_removal"])
        print("with:   ", report["mean_context_precision_with_background_removal"])
        print("paired delta:", report["mean_paired_delta_with_minus_without"])
        print(f"Plot: {OUT / 'background_ablation.png'}")
        print(f"Results: {result}")
    rerank_report_path = OUT / f"background_reranking_ablation{suffix}_results.json"
    rerank_report = json.loads(rerank_report_path.read_text(encoding="utf-8"))
    print("\n2x2 reranking ablation means (95% bootstrap CI)")
    for condition, values in rerank_report["conditions"].items():
        print(f"{condition}: {values}")
    for effect, values in rerank_report["paired_effects"].items():
        print(f"{effect}: {values}")
    print(f"Plot: {OUT / f'background_reranking_comparison{suffix}.png'}")


if __name__ == "__main__":
    main()
