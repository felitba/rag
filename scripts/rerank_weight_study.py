"""Which part of the hybrid rerank helps or hurts, by the judged (independent) labels.

Reuses the grades saved by scripts/judged_context_precision.py, so it makes no LLM calls:
the 30 retrieved garments per query are the same for every weighting, only their order changes.

Usage: python scripts/rerank_weight_study.py [logs/judged_context_precision_100.json]
"""
import json, math, os, statistics, sys
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO); os.chdir(REPO)
from home_made_rag.catalog import Catalog
from home_made_rag.chroma_db import ChromaDB
from home_made_rag.config import CONFIG
from home_made_rag.embedding import ClipEmbedding
from home_made_rag.fashion_clip_classifier import FashionClipCategoryClassifier
from home_made_rag.hybrid_reranker import HybridGarmentReranker
from home_made_rag.image_preprocessing import ForegroundPreprocessor
from home_made_rag.rag import RAG

SOURCE = sys.argv[1] if len(sys.argv) > 1 else "logs/judged_context_precision_100.json"
# name -> (cosine, colour, category)
CONFIGURATIONS = {
    "cosine only (no rerank)": (1.0, 0.0, 0.0),
    "+ category (0.5 / 0.25)": (0.5, 0.0, 0.25),
    "+ colour (0.5 / 0.25)": (0.5, 0.25, 0.0),
    "hybrid, current (0.5 / 0.25 / 0.25)": (0.5, 0.25, 0.25),
    "light category (0.9 / 0.1)": (0.9, 0.0, 0.1),
    "light colour (0.9 / 0.1)": (0.9, 0.1, 0.0),
    "light hybrid (0.8 / 0.1 / 0.1)": (0.8, 0.1, 0.1),
}

def average_precision(labels):
    hits, total = 0, 0.0
    for rank, relevant in enumerate(labels, 1):
        if relevant:
            hits += 1; total += hits / rank
    return total / hits if hits else 0.0

def ndcg(grades, k=10):
    dcg = lambda values: sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(values[:k]))
    ideal = dcg(sorted(grades, reverse=True))
    return dcg(grades) / ideal if ideal else 0.0

rows = json.load(open(SOURCE, encoding="utf-8"))["rows"]
catalog = Catalog(); settings = CONFIG["image_preprocessing"]
rag = RAG(ClipEmbedding(), ChromaDB(), catalog,
          attribute_classifier=FashionClipCategoryClassifier(labels=[i["category"] for i in catalog.items.values()]),
          image_preprocessor=ForegroundPreprocessor(enabled=settings["enabled"], model_name=settings["model_name"],
              max_side=settings["max_side"], background_color=settings["background_color"], padding_ratio=settings["padding_ratio"]))
top_k = CONFIG["vector_store"]["top_k"]
scores = {name: [] for name in CONFIGURATIONS}
category_right = []
for row in rows:
    query_id = row["query_id"]
    image = rag.image_preprocessor.process(catalog.image_path(query_id))
    retrieved = [m for m in rag._search(rag.embedding.embed_image(image), top_k + 1) if m["item_id"] != query_id][:top_k]
    if [m["item_id"] for m in retrieved] != row["before"]:
        raise SystemExit(f"Retrieval for {query_id} differs from the saved run; re-run judged_context_precision.py")
    attributes = rag.classify_query_image(image)
    category_right.append((attributes.get("category") or {}).get("label") == catalog.items[query_id]["category"])
    for name, (cosine, colour, category) in CONFIGURATIONS.items():
        reranker = HybridGarmentReranker({"cosine_similarity": cosine, "color": colour, "category": category})
        grades = [row["grades"][m["item_id"]] for m in reranker.rerank(retrieved, attributes)]
        scores[name].append({
            "ap_same_type": average_precision([g >= 1 for g in grades]),
            "ap_similar": average_precision([g >= 2 for g in grades]),
            "ndcg10": ndcg(grades),
            "mean_grade_top5": sum(grades[:5]) / 5,
        })

baseline = scores["cosine only (no rerank)"]
report = {"source": SOURCE, "queries": len(rows), "category_detected_correctly": sum(category_right), "configurations": {}}
print(f"{len(rows)} queries; photo category detected correctly in {sum(category_right)}\n")
print(f"{'configuration':38s} {'AP same type':>13s} {'AP similar':>11s} {'NDCG@10':>8s} {'grade top5':>11s}   NDCG change vs cosine (± s.e.)   better/worse")
for name, values in scores.items():
    mean = {key: statistics.mean(v[key] for v in values) for key in values[0]}
    diffs = [v["ndcg10"] - b["ndcg10"] for v, b in zip(values, baseline)]
    change, error = statistics.mean(diffs), statistics.stdev(diffs) / len(diffs) ** 0.5
    better, worse = sum(d > 1e-9 for d in diffs), sum(d < -1e-9 for d in diffs)
    split = {label: round(statistics.mean(d for d, ok in zip(diffs, category_right) if ok == flag), 4) if any(ok == flag for ok in category_right) else None
             for label, flag in (("category_right", True), ("category_wrong", False))}
    report["configurations"][name] = {**{k: round(v, 4) for k, v in mean.items()}, "ndcg10_change": round(change, 4),
                                      "ndcg10_change_se": round(error, 4), "better": better, "worse": worse, "ndcg10_change_by_detection": split}
    print(f"{name:38s} {mean['ap_same_type']:13.3f} {mean['ap_similar']:11.3f} {mean['ndcg10']:8.3f} {mean['mean_grade_top5']:11.3f}   {change:+.3f} ± {error:.3f}   {better}/{worse}   right {split['category_right']} wrong {split['category_wrong']}")
os.makedirs("logs", exist_ok=True)
with open("logs/rerank_weight_study.json", "w", encoding="utf-8") as output:
    json.dump(report, output, indent=1)
print("\nSaved logs/rerank_weight_study.json")
