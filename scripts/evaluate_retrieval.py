import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from home_made_rag.catalog import Catalog
from home_made_rag.chroma_db import ChromaDB
from home_made_rag.context_precision import evaluate_context_precision
from home_made_rag.hybrid_reranker import HybridGarmentReranker

MAX_OUTFITS = 5
COLOR_KEYS = ("color", "color_confidence", "color_r", "color_g", "color_b")

# name -> weights of the hybrid reranker (cosine, colour, category)
CONFIGURATIONS = {
    "naive RAG": {"cosine_similarity": 1.0, "color": 0.0, "category": 0.0},
    "+ category": {"cosine_similarity": 0.5, "color": 0.0, "category": 0.25},
    "+ colour": {"cosine_similarity": 0.5, "color": 0.25, "category": 0.0},
    "hybrid (current)": {"cosine_similarity": 0.5, "color": 0.25, "category": 0.25},
}
class Evaluator:
    def __init__(self):
        self.catalog = Catalog()
        self.catalog.enrich_missing_colors(persist=False)
        stored = ChromaDB().collection.get(include=["embeddings"])
        self.ids = stored["ids"]
        self.row = {item_id: index for index, item_id in enumerate(self.ids)}
        self.vectors = np.array(stored["embeddings"])
        self.similarity = self.vectors @ self.vectors.T
        self.category = {i: self.catalog.items[i]["category"] for i in self.ids}
        self.by_category = defaultdict(list)
        for item_id in self.ids:
            self.by_category[self.category[item_id]].append(item_id)
    def retrieve(self, query_id, top_k):
        """Same output as RAG._search, using the stored vector of a catalog garment as the query."""
        scores = self.similarity[self.row[query_id]]
        matches = []
        for index in np.argsort(-scores):
            item_id = self.ids[index]
            if item_id == query_id:
                continue
            item = self.catalog.items[item_id]
            match = {
                "item_id": item_id,
                "distance": float(1 - scores[index]),
                "category": self.category[item_id],
                "outfit_ids": self.catalog.outfits_by_item[item_id],
            }
            match.update({key: item.get(key) for key in COLOR_KEYS})
            matches.append(match)
            if len(matches) == top_k:
                break
        return matches

    def query_attributes(self, query_id):
        item = self.catalog.items[query_id]
        return {
            "category": {"label": self.category[query_id]},
            "color": {key: item.get(key) for key in COLOR_KEYS},
        }

    def collect_outfits(self, matches, held_out):
        """Same rule as RAG._collect_outfits: outfits in order of first appearance."""
        outfits = []
        for match in matches:
            for outfit_id in match["outfit_ids"]:
                if outfit_id not in held_out and outfit_id not in outfits:
                    outfits.append(outfit_id)
        return outfits[:MAX_OUTFITS]
    def companion_targets(self, query_id, held_out):
        """For each real companion of the query: its 10 nearest catalog garments of the same category."""
        own_items = {i for outfit_id in held_out for i in self.catalog.outfits[outfit_id]}
        targets = []
        for companion in own_items - {query_id}:
            pool = [i for i in self.by_category[self.category[companion]] if i not in own_items]
            if len(pool) < 10:
                continue
            pool.sort(key=lambda i: -self.similarity[self.row[companion], self.row[i]])
            targets.append(set(pool[:10]))
        return targets

    def companion_hit(self, outfits, targets):
        if not targets:
            return None
        recommended = {i for outfit_id in outfits for i in self.catalog.outfits[outfit_id]}
        return sum(1 for target in targets if recommended & target) / len(targets)

    def diversity(self, outfits):
        centroids = []
        for outfit_id in outfits:
            rows = [self.row[i] for i in self.catalog.outfits[outfit_id]]
            mean = self.vectors[rows].mean(axis=0)
            centroids.append(mean / np.linalg.norm(mean))
        distances = [
            1 - float(first @ second)
            for position, first in enumerate(centroids)
            for second in centroids[position + 1:]
        ]
        return sum(distances) / len(distances) if distances else None
    def run(self, queries, weights, top_k):
        reranker = HybridGarmentReranker(weights=weights)
        values = defaultdict(list)
        for query_id in queries:
            held_out = self.catalog.outfits_by_item[query_id]
            attributes = self.query_attributes(query_id)
            retrieved = self.retrieve(query_id, top_k)
            reranked = reranker.rerank(retrieved, attributes)
            outfits = self.collect_outfits(reranked, held_out)
            naive_outfits = self.collect_outfits(retrieved, held_out)
            targets = self.companion_targets(query_id, held_out)

            precision = evaluate_context_precision(reranked, attributes, self.catalog)["score"]
            same_category = [m["category"] == self.category[query_id] for m in reranked[:5]]
            values["context_precision_proxy"].append(precision)
            values["companion_hit@10"].append(self.companion_hit(outfits, targets))
            values["top5_same_category"].append(sum(same_category) / len(same_category))
            values["diversity"].append(self.diversity(outfits))
            values["outfits_changed"].append(len(set(outfits) - set(naive_outfits)))
        return average(values)

    def run_random(self, queries, seed):
        rng = random.Random(seed)
        values = defaultdict(list)
        for query_id in queries:
            held_out = self.catalog.outfits_by_item[query_id]
            candidates = [o for o in self.catalog.outfits if o not in held_out]
            outfits = rng.sample(candidates, MAX_OUTFITS)
            targets = self.companion_targets(query_id, held_out)
            values["companion_hit@10"].append(self.companion_hit(outfits, targets))
            values["diversity"].append(self.diversity(outfits))
        return average(values)
def average(values):
    result = {}
    for name, numbers in values.items():
        numbers = [number for number in numbers if number is not None]
        result[name] = round(sum(numbers) / len(numbers), 3) if numbers else None
    return result


def print_table(title, rows):
    columns = list(next(iter(rows.values())))
    print(f"\n{title}")
    print(f"{'configuration':<22}" + "".join(f"{column:>25}" for column in columns))
    for name, metrics in rows.items():
        cells = "".join(f"{str(metrics.get(column, '-')):>25}" for column in columns)
        print(f"{name:<22}{cells}")
def main():
    parser = argparse.ArgumentParser(description="Ablation of the retrieval stage.")
    parser.add_argument("--queries", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=30)
    args = parser.parse_args()

    evaluator = Evaluator()
    queries = random.Random(args.seed).sample(evaluator.ids, args.queries)

    ablation = {
        name: evaluator.run(queries, weights, args.top_k)
        for name, weights in CONFIGURATIONS.items()
    }
    ablation["random outfits"] = evaluator.run_random(queries, args.seed)
    print_table(f"Ablation ({args.queries} queries, top_k={args.top_k})", ablation)

    current = CONFIGURATIONS["hybrid (current)"]
    top_k_sweep = {
        f"top_k = {top_k}": evaluator.run(queries, current, top_k)
        for top_k in (10, 30, 60)
    }
    print_table("Sweep: top_k (hybrid weights)", top_k_sweep)

    weight_sweep = {
        f"category = {weight}": evaluator.run(queries, {**current, "category": weight}, args.top_k)
        for weight in (0.0, 0.25, 0.5, 1.0)
    }
    print_table("Sweep: category weight (cosine 0.5, colour 0.25)", weight_sweep)

    output = ROOT_DIR / "logs" / "evaluation_retrieval.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "queries": args.queries,
        "seed": args.seed,
        "top_k": args.top_k,
        "ablation": ablation,
        "top_k_sweep": top_k_sweep,
        "category_weight_sweep": weight_sweep,
    }
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved {output}")
    
    try:
        from plot_evaluation import plot_retrieval
    except ImportError as error:
        print(f"Figures skipped: {error}")
    else:
        plot_retrieval(report)



if __name__ == "__main__":
    main()
