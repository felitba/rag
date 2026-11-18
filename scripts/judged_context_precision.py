"""Context precision with independent, graded labels from a vision LLM judge.

Usage: python scripts/judged_context_precision.py [number of queries, default 30]
Writes logs/judged_context_precision_<N>.json. Costs about USD 0.007 per query.

Per query garment (a catalog photo, held out of its own results): the app's own code path
(background removal -> CLIP -> top-30 -> hybrid rerank). The judge sees the query photo and
ONE retrieved photo per call, no metadata, and grades it 0-3. AP@30 is computed before and
after the rerank at two relevance thresholds, next to the proxy (category + colour).
"""
import base64, io, json, math, os, random, statistics, sys, time
from concurrent.futures import ThreadPoolExecutor
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO); os.chdir(REPO)
from dotenv import load_dotenv; load_dotenv(".env")
from openai import OpenAI
from PIL import Image
from pydantic import BaseModel, Field
from home_made_rag.catalog import Catalog
from home_made_rag.chroma_db import ChromaDB
from home_made_rag.config import CONFIG
from home_made_rag.context_precision import evaluate_context_precision
from home_made_rag.embedding import ClipEmbedding
from home_made_rag.fashion_clip_classifier import FashionClipCategoryClassifier
from home_made_rag.image_preprocessing import ForegroundPreprocessor
from home_made_rag.rag import RAG

N_QUERIES = int(sys.argv[1]) if len(sys.argv) > 1 else 30
SEED, TOP_K = 42, CONFIG["vector_store"]["top_k"]
gen = CONFIG["generator"]
client = OpenAI(api_key=os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY"),
                base_url=os.getenv("OPENAI_BASE_URL", gen["base_url"]), timeout=60, max_retries=3)
MODEL = os.getenv("OPENAI_MODEL", gen["model"])

class Grade(BaseModel):
    grade: int = Field(ge=0, le=3)

JUDGE = (
    "You compare two fashion product photos: the QUERY garment and one CANDIDATE. Grade how good "
    "the candidate is as a result for a shopper searching with the query photo:\n"
    "0 = a different type of garment or accessory (a skirt for a jacket, a bag for a shoe).\n"
    "1 = the same type of garment, but a clearly different look (different colour family and style).\n"
    "2 = the same type and a similar look: shares the colour family, or the material, cut or style.\n"
    "3 = nearly the same garment.\n"
    "Judge only what is visible. Return the grade."
)

def data_url(path, side=384):
    image = Image.open(path).convert("RGB"); image.thumbnail((side, side))
    buffer = io.BytesIO(); image.save(buffer, "JPEG", quality=80)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()

def judge(query_path, candidate_path):
    response = client.chat.completions.parse(
        model=MODEL, temperature=0, response_format=Grade, max_completion_tokens=20,
        messages=[{"role": "system", "content": JUDGE}, {"role": "user", "content": [
            {"type": "text", "text": "QUERY"}, {"type": "image_url", "image_url": {"url": data_url(query_path)}},
            {"type": "text", "text": "CANDIDATE"}, {"type": "image_url", "image_url": {"url": data_url(candidate_path)}}]}],
    )
    return response.choices[0].message.parsed.grade, response.usage.prompt_tokens, response.usage.completion_tokens

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

catalog = Catalog(); settings = CONFIG["image_preprocessing"]
rag = RAG(ClipEmbedding(), ChromaDB(), catalog,
          attribute_classifier=FashionClipCategoryClassifier(labels=[i["category"] for i in catalog.items.values()]),
          image_preprocessor=ForegroundPreprocessor(enabled=settings["enabled"], model_name=settings["model_name"],
              max_side=settings["max_side"], background_color=settings["background_color"], padding_ratio=settings["padding_ratio"]))
rng = random.Random(SEED)
queries = rng.sample(sorted(catalog.items), N_QUERIES)
rows, tokens, started = [], [0, 0], time.time()
for index, query_id in enumerate(queries, 1):
    path = catalog.image_path(query_id)
    image = rag.image_preprocessor.process(path)
    retrieved = [m for m in rag._search(rag.embedding.embed_image(image), TOP_K + 1) if m["item_id"] != query_id][:TOP_K]
    attributes = rag.classify_query_image(image)
    reranked = rag.hybrid_reranker.rerank(retrieved, attributes)
    # Controls for the judge itself, not part of the ranking: the query's own photo (expect 3)
    # and two garments of another category (expect 0).
    others = [i for i in sorted(catalog.items) if catalog.items[i]["category"] != catalog.items[query_id]["category"]]
    controls = [query_id, *rng.sample(others, 2)]
    to_judge = [m["item_id"] for m in retrieved] + controls
    with ThreadPoolExecutor(8) as pool:
        results = list(pool.map(lambda item_id: judge(path, catalog.image_path(item_id)), to_judge))
    for _, t_in, t_out in results:
        tokens[0] += t_in; tokens[1] += t_out
    grades = [g for g, _, _ in results]
    grade = dict(zip(to_judge[:len(retrieved)], grades))
    control_grades = grades[len(retrieved):]
    proxy_before = evaluate_context_precision(retrieved, attributes, catalog)
    proxy_after = evaluate_context_precision(reranked, attributes, catalog)
    proxy_label = {m["item_id"]: bool(m["context_relevance"]["relevant"]) for m in proxy_before["annotated_matches"]}
    before = [grade[m["item_id"]] for m in retrieved]; after = [grade[m["item_id"]] for m in reranked]
    row = {"query_id": query_id, "true_category": catalog.items[query_id]["category"],
           "detected_category": (attributes.get("category") or {}).get("label"),
           "control_self": control_grades[0], "control_other": control_grades[1:],
           "proxy_ap_before": proxy_before["score"], "proxy_ap_after": proxy_after["score"],
           "ndcg10_before": ndcg(before), "ndcg10_after": ndcg(after),
           "mean_grade_top5_before": sum(before[:5]) / 5, "mean_grade_top5_after": sum(after[:5]) / 5,
           "before": [m["item_id"] for m in retrieved], "after": [m["item_id"] for m in reranked],
           "grades": grade, "proxy_labels": proxy_label}
    for name, threshold in (("same_type", 1), ("similar", 2)):
        row[f"n_{name}"] = sum(g >= threshold for g in before)
        row[f"ap_{name}_before"] = average_precision([g >= threshold for g in before])
        row[f"ap_{name}_after"] = average_precision([g >= threshold for g in after])
    rows.append(row)
    print(f"[{index}/{N_QUERIES}] {query_id} {row['true_category']:10s} same-type {row['n_same_type']:2d} similar {row['n_similar']:2d} | "
          f"AP similar {row['ap_similar_before']:.3f}->{row['ap_similar_after']:.3f} | proxy {row['proxy_ap_before']}->{row['proxy_ap_after']}", flush=True)

def mean(key):
    values = [r[key] for r in rows if r[key] is not None]
    return round(sum(values) / len(values), 4)
def diff(a, b):
    d = [r[b] - r[a] for r in rows]
    return {"mean": round(statistics.mean(d), 4), "se": round(statistics.stdev(d) / len(d) ** 0.5, 4),
            "improved": sum(x > 1e-9 for x in d), "worsened": sum(x < -1e-9 for x in d)}
summary = {key: mean(key) for key in ("proxy_ap_before", "proxy_ap_after", "ap_same_type_before", "ap_same_type_after",
           "ap_similar_before", "ap_similar_after", "ndcg10_before", "ndcg10_after",
           "mean_grade_top5_before", "mean_grade_top5_after", "n_same_type", "n_similar")}
summary["change_ap_same_type"] = diff("ap_same_type_before", "ap_same_type_after")
summary["change_ap_similar"] = diff("ap_similar_before", "ap_similar_after")
summary["change_ndcg10"] = diff("ndcg10_before", "ndcg10_after")
all_grades = [(g, r["proxy_labels"][i]) for r in rows for i, g in r["grades"].items()]
summary["grade_distribution"] = {g: sum(x == g for x, _ in all_grades) for g in range(4)}
summary["mean_grade_by_clip_rank"] = {name: round(statistics.mean(r["grades"][i] for r in rows for i in r["before"][lo:hi]), 3)
                                      for name, lo, hi in (("1-5", 0, 5), ("6-15", 5, 15), ("16-30", 15, 30))}
summary["proxy_relevant_judged_similar"] = round(sum(g >= 2 for g, p in all_grades if p) / max(1, sum(p for _, p in all_grades)), 3)
summary["judged_similar_proxy_relevant"] = round(sum(p for g, p in all_grades if g >= 2) / max(1, sum(g >= 2 for g, _ in all_grades)), 3)
summary["controls"] = {"self_grades": {g: sum(r["control_self"] == g for r in rows) for g in range(4)},
                       "other_category_grades": {g: sum(x == g for r in rows for x in r["control_other"]) for g in range(4)}}
summary.update(queries=N_QUERIES, seed=SEED, top_k=TOP_K, judge_model=MODEL, input_tokens=tokens[0], output_tokens=tokens[1],
               cost_usd=round((tokens[0] * gen["input_usd_per_million_tokens"] + tokens[1] * gen["output_usd_per_million_tokens"]) / 1e6, 4),
               seconds=round(time.time() - started))
os.makedirs("logs", exist_ok=True)
with open(os.path.join("logs", f"judged_context_precision_{N_QUERIES}.json"), "w", encoding="utf-8") as output:
    json.dump({"summary": summary, "rows": rows}, output, indent=1)
print("SUMMARY " + json.dumps(summary))
