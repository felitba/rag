"""Precision@k from independent vision-LLM grades, across reranker colour weights.

Usage: python scripts/judged_context_precision.py [N] [--device auto|cpu|cuda|mps]
Writes logs/judged_context_precision_<N>.json and reports/context_precision_judged_<N>.png.
Costs about USD 0.005 per query. --device cpu forces CLIP, FashionCLIP and rembg onto CPU.

Per query garment (held out of its own results): background removal -> CLIP -> top-30.
The judge sees the query photo and ONE retrieved photo per call, no metadata, and grades
it 0-3. A grade >= 2 counts as relevant. The same grades are reused for every reranker
configuration, so the colour-weight sweep needs no extra judge calls. There is no proxy
evaluator: the judge labels are the only source of relevance.
"""
import argparse
import base64, io, json, os, random, statistics, sys, time
from concurrent.futures import ThreadPoolExecutor
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO); os.chdir(REPO)
from home_made_rag.config import CONFIG

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("n_queries", nargs="?", type=int, default=30, help="number of queries (default: 30)")
parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default=CONFIG["embeddings"]["device"],
                    help="device for CLIP, FashionCLIP and rembg (default: config value)")
ARGS = parser.parse_args()
if ARGS.device == "cpu":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # set before torch and rembg are imported

from dotenv import load_dotenv; load_dotenv(".env")
from openai import OpenAI
from PIL import Image
from pydantic import BaseModel, Field
from home_made_rag.catalog import Catalog
from home_made_rag.chroma_db import ChromaDB
from home_made_rag.embedding import ClipEmbedding
from home_made_rag.fashion_clip_classifier import FashionClipCategoryClassifier
from home_made_rag.hybrid_reranker import HybridGarmentReranker
from home_made_rag.image_preprocessing import ForegroundPreprocessor
from home_made_rag.rag import RAG

N_QUERIES = ARGS.n_queries
SEED, TOP_K = 42, CONFIG["vector_store"]["top_k"]
K_VALUES = (5, 10, 30)
CATEGORY_WEIGHT = CONFIG["hybrid_reranker"]["weights"]["category"]
COLOUR_WEIGHTS = (0.0, 0.05, 0.15, 0.25, 0.35)
NO_RERANK = "no rerank (CLIP order)"
METRICS = [f"{prefix}_at_{k}" for k in K_VALUES for prefix in ("relevant", "precision")]
gen = CONFIG["generator"]
client = OpenAI(api_key=os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY"),
                base_url=os.getenv("OPENAI_BASE_URL", gen["base_url"]), timeout=60, max_retries=3)
MODEL = os.getenv("OPENAI_MODEL", gen["model"])

RERANKERS = {NO_RERANK: None}
for colour in COLOUR_WEIGHTS:
    cosine = round(1.0 - CATEGORY_WEIGHT - colour, 6)
    if cosine <= 0:
        raise ValueError(f"Colour weight {colour} leaves no cosine weight")
    RERANKERS[f"colour {colour:.2f}"] = HybridGarmentReranker({
        "cosine_similarity": cosine, "color": colour, "category": CATEGORY_WEIGHT,
    })

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

catalog = Catalog(); settings = CONFIG["image_preprocessing"]
rag = RAG(ClipEmbedding(device=ARGS.device), ChromaDB(), catalog,
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
    to_judge = [m["item_id"] for m in retrieved]
    with ThreadPoolExecutor(8) as pool:
        results = list(pool.map(lambda item_id: judge(path, catalog.image_path(item_id)), to_judge))
    for _, t_in, t_out in results:
        tokens[0] += t_in; tokens[1] += t_out
    grades_by_id = {item_id: grade for item_id, (grade, _, _) in zip(to_judge, results)}
    configurations = {}
    for name, reranker in RERANKERS.items():
        ranked = retrieved if reranker is None else reranker.rerank(retrieved, attributes)
        grades = [grades_by_id[m["item_id"]] for m in ranked]
        values = {}
        for k in K_VALUES:
            relevant = sum(g >= 2 for g in grades[:k])
            values[f"relevant_at_{k}"] = relevant
            values[f"precision_at_{k}"] = round(relevant / k, 4)
        configurations[name] = values
    row = {"query_id": query_id, "true_category": catalog.items[query_id]["category"],
           "detected_category": (attributes.get("category") or {}).get("label"),
           "configurations": configurations}
    rows.append(row)
    print(f"[{index}/{N_QUERIES}] {query_id} {row['true_category']:10s} " +
          " ".join(f"{name}={values['precision_at_10']:.2f}" for name, values in configurations.items()), flush=True)

def mean(values):
    usable = [value for value in values if value is not None]
    return round(sum(usable) / len(usable), 4) if usable else None

def diff(baseline, comparison):
    changes = [b - a for a, b in zip(baseline, comparison)]
    error = statistics.stdev(changes) / len(changes) ** 0.5 if len(changes) > 1 else 0.0
    return {"mean": round(statistics.mean(changes), 4), "se": round(error, 4),
            "improved": sum(x > 1e-9 for x in changes), "worsened": sum(x < -1e-9 for x in changes)}

summary = {"configurations": {}}
for name in RERANKERS:
    values = {metric: mean([r["configurations"][name][metric] for r in rows]) for metric in METRICS}
    if name != NO_RERANK:
        values["change_vs_no_rerank"] = {
            f"precision_at_{k}": diff(
                [r["configurations"][NO_RERANK][f"precision_at_{k}"] for r in rows],
                [r["configurations"][name][f"precision_at_{k}"] for r in rows],
            ) for k in K_VALUES
        }
    summary["configurations"][name] = values
summary.update(relevance_threshold=2, k_values=list(K_VALUES), category_weight=CATEGORY_WEIGHT,
               colour_weights=list(COLOUR_WEIGHTS), queries=N_QUERIES, seed=SEED, top_k=TOP_K,
               device=ARGS.device, judge_model=MODEL, input_tokens=tokens[0], output_tokens=tokens[1],
               cost_usd=round((tokens[0] * gen["input_usd_per_million_tokens"] + tokens[1] * gen["output_usd_per_million_tokens"]) / 1e6, 4),
               seconds=round(time.time() - started))
os.makedirs("logs", exist_ok=True)
with open(os.path.join("logs", f"judged_context_precision_{N_QUERIES}.json"), "w", encoding="utf-8") as output:
    json.dump({"summary": summary, "rows": rows}, output, indent=1)
print("SUMMARY " + json.dumps(summary))
try:
    from plot_evaluation import plot_judged_context_precision
except ImportError as error:
    print(f"Figure skipped: {error}")
else:
    plot_judged_context_precision({"summary": summary, "rows": rows})
