"""Ablate generator inputs: catalog metadata only vs metadata plus retrieved photos.

The retrieval stage is the production RAG path: the configured query rewriter runs,
the default HybridGarmentReranker reranks CLIP hits, and the top_n_outfits candidates
are collected once.  Two generators then receive those same candidates.  They are
identical except for use_images, so the only experimental difference is whether the
model sees the catalog photos in addition to the normal JSON metadata.

Compatible generated outfits are scored with the type-aware compatibility model.
Usage from query rewriting and from both generator conditions is recorded, so the
full experiment cost is printed and saved with the results.

Usage:
    python scripts/evaluate_generator_inputs.py 30 --device cpu

Results: logs/generator_inputs_<N>.json
Figure:  reports/generator_inputs_compatibility_<N>.png
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
os.chdir(REPO)

from home_made_rag.config import CONFIG

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("n_queries", nargs="?", type=int, default=30, help="number of requests (default: 30)")
parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"),
                    default=CONFIG["embeddings"]["device"],
                    help="device for CLIP and the type-aware scorer (default: config value)")
parser.add_argument("--query-set", type=Path, default=None,
                    help="fixed query JSON (default: logs/judged_text_queries_<N>.json)")
parser.add_argument("--max-images", type=int, default=CONFIG["generator"].get("max_images", 60),
                    help="maximum catalog photos attached in the with_images condition")
ARGS = parser.parse_args()

if ARGS.n_queries < 1 or ARGS.max_images < 1:
    raise ValueError("--n_queries and --max-images must be positive")
if ARGS.device == "cpu":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # before torch, rembg and FashionCLIP are imported

from dotenv import load_dotenv
from home_made_rag.catalog import Catalog
from home_made_rag.chroma_db import ChromaDB
from home_made_rag.embedding import ClipEmbedding
from home_made_rag.fashion_clip_classifier import FashionClipCategoryClassifier
from home_made_rag.generator import OutfitGenerator
from home_made_rag.hybrid_reranker import HybridGarmentReranker
from home_made_rag.image_preprocessing import ForegroundPreprocessor
from home_made_rag.rag import OutOfScopeRequest, RAG
from home_made_rag.rewriting import QueryRewriter
from home_made_rag.type_aware_compatibility import TypeAwareCompatibilityScorer


CONDITIONS = ("metadata_only", "with_images")


class UsageRecorder:
    """Capture LLM usage/traces without writing to the application monitoring DB."""

    def __init__(self) -> None:
        self.usage: dict[str, list[dict[str, Any]]] = {}
        self.rewriting: dict[str, dict[str, Any]] = {}

    def record_llm_call_start(self, request_id, model, credential_source, credential_fingerprint):
        pass

    def record_llm_call_error(self, request_id):
        pass

    def record_llm_usage(self, request_id, *, input_tokens, cached_input_tokens=0,
                         output_tokens, estimated_cost_usd=0.0):
        self.usage.setdefault(str(request_id), []).append({
            "input": int(input_tokens or 0),
            "cached_input": int(cached_input_tokens or 0),
            "output": int(output_tokens or 0),
            "cost_usd": float(estimated_cost_usd or 0.0),
        })

    def set_query_rewriting(self, request_id, trace):
        self.rewriting[str(request_id)] = trace

    # Other RAG monitor hooks; no storage needed for this study.
    def begin_request(self, *args, **kwargs):
        pass

    def finish_request(self, *args, **kwargs):
        pass

    def record_query_images(self, *args, **kwargs):
        pass

    def record_candidates(self, *args, **kwargs):
        pass

    def set_context_precision(self, *args, **kwargs):
        pass

    def set_stage_timings(self, *args, **kwargs):
        pass

    def set_pipeline_trace(self, *args, **kwargs):
        pass

    def totals(self, request_ids: list[str]) -> dict[str, float | int]:
        records = [record for request_id in request_ids for record in self.usage.get(request_id, [])]
        return {
            "input": sum(record["input"] for record in records),
            "cached_input": sum(record["cached_input"] for record in records),
            "output": sum(record["output"] for record in records),
            "cost_usd": round(sum(record["cost_usd"] for record in records), 6),
        }


def load_queries() -> tuple[list[str], dict[str, Any]]:
    path = ARGS.query_set or REPO / "logs" / f"judged_text_queries_{ARGS.n_queries}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    queries = payload.get("queries", [])
    if not isinstance(queries, list) or len(queries) != ARGS.n_queries:
        raise ValueError(
            f"{path} does not contain exactly {ARGS.n_queries} queries. "
            "Use --query-set with a matching file or change the query count."
        )
    relative = path.relative_to(REPO) if path.is_relative_to(REPO) else path
    info = {"source": "loaded", "path": str(relative), **payload.get("generation", {})}
    return queries, info


def paired_change(rows: list[dict[str, Any]]) -> dict[str, Any]:
    pairs = []
    for row in rows:
        before = row["conditions"]["metadata_only"].get("mean_compatibility_score")
        after = row["conditions"]["with_images"].get("mean_compatibility_score")
        if before is not None and after is not None:
            pairs.append(after - before)
    if not pairs:
        return {"mean": None, "se": None, "improved": 0, "worsened": 0, "unchanged": 0, "n": 0}
    se = statistics.stdev(pairs) / len(pairs) ** 0.5 if len(pairs) > 1 else 0.0
    return {
        "mean": round(statistics.mean(pairs), 6),
        "se": round(se, 6),
        "improved": sum(diff > 1e-9 for diff in pairs),
        "worsened": sum(diff < -1e-9 for diff in pairs),
        "unchanged": sum(abs(diff) <= 1e-9 for diff in pairs),
        "n": len(pairs),
    }


def generate_condition(
    rag: RAG,
    generator: OutfitGenerator,
    retrieved: list[dict[str, Any]],
    text: str,
    request_id: str,
    monitor: UsageRecorder,
) -> dict[str, Any]:
    attempts: list[dict[str, Any]] = []
    gen_request_id = f"{request_id}::{generator.condition_name}"
    result: dict[str, Any] = {
        "use_images": generator.providers[0]["supports_images"],
        "provider": None,
        "model": None,
        "generated_outfit_count": 0,
        "accepted_outfit_count": 0,
        "outfits": [],
        "mean_compatibility_score": None,
        "error": None,
        "attempts": attempts,
    }
    try:
        generated = generator.generate(
            retrieved,
            prompt=text,
            request_id=gen_request_id,
            attempt_log=attempts,
            photo=None,  # text-only study, as in the production text endpoint
        )
        result["generated_outfit_count"] = len(generated.get("outfits") or [])
        # The production resolver trims drafts using self.generator.max_generated_outfits.
        rag.generator = generator
        verdicts: list[dict[str, Any]] = []
        resolved = rag._resolve_generated(generated, retrieved, report=verdicts)
        if attempts:
            attempts[-1]["drafts"] = verdicts
        if not resolved:
            raise ValueError("No valid generated outfits")

        rag._evaluate_outfits(resolved, {}, {}, "generator_output")
        scores = []
        for outfit in resolved:
            report = outfit.get("type_aware_compatibility") or {}
            score = report.get("compatibility_score")
            if score is not None:
                scores.append(float(score))
            result["outfits"].append({
                "outfit_id": outfit.get("outfit_id"),
                "garment_ids": [item["garment_id"] for item in outfit.get("items", [])],
                "look_kind": outfit.get("look_kind"),
                "compatibility_score": score,
                "status": report.get("status"),
                "pair_count": report.get("pair_count", 0),
            })
        if not scores:
            raise RuntimeError("No generated outfit could be scored by the type-aware model")
        result["mean_compatibility_score"] = round(statistics.mean(scores), 6)
        result["accepted_outfit_count"] = len(resolved)
        if attempts:
            result["provider"] = attempts[-1].get("provider")
            result["model"] = attempts[-1].get("model")
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {str(error)[:300]}"

    result["usage"] = monitor.totals([gen_request_id])
    return result


def main() -> None:
    started = time.time()
    load_dotenv(REPO / ".env")

    settings = CONFIG["generator"]
    api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Set LLM_API_KEY or OPENAI_API_KEY before running this study")
    credential_name = "LLM_API_KEY" if os.getenv("LLM_API_KEY") else "OPENAI_API_KEY"
    model = os.getenv("OPENAI_MODEL", settings.get("model", "gpt-4.1-mini"))
    base_url = os.getenv("OPENAI_BASE_URL", settings.get("base_url"))
    queries, query_set_info = load_queries()

    monitor = UsageRecorder()
    catalog = Catalog()
    embedding = ClipEmbedding(device=ARGS.device)
    chroma = ChromaDB()

    image_settings = CONFIG.get("image_preprocessing", {})
    image_preprocessor = ForegroundPreprocessor(
        enabled=image_settings.get("enabled", True),
        model_name=image_settings.get("model_name", "u2net"),
        max_side=image_settings.get("max_side", 1024),
        background_color=image_settings.get("background_color", "#f1f0ec"),
        padding_ratio=image_settings.get("padding_ratio", 0.10),
    )
    attribute_classifier = (
        FashionClipCategoryClassifier(
            labels=[item.get("category") for item in catalog.items.values() if item.get("category")]
        )
        if CONFIG.get("category_classifier", {}).get("enabled", False)
        else None
    )
    rewriting_settings = CONFIG.get("query_rewriting", {})
    rewriter = (
        QueryRewriter(
            api_key=api_key,
            model=model,
            base_url=base_url,
            max_queries=rewriting_settings.get("max_queries", 4),
            mode=rewriting_settings.get("mode", "auto"),
            input_usd_per_million_tokens=settings.get("input_usd_per_million_tokens", 0.4),
            output_usd_per_million_tokens=settings.get("output_usd_per_million_tokens", 1.6),
            monitor=monitor,
        )
        if rewriting_settings.get("enabled", False)
        else None
    )
    compatibility_settings = CONFIG.get("type_aware_compatibility", {})
    scorer = TypeAwareCompatibilityScorer(
        enabled=compatibility_settings.get("enabled", False),
        checkpoint_path=REPO / compatibility_settings.get(
            "checkpoint_path", "models/type_aware/model_best.pth.tar"
        ),
        typespaces_path=REPO / compatibility_settings.get(
            "typespaces_path", "models/type_aware/typespaces.p"
        ),
        device=ARGS.device,
        category_aliases=compatibility_settings.get("category_aliases", {}),
        max_batch_size=compatibility_settings.get("max_batch_size", 16),
    )
    if not scorer.available:
        raise RuntimeError(f"Type-aware compatibility scorer is unavailable: {scorer.reason}")

    rag = RAG(
        embedding,
        chroma,
        catalog,
        attribute_classifier=attribute_classifier,
        hybrid_reranker=HybridGarmentReranker(),  # production weights from config.json
        monitor=monitor,
        image_preprocessor=image_preprocessor,
        rewriter=rewriter,
        compatibility_scorer=scorer,
    )

    generators: dict[str, OutfitGenerator] = {}
    for condition in CONDITIONS:
        generator = OutfitGenerator(
            api_key=api_key,
            model=model,
            base_url=base_url,
            max_generated_outfits=settings.get("max_generated_outfits", 3),
            timeout_seconds=settings.get("timeout_seconds", 25),
            input_usd_per_million_tokens=settings.get("input_usd_per_million_tokens", 0.4),
            cached_input_usd_per_million_tokens=settings.get("cached_input_usd_per_million_tokens", 0.1),
            output_usd_per_million_tokens=settings.get("output_usd_per_million_tokens", 1.6),
            monitor=monitor,
            credential_source=credential_name,
            use_images=condition == "with_images",
            max_images=ARGS.max_images,
            extra_drafts=settings.get("extra_drafts", 2),
        )
        generator.condition_name = condition  # type: ignore[attr-defined]
        generators[condition] = generator

    rows: list[dict[str, Any]] = []
    candidate_limit = settings.get("top_n_outfits", CONFIG["retrieval"]["max_outfits"])
    print("Retrieval runs once per request with production rewriting and reranking; "
          "only the generator input changes.", flush=True)
    for index, text in enumerate(queries, start=1):
        request_id = f"g{index:03d}"
        try:
            retrieved, query_attributes = rag.recommend(
                text=text,
                max_outfits=candidate_limit,
                return_query_attributes=True,
                request_id=request_id,
            )
        except OutOfScopeRequest as error:
            rows.append({"query": text, "request_id": request_id,
                         "retrieval_error": f"OutOfScopeRequest: {error}", "conditions": {}})
            print(f"[{index}/{len(queries)}] {text!r} | out of scope", flush=True)
            continue

        trace = monitor.rewriting.get(request_id, {})
        conditions = {
            condition: generate_condition(rag, generator, retrieved, text, request_id, monitor)
            for condition, generator in generators.items()
        }
        rows.append({
            "query": text,
            "request_id": request_id,
            "rewritten_queries": [
                {key: value for key, value in query.items() if key != "embedding"}
                for query in trace.get("queries", [])
            ],
            "query_attributes": query_attributes,
            "retrieved_outfit_ids": [outfit["outfit_id"] for outfit in retrieved],
            "conditions": conditions,
        })

        values = " | ".join(
            f"{condition}="
            + (f"{conditions[condition]['mean_compatibility_score']:.3f}"
               if conditions[condition]["mean_compatibility_score"] is not None else "FAILED")
            for condition in CONDITIONS
        )
        print(f"[{index}/{len(queries)}] {text[:55]!r} | {values}", flush=True)

    summary: dict[str, Any] = {}
    for condition in CONDITIONS:
        values = [
            row["conditions"][condition]["mean_compatibility_score"]
            for row in rows
            if condition in row["conditions"]
            and row["conditions"][condition]["mean_compatibility_score"] is not None
        ]
        summary[condition] = {
            "mean_compatibility_score": round(statistics.mean(values), 6) if values else None,
            "valid_requests": len(values),
            "failed_requests": sum(
                condition in row["conditions"]
                and row["conditions"][condition]["mean_compatibility_score"] is None
                for row in rows
            ),
            "accepted_outfits": sum(
                row["conditions"][condition].get("accepted_outfit_count", 0)
                for row in rows if condition in row["conditions"]
            ),
        }
    summary["change_vs_metadata_only"] = paired_change(rows)

    usage_by_component = {
        "query_rewriting": monitor.totals([row["request_id"] for row in rows]),
        **{
            condition: monitor.totals([
                f"{row['request_id']}::{condition}" for row in rows
            ])
            for condition in CONDITIONS
        },
    }
    total = {
        key: sum(component[key] for component in usage_by_component.values())
        for key in ("input", "cached_input", "output")
    }
    total["cost_usd"] = round(sum(component["cost_usd"] for component in usage_by_component.values()), 6)
    summary["cost"] = {
        "components": usage_by_component,
        "total": total,
        "rates_usd_per_million_tokens": {
            "generator_input": settings.get("input_usd_per_million_tokens", 0.4),
            "generator_cached_input": settings.get("cached_input_usd_per_million_tokens", 0.1),
            "generator_output": settings.get("output_usd_per_million_tokens", 1.6),
            "query_rewriting_input": settings.get("input_usd_per_million_tokens", 0.4),
            "query_rewriting_output": settings.get("output_usd_per_million_tokens", 1.6),
        },
    }
    pipeline = rag._config_snapshot()
    pipeline.update({
        "generator": True,
        "generator_model": model,
        "top_n_outfits": candidate_limit,
        "max_generated_outfits": settings.get("max_generated_outfits", 3),
        "generator_uses_images": "ablation: metadata_only=False, with_images=True",
    })
    summary.update({
        "queries": len(rows),
        "device": ARGS.device,
        "model": model,
        "max_images": ARGS.max_images,
        "query_set": query_set_info,
        "scorer": scorer.status(),
        "pipeline": pipeline,
        "ablation": {
            "only_difference": "generator.use_images",
            "conditions": {"metadata_only": False, "with_images": True},
            "retrieval_calls_per_request": 1,
            "fallbacks_enabled": False,
        },
        "seconds": round(time.time() - started, 1),
    })

    output_path = REPO / "logs" / f"generator_inputs_{len(rows)}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2, ensure_ascii=False),
                           encoding="utf-8")
    print("COST " + json.dumps(summary["cost"], ensure_ascii=False))
    print("SUMMARY " + json.dumps(summary, ensure_ascii=False))
    print(f"Saved {output_path}")

    try:
        from plot_evaluation import plot_generator_inputs
    except ImportError as error:
        print(f"Figure skipped: {error}")
    else:
        try:
            plot_generator_inputs({"summary": summary, "rows": rows})
        except ValueError as error:
            print(f"Figure skipped: {error}")


if __name__ == "__main__":
    main()
