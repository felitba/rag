import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
import logging
import os
import time
import uuid
from dotenv import load_dotenv, find_dotenv


from fastapi import FastAPI, Request, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from home_made_rag.catalog import Catalog
from home_made_rag.chroma_db import ChromaDB
from home_made_rag.config import CONFIG, IMAGES_DIR, MONITORING_DB
from home_made_rag.embedding import ClipEmbedding
from home_made_rag.fashion_clip_classifier import FashionClipCategoryClassifier
from home_made_rag.generator import OutfitGenerator
from home_made_rag.image_preprocessing import ForegroundPreprocessor
from home_made_rag.monitoring import MonitoringService
from home_made_rag.rag import RAG


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load credentials before constructing any OpenAI-backed component.
    load_dotenv(find_dotenv())
    project_dir = Path(__file__).resolve().parent.parent
    log_file = project_dir / "Orag.log"
    logging.basicConfig(
        level=getattr(logging, CONFIG["project"].get("log_level", "INFO").upper(), logging.INFO),
        format="[%(levelname)s] - %(message)s",
        handlers=[logging.FileHandler(log_file, mode="a", encoding="utf-8"), logging.StreamHandler()],
        force=True,
    )
    logger = logging.getLogger(__name__)
    logger.info("Starting fashion RAG application")

    catalog = Catalog()
    embedding = ClipEmbedding()
    chroma = ChromaDB()
    image_settings = CONFIG.get("image_preprocessing", {})
    image_preprocessor = ForegroundPreprocessor(
        enabled=image_settings.get("enabled", True),
        model_name=image_settings.get("model_name", "u2net"),
        max_side=image_settings.get("max_side", 1024),
        background_color=image_settings.get("background_color", "#f1f0ec"),
        padding_ratio=image_settings.get("padding_ratio", 0.10),
    )
    monitor_settings = CONFIG.get("monitoring", {})
    monitor = None
    if monitor_settings.get("enabled", True):
        try:
            monitor = MonitoringService(
                database_path=MONITORING_DB,
                retention_days=monitor_settings.get("retention_days", 30),
            )
        except Exception:
            logger.exception("Monitoring could not start; recommendations will continue without it")
    attribute_classifier = (
        FashionClipCategoryClassifier(
            labels=[item.get("category") for item in catalog.items.values() if item.get("category")]
        )
        if CONFIG.get("category_classifier", {}).get("enabled", False)
        else None
    )
    generator = None
    generator_settings = CONFIG.get("generator", {})
    if generator_settings.get("enabled", False):
        # ITBA's LiteLLM proxy expects its own credential (LLM_API_KEY).
        # Keep OPENAI_API_KEY as a backwards-compatible fallback.
        api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
        credential_name = "LLM_API_KEY" if os.getenv("LLM_API_KEY") else "OPENAI_API_KEY"
        fallback_providers = []
        for name in ("openrouter", "huggingface"):
            fallback_settings = generator_settings.get("fallbacks", {}).get(name, {})
            if not fallback_settings.get("enabled", True):
                continue
            credential_env = fallback_settings.get("api_key_env")
            fallback_key = os.getenv(credential_env) if credential_env else None
            if not fallback_key:
                continue
            model_env = fallback_settings.get("model_env")
            fallback_provider = {
                "name": name,
                "api_key": fallback_key,
                "model": (os.getenv(model_env) if model_env else None) or fallback_settings.get("model"),
                "base_url": fallback_settings.get("base_url"),
                "credential_source": credential_env,
                "input_rate": fallback_settings.get("input_usd_per_million_tokens", 0.0),
                "cached_input_rate": fallback_settings.get("cached_input_usd_per_million_tokens", 0.0),
                "output_rate": fallback_settings.get("output_usd_per_million_tokens", 0.0),
                "json_object_mode": fallback_settings.get("json_object_mode", False),
                "max_completion_tokens": fallback_settings.get("max_completion_tokens", 1400),
            }
            extra_body = {}
            if fallback_settings.get("require_parameters"):
                extra_body["provider"] = {"require_parameters": True}
            if fallback_settings.get("reasoning_effort"):
                extra_body["reasoning_effort"] = fallback_settings["reasoning_effort"]
            if extra_body:
                fallback_provider["extra_body"] = extra_body
            fallback_providers.append(fallback_provider)

        openrouter_settings = generator_settings.get("fallbacks", {}).get("openrouter", {})
        if openrouter_settings.get("enabled", True) and not os.getenv(
            openrouter_settings.get("api_key_env", "OPENROUTER_API_KEY")
        ):
            logger.warning(
                "OpenRouter free fallback is enabled but its API key is missing; "
                "set OPENROUTER_API_KEY in .env."
            )

        if api_key or fallback_providers:
            generator = OutfitGenerator(
                api_key=api_key,
                model=os.getenv("OPENAI_MODEL", generator_settings.get("model", "gpt-4.1-mini")),
                base_url=os.getenv("OPENAI_BASE_URL", generator_settings.get("base_url")),
                max_generated_outfits=generator_settings.get("max_generated_outfits", 3),
                timeout_seconds=generator_settings.get("timeout_seconds", 25),
                input_usd_per_million_tokens=generator_settings.get("input_usd_per_million_tokens", 0.4),
                cached_input_usd_per_million_tokens=generator_settings.get("cached_input_usd_per_million_tokens", 0.1),
                output_usd_per_million_tokens=generator_settings.get("output_usd_per_million_tokens", 1.6),
                monitor=monitor,
                credential_source=credential_name,
                fallbacks=fallback_providers,
            )
            provider_order = ", ".join(
                f"{provider['name']}:{provider['model']}" for provider in generator.providers
            )
            logger.info("Outfit generator fallback chain: %s", provider_order)
        else:
            logger.warning(
                "Generator is enabled but no provider keys are set; using retrieval fallback. "
                "Configure LLM_API_KEY, HF_TOKEN, or OPENROUTER_API_KEY."
            )

    app.state.rag = RAG(
        embedding,
        chroma,
        catalog,
        generator=generator,
        attribute_classifier=attribute_classifier,
        monitor=monitor,
        image_preprocessor=image_preprocessor,
    )
    app.state.monitor = monitor
    logger.info("Ready. Indexed garments: %d", chroma.count())
    
    yield

app = FastAPI(lifespan=lifespan)


@app.middleware("http")
async def monitor_recommendation_requests(request: Request, call_next):
    monitor = getattr(request.app.state, "monitor", None)
    if monitor is None or request.url.path != "/api/recommend" or request.method != "POST":
        return await call_next(request)

    request_id = uuid.uuid4().hex
    started = time.perf_counter()
    try:
        monitor.begin_request(request_id, request.method, request.url.path)
    except Exception:
        logging.getLogger(__name__).exception("Could not record request in monitoring database")
        request_id = None
    request.state.monitor_request_id = request_id
    try:
        response = await call_next(request)
    except Exception as error:
        if request_id:
            try:
                monitor.finish_request(
                    request_id,
                    status_code=500,
                    duration_ms=(time.perf_counter() - started) * 1000,
                    error_type=type(error).__name__,
                )
            except Exception:
                logging.getLogger(__name__).exception("Could not finish request in monitoring database")
        raise

    if request_id:
        try:
            monitor.finish_request(
                request_id,
                status_code=response.status_code,
                duration_ms=(time.perf_counter() - started) * 1000,
                error_type="HTTPError" if response.status_code >= 400 else None,
            )
            response.headers["X-Request-ID"] = request_id
        except Exception:
            logging.getLogger(__name__).exception("Could not finish request in monitoring database")
    return response

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
    
app.mount("/images", StaticFiles(directory=IMAGES_DIR), name="images") #  makes data/images/123.jpg available at http://localhost:8000/images/123.jpg

@app.get("/health")
def health(request: Request):
    return {"status": "ok", "indexed_garments": request.app.state.rag.chroma.count()}


@app.get("/api/monitor/summary")
def monitor_summary(request: Request):
    monitor = getattr(request.app.state, "monitor", None)
    if monitor is None:
        raise HTTPException(status_code=503, detail="Monitoring is disabled")
    return monitor.summary()


@app.get("/api/monitor/requests")
def monitor_requests(request: Request, limit: int = 50):
    monitor = getattr(request.app.state, "monitor", None)
    if monitor is None:
        raise HTTPException(status_code=503, detail="Monitoring is disabled")
    return {"requests": monitor.list_requests(limit)}


@app.get("/api/monitor/requests/{request_id}")
def monitor_request_detail(request: Request, request_id: str):
    monitor = getattr(request.app.state, "monitor", None)
    if monitor is None:
        raise HTTPException(status_code=503, detail="Monitoring is disabled")
    result = monitor.get_request(request_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Request not found")
    return result


@app.get("/api/monitor/requests/{request_id}/image/{variant}")
def monitor_request_image(request: Request, request_id: str, variant: str):
    monitor = getattr(request.app.state, "monitor", None)
    if monitor is None:
        raise HTTPException(status_code=503, detail="Monitoring is disabled")
    image_path = monitor.query_image_path(request_id, variant)
    if image_path is None:
        raise HTTPException(status_code=404, detail="Request image not found")
    return FileResponse(image_path)

# ------------- user prompt receive -------------- #
@app.post("/api/recommend")
def recommend(request: Request, file: UploadFile | None = None, message: str | None = Form(None)):
    
    rag = request.app.state.rag
    text = (message or "").strip()
    monitor = getattr(request.app.state, "monitor", None)
    request_id = getattr(request.state, "monitor_request_id", None)
    if monitor:
        input_type = "image+text" if file is not None and text else "image" if file is not None else "text"
        try:
            monitor.set_request_input(request_id, input_type)
        except Exception:
            logging.getLogger(__name__).exception("Could not record request input type")
    
    if file is not None:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temporary:
            temporary.write(file.file.read())
            upload_path = Path(temporary.name)

        try:
            outfits, explanation, generation_status, retrieved_context, query_attributes, context_precision = rag.recommend_with_generation(
                image_path=upload_path,
                text=text or None,
                request_id=request_id,
            )
            metadata_matches = rag.compare_retrieved_metadata(retrieved_context, query_attributes)
        finally:
            upload_path.unlink()
    elif text:
        outfits, explanation, generation_status, retrieved_context, query_attributes, context_precision = rag.recommend_with_generation(
            text=text,
            request_id=request_id,
        )
        metadata_matches = []
    else:
        raise HTTPException(status_code=400, detail="Enviá una foto o un mensaje.")

    if monitor:
        try:
            monitor.set_generation_status(request_id, generation_status)
            monitor.set_query_attributes(request_id, query_attributes)
        except Exception:
            logging.getLogger(__name__).exception("Could not record request analysis")

    for recommendation_set in (retrieved_context, outfits):
        for outfit in recommendation_set:
            for item in outfit["items"]:
                for field in list(item):
                    if field.startswith("_"):
                        item.pop(field, None)

    return {
        "recommendations": outfits,
        "retrieved_outfits": retrieved_context,
        "explanation": explanation,
        "query_attributes": query_attributes,
        "context_precision": context_precision,
        "retrieved_metadata_matches": metadata_matches,
        "generation_enabled": bool(CONFIG.get("generator", {}).get("enabled", False)),
        "generation_status": generation_status,
        "request_id": request_id,
    }
