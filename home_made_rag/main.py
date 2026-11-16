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
from fastapi.staticfiles import StaticFiles

from home_made_rag.catalog import Catalog
from home_made_rag.chroma_db import ChromaDB
from home_made_rag.config import CONFIG, IMAGES_DIR, MONITORING_DB
from home_made_rag.embedding import ClipEmbedding
from home_made_rag.fashion_clip_classifier import FashionClipCategoryClassifier
from home_made_rag.generator import OutfitGenerator
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
        credential_name = "LLM_API_KEY" if os.getenv("LLM_API_KEY") else "OPENAI_API_KEY"
        api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
        if api_key:
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
            )
            logger.info("Outfit generator enabled (model=%s)", generator.model)
            logger.info("Outfit generator credential loaded from %s", credential_name)
        else:
            logger.warning("Generator is enabled but LLM_API_KEY/OPENAI_API_KEY is missing; using retrieval fallback")

    app.state.rag = RAG(
        embedding,
        chroma,
        catalog,
        generator=generator,
        attribute_classifier=attribute_classifier,
        monitor=monitor,
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
            outfits, explanation, generation_status, retrieved_context, query_attributes = rag.recommend_with_generation(
                image_path=upload_path,
                text=text or None,
                request_id=request_id,
            )
            metadata_matches = rag.compare_retrieved_metadata(retrieved_context, query_attributes)
        finally:
            upload_path.unlink()
    elif text:
        outfits, explanation, generation_status, _, query_attributes = rag.recommend_with_generation(
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

    for outfit in outfits:
        for item in outfit["items"]:
            for field in list(item):
                if field.startswith("_"):
                    item.pop(field, None)

    return {
        "recommendations": outfits,
        "explanation": explanation,
        "query_attributes": query_attributes,
        "retrieved_metadata_matches": metadata_matches,
        "generation_enabled": bool(CONFIG.get("generator", {}).get("enabled", False)),
        "generation_status": generation_status,
        "request_id": request_id,
    }
