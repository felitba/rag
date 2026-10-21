from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps, UnidentifiedImageError
from starlette.concurrency import run_in_threadpool

from app.config import (
    CHROMA_PATH,
    CLIP_MODEL,
    COLLECTION_NAME,
    MAX_IMAGE_PIXELS,
    MAX_OUTFITS,
    MAX_UPLOAD_BYTES,
    OPENAI_MODEL,
    PROJECT_ROOT,
    TOP_K_GARMENTS,
    UPLOADS_DIR,
    MAX_IMAGES_PER_OUTFIT_FOR_LLM,
    CATALOG_PATH,
)
from app.services.catalog_service import CatalogService
from app.services.chroma_service import ChromaService
from app.services.clip_service import ClipService
from app.services.openai_service import OpenAIRecommendationService
from app.services.outfit_service import OutfitService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fashion_rag")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Cargando catálogo...")
    catalog = CatalogService(PROJECT_ROOT)

    logger.info("Cargando CLIP (%s)...", CLIP_MODEL)
    clip_service = ClipService(model_name=CLIP_MODEL)

    chroma_service = ChromaService(
        persist_directory=CHROMA_PATH,
        collection_name=COLLECTION_NAME,
        clip_service=clip_service,
        reset_collection=False,
    )

    outfit_service = OutfitService(
        catalog=catalog,
        clip_service=clip_service,
        chroma_service=chroma_service,
    )

    openai_service = None
    import os
    if os.getenv("OPENAI_API_KEY") and os.getenv("OPENAI_API_KEY") != "replace_me":
        try:
            openai_service = OpenAIRecommendationService(
                model_name=OPENAI_MODEL,
                max_images_per_outfit=MAX_IMAGES_PER_OUTFIT_FOR_LLM,
            )
        except Exception:
            logger.exception("No se pudo inicializar ChatOpenAI; se usará fallback.")
    else:
        logger.warning(
            "OPENAI_API_KEY no está configurada. El endpoint podrá recuperar outfits, "
            "pero devolverá una explicación de fallback hasta configurar .env."
        )

    app.state.catalog = catalog
    app.state.clip_service = clip_service
    app.state.chroma_service = chroma_service
    app.state.outfit_service = outfit_service
    app.state.openai_service = openai_service

    logger.info("Fashion RAG listo. Prendas indexadas: %s", chroma_service.count())
    yield


app = FastAPI(
    title="Fashion RAG API",
    description=(
        "Recupera prendas visualmente similares con CLIP + ChromaDB y "
        "explica outfits prearmados mediante LangChain + OpenAI."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# The catalog images are public storefront assets. Uploaded user images are not
# saved under a publicly served path for longer than the request duration.
CATALOG_PATH.mkdir(parents=True, exist_ok=True)
app.mount(
    "/catalog",
    StaticFiles(directory=str(CATALOG_PATH)),
    name="catalog",
)


def _make_public_recommendations(recommendations: list[dict]) -> list[dict]:
    public_outfits = []
    for outfit in recommendations:
        public_items = []
        for item in outfit["items"]:
            public_items.append({
                "garment_id": item["garment_id"],
                "category": item.get("category", "unknown"),
                "color": item.get("color", "unknown"),
                "style": item.get("style", "unknown"),
                "price": item.get("price"),
                "available": item.get("available", True),
                "image_url": item["image_url"],
            })
        public_outfits.append({
            "outfit_id": outfit["outfit_id"],
            "score": outfit["score"],
            "matched_garment_ids": outfit["matched_garment_ids"],
            "items": public_items,
        })
    return public_outfits


def _prepare_upload(raw_bytes: bytes) -> Path:
    try:
        with Image.open(BytesIO(raw_bytes)) as opened:
            if opened.format not in {"JPEG", "PNG", "WEBP"}:
                raise HTTPException(
                    status_code=415,
                    detail="Formato no soportado. Utilizá JPEG, PNG o WEBP.",
                )
            if opened.width * opened.height > MAX_IMAGE_PIXELS:
                raise HTTPException(
                    status_code=413,
                    detail="La imagen tiene demasiados píxeles.",
                )
            image = ImageOps.exif_transpose(opened).convert("RGB")
            image.thumbnail((2048, 2048))

        destination = UPLOADS_DIR / f"{uuid4().hex}.jpg"
        image.save(destination, format="JPEG", quality=92, optimize=True)
        return destination
    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(
            status_code=400,
            detail="El archivo no contiene una imagen válida.",
        ) from exc


@app.get("/health")
def health(request: Request) -> dict:
    chroma_service: ChromaService = request.app.state.chroma_service
    return {
        "status": "ok",
        "service": "fashion-rag",
        "indexed_garments": chroma_service.count(),
        "openai_explanations_enabled": request.app.state.openai_service is not None,
    }


@app.post("/api/recommend")
async def recommend_outfits(
    request: Request,
    file: UploadFile = File(...),
) -> dict:
    upload_path: Path | None = None

    try:
        raw_bytes = await file.read(MAX_UPLOAD_BYTES + 1)
        if not raw_bytes:
            raise HTTPException(status_code=400, detail="El archivo está vacío.")
        if len(raw_bytes) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"La imagen supera el límite de {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.",
            )
        upload_path = _prepare_upload(raw_bytes)

        chroma_service: ChromaService = request.app.state.chroma_service
        if chroma_service.count() == 0:
            raise HTTPException(
                status_code=503,
                detail="ChromaDB está vacío. Ejecutá scripts/index_catalog.py primero.",
            )

        outfit_service: OutfitService = request.app.state.outfit_service
        recommendations = await run_in_threadpool(
            outfit_service.recommend,
            upload_path,
            TOP_K_GARMENTS,
            MAX_OUTFITS,
        )

        if not recommendations:
            return {
                "recommendations": [],
                "explanation": (
                    "No encontramos outfits prearmados disponibles asociados "
                    "con las prendas visualmente similares."
                ),
                "llm_available": request.app.state.openai_service is not None,
            }

        openai_service: OpenAIRecommendationService | None = (
            request.app.state.openai_service
        )
        if openai_service is None:
            explanation = (
                "Encontramos outfits relacionados con la prenda. Configurá "
                "OPENAI_API_KEY en el archivo .env para generar explicaciones "
                "personalizadas con el modelo multimodal."
            )
            llm_available = False
        else:
            try:
                explanation = await run_in_threadpool(
                    openai_service.explain,
                    upload_path,
                    recommendations,
                )
                llm_available = True
            except Exception:
                logger.exception("Falló la explicación mediante OpenAI.")
                explanation = (
                    "Encontramos outfits relacionados con la prenda, pero no "
                    "se pudo generar la explicación automática en este momento. "
                    "Las recomendaciones recuperadas siguen disponibles."
                )
                llm_available = False

        return {
            "recommendations": _make_public_recommendations(recommendations),
            "explanation": explanation,
            "llm_available": llm_available,
        }

    finally:
        await file.close()
        if upload_path is not None:
            upload_path.unlink(missing_ok=True)
