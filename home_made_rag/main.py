import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
import logging
import os
from dotenv import load_dotenv, find_dotenv


from fastapi import FastAPI, Request, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from home_made_rag.catalog import Catalog
from home_made_rag.chroma_db import ChromaDB
from home_made_rag.config import CONFIG, IMAGES_DIR
from home_made_rag.embedding import ClipEmbedding
from home_made_rag.fashion_clip_reranker import FashionClipReranker
from home_made_rag.generator import OutfitGenerator
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
    reranker = FashionClipReranker() if CONFIG["reranker"]["enabled"] else None
    generator = None
    generator_settings = CONFIG.get("generator", {})
    if generator_settings.get("enabled", False):
        api_key = os.getenv("OPENAI_API_KEY")
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
            )
            logger.info("Outfit generator enabled (model=%s)", generator.model)
        else:
            logger.warning("Generator is enabled but OPENAI_API_KEY is missing; using retrieval fallback")

    app.state.rag = RAG(embedding, None, chroma, catalog, reranker, generator)
    logger.info("Ready. Indexed garments: %d", chroma.count())
    
    yield

app = FastAPI(lifespan=lifespan)

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

# ------------- user prompt receive -------------- #
@app.post("/api/recommend")
def recommend(request: Request, file: UploadFile | None = None, message: str | None = Form(None)):
    
    rag = request.app.state.rag
    text = (message or "").strip()
    
    if file is not None:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temporary:
            temporary.write(file.file.read())
            upload_path = Path(temporary.name)

        try:
            outfits, explanation, generation_status = rag.recommend_with_generation(
                image_path=upload_path,
                text=text or None,
            )
        finally:
            upload_path.unlink()
    elif text:
        outfits, explanation, generation_status = rag.recommend_with_generation(text=text)
    else:
        raise HTTPException(status_code=400, detail="Enviá una foto o un mensaje.")

    for outfit in outfits:
        for item in outfit["items"]:
            for field in list(item):
                if field.startswith("_"):
                    item.pop(field, None)

    return {
        "recommendations": outfits,
        "explanation": explanation,
        "generation_enabled": bool(CONFIG.get("generator", {}).get("enabled", False)),
        "generation_status": generation_status,
    }
