import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
import logging
from dotenv import load_dotenv, find_dotenv


from fastapi import FastAPI, Request, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from home_made_rag.catalog import Catalog
from home_made_rag.chroma_db import ChromaDB
from home_made_rag.config import IMAGES_DIR
from home_made_rag.embedding import ClipEmbedding
from home_made_rag.rag import RAG


@asynccontextmanager
async def lifespan(app: FastAPI):
    
    # 1. Startup: execute once when the app starts
    print("Starting application...")
    
    # 2. Basic setup
        
    catalog = Catalog()
    embedding = ClipEmbedding()
    chroma = ChromaDB()
    app.state.rag = RAG(embedding, None, chroma, catalog)
    print(f"Listo. Prendas indexadas: {chroma.count()}")

    # -------------- Logger set up -------------- #

    PROJECT_DIR = Path(__file__).resolve().parent.parent
    LOG_FILE = PROJECT_DIR / "Orag.log"

    _ = load_dotenv(find_dotenv())

    logging.basicConfig(level=logging.DEBUG,
                        format='[%(levelname)s] - %(message)s ',
                        handlers=[
                                logging.FileHandler(LOG_FILE, mode='w', encoding='utf-8'),
                                logging.StreamHandler(),
                        ],
                        force=True)
    logger = logging.getLogger(__name__)
    logger.info("Langchain Demo Initialized")
    
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
            outfits = rag.recommend(image_path=upload_path)
        finally:
            upload_path.unlink()
    elif text:
        outfits = rag.recommend(text=text)
    else:
        raise HTTPException(status_code=400, detail="Enviá una foto o un mensaje.")

    for outfit in outfits:
        for item in outfit["items"]:
            del item["_image_abs_path"]

    return {
        "recommendations": outfits,
        "explanation": "Outfits recuperados sin LLM (prueba de componentes).",
    }
