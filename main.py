import langchain
import os
import logging
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
from chroma_db import ChromaDB
from config import CONFIG
from embedding import ClipEmbedding
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from llm import LLM
from rag import RAG
import rag


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. Startup: execute once when the app starts
    print("Starting application...")

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


    # --------------TODO: Data preprocessing and loading-------------- #



    # --------------Db set up -------------- #

    chroma_db = ChromaDB()

    # --------------Embedding set up -------------- #

    embedding_function = ClipEmbedding(model_name=CONFIG["embeddings"]["model_name"])

    # -------------- TODO : Data Insertion -------------- #


    # -------------- TODO : LLM set up (openAI API Key set up missing) -------------- #
    
    model  = LLM()

    # --------------RAG set up -------------- #
    app.state.rag = RAG(embedding_function, model, chroma_db)
    
    yield

    # 2. Shutdown: execute when the app stops
    print("Shutting down application...")

# ------------- API set up -------------- #

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ------------- user prompt receive -------------- #
@app.get("/recommendations")
async def get_recommendations(user_prompt: str):
    # 1. Process the user prompt and generate recommendations
    recommendations = app.state.rag.retrieve(user_prompt).generate_response()

    # 2. Return the recommendations as a JSON response
    return {"recommendations": recommendations}
