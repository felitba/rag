import langchain
import os
import logging
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
from chroma_db import ChromaDB
from config import CONFIG
from embedding import ClipEmbedding


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

# --------------Data preprocessing and loading-------------- #



# --------------Db set up -------------- #

chroma_db = ChromaDB()

# --------------Embedding set up -------------- #

embedding_function = ClipEmbedding(model_name=CONFIG["embeddings"]["model_name"])

# --------------Data Insertion -------------- #

