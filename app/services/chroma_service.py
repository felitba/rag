from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import chromadb
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings

from app.services.clip_service import ClipService


class ClipImageEmbeddings(Embeddings):
    """LangChain Embeddings adapter that treats each input string as an image path.

    This allows LangChain's Chroma integration to call CLIP both when indexing
    documents and when searching with a query image. Text embeddings are not used.
    """

    def __init__(self, clip_service: ClipService) -> None:
        self.clip_service = clip_service

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.clip_service.embed_image(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self.clip_service.embed_image(text)


class ChromaService:
    def __init__(
        self,
        persist_directory: Path,
        collection_name: str,
        clip_service: ClipService,
        reset_collection: bool = False,
    ) -> None:
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        self.embedding_function = ClipImageEmbeddings(clip_service)

        self.vector_store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embedding_function,
            persist_directory=str(self.persist_directory),
            collection_metadata={"hnsw:space": "cosine"},
        )
        if reset_collection:
            # Full rebuild: this also removes IDs no longer present in garments.json.
            self.vector_store.reset_collection()

    def count(self) -> int:
        # Chroma's internal collection attribute name differs between package
        # versions. Prefer the native count method; fall back to public get().
        collection = getattr(self.vector_store, "_chroma_collection", None)
        if collection is None:
            collection = getattr(self.vector_store, "_collection", None)
        if collection is not None:
            return int(collection.count())
        return len(self.vector_store.get().get("ids", []))

    def index_garment(
        self,
        garment: dict[str, Any],
        image_path: Path,
        outfit_ids: list[str],
    ) -> None:
        metadata: dict[str, Any] = {
            "garment_id": garment["garment_id"],
            "image_path": garment["image_path"],
            "category": str(garment.get("category", "unknown")),
            "color": str(garment.get("color", "unknown")),
            "style": str(garment.get("style", "unknown")),
            "available": bool(garment.get("available", True)),
            # JSON string is portable across Chroma versions, including versions
            # where metadata values are restricted to scalar types.
            "outfit_ids_json": json.dumps(outfit_ids, ensure_ascii=False),
        }
        if garment.get("price") is not None:
            metadata["price"] = float(garment["price"])

        self.vector_store.add_texts(
            texts=[str(image_path.resolve())],
            metadatas=[metadata],
            ids=[garment["garment_id"]],
        )

    def search_similar(
        self,
        query_image_path: Path,
        top_k: int = 30,
    ) -> list[dict[str, Any]]:
        total = self.count()
        if total == 0 or top_k <= 0:
            return []

        results = self.vector_store.similarity_search_with_score(
            query=str(query_image_path.resolve()),
            k=min(top_k, total),
            filter={"available": True},
        )

        matches: list[dict[str, Any]] = []
        for document, distance in results:
            try:
                outfit_ids = json.loads(
                    document.metadata.get("outfit_ids_json", "[]")
                )
            except (TypeError, json.JSONDecodeError):
                outfit_ids = []

            matches.append({
                "garment_id": document.metadata.get("garment_id", document.id),
                "outfit_ids": outfit_ids,
                "distance": float(distance),
                "metadata": document.metadata,
            })
        return matches
