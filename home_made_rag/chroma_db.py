import chromadb

from home_made_rag.config import CONFIG, PROJECT_DIR


class ChromaDB:
    def __init__(self):
        settings = CONFIG["vector_store"]

        db_path = PROJECT_DIR / settings["persist_directory"]

        self.client = chromadb.PersistentClient(
            path=str(db_path)
        )

        self.collection = self.client.get_or_create_collection(
            name=settings["collection_name"],
            metadata={
                "hnsw:space": settings["distance_metric"]
            },
        )

    def add_image(self, item_id, embedding, metadata):
        self.collection.upsert(
            ids=[str(item_id)],
            embeddings=[embedding],
            metadatas=[metadata],
        )

    def retrieve(self, embedding, top_k=None):
        if top_k is None:
            top_k = CONFIG["vector_store"]["top_k"]

        return self.collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
            include=["metadatas", "distances"],
        )