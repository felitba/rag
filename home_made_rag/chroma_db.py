import chromadb

from home_made_rag.config import CONFIG, CHROMA_DIR


class ChromaDB:
    def __init__(self):
        settings = CONFIG["vector_store"]

        self.client = chromadb.PersistentClient(
            path=str(CHROMA_DIR)
        )

        self.collection = self.client.get_or_create_collection(
            name=settings["collection_name"],
            metadata={
                "hnsw:space": settings["distance_metric"]
            },
        )

    def retrieve(self, embedding, top_k=None, where=None):
        if top_k is None:
            top_k = CONFIG["vector_store"]["top_k"]
            
        if top_k < 1:
            raise ValueError("top_k must be greater than zero")
        
        count = self.count()
        
        if count == 0:
            return {
                "ids": [[]],
                "metadatas":[[]],
                "documents":[[]],
                "distances":[[]]
            }

        return self.collection.query(
            query_embeddings=[embedding],
            n_results=min(top_k, count),
            where=where,
            include=["metadatas", "documents", "distances"],
        )
    
    def count(self):
        return self.collection.count()
    
    def add_batch(self, ids, embeddings, metadatas, documents):
        if not ids:
            return
        
        self.collection.upsert(
            ids = [str(item_id) for item_id in ids],
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents
        )
        
    def _get_collection(self):
        settings = CONFIG["vector_store"]

        return self.client.get_or_create_collection(
            name=settings["collection_name"],
            metadata={
                "hnsw:space": settings["distance_metric"],
            },
        )
        
    def reset(self):
        self.client.delete_collection(
            name=CONFIG["vector_store"]["collection_name"]
        )
        self.collection = self._get_collection()
