from core.logger import logger
import chromadb

class VectorStore:

    def __init__(self, persist_directory: str, collection_name: str = "phones") -> None:
        logger.info("Initializing ChromaDB client")
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert(self, ids: list[str], texts: list[str], embeddings, metadatas: list[dict]):
        # upsert, not add — safe to re-run on updated data without duplicating
        logger.info(f"Upserting {len(ids)} items into ChromaDB")
        self.collection.upsert(
            ids=ids, documents=texts, embeddings=embeddings, metadatas=metadatas
        )

    def query(self, query_embedding, top_k: int = 5, where: dict | None = None):
        logger.info(f"Querying vector store for top_k={top_k}")
        return self.collection.query(query_embeddings=[query_embedding], n_results=top_k, where=where)

    def count(self):
        return self.collection.count()