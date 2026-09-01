# services/rag_service.py

import pandas as pd

from services.text_embedder import TextEmbedder
from services.VectorStore import VectorStore
from core.logger import logger


class RAGService:

    def __init__(
        self,
        embedder: TextEmbedder = None,
        vector_store: VectorStore = None
    ):
        self.embedder = embedder or TextEmbedder()
        self.vector_store = vector_store or VectorStore()

    def index_phones(self, grouped_df: pd.DataFrame) -> int:

        valid = grouped_df[
            grouped_df["spec_text"]
            .fillna("")
            .str.strip()
            .str.len()
            > 0
        ]

        if valid.empty:
            logger.warning("No valid phones found for indexing.")
            return 0

        logger.info(f"Indexing {len(valid)} phones")

        texts = valid["spec_text"].tolist()

        embeddings = self.embedder.encode(texts).tolist()

        metadatas = valid[
            [
                "brand",
                "model",
                "price_min",
                "price_max",
                "in_stock"
            ]
        ].to_dict("records")

        ids = valid["model_id"].astype(str).tolist()

        self.vector_store.upsert(
            ids=ids,
            texts=texts,
            embeddings=embeddings,
            metadatas=metadatas
        )

        logger.info(f"Successfully indexed {len(valid)} phones")

        return len(valid)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        where: dict | None = None
    ):
        if not query or not query.strip():
            logger.warning("Empty query received.")
            return None

        logger.info(f"Retrieving top {top_k} results for query")

        query_embedding = self.embedder.encode_query(query).tolist()

        return self.vector_store.query(
            query_embedding,
            top_k=top_k,
            where=where
        )