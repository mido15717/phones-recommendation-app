from sentence_transformers import SentenceTransformer
import numpy as np
from core.logger import logger


class TextEmbedder:

    def __init__(self, model_name: str):
        logger.info(f"Initializing embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)

    def encode(
        self,
        texts: list[str],
        batch_size: int = 32
    ) -> np.ndarray:

        valid_texts = [
            text for text in texts
            if isinstance(text, str) and text.strip()
        ]

        if not valid_texts:
            return np.array([])

        return self.model.encode(
            valid_texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )

    def encode_query(self, query: str) -> np.ndarray:

        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query cannot be empty")

        return self.model.encode(
            query,
            convert_to_numpy=True
        )