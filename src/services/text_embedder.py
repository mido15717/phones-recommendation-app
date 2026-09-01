from core.logger import logger

class TextEmbedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            logger.error("Please install the 'sentence-transformers' package to use TextEmbedder.")
            raise

        self.model = SentenceTransformer(model_name)
        logger.info(f"Initialized TextEmbedder with model '{model_name}'")

    def embed_text(self, text: str):
        if not isinstance(text, str) or not text.strip():
            logger.warning("Empty or invalid text provided for embedding.")
            return None
        embedding = self.model.encode(text)
        logger.debug(f"Generated embedding of length {len(embedding)} for text: {text[:30]}...")
        return embedding