from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI

from helpers.config import get_settings
from core.logger import logger
from services.Rag_Service import RAGService
from services.recommendation_service import RecommendationService
from routes import base, rag

@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()

    app.state.rag_service = RAGService()
    app.state.grouped_df = pd.read_csv(settings.DATAFRAME_PROCESSED_LOCATION)

    indexed_count = app.state.rag_service.vector_store.count()
    expected_count = len(app.state.grouped_df)

    if indexed_count < expected_count:
        logger.info(
            f"Vector store has {indexed_count} items, "
            f"expected {expected_count} — reindexing."
        )
        app.state.rag_service.index_phones(app.state.grouped_df)

    app.state.recommendation_service = RecommendationService(
        rag_service=app.state.rag_service,
        grouped_df=app.state.grouped_df,
    )

    yield


app = FastAPI(lifespan=lifespan)

app.include_router(base.router)
app.include_router(rag.router)