from services.lookup_service import ListingService
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI

from helpers.config import get_settings
from core.logger import logger
from services.Rag_Service import RAGService
from services.chat_recommendation_service import ChatRecommendationService
from services.recommendation_service import RecommendationService
from routes import base, rag

@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()

    app.state.rag_service = RAGService()
    app.state.grouped_df = pd.read_csv(settings.DATAFRAME_PROCESSED_LOCATION)
    app.state.cleaned_df = pd.read_csv(settings.DATAFRAME_CLEANED_LOCATION)

    logger.info(
        f"Reindexing vector store with {len(app.state.grouped_df)} phones from current dataset."
    )
    app.state.rag_service.index_phones(app.state.grouped_df)

    app.state.listing_service = ListingService(cleaned_df=app.state.cleaned_df)

    app.state.recommendation_service = RecommendationService(
        rag_service=app.state.rag_service,
        grouped_df=app.state.grouped_df,
    )
    app.state.chat_recommendation_service = ChatRecommendationService(
        recommendation_service=app.state.recommendation_service,
        grouped_df=app.state.grouped_df,
        listing_service=app.state.listing_service,
    )

    yield

app = FastAPI(lifespan=lifespan)

app.include_router(base.router)
app.include_router(rag.router)