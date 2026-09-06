from fastapi import APIRouter, Depends, HTTPException, Request
from models.chat import ChatRecommendationRequest, ChatTurnRequest
from services.Rag_Service import RAGService
from services.chat_recommendation_service import ChatRecommendationService
from services.recommendation_service import RecommendationService
from helpers.LLM_Finder import LLMFinder          

router = APIRouter(
    prefix="/api/v1/rag",
    tags=["api_v1", "rag"],
)

def get_rag_service(request: Request) -> RAGService:
    return request.app.state.rag_service

def get_recommendation_service(request: Request) -> RecommendationService:
    return request.app.state.recommendation_service

def get_chat_recommendation_service(request: Request) -> ChatRecommendationService:
    return request.app.state.chat_recommendation_service

@router.get("/search")
def search_phones(query: str, top_k: int = 5, rag: RAGService = Depends(get_rag_service)):
    return rag.retrieve(query, top_k=top_k)

@router.get("/local-models")
def list_local_models():
    return {"models": LLMFinder.discover_local_models()}

@router.get("/recommend")
def recommend_phones(
    query: str,
    budget: float | None = None,
    brand: str | None = None,
    min_storage: float | None = None,
    min_ram: float | None = None,
    network: str | None = None,
    category: str | None = None,
    top_k: int = 5,
    service: RecommendationService = Depends(get_recommendation_service),
):
    return service.hybrid_retrieve(
        query=query, budget=budget, brand=brand,
        min_storage=min_storage, min_ram=min_ram,
        network=network, category=category, top_k=top_k,
    )

@router.post("/chat/recommend")
def chat_recommend_phones(
    payload: ChatRecommendationRequest,
    service: ChatRecommendationService = Depends(get_chat_recommendation_service),
):
    try:
        return service.recommend(
            message=payload.message,
            top_k=payload.top_k,
            explicit_preferences=payload.preferences,
        )
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error

@router.post("/chat/turn")
def chat_turn(
    payload: ChatTurnRequest,
    service: ChatRecommendationService = Depends(get_chat_recommendation_service),
):
    try:
        return service.chat_turn(
            message=payload.message,
            history=payload.history,
            preferences=payload.preferences,
            use_case=payload.use_case,
            completed_slots=payload.completed_slots,
            top_k=payload.top_k,
            llm_config=payload.llm_config,
        )
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error