from fastapi import APIRouter, Depends
from helpers.config import get_settings, Settings
from services.Rag_Service import RAGService

router = APIRouter(
    prefix="/api/v1/rag",
    tags=["api_v1","rag"],
)

def get_rag_service() -> RAGService:
    return RAGService()

@router.get("/search")
def search_phones(query: str, top_k: int = 5, rag: RAGService = Depends(get_rag_service)):
    results = rag.retrieve(query, top_k=top_k)
    return results