from fastapi import APIRouter, Depends
from helpers.config import get_settings, Settings

router = APIRouter(
    prefix="/api/v1/rag",
    tags=["api_v1","rag"],
)