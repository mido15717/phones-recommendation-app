from fastapi import APIRouter, Depends
from helpers.config import get_settings, Settings

router = APIRouter(
    prefix="/api/v1",
    tags=["api_v1","rag"],
)

@router.get("/")
def welcomeapp(settings: Settings = Depends(get_settings)):
    return {"message": f"Welcome to {settings.APP_NAME} v{settings.APP_VERSION}"}