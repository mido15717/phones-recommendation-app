from fastapi import FastAPI, APIRouter
import os

router = APIRouter(
    prefix = "/api/v1",
    tags = ["api_v1"],
)

@router.get("/")
def welcomeapp():
    app_name = os.getenv("APP_NAME")
    app_version = os.getenv("APP_VERSION")
    return {"message": f"Welcome to {app_name} v{app_version}"}