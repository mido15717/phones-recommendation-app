from fastapi import FastAPI
from routes import base, rag

app = FastAPI()
app.include_router(base.router)
app.include_router(rag.router)