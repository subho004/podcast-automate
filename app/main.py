from fastapi import FastAPI
from app.api.routes import router as api_router
from app.core.config import settings

app = FastAPI(title=settings.PROJECT_NAME, version=settings.VERSION)

app.include_router(api_router, prefix="/api")

@app.get("/")
async def root():
    return {"message": f"Welcome to {settings.PROJECT_NAME}"}
