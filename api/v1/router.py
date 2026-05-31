from fastapi import APIRouter
from api.v1.endpoints import health, optimizer

api_router = APIRouter()

# Register endpoint routers
api_router.include_router(health.router)
api_router.include_router(optimizer.router)
