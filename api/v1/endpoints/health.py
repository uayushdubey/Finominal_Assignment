from fastapi import APIRouter
from datetime import datetime, timezone
from typing import Dict, Any
from core.config import settings

router = APIRouter()

@router.get("/health", response_model=Dict[str, Any], tags=["System"])
def health_check() -> Dict[str, Any]:
    """
    Health check endpoint to verify system status, environment config, and application lifelines.
    """
    return {
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "environment": settings.ENV,
        "app_name": settings.APP_NAME,
        "version": "1.0.0"
    }
