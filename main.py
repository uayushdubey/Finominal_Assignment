import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from core.config import settings
from core.logging import setup_logging
from core.exception_handlers import register_exception_handlers
from api.v1.router import api_router
from api.v1.endpoints.health import router as health_router

# Initialize structured logging setup
setup_logging()
logger = logging.getLogger("app.main")

app = FastAPI(
    title=settings.APP_NAME,
    description="Production-grade Portfolio Optimizer API backend.",
    version="1.0.0",
    docs_url="/docs" if settings.is_dev else None,
    redoc_url="/redoc" if settings.is_dev else None,
)

# Apply CORS middleware using allowed origins from environment
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Exception Handler setup
register_exception_handlers(app)

# Include APIs
# Root level health endpoint (matches task: GET /health)
app.include_router(health_router)

# Versioned API routes under /api/v1
app.include_router(api_router, prefix="/api/v1")

logger.info(f"Application {settings.APP_NAME} starting up in {settings.ENV} mode.")
