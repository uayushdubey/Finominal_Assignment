from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )

    APP_NAME: str = "Portfolio Optimizer API"
    ENV: str = "development"
    LOG_LEVEL: str = "INFO"
    
    HOST: str = "127.0.0.1"
    PORT: int = 8000
    
    ALLOWED_ORIGINS: str = "http://localhost:3000,http://localhost:8000,http://127.0.0.1:8000"

    @property
    def cors_origins(self) -> List[str]:
        return [origin.strip() for origin in self.ALLOWED_ORIGINS.split(",") if origin.strip()]

    @property
    def is_dev(self) -> bool:
        return self.ENV.lower() == "development"

# Instantiate settings to be imported across the application
settings = Settings()
