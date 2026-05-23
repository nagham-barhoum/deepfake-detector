from pydantic_settings import BaseSettings
from pathlib import Path

class Settings(BaseSettings):
    # App
    APP_NAME: str
    APP_VERSION: str
    DEBUG: bool
    SECRET_KEY: str

    # Database
    DATABASE_URL: str

    # Upload
    MAX_IMAGE_SIZE_MB: int
    MAX_VIDEO_SIZE_MB: int
    MAX_VIDEO_DURATION_SEC: int
    UPLOAD_DIR: str

    # ML
    ML_MODELS_DIR: str
    CONFIDENCE_THRESHOLD: float

    class Config:
        env_file = ".env"

settings = Settings()
