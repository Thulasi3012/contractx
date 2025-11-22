import os
import urllib.parse
from pydantic_settings import BaseSettings


class AppSettings(BaseSettings):
    """
    Unified configuration for the Contract Analyzer application.
    Reads environment variables from `.env` or the system.
    """

    # ============================================================
    # 🌐 App Configuration
    # ============================================================
    APP_NAME: str = os.getenv("APP_NAME", "Contract Analyzer")
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "Development")
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "info")
    API_PREFIX: str = os.getenv("API_PREFIX", "/api")

    # ============================================================
    # 🧠 Gemini / AI Agent Settings
    # ============================================================
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-exp")

    # New fields from .env
    DEFAULT_CHUNK_SIZE: int = int(os.getenv("DEFAULT_CHUNK_SIZE", 20))
    DEFAULT_OVERLAP_PAGES: int = int(os.getenv("DEFAULT_OVERLAP_PAGES", 2))
    DEFAULT_MAX_WORKERS: int = int(os.getenv("DEFAULT_MAX_WORKERS", 4))

    # Optional delay control for API rate management
    API_DELAY: float = float(os.getenv("API_DELAY", 0.5))

    # ============================================================
    # 🗄️ Database Settings
    # ============================================================
    DB_HOST: str = os.getenv("DB_HOST", "localhost")
    DB_PORT: int = int(os.getenv("DB_PORT", 5432))
    DB_NAME: str = os.getenv("DB_NAME", "Contract")
    DB_USER: str = os.getenv("DB_USER", "Thulasi")
    DB_PASSWORD: str = os.getenv("DB_PASSWORD", "Thulasi@3012")

    # ============================================================
    # 🔐 Authentication & Security
    # ============================================================
    AUTH_REQUIRED: bool = os.getenv("AUTH_REQUIRED", "true").lower() == "true"
    API_KEY: str = os.getenv("API_KEY", "your-secure-api-key-here")

    # ============================================================
    # ⚙️ Server Settings
    # ============================================================
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", 8000))

    # ============================================================
    # 📂 Directory Settings
    # ============================================================
    UPLOAD_FOLDER: str = os.getenv("UPLOAD_FOLDER", "uploads")
    OUTPUT_FOLDER: str = os.getenv("OUTPUT_FOLDER", "outputs")

    @property
    def DATABASE_URL(self) -> str:
        """
        Build a properly escaped PostgreSQL connection string.
        """
        escaped_password = urllib.parse.quote_plus(self.DB_PASSWORD)
        return f"postgresql://{self.DB_USER}:{escaped_password}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# === Instantiate settings and ensure folders exist ===
settings = AppSettings()

# Ensure upload/output directories exist
os.makedirs(settings.UPLOAD_FOLDER, exist_ok=True)
os.makedirs(settings.OUTPUT_FOLDER, exist_ok=True)

# Validate required API keys
if not settings.GEMINI_API_KEY:
    raise ValueError("❌ GEMINI_API_KEY is missing in environment variables")

# Print summary for development environments
if settings.ENVIRONMENT.lower() != "production":
    print(f"✅ Environment: {settings.ENVIRONMENT}")
    print(f"✅ Database URL: {settings.DATABASE_URL}")
    print(f"✅ Upload folder: {settings.UPLOAD_FOLDER}")
    print(f"✅ Output folder: {settings.OUTPUT_FOLDER}")
    print(f"✅ Chunk Size: {settings.DEFAULT_CHUNK_SIZE}")
    print(f"✅ Overlap Pages: {settings.DEFAULT_OVERLAP_PAGES}")
    print(f"✅ Max Workers: {settings.DEFAULT_MAX_WORKERS}")
    print(f"✅ Gemini Model: {settings.GEMINI_MODEL}")
    print(f"🔑 Gemini API Key: {settings.GEMINI_API_KEY[:4]}...{settings.GEMINI_API_KEY[-4:]}")
