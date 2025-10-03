import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "Development")
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "info")
    # App Configuration
    APP_NAME: str = "API for analysing the Contract Document"
    API_PREFIX: str = os.getenv("API_PREFIX", "/api")
    
    # Database Configuration
    DB_HOST: str = os.getenv("DB_HOST", "localhost")
    DB_PORT: int = int(os.getenv("DB_PORT", "5432"))
    DB_NAME: str = os.getenv("DB_NAME", "")
    DB_USER: str = os.getenv("DB_USER", "")
    DB_PASSWORD: str = os.getenv("DB_PASSWORD", "")
    AUTH_REQUIRED: bool = os.getenv("AUTH_REQUIRED", "true").lower() == "true"
    
    @property
    def DATABASE_URL(self) -> str:
        """Create properly escaped database URL"""
        # Escape special characters in password
        import urllib.parse
        escaped_password = urllib.parse.quote_plus(self.DB_PASSWORD)
        return f"postgresql://{self.DB_USER}:{escaped_password}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
    
    # OpenAI Configuration
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-5")
    OPENAI_MODEL_MINI: str = os.getenv("OPENAI_MODEL_MINI", "gpt-5-mini")

    # API Key security
    API_KEY: str = os.getenv("API_KEY", "")
    class Config:
        env_file = ".env"

settings = Settings()
print("DB_USER from settings:", settings.DB_USER)
print("openAI Key from settings:", settings.OPENAI_API_KEY)
print("DB_NAME from settings:", settings.DB_NAME)
print("DATABASE_URL from settings:", settings.DATABASE_URL)