from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import urllib.parse
from app.config.config import settings

# Direct database credentials
# DB_USER = "Thulasi"
# DB_PASSWORD = "Thulasi@30125"
# DB_HOST = "localhost"
# DB_PORT = 5432
# DB_NAME = "Contract"

# Properly escape password
escaped_password = urllib.parse.quote_plus(settings.DB_PASSWORD)

# Build database URL
# DATABASE_URL = f"postgresql://{DB_USER}:{escaped_password}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Create engine
engine = create_engine(settings.DATABASE_URL)

# SessionLocal class
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class
Base = declarative_base()

# Dependency for FastAPI
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
