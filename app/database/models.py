from sqlalchemy import Column, Integer, String, JSON, TIMESTAMP,Text,DateTime,text
from datetime import datetime
from app.database.database import Base
from app.database.database import engine
import uuid
from sqlalchemy.dialects.postgresql import UUID

class Document(Base):
    """Database model for storing processed documents"""
    __tablename__ = "documents"

    id = Column(Integer, autoincrement=True, primary_key=True, index=True)
    document_uuid = Column(String(36), unique=True, default=lambda: str(uuid.uuid4()), nullable=False)
    document_name = Column(String(255), nullable=False, index=True)
    document_type = Column(String(100), nullable=True, index=True)
    buyer = Column(String(500), nullable=True)
    seller = Column(String(500), nullable=True)
    summary = Column(Text, nullable=True)
    deadlines = Column(JSON, nullable=True)
    obligations = Column(JSON, nullable=True)
    alerts = Column(JSON, nullable=True)
    cleaned_text = Column(Text, nullable=True)
    text_as_json = Column(JSON, nullable=True)
    page_count = Column(Integer, nullable=True)
    extraction_method = Column(String(100), nullable=True)
    processing_time_seconds = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

# Create tables
Base.metadata.create_all(bind=engine)