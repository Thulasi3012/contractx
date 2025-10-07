from sqlalchemy import Column, Integer, String, JSON, TIMESTAMP,Text,DateTime,text
from datetime import datetime
from app.database.database import Base
from app.database.database import engine

class Template(Base):
    __tablename__ = "templates"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    template_type = Column(String, nullable=False)
    sector = Column(String, nullable=False)
    keywords = Column(JSON, nullable=False)
    created_on = Column(TIMESTAMP, server_default=text("CURRENT_TIMESTAMP"))
    updated_on = Column(TIMESTAMP, server_default=text("CURRENT_TIMESTAMP"), onupdate=datetime.utcnow)

class Document(Base):
    """Enhanced Document model with comprehensive fields"""
    __tablename__ = "documents"
    
    id = Column(String(36), primary_key=True, index=True)
    document_name = Column(String(255), nullable=False)
    uploaded_on = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Analysis fields
    summary = Column(Text, nullable=True)
    document_type = Column(String(100), nullable=True)
    document_version = Column(String(50), nullable=True)
    
    # Parties information
    buyer = Column(String(500), nullable=True)
    seller = Column(String(500), nullable=True)
    parties_json = Column(JSON, nullable=True)  # All parties involved
    
    # Critical information
    deadlines = Column(JSON, nullable=True)  # List of deadlines
    alerts = Column(JSON, nullable=True)  # Critical alerts
    obligations = Column(JSON, nullable=True)  # Party obligations
    
    # Text data
    cleaned_text = Column(Text, nullable=False)
    text_as_json = Column(JSON, nullable=False)
    
    # Metadata
    page_count = Column(Integer, nullable=True)
    extraction_method = Column(String(50), nullable=True)

# Create tables
Base.metadata.create_all(bind=engine)