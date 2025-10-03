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
    __tablename__ = "documents"
    
    id = Column(String(36), primary_key=True, index=True)  # UUID
    document_name = Column(String(255), nullable=False)
    uploaded_on = Column(DateTime, default=datetime.utcnow, nullable=False)
    summary = Column(Text, nullable=True)  # Can be populated later or via additional processing
    cleaned_text = Column(Text, nullable=False)  # Raw extracted text from Doctr/pdfplumber
    text_as_json = Column(JSON, nullable=False)  # Structured JSON

Base.metadata.create_all(bind=engine)