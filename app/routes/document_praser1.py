"""
Main API Server - Optimized Version with MySQL Database Integration and UUID
Integration with Text, Table, Image Agents using smart detection
Features:
- Microsoft Table Transformer (DETR-based) for table detection
- Image detection via PyMuPDF (only call ImageAgent when images exist)
- Comprehensive logging for all operations
- PostgreSQL database storage for extracted document data
- UUID for unique document identification
"""
from fastapi import FastAPI, File, UploadFile, HTTPException, APIRouter, Depends
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from datetime import datetime
import fitz  # PyMuPDF
import os
import tempfile
import time
import logging
from typing import Dict, Any, List, Optional
from dataclasses import asdict
import numpy as np
from PIL import Image
import io
import json
import uuid
from app.database.models import Document
# Import agents
from app.service.text_agent import TextAgent, TextContent
from app.service.table_agent import TableAgent, TableContent
from app.service.image_agent import ImageAgent, ImageContent
from app.config.config import settings
from dotenv import load_dotenv
from app.database.database import get_db

# Load environment variables
load_dotenv()

def extract_buyer_from_json(pages_data: List[Dict]) -> Optional[str]:
    """Extract buyer name from first occurrence in pages"""
    for page in pages_data:
        if page.get("text") and page["text"].get("buyer"):
            buyer_info = page["text"]["buyer"]
            if isinstance(buyer_info, dict):
                return buyer_info.get("name")
            elif isinstance(buyer_info, str):
                return buyer_info
    return None

def extract_seller_from_json(pages_data: List[Dict]) -> Optional[str]:
    """Extract seller name from first occurrence in pages"""
    for page in pages_data:
        if page.get("text") and page["text"].get("seller"):
            seller_info = page["text"]["seller"]
            if isinstance(seller_info, dict):
                return seller_info.get("name")
            elif isinstance(seller_info, str):
                return seller_info
    return None

def extract_deadlines_from_json(pages_data: List[Dict]) -> List[Dict]:
    """Extract all deadlines from pages"""
    deadlines = []
    for page in pages_data:
        if page.get("text") and page["text"].get("deadlines"):
            page_deadlines = page["text"]["deadlines"]
            if isinstance(page_deadlines, list):
                deadlines.extend(page_deadlines)
            elif page_deadlines:
                deadlines.append(page_deadlines)
    return deadlines

def extract_obligations_from_json(pages_data: List[Dict]) -> List[Dict]:
    """Extract all obligations from pages"""
    obligations = []
    for page in pages_data:
        if page.get("text") and page["text"].get("obligations"):
            page_obligations = page["text"]["obligations"]
            if isinstance(page_obligations, list):
                obligations.extend(page_obligations)
            elif page_obligations:
                obligations.append(page_obligations)
    return obligations

def extract_alerts_from_json(pages_data: List[Dict]) -> List[Dict]:
    """Extract all alerts from pages"""
    alerts = []
    for page in pages_data:
        if page.get("text") and page["text"].get("alerts"):
            page_alerts = page["text"]["alerts"]
            if isinstance(page_alerts, list):
                alerts.extend(page_alerts)
            elif page_alerts:
                alerts.append(page_alerts)
    return alerts

def extract_cleaned_text_from_json(pages_data: List[Dict]) -> str:
    """Extract and combine all cleaned text from pages"""
    text_parts = []
    for page in pages_data:
        if page.get("text") and page["text"].get("cleaned_text"):
            text_parts.append(f"--- Page {page['page']} ---\n{page['text']['cleaned_text']}")
    return "\n\n".join(text_parts)

def extract_summary_from_json(pages_data: List[Dict]) -> Optional[str]:
    """Extract summary from pages or generate from text"""
    summaries = []
    for page in pages_data:
        if page.get("text") and page["text"].get("summary"):
            summaries.append(page["text"]["summary"])
    
    if summaries:
        return " ".join(summaries)
    return None

def save_document_to_db(db: Session, result: Dict[str, Any], document_uuid: str) -> Document:
    """
    Save processed document results to database with UUID
    
    Args:
        db: Database session
        result: Complete JSON result from HeartLLM.process_document()
        document_uuid: Unique identifier for the document
    
    Returns:
        Created Document instance
    """
    try:
        # Extract metadata
        metadata = result.get("document_metadata", {})
        pages_data = result.get("pages", [])
        summary_data = result.get("summary", {})
        optimization_metrics = result.get("optimization_metrics", {})
        processing_metadata = result.get("processing_metadata", {})
        
        # Extract fields from JSON
        buyer = extract_buyer_from_json(pages_data)
        seller = extract_seller_from_json(pages_data)
        deadlines = extract_deadlines_from_json(pages_data)
        obligations = extract_obligations_from_json(pages_data)
        alerts = extract_alerts_from_json(pages_data)
        cleaned_text = extract_cleaned_text_from_json(pages_data)
        summary = extract_summary_from_json(pages_data)
        
        # Create document instance with UUID
        document = Document(
            document_uuid=document_uuid,
            document_name=metadata.get("document_name", "Unknown"),
            document_type=metadata.get("document_type", "Unknown"),
            buyer=buyer,
            seller=seller,
            summary=summary,
            deadlines=deadlines if deadlines else None,
            obligations=obligations if obligations else None,
            alerts=alerts if alerts else None,
            cleaned_text=cleaned_text,
            text_as_json=result,  # Store entire JSON
            page_count=metadata.get("total_pages", 0),
            extraction_method=processing_metadata.get("detection_method", "LLM + SmartDetector"),
            processing_time_seconds=int(processing_metadata.get("total_processing_time_seconds", 0))
        )
        
        # Add to database
        db.add(document)
        db.commit()
        db.refresh(document)
        
        logger.info(f"✅ Document saved to database with ID: {document.id}")
        logger.info(f"   UUID: {document.document_uuid}")
        logger.info(f"   Document: {document.document_name}")
        logger.info(f"   Type: {document.document_type}")
        logger.info(f"   Pages: {document.page_count}")
        logger.info(f"   Buyer: {document.buyer}")
        logger.info(f"   Seller: {document.seller}")
        logger.info(f"   Deadlines: {len(deadlines) if deadlines else 0}")
        logger.info(f"   Obligations: {len(obligations) if obligations else 0}")
        logger.info(f"   Alerts: {len(alerts) if alerts else 0}")
        
        return document
        
    except Exception as e:
        logger.error(f"❌ Failed to save document to database: {e}", exc_info=True)
        db.rollback()
        raise

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================
import sys

# Create file handler with UTF-8 encoding
file_handler = logging.FileHandler('nnx_agent.log', encoding='utf-8')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

# Create console handler with UTF-8 encoding
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

# Force UTF-8 encoding on console handler (Windows fix)
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    handlers=[file_handler, console_handler]
)

logger = logging.getLogger('NNXAgent')

# ============================================================================
# TABLE DETECTION MODEL SETUP
# ============================================================================
try:
    from transformers import AutoImageProcessor, TableTransformerForObjectDetection
    import torch
    table_detector = None
    table_processor = None
    logger.info("Table Transformer import successful")
except ImportError as e:
    logger.warning(f"Table Transformer not available: {e}")
    logger.warning("Table detection will fall back to basic heuristics")
    table_detector = None
    table_processor = None

# Global Heart LLM instance
heart_llm = None

class SmartDetector:
    """Smart detection layer"""
    def __init__(self, table_model=None, table_processor=None):
        self.table_model = table_model
        self.table_processor = table_processor
        logger.info("SmartDetector initialized")
    
    def detect_tables_ml(self, page: fitz.Page) -> bool:
        try:
            logger.info(f"Running ML-based table detection on page")
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            img_data = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_data)).convert("RGB")
            
            inputs = self.table_processor(images=img, return_tensors="pt")
            with torch.no_grad():
                outputs = self.table_model(**inputs)
            
            target_sizes = torch.tensor([img.size[::-1]])
            results = self.table_processor.post_process_object_detection(
                outputs, threshold=0.85, target_sizes=target_sizes
            )[0]
            
            num_tables = len(results['scores'])
            if num_tables > 0:
                valid_tables = 0
                boxes = results['boxes']
                img_width, img_height = img.size
                
                for box in boxes:
                    x1, y1, x2, y2 = box.tolist()
                    width = x2 - x1
                    height = y2 - y1
                    area = width * height
                    img_area = img_width * img_height
                    
                    if (area / img_area) > 0.05 and width > 100 and height > 50:
                        valid_tables += 1
                
                has_tables = valid_tables > 0
                if has_tables:
                    logger.info(f"✓ ML detected {valid_tables} table(s)")
                else:
                    logger.info(f"✗ ML detected no valid tables")
            else:
                has_tables = False
                logger.info("✗ ML detected no tables")
            
            return has_tables
        except Exception as e:
            logger.error(f"ML table detection failed: {e}")
            return self.detect_tables_heuristic(page)
    
    def detect_tables_heuristic(self, page: fitz.Page) -> bool:
        try:
            text = page.get_text("dict")
            blocks = text.get("blocks", [])
            aligned_blocks = 0
            
            for block in blocks:
                if "lines" in block:
                    lines = block["lines"]
                    if len(lines) > 2:
                        spans_per_line = [len(line.get("spans", [])) for line in lines]
                        if len(set(spans_per_line)) == 1 and spans_per_line[0] > 1:
                            aligned_blocks += 1
            
            drawings = page.get_drawings()
            rect_count = sum(1 for d in drawings if d.get("type") == "re")
            has_tables = aligned_blocks >= 2 or rect_count > 10
            return has_tables
        except Exception as e:
            logger.error(f"Heuristic table detection failed: {e}")
            return False
    
    def detect_tables(self, page: fitz.Page, page_number: int) -> bool:
        if self.table_model and self.table_processor:
            return self.detect_tables_ml(page)
        return self.detect_tables_heuristic(page)
    
    def detect_images(self, page: fitz.Page, page_number: int) -> bool:
        try:
            image_list = page.get_images(full=True)
            has_images = len(image_list) > 0
            if has_images:
                logger.info(f"[Page {page_number}] ✓ Detected {len(image_list)} image(s)")
            return has_images
        except Exception as e:
            logger.error(f"Image detection failed: {e}")
            return False


class HeartLLM:
    """Core orchestrator"""
    def __init__(self, gemini_api_key: str, table_detector_model=None, table_processor=None):
        logger.info("="*70)
        logger.info("Initializing HeartLLM")
        logger.info("="*70)
        
        self.detector = SmartDetector(table_model=table_detector_model, table_processor=table_processor)
        self.text_agent = TextAgent(api_key=gemini_api_key)
        self.table_agent = TableAgent(api_key=gemini_api_key)
        self.image_agent = ImageAgent(api_key=gemini_api_key)
        
        logger.info("✅ HeartLLM initialization complete")
    
    def process_page(self, page: fitz.Page, page_number: int) -> Dict[str, Any]:
        logger.info(f"📄 PROCESSING PAGE {page_number}")
        start_time = time.time()
        
        # Text extraction
        try:
            text_result = self.text_agent.process_page(page, page_number)
        except Exception as e:
            logger.error(f"TextAgent failed: {e}")
            text_result = None
        
        # Table detection & extraction
        table_detected = self.detector.detect_tables(page, page_number)
        if table_detected:
            logger.info(f"[Page {page_number}] ✓ Table detected")
            try:
                table_result = self.table_agent.process_page(page, page_number)
                logger.info(f"✓ Table extraction complete on page {page_number}")
            except Exception as e:
                logger.error(f"TableAgent failed: {e}")
                table_result = None
        else:
            table_result = None

#   Step 3: Image Detection & Processing (CONDITIONAL)
        logger.info(f"[Page {page_number}] STEP 3: Image Detection & Extraction")
        image_detected = self.detector.detect_images(page, page_number)
        
        if image_detected:
            logger.info(f"[Page {page_number}] → Images detected, calling ImageAgent...")
            image_start = time.time()
            
            try:
                image_result = self.image_agent.process_page(page, page_number)
                image_time = round(time.time() - image_start, 2)
                logger.info(f"[Page {page_number}] ✓ ImageAgent completed in {image_time}s")
                logger.info(f"[Page {page_number}]   Images found: {image_result.image_count}")
            except Exception as e:
                image_time = round(time.time() - image_start, 2)
                logger.error(f"[Page {page_number}] ✗ ImageAgent failed: {e}", exc_info=True)
                image_result = None
        else:
            logger.info(f"[Page {page_number}] → No images detected, skipping ImageAgent")
            logger.info(f"[Page {page_number}] 💰 Saved 1 LLM call")
            image_result = None
            image_time = 0
        
        page_data = {
            "page": page_number,
            "text": asdict(text_result) if text_result else None,
            "tables": asdict(table_result) if table_result else None,
            "images": asdict(image_result) if image_result else None,
            "processing_info": {
                "table_detected": table_detected,
                "image_detected": image_detected,
                "agents_called": {
                    "text_agent": True,
                    "table_agent": table_detected,
                    "image_agent": image_detected
                },
                "timing": {
                    "total_page_seconds": round(time.time() - start_time, 2)
                }
            }
        }
        
        return page_data
    
    def process_document(self, pdf_path: str, document_type: str = "Unknown") -> Dict[str, Any]:
        logger.info("🫀 NNX AGENT - DOCUMENT PROCESSING STARTED")
        overall_start = time.time()
        
        doc = fitz.open(pdf_path)
        document_name = os.path.basename(pdf_path)
        total_pages = len(doc)
        file_size = os.path.getsize(pdf_path)
        
        all_pages_data = []
        total_llm_calls = 0
        total_llm_calls_saved = 0
        
        for page_num in range(total_pages):
            page = doc[page_num]
            page_data = self.process_page(page, page_num + 1)
            all_pages_data.append(page_data)
            
            agents_called = sum([
                True,
                page_data["processing_info"]["table_detected"],
                page_data["processing_info"]["image_detected"]
            ])
            total_llm_calls += agents_called
            total_llm_calls_saved += (3 - agents_called)
            
            if page_num < total_pages - 1:
                time.sleep(0.5)
        
        doc.close()
        
        processing_time = round(time.time() - overall_start, 2)
        
        text_success = sum(1 for p in all_pages_data if p["text"] and p["text"]["status"] == "success")
        total_tables = sum(p["tables"]["table_count"] for p in all_pages_data if p["tables"])
        total_images = sum(p["images"]["image_count"] for p in all_pages_data if p["images"])
        
        final_output = {
            "document_metadata": {
                "document_name": document_name,
                "document_type": document_type,
                "total_pages": total_pages,
                "file_size_bytes": file_size,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            },
            "pages": all_pages_data,
            "summary": {
                "total_text_extracted": text_success,
                "total_tables_found": total_tables,
                "total_images_found": total_images
            },
            "optimization_metrics": {
                "total_llm_calls_made": total_llm_calls,
                "total_llm_calls_saved": total_llm_calls_saved,
                "cost_reduction_percentage": round(total_llm_calls_saved/(total_llm_calls+total_llm_calls_saved)*100, 2)
            },
            "processing_metadata": {
                "total_processing_time_seconds": processing_time,
                "average_time_per_page_seconds": round(processing_time/total_pages, 2),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "detection_method": "ML-based" if table_detector else "Heuristic-based"
            }
        }
        
        logger.info("✅ NNX AGENT - PROCESSING COMPLETE")
        return final_output


# ============================================================================
# FASTAPI APPLICATION
# ============================================================================
app = FastAPI(
    title="NNX Document Intelligence API",
    description="AI-powered document analysis with MySQL database storage and UUID",
    version="4.1.0",
)

router = APIRouter(prefix="/Document_parser", tags=["Document Parser"])


@router.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "NNX Agent API with Database and UUID is running",
        "version": "4.1.0",
        "database": "POSTGRESQL",
        "features": ["UUID support", "Document tracking"],
        "status": "active" if heart_llm else "not initialized"
    }


@router.on_event("startup")
async def startup_event():
    """Initialize Heart LLM and database on server startup"""
    global heart_llm, table_detector, table_processor
    
    logger.info("🚀 STARTING NNX AGENT API SERVER WITH DATABASE")
    
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        logger.error("⚠️  CRITICAL: GEMINI_API_KEY not set!")
        return
    
    # Initialize Table Transformer
    try:
        if table_detector is None and 'TableTransformerForObjectDetection' in globals():
            model_name = "microsoft/table-transformer-detection"
            table_processor = AutoImageProcessor.from_pretrained(model_name)
            table_detector = TableTransformerForObjectDetection.from_pretrained(model_name)
            table_detector.eval()
            logger.info("✓ Table Transformer loaded")
    except Exception as e:
        logger.warning(f"Table Transformer unavailable: {e}")
    
    # Initialize Heart LLM
    heart_llm = HeartLLM(
        gemini_api_key=gemini_api_key,
        table_detector_model=table_detector,
        table_processor=table_processor
    )
    logger.info("✅ NNX Agent initialized with database support")


@router.post("/process-pdf")
async def process_pdf(
    file: UploadFile = File(...),
    document_type: str = "Unknown",
    db: Session = Depends(get_db)
):
    """
    Process PDF and save to database with UUID
    
    Args:
        file: PDF file
        document_type: Type of document
        db: Database session (auto-injected)
    
    Returns:
        JSON with extracted data + database ID + UUID
    """
    logger.info("📤 NEW PDF UPLOAD REQUEST")
    
    if not heart_llm:
        raise HTTPException(status_code=500, detail="Heart LLM not initialized")
    
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files supported")
    
    # Generate UUID for this document
    document_uuid = str(uuid.uuid4())
    logger.info(f"Generated UUID: {document_uuid}")
    
    temp_pdf_path = os.path.join(tempfile.gettempdir(), f"upload_{int(time.time())}_{file.filename}")
    
    try:
        # Save uploaded file
        with open(temp_pdf_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        logger.info(f"✓ File saved ({len(content)/1024:.2f} KB)")
        
        # Process document
        result = heart_llm.process_document(
            pdf_path=temp_pdf_path,
            document_type=document_type
        )
        
        # Save to database with UUID
        document = save_document_to_db(db, result, document_uuid)
        
        # Add database information to response
        result["database_id"] = document.id
        result["document_uuid"] = document.document_uuid
        result["database_status"] = "saved"
        result["created_at"] = document.created_at.isoformat()
        
        logger.info(f"✅ Processing complete - Database ID: {document.id}, UUID: {document.document_uuid}")
        return JSONResponse(content=result)
    
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        if os.path.exists(temp_pdf_path):
            os.remove(temp_pdf_path)


@router.get("/documents")
async def list_documents(
    skip: int = 0,
    limit: int = 10,
    document_type: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """
    List all processed documents from database
    
    Args:
        skip: Number of records to skip
        limit: Maximum records to return
        document_type: Filter by document type (optional)
    """
    query = db.query(Document)
    
    if document_type:
        query = query.filter(Document.document_type == document_type)
    
    total = query.count()
    documents = query.order_by(Document.created_at.desc()).offset(skip).limit(limit).all()
    
    return {
        "total": total,
        "skip": skip,
        "limit": limit,
        "documents": [
            {
                "id": doc.id,
                "document_uuid": doc.document_uuid,
                "document_name": doc.document_name,
                "document_type": doc.document_type,
                "buyer": doc.buyer,
                "seller": doc.seller,
                "page_count": doc.page_count,
                "created_at": doc.created_at.isoformat(),
                "processing_time_seconds": doc.processing_time_seconds
            }
            for doc in documents
        ]
    }


@router.get("/documents/{document_identifier}")
async def get_document(document_identifier: str, db: Session = Depends(get_db)):
    """
    Get full document details by ID or UUID
    
    Args:
        document_identifier: Either integer ID or UUID string
    """
    # Try to parse as integer ID first
    try:
        doc_id = int(document_identifier)
        document = db.query(Document).filter(Document.id == doc_id).first()
    except ValueError:
        # If not an integer, treat as UUID
        document = db.query(Document).filter(Document.document_uuid == document_identifier).first()
    
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    return {
        "id": document.id,
        "document_uuid": document.document_uuid,
        "document_name": document.document_name,
        "document_type": document.document_type,
        "buyer": document.buyer,
        "seller": document.seller,
        "summary": document.summary,
        "deadlines": document.deadlines,
        "obligations": document.obligations,
        "alerts": document.alerts,
        "cleaned_text": document.cleaned_text,
        "page_count": document.page_count,
        "extraction_method": document.extraction_method,
        "processing_time_seconds": document.processing_time_seconds,
        "created_at": document.created_at.isoformat(),
        "full_json": document.text_as_json
    }


@router.delete("/documents/{document_identifier}")
async def delete_document(document_identifier: str, db: Session = Depends(get_db)):
    """
    Delete document by ID or UUID
    
    Args:
        document_identifier: Either integer ID or UUID string
    """
    # Try to parse as integer ID first
    try:
        doc_id = int(document_identifier)
        document = db.query(Document).filter(Document.id == doc_id).first()
    except ValueError:
        # If not an integer, treat as UUID
        document = db.query(Document).filter(Document.document_uuid == document_identifier).first()
    
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    doc_uuid = document.document_uuid
    doc_id = document.id
    
    db.delete(document)
    db.commit()
    
    return {
        "message": f"Document deleted successfully",
        "id": doc_id,
        "uuid": doc_uuid
    }


@router.get("/health")
async def health_check(db: Session = Depends(get_db)):
    """Health check with database connection test"""
    try:
        # Test database connection
        db.execute("SELECT 1")
        db_status = "connected"
    except Exception as e:
        db_status = f"error: {str(e)}"
    
    return {
        "status": "healthy" if heart_llm else "degraded",
        "version": "4.1.0",
        "heart_llm": heart_llm is not None,
        "database": db_status,
        "features": ["UUID support", "Dual identifier lookup"],
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

# Include router
app.include_router(router)
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting NNX Agent API Server with Database and UUID")
    uvicorn.run(app, host="0.0.0.0", port=8000)