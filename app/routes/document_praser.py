"""
Main API Server - Optimized Version
Integration with Text, Table, Image Agents using smart detection
Features:
- Microsoft Table Transformer (DETR-based) for table detection
- Image detection via PyMuPDF (only call ImageAgent when images exist)
- Comprehensive logging for all operations
"""
from fastapi import FastAPI, File, UploadFile, HTTPException, APIRouter
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
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

# Import agents
from app.service.text_agent import TextAgent, TextContent
from app.service.table_agent import TableAgent, TableContent
from app.service.image_agent import ImageAgent, ImageContent
from app.config.config import settings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================
# Fix for Windows Unicode encoding issues
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
        # Reconfigure stdout to use UTF-8
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        # Python < 3.7 fallback
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
    # Import Microsoft Table Transformer
    # Note: Install with: pip install transformers torch torchvision
    from transformers import AutoImageProcessor, TableTransformerForObjectDetection
    import torch
    table_detector = None  # Will be initialized in startup
    table_processor = None  # Image processor for Table Transformer
    logger.info("Table Transformer import successful")
except ImportError as e:
    logger.warning(f"Table Transformer not available: {e}")
    logger.warning("Table detection will fall back to basic heuristics")
    table_detector = None
    table_processor = None

# Global Heart LLM instance
heart_llm = None


class SmartDetector:
    """
    🔍 Smart detection layer that determines which agents to call
    Uses Microsoft Table Transformer for table detection and PyMuPDF for image detection
    """
    
    def __init__(self, table_model=None, table_processor=None):
        """
        Initialize smart detector
        
        Args:
            table_model: Table Transformer model instance (optional)
            table_processor: Table Transformer image processor (optional)
        """
        self.table_model = table_model
        self.table_processor = table_processor
        logger.info("SmartDetector initialized")
        logger.info(f"Table detection mode: {'ML-based (CascadeTabNet)' if table_model else 'Heuristic-based'}")
    
    def detect_tables_ml(self, page: fitz.Page) -> bool:
        """
        Detect tables using Microsoft Table Transformer ML model
        
        Args:
            page: PyMuPDF Page object
        
        Returns:
            True if tables detected, False otherwise
        """
        try:
            logger.info(f"Running ML-based table detection on page")
            
            # Convert page to image
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom for better detection
            img_data = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_data)).convert("RGB")
            
            logger.debug(f"Page converted to image: size={img.size}")
            
            # Prepare image for model
            inputs = self.table_processor(images=img, return_tensors="pt")
            
            # Run detection
            with torch.no_grad():
                outputs = self.table_model(**inputs)
            
            # Process outputs - get predicted boxes and scores
            # INCREASED threshold from 0.7 to 0.85 for better precision
            target_sizes = torch.tensor([img.size[::-1]])
            results = self.table_processor.post_process_object_detection(
                outputs, 
                threshold=0.85,  # Higher confidence threshold to reduce false positives
                target_sizes=target_sizes
            )[0]
            
            # Count detected tables with additional validation
            num_tables = len(results['scores'])
            
            # Additional validation: check if detected boxes are reasonable size
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
                    
                    # Filter out very small detections (likely false positives)
                    # Table should be at least 5% of page area and have minimum dimensions
                    if (area / img_area) > 0.05 and width > 100 and height > 50:
                        valid_tables += 1
                        logger.debug(f"Valid table detected: area={area:.0f}px ({area/img_area*100:.1f}% of page)")
                    else:
                        logger.debug(f"Filtered small detection: area={area:.0f}px ({area/img_area*100:.1f}% of page)")
                
                has_tables = valid_tables > 0
                
                if has_tables:
                    logger.info(f"✓ ML detected {valid_tables} table(s) (filtered from {num_tables} detections)")
                else:
                    logger.info(f"✗ ML detected no valid tables (filtered {num_tables} small detections)")
            else:
                has_tables = False
                logger.info("✗ ML detected no tables")
            
            return has_tables
            
        except Exception as e:
            logger.error(f"ML table detection failed: {e}", exc_info=True)
            logger.info("Falling back to heuristic detection")
            return self.detect_tables_heuristic(page)
    
    def detect_tables_heuristic(self, page: fitz.Page) -> bool:
        """
        Fallback: Detect tables using heuristic methods (text blocks, lines)
        
        Args:
            page: PyMuPDF Page object
        
        Returns:
            True if tables likely present, False otherwise
        """
        try:
            logger.info("Running heuristic-based table detection")
            
            # Method 1: Check for table-like structures in page
            text = page.get_text("dict")
            blocks = text.get("blocks", [])
            
            # Count blocks with consistent spacing (table indicator)
            aligned_blocks = 0
            for block in blocks:
                if "lines" in block:
                    lines = block["lines"]
                    if len(lines) > 2:
                        # Check if spans are aligned (table columns)
                        spans_per_line = [len(line.get("spans", [])) for line in lines]
                        if len(set(spans_per_line)) == 1 and spans_per_line[0] > 1:
                            aligned_blocks += 1
            
            # Method 2: Check for drawing objects (table borders)
            drawings = page.get_drawings()
            rect_count = sum(1 for d in drawings if d.get("type") == "re")
            
            has_tables = aligned_blocks >= 2 or rect_count > 10
            
            if has_tables:
                logger.info(f"✓ Heuristic detected table indicators (aligned_blocks={aligned_blocks}, rectangles={rect_count})")
            else:
                logger.info(f"✗ Heuristic detected no tables (aligned_blocks={aligned_blocks}, rectangles={rect_count})")
            
            return has_tables
            
        except Exception as e:
            logger.error(f"Heuristic table detection failed: {e}", exc_info=True)
            return False
    
    def detect_tables(self, page: fitz.Page, page_number: int) -> bool:
        """
        Main table detection method (uses ML if available, else heuristic)
        
        Args:
            page: PyMuPDF Page object
            page_number: Page number for logging
        
        Returns:
            True if tables detected, False otherwise
        """
        logger.info(f"[Page {page_number}] Starting table detection")
        
        if self.table_model and self.table_processor:
            result = self.detect_tables_ml(page)
        else:
            result = self.detect_tables_heuristic(page)
        
        logger.info(f"[Page {page_number}] Table detection result: {result}")
        return result
    
    def detect_images(self, page: fitz.Page, page_number: int) -> bool:
        """
        Detect if page contains images using PyMuPDF
        
        Args:
            page: PyMuPDF Page object
            page_number: Page number for logging
        
        Returns:
            True if images detected, False otherwise
        """
        logger.info(f"[Page {page_number}] Starting image detection")
        
        try:
            # Get all images on the page
            image_list = page.get_images(full=True)
            
            has_images = len(image_list) > 0
            
            if has_images:
                logger.info(f"[Page {page_number}] ✓ Detected {len(image_list)} image(s)")
                for idx, img in enumerate(image_list[:5]):  # Log first 5 images
                    xref = img[0]
                    logger.debug(f"  Image {idx+1}: xref={xref}")
            else:
                logger.info(f"[Page {page_number}] ✗ No images detected")
            
            return has_images
            
        except Exception as e:
            logger.error(f"[Page {page_number}] Image detection failed: {e}", exc_info=True)
            return False


class HeartLLM:
    """
    🫀 Core orchestrator that manages the document processing workflow.
    Coordinates Text, Table, Image agents with smart detection.
    """
    
    def __init__(self, gemini_api_key: str, table_detector_model=None, table_processor=None):
        """
        Initialize Heart LLM with all agents and smart detector.
        
        Args:
            gemini_api_key: Google Gemini API key for LLM-based agents
            table_detector_model: Table Transformer model instance (optional)
            table_processor: Table Transformer processor instance (optional)
        """
        logger.info("="*70)
        logger.info("Initializing HeartLLM")
        logger.info("="*70)
        
        # Initialize smart detector
        self.detector = SmartDetector(table_model=table_detector_model, table_processor=table_processor)
        logger.info("✓ SmartDetector initialized")
        
        # Initialize agents
        logger.info("Initializing agents...")
        self.text_agent = TextAgent(api_key=gemini_api_key)
        logger.info("✓ TextAgent initialized")
        
        self.table_agent = TableAgent(api_key=gemini_api_key)
        logger.info("✓ TableAgent initialized")
        
        self.image_agent = ImageAgent(api_key=gemini_api_key)
        logger.info("✓ ImageAgent initialized")
        
        logger.info("="*70)
        logger.info("✅ HeartLLM initialization complete")
        logger.info("="*70)
        
    def process_page(self, page: fitz.Page, page_number: int) -> Dict[str, Any]:
        """
        Process a single page with smart agent selection.
        
        Flow:
        1. Always call TextAgent (for main content)
        2. Detect tables → only call TableAgent if detected
        3. Detect images → only call ImageAgent if detected
        
        Args:
            page: PyMuPDF Page object
            page_number: Page number (1-indexed)
        
        Returns:
            Combined dictionary with all agent outputs
        """
        logger.info("="*70)
        logger.info(f"📄 PROCESSING PAGE {page_number}")
        logger.info("="*70)
        
        start_time = time.time()
        
        # Step 1: Text Agent (ALWAYS called)
        logger.info(f"[Page {page_number}] STEP 1: Text Extraction")
        logger.info(f"[Page {page_number}] → Calling TextAgent...")
        text_start = time.time()
        
        try:
            text_result = self.text_agent.process_page(page, page_number)
            text_time = round(time.time() - text_start, 2)
            logger.info(f"[Page {page_number}] ✓ TextAgent completed in {text_time}s")
            logger.info(f"[Page {page_number}]   Status: {text_result.status}")
        except Exception as e:
            text_time = round(time.time() - text_start, 2)
            logger.error(f"[Page {page_number}] ✗ TextAgent failed: {e}", exc_info=True)
            text_result = None
        
        # Step 2: Table Detection & Processing (CONDITIONAL)
        logger.info(f"[Page {page_number}] STEP 2: Table Detection & Extraction")
        table_detected = self.detector.detect_tables(page, page_number)
        
        if table_detected:
            logger.info(f"[Page {page_number}] → Table detected, calling TableAgent...")
            table_start = time.time()
            
            try:
                table_result = self.table_agent.process_page(page, page_number)
                table_time = round(time.time() - table_start, 2)
                logger.info(f"[Page {page_number}] ✓ TableAgent completed in {table_time}s")
                logger.info(f"[Page {page_number}]   Tables found: {table_result.table_count}")
            except Exception as e:
                table_time = round(time.time() - table_start, 2)
                logger.error(f"[Page {page_number}] ✗ TableAgent failed: {e}", exc_info=True)
                table_result = None
        else:
            logger.info(f"[Page {page_number}] → No tables detected, skipping TableAgent")
            logger.info(f"[Page {page_number}] 💰 Saved 1 LLM call")
            table_result = None
            table_time = 0
        
        # Step 3: Image Detection & Processing (CONDITIONAL)
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
        
        # Combine all results
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
                    "text_processing_seconds": text_time if text_result else 0,
                    "table_processing_seconds": table_time,
                    "image_processing_seconds": image_time,
                    "total_page_seconds": round(time.time() - start_time, 2)
                }
            }
        }
        
        total_time = round(time.time() - start_time, 2)
        agents_called = sum([True, table_detected, image_detected])
        agents_saved = 2 - (agents_called - 1)
        
        logger.info("="*70)
        logger.info(f"✅ PAGE {page_number} COMPLETE")
        logger.info(f"⏱️  Total time: {total_time}s")
        logger.info(f"🤖 Agents called: {agents_called}/3")
        logger.info(f"💰 LLM calls saved: {agents_saved}")
        logger.info("="*70)
        
        return page_data
    
    def process_document(self, 
                        pdf_path: str, 
                        document_type: str = "Unknown") -> Dict[str, Any]:
        """
        🫀 Main processing pipeline for the entire document.
        
        Flow:
        1. Load PDF and split into pages
        2. Process each page through smart detection + agents
        3. Combine all page results into structured output
        
        Args:
            pdf_path: Path to PDF file
            document_type: Type of document
        
        Returns:
            Complete structured JSON with all outputs
        """
        logger.info("\n" + "="*70)
        logger.info("🫀 NNX AGENT - DOCUMENT PROCESSING STARTED")
        logger.info("="*70)
        
        overall_start = time.time()
        
        # Step 1: Load PDF document
        logger.info("STEP 1: Loading PDF Document")
        try:
            doc = fitz.open(pdf_path)
            document_name = os.path.basename(pdf_path)
            total_pages = len(doc)
            file_size = os.path.getsize(pdf_path)
            
            logger.info(f"✓ Document loaded successfully")
            logger.info(f"  File name: {document_name}")
            logger.info(f"  File size: {file_size} bytes ({file_size/1024:.2f} KB)")
            logger.info(f"  Total pages: {total_pages}")
            logger.info(f"  Document type: {document_type}")
            
        except Exception as e:
            logger.error(f"✗ Failed to load PDF: {e}", exc_info=True)
            raise Exception(f"Failed to load PDF: {str(e)}")
        
        # Step 2: Process each page
        logger.info("\n" + "="*70)
        logger.info(f"STEP 2: PROCESSING {total_pages} PAGE(S)")
        logger.info("="*70)
        
        all_pages_data = []
        total_llm_calls = 0
        total_llm_calls_saved = 0
        
        for page_num in range(total_pages):
            page = doc[page_num]
            
            logger.info(f"\nProcessing page {page_num + 1}/{total_pages}...")
            
            # Process page through smart detection + agents
            page_data = self.process_page(page, page_num + 1)
            all_pages_data.append(page_data)
            
            # Count LLM calls
            agents_called = sum([
                True,  # TextAgent always called
                page_data["processing_info"]["table_detected"],
                page_data["processing_info"]["image_detected"]
            ])
            total_llm_calls += agents_called
            total_llm_calls_saved += (3 - agents_called)
            
            # Rate limiting delay
            if page_num < total_pages - 1:
                logger.info(f"⏸️  Rate limiting delay (0.5s)...")
                time.sleep(0.5)
        
        doc.close()
        logger.info("✓ PDF document closed")
        
        # Step 3: Create final output
        logger.info("\n" + "="*70)
        logger.info("STEP 3: COMPILING FINAL OUTPUT")
        logger.info("="*70)
        
        processing_time = round(time.time() - overall_start, 2)
        
        # Calculate statistics
        text_success = sum(1 for p in all_pages_data if p["text"] and p["text"]["status"] == "success")
        total_tables = sum(p["tables"]["table_count"] for p in all_pages_data if p["tables"])
        total_images = sum(p["images"]["image_count"] for p in all_pages_data if p["images"])
        pages_with_tables = sum(1 for p in all_pages_data if p["processing_info"]["table_detected"])
        pages_with_images = sum(1 for p in all_pages_data if p["processing_info"]["image_detected"])
        
        logger.info(f"📊 Processing Statistics:")
        logger.info(f"  ✓ Text extracted: {text_success}/{total_pages} pages")
        logger.info(f"  ✓ Tables found: {total_tables} across {pages_with_tables} pages")
        logger.info(f"  ✓ Images found: {total_images} across {pages_with_images} pages")
        logger.info(f"  🤖 Total LLM calls made: {total_llm_calls}")
        logger.info(f"  💰 Total LLM calls saved: {total_llm_calls_saved}")
        logger.info(f"  📉 Cost reduction: {(total_llm_calls_saved/(total_llm_calls+total_llm_calls_saved)*100):.1f}%")
        
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
                "total_images_found": total_images,
                "pages_with_tables": pages_with_tables,
                "pages_with_images": pages_with_images
            },
            "optimization_metrics": {
                "total_llm_calls_made": total_llm_calls,
                "total_llm_calls_saved": total_llm_calls_saved,
                "cost_reduction_percentage": round(total_llm_calls_saved/(total_llm_calls+total_llm_calls_saved)*100, 2),
                "average_agents_per_page": round(total_llm_calls/total_pages, 2)
            },
            "processing_metadata": {
                "total_processing_time_seconds": processing_time,
                "average_time_per_page_seconds": round(processing_time/total_pages, 2),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "detection_method": "ML-based (CascadeTabNet)" if table_detector else "Heuristic-based"
            }
        }
        
        logger.info("\n" + "="*70)
        logger.info("✅ NNX AGENT - PROCESSING COMPLETE")
        logger.info("="*70)
        logger.info(f"⏱️  Total time: {processing_time}s ({processing_time/total_pages:.2f}s per page)")
        logger.info(f"📊 Results: {total_pages} pages | {total_tables} tables | {total_images} images")
        logger.info(f"💰 Optimization: {total_llm_calls_saved} LLM calls saved ({(total_llm_calls_saved/(total_llm_calls+total_llm_calls_saved)*100):.1f}%)")
        logger.info("="*70 + "\n")
        
        return final_output


# ============================================================================
# FASTAPI APPLICATION
# ============================================================================

app = FastAPI(
    title="NNX Document Intelligence API",
    description="Optimized AI-powered document analysis with smart agent selection",
    version="3.0.0",
)

router = APIRouter(prefix="/Document_parser", tags=["Document Parser"])


@router.get("/")
async def root():
    """Health check endpoint"""
    logger.info("Root endpoint called")
    return {
        "message": "NNX Agent API is running",
        "version": "3.0.0",
        "agents": ["TextAgent", "TableAgent", "ImageAgent"],
        "features": ["Smart Detection", "Table Transformer", "Cost Optimization"],
        "status": "active" if heart_llm else "not initialized"
    }


@router.on_event("startup")
async def startup_event():
    """Initialize Heart LLM and table detector on server startup"""
    global heart_llm, table_detector, table_processor
    
    logger.info("="*70)
    logger.info("🚀 STARTING NNX AGENT API SERVER")
    logger.info("="*70)
    
    # Get API key from environment
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    
    if not gemini_api_key:
        logger.error("⚠️  CRITICAL: GEMINI_API_KEY not set in environment variables!")
        logger.error("   Set it with: export GEMINI_API_KEY='your-api-key'")
        return
    
    logger.info(f"✓ Gemini API Key loaded: {gemini_api_key[:4]}...{gemini_api_key[-4:]}")
    
    # Initialize Microsoft Table Transformer model
    try:
        if table_detector is None and 'TableTransformerForObjectDetection' in globals():
            logger.info("Initializing Table Transformer model...")
            model_name = "microsoft/table-transformer-detection"
            table_processor = AutoImageProcessor.from_pretrained(model_name)
            table_detector = TableTransformerForObjectDetection.from_pretrained(model_name)
            # Set to eval mode
            table_detector.eval()
            logger.info("✓ Table Transformer model loaded successfully")
    except Exception as e:
        logger.warning(f"Could not initialize Table Transformer: {e}")
        logger.warning("Will use heuristic-based table detection")
        table_detector = None
        table_processor = None
    
    # Initialize Heart LLM
    try:
        heart_llm = HeartLLM(
            gemini_api_key=gemini_api_key,
            table_detector_model=table_detector,
            table_processor=table_processor
        )
        logger.info("✅ NNX Agent initialized successfully")
        logger.info("="*70)
    except Exception as e:
        logger.error(f"Failed to initialize NNX Agent: {e}", exc_info=True)


@router.post("/process-pdf")
async def process_pdf(
    file: UploadFile = File(...),
    document_type: str = "Unknown"
):
    """
    Main endpoint to process PDF files with smart agent selection.
    
    Args:
        file: PDF file uploaded by user
        document_type: Type of document (e.g., "Contract", "Invoice", "Report")
    
    Returns:
        Complete JSON with all extracted content and optimization metrics
    """
    logger.info("\n" + "="*70)
    logger.info("📤 NEW PDF UPLOAD REQUEST RECEIVED")
    logger.info("="*70)
    
    # Check if Heart LLM is initialized
    if not heart_llm:
        logger.error("Heart LLM not initialized - rejecting request")
        raise HTTPException(
            status_code=500,
            detail="Heart LLM not initialized. Please set GEMINI_API_KEY environment variable."
        )
    
    # Validate file type
    if not file.filename.endswith('.pdf'):
        logger.warning(f"Invalid file type uploaded: {file.filename}")
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported"
        )
    
    logger.info(f"File: {file.filename}")
    logger.info(f"Document type: {document_type}")
    
    # Create temporary file
    temp_dir = tempfile.gettempdir()
    temp_pdf_path = os.path.join(temp_dir, f"upload_{int(time.time())}_{file.filename}")
    
    try:
        # Save uploaded file
        logger.info(f"Saving file to temporary location: {temp_pdf_path}")
        with open(temp_pdf_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        logger.info(f"✓ File saved ({len(content)} bytes, {len(content)/1024:.2f} KB)")
        
        # Process document
        logger.info("Starting document processing...")
        result = heart_llm.process_document(
            pdf_path=temp_pdf_path,
            document_type=document_type
        )
        
        logger.info("✅ Processing complete, returning results")
        return JSONResponse(content=result)
    
    except Exception as e:
        logger.error(f"Error processing PDF: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing PDF: {str(e)}"
        )
    
    finally:
        # Cleanup
        if os.path.exists(temp_pdf_path):
            os.remove(temp_pdf_path)
            logger.info(f"🧹 Cleaned up temporary file: {temp_pdf_path}")


@router.post("/process-pdf-url")
async def process_pdf_url(
    pdf_url: str,
    document_type: str = "Unknown"
):
    """
    Process PDF from URL with smart agent selection.
    
    Args:
        pdf_url: URL to PDF file
        document_type: Type of document (optional)
    
    Returns:
        Complete JSON with extracted content and optimization metrics
    """
    logger.info("\n" + "="*70)
    logger.info("🌐 NEW PDF URL REQUEST RECEIVED")
    logger.info("="*70)
    logger.info(f"URL: {pdf_url}")
    logger.info(f"Document type: {document_type}")
    
    if not heart_llm:
        logger.error("Heart LLM not initialized - rejecting request")
        raise HTTPException(
            status_code=500,
            detail="Heart LLM not initialized. Please set GEMINI_API_KEY."
        )
    
    try:
        import requests
        
        # Download PDF
        logger.info("Downloading PDF from URL...")
        response = requests.get(pdf_url, timeout=30)
        response.raise_for_status()
        logger.info(f"✓ PDF downloaded ({len(response.content)} bytes)")
        
        # Save to temporary file
        temp_dir = tempfile.gettempdir()
        temp_pdf_path = os.path.join(temp_dir, f"url_download_{int(time.time())}.pdf")
        
        with open(temp_pdf_path, "wb") as f:
            f.write(response.content)
        
        logger.info(f"✓ Saved to: {temp_pdf_path}")
        
        # Process document
        logger.info("Starting document processing...")
        result = heart_llm.process_document(
            pdf_path=temp_pdf_path,
            document_type=document_type
        )
        
        # Cleanup
        os.remove(temp_pdf_path)
        logger.info("🧹 Cleaned up temporary file")
        
        logger.info("✅ Processing complete, returning results")
        return JSONResponse(content=result)
    
    except requests.RequestException as e:
        logger.error(f"Failed to download PDF from URL: {e}", exc_info=True)
        raise HTTPException(
            status_code=400,
            detail=f"Failed to download PDF from URL: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Error processing PDF URL: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing PDF URL: {str(e)}"
        )


@router.get("/health")
async def health_check():
    """Detailed health check with system status"""
    logger.info("Health check endpoint called")
    
    health_status = {
        "status": "healthy" if heart_llm else "degraded",
        "version": "3.0.0",
        "heart_llm_initialized": heart_llm is not None,
        "agents": {
            "text_agent": "active" if heart_llm else "inactive",
            "table_agent": "active" if heart_llm else "inactive",
            "image_agent": "active" if heart_llm else "inactive"
        },
        "features": {
            "smart_detection": True,
            "table_detection": "ML-based (CascadeTabNet)" if table_detector else "Heuristic-based",
            "cost_optimization": True
        },
        "detection_models": {
            "table_transformer": "loaded" if table_detector else "not available"
        },
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    logger.info(f"Health status: {health_status['status']}")
    return health_status


@router.get("/stats")
async def get_stats():
    """Get API statistics and capabilities"""
    logger.info("Stats endpoint called")
    
    return {
        "api_version": "3.0.0",
        "capabilities": {
            "text_extraction": True,
            "table_detection": True,
            "table_extraction": True,
            "image_detection": True,
            "image_analysis": True,
            "smart_agent_selection": True,
            "cost_optimization": True
        },
        "detection_methods": {
            "tables": "ML-based (CascadeTabNet)" if table_detector else "Heuristic-based",
            "images": "PyMuPDF native detection"
        },
        "supported_formats": ["PDF"],
        "max_file_size": "No explicit limit (depends on server resources)",
        "optimization": {
            "description": "Only calls agents when content is detected",
            "potential_savings": "Up to 67% LLM call reduction"
        }
    }


# Include router in app
app.include_router(router)


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    logger.info("="*70)
    logger.info("Starting NNX Agent API Server")
    logger.info("Host: 0.0.0.0")
    logger.info("Port: 8000")
    logger.info("="*70)
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
