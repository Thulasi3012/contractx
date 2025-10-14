"""
Main API Server
Complete integration with Text, Table, Image, and JSON Formatter Agents
"""
from fastapi import FastAPI, File, UploadFile, HTTPException, APIRouter
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import fitz  # PyMuPDF
import os
import tempfile
import time
from typing import Dict, Any, List
from dataclasses import asdict

# Import agents (these should be in separate files)
from app.service.text_agent import TextAgent, TextContent
from app.service.table_agent import TableAgent, TableContent
from app.service.image_agent import ImageAgent, ImageContent
from app.service.json_formator import JSONFormatterAgent
from app.config.config import settings
from dotenv import load_dotenv
load_dotenv()  # Load .env file if present

# Global Heart LLM instance
heart_llm = None
print(f"gemini_api_key from env: {heart_llm[:4]}...{heart_llm[-4:]}" if heart_llm else "No gem")
class HeartLLM:
    """
    🫀 Core orchestrator that manages the document processing workflow.
    Coordinates Text, Table, Image agents and formats output with JSON Formatter.
    """
    
    def __init__(self, gemini_api_key: str):
        """
        Initialize Heart LLM with all agents.
        
        Args:
            gemini_api_key: Google Gemini API key for LLM-based agents
        """
        # Initialize base extraction agents
        self.text_agent = TextAgent(api_key=gemini_api_key)
        self.table_agent = TableAgent(api_key=gemini_api_key)
        self.image_agent = ImageAgent(api_key=gemini_api_key)
        self.json_formatter = JSONFormatterAgent(api_key=gemini_api_key)
        
    def process_page(self, page: fitz.Page, page_number: int) -> Dict[str, Any]:
        """
        Process a single page through all three extraction agents.
        
        Args:
            page: PyMuPDF Page object
            page_number: Page number (1-indexed)
        
        Returns:
            Combined dictionary with all agent outputs
        """
        print(f"\n📄 Processing Page {page_number}...")
        
        # Send page to Text Agent
        print(f"  → Text Agent processing...")
        text_result = self.text_agent.process_page(page, page_number)
        
        # Send page to Table Agent
        print(f"  → Table Agent processing...")
        table_result = self.table_agent.process_page(page, page_number)
        
        # Send page to Image Agent
        print(f"  → Image Agent processing...")
        image_result = self.image_agent.process_page(page, page_number)
        
        # Combine all agent outputs for this page
        page_data = {
            "page": page_number,
            "text": asdict(text_result),
            "tables": asdict(table_result),
            "images": asdict(image_result)
        }
        
        print(f"  ✓ Page {page_number} completed")
        return page_data
    
    def process_document(self, 
                        pdf_path: str, 
                        document_type: str = "Unknown",
                        format_output: bool = True) -> Dict[str, Any]:
        """
        🫀 Main processing pipeline for the entire document.
        
        Flow:
        1. Load PDF and split into pages
        2. Process each page through Text, Table, Image agents
        3. Combine all page results into raw output
        4. Send raw output to JSON Formatter for intelligent structuring
        5. Return both formatted and raw outputs
        
        Args:
            pdf_path: Path to PDF file
            document_type: Type of document
            format_output: Whether to apply JSON formatting (default: True)
        
        Returns:
            Complete structured JSON with formatted and raw outputs
        """
        print("=" * 70)
        print("🫀 NNX Agent - Document Processing Started")
        print("=" * 70)
        
        start_time = time.time()
        
        # Step 1: Load PDF document
        try:
            doc = fitz.open(pdf_path)
            document_name = os.path.basename(pdf_path)
            total_pages = len(doc)
            print(f"✓ Loaded: {document_name}")
            print(f"✓ Total Pages: {total_pages}")
        except Exception as e:
            raise Exception(f"Failed to load PDF: {str(e)}")
        
        # Step 2: Process each page through all extraction agents
        print(f"\n{'='*70}")
        print(f"⚙️  STAGE 1: EXTRACTING CONTENT FROM {total_pages} PAGE(S)")
        print(f"{'='*70}")
        
        all_pages_data = []
        
        for page_num in range(total_pages):
            page = doc[page_num]
            
            # Process page through all agents
            page_data = self.process_page(page, page_num + 1)
            all_pages_data.append(page_data)
            
            # Small delay to avoid API rate limits
            if page_num < total_pages - 1:
                time.sleep(0.5)
        
        doc.close()
        
        # Step 3: Create raw combined output
        extraction_time = round(time.time() - start_time, 2)
        
        raw_output = {
            "document_metadata": {
                "document_name": document_name,
                "document_type": document_type,
                "total_pages": total_pages,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            },
            "pages": all_pages_data,
            "summary": {
                "total_text_extracted": sum(
                    1 for p in all_pages_data 
                    if p["text"]["status"] == "success"
                ),
                "total_tables_found": sum(
                    p["tables"]["table_count"] for p in all_pages_data
                ),
                "total_images_found": sum(
                    p["images"]["image_count"] for p in all_pages_data
                )
            }
        }
        
        print(f"\n{'='*70}")
        print(f"✅ STAGE 1 COMPLETE - Raw Extraction Done")
        print(f"⏱️  Time: {extraction_time}s")
        print(f"{'='*70}")
        
        # Step 4: Apply JSON formatting if requested
        if format_output:
            print(f"\n{'='*70}")
            print(f"⚙️  STAGE 2: INTELLIGENT JSON FORMATTING")
            print(f"{'='*70}\n")
            
            formatting_start = time.time()
            
            # Send raw output to JSON Formatter Agent
            formatted_result = self.json_formatter.process_document(
                raw_combined_data=raw_output,
                document_name=document_name,
                document_type=document_type
            )
            
            formatting_time = round(time.time() - formatting_start, 2)
            
            print(f"\n{'='*70}")
            print(f"✅ STAGE 2 COMPLETE - JSON Formatting Done")
            print(f"⏱️  Time: {formatting_time}s")
            print(f"📊 Status: {formatted_result.status}")
            print(f"{'='*70}")
        else:
            formatted_result = None
            formatting_time = 0
        
        # Step 5: Create final comprehensive output
        total_time = round(time.time() - start_time, 2)
        
        final_output = {
            "formatted_output": formatted_result.formatted_data if formatted_result else None,
            "raw_output": raw_output,
            "processing_metadata": {
                "total_processing_time_seconds": total_time,
                "extraction_time_seconds": extraction_time,
                "formatting_time_seconds": formatting_time,
                "formatting_applied": format_output,
                "formatting_status": formatted_result.status if formatted_result else "not_applied",
                "total_pages": total_pages,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
        }
        
        print("\n" + "=" * 70)
        print("✅ NNX Agent - Processing Complete")
        print(f"⏱️  Total Time: {total_time}s")
        print(f"📊 Pages: {total_pages} | Tables: {raw_output['summary']['total_tables_found']} | Images: {raw_output['summary']['total_images_found']}")
        print(f"🎨 Formatting: {'Applied' if format_output else 'Skipped'}")
        print("=" * 70)
        
        return final_output


# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Document Analysis API",
    description="AI For Analyzing Documents With NNX Document Intelligence",
    version="2.0.0",
)

# Create router
router = APIRouter(prefix="/Document_parser", tags=["Document Parser"])


@router.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "NNX Agent API is running",
        "version": "2.0.0",
        "agents": ["TextAgent", "TableAgent", "ImageAgent", "JSONFormatterAgent"],
        "status": "active" if heart_llm else "not initialized"
    }

@router.on_event("startup")
async def startup_event():
    """Initialize Heart LLM on server startup"""
    global heart_llm
    
    # Get API key from environment variable
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    
    if not gemini_api_key:
        print("⚠  WARNING: GEMINI_API_KEY not set in environment variables!")
        print("   Set it with: export GEMINI_API_KEY='your-api-key'")
    else:
        heart_llm = HeartLLM(gemini_api_key=gemini_api_key)
        print(f"The API used is: {gemini_api_key[:4]}...{gemini_api_key[-4:]}")
        print("✅ NNX Agent initialized successfully")

@router.post("/process-pdf")
async def process_pdf(
    file: UploadFile = File(...),
    document_type: str = "Unknown",
    format_output: bool = True
):
    """
    Main endpoint to process PDF files with intelligent formatting.
    
    Args:
        file: PDF file uploaded by user
        document_type: Type of document (e.g., "Contract", "Invoice", "Report")
        format_output: Apply intelligent JSON formatting (default: True)
    
    Returns:
        Complete JSON with formatted output, raw output, and metadata
        
    Response Structure:
    {
        "formatted_output": {...},  # Intelligently structured JSON
        "raw_output": {...},         # Raw extraction data
        "processing_metadata": {...} # Processing information
    }
    """
    # Check if Heart LLM is initialized
    if not heart_llm:
        raise HTTPException(
            status_code=500,
            detail="Heart LLM not initialized. Please set GEMINI_API_KEY environment variable."
        )
    
    # Validate file type
    if not file.filename.endswith('.pdf'):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported"
        )
    
    # Create temporary file to save uploaded PDF
    temp_dir = tempfile.gettempdir()
    temp_pdf_path = os.path.join(temp_dir, f"upload_{int(time.time())}_{file.filename}")
    
    try:
        # Save uploaded file
        with open(temp_pdf_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        print(f"\n📤 Received file: {file.filename} ({len(content)} bytes)")
        
        # Process document through Heart LLM
        result = heart_llm.process_document(
            pdf_path=temp_pdf_path,
            document_type=document_type,
            format_output=format_output
        )
        
        return JSONResponse(content=result)
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing PDF: {str(e)}"
        )
    
    finally:
        # Cleanup temporary file
        if os.path.exists(temp_pdf_path):
            os.remove(temp_pdf_path)
            print(f"🧹 Cleaned up temporary file")


@router.post("/process-pdf-raw")
async def process_pdf_raw(
    file: UploadFile = File(...),
    document_type: str = "Unknown"
):
    """
    Process PDF and return only raw output (no formatting).
    Faster processing without JSON formatting stage.
    
    Args:
        file: PDF file uploaded by user
        document_type: Type of document (optional)
    
    Returns:
        Raw JSON output without intelligent formatting
    """
    return await process_pdf(file, document_type, format_output=False)


@router.post("/process-pdf-url")
async def process_pdf_url(
    pdf_url: str,
    document_type: str = "Unknown",
    format_output: bool = True
):
    """
    Process PDF from URL with intelligent formatting.
    
    Args:
        pdf_url: URL to PDF file
        document_type: Type of document (optional)
        format_output: Apply intelligent JSON formatting (default: True)
    
    Returns:
        Complete JSON with formatted and raw outputs
    """
    if not heart_llm:
        raise HTTPException(
            status_code=500,
            detail="Heart LLM not initialized. Please set GEMINI_API_KEY."
        )
    
    try:
        import requests
        
        # Download PDF
        response = requests.get(pdf_url)
        response.raise_for_status()
        
        # Save to temporary file
        temp_dir = tempfile.gettempdir()
        temp_pdf_path = os.path.join(temp_dir, f"url_download_{int(time.time())}.pdf")
        
        with open(temp_pdf_path, "wb") as f:
            f.write(response.content)
        
        # Process document
        result = heart_llm.process_document(
            pdf_path=temp_pdf_path,
            document_type=document_type,
            format_output=format_output
        )
        
        # Cleanup
        os.remove(temp_pdf_path)
        
        return JSONResponse(content=result)
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing PDF URL: {str(e)}"
        )


@router.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "heart_llm_initialized": heart_llm is not None,
        "agents": {
            "text_agent": "active" if heart_llm else "inactive",
            "table_agent": "active" if heart_llm else "inactive",
            "image_agent": "active" if heart_llm else "inactive",
            "json_formatter": "active" if heart_llm else "inactive"
        },
        "version": "2.0.0"
    }


# CRITICAL: Include the router in the app
app.include_router(router)


# Run with: uvicorn main:app --reload --host 0.0.0.0 --port 8000
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)