"""
Main API Server
Complete integration with Text, Table, and Image Agents
"""

from fastapi import FastAPI, File, UploadFile, HTTPException,APIRouter
from fastapi.responses import JSONResponse
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


app = FastAPI(
    title="Document Analysis API",
    description="AI For Analyzing Documents With NNX Docuement Intelligence",
    version="1.0.0"
)

router = APIRouter(prefix="/Document_parser", tags=["Document Parser"])

class HeartLLM:
    """
    Core orchestrator that manages the document processing workflow.
    Coordinates Text, Table, and Image agents.
    """
    
    def __init__(self, gemini_api_key: str):
        """
        Initialize Heart LLM with all agents.
        
        Args:
            gemini_api_key: Google Gemini API key for LLM-based agents
        """
        self.text_agent = TextAgent(api_key=gemini_api_key)
        self.table_agent = TableAgent(api_key=gemini_api_key)
        self.image_agent = ImageAgent()
        
    def process_page(self, page: fitz.Page, page_number: int) -> Dict[str, Any]:
        """
        Process a single page through all three agents.
        
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
    
    def process_document(self, pdf_path: str, document_type: str = "Unknown") -> Dict[str, Any]:
        """
        Main processing pipeline for the entire document.
        
        Args:
            pdf_path: Path to PDF file
            document_type: Type of document
        
        Returns:
            Complete structured JSON with all pages and agent outputs
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
        
        # Step 2: Process each page through all agents
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
        
        # Step 3: Create final combined output
        processing_time = round(time.time() - start_time, 2)
        
        final_output = {
            "document_metadata": {
                "document_name": document_name,
                "document_type": document_type,
                "total_pages": total_pages,
                "processing_time_seconds": processing_time,
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
        
        print("\n" + "=" * 70)
        print("✅ NNX Agent - Processing Complete")
        print(f"⏱️  Time: {processing_time}s")
        print(f"📊 Pages: {total_pages} | Tables: {final_output['summary']['total_tables_found']} | Images: {final_output['summary']['total_images_found']}")
        print("=" * 70)
        
        return final_output

# Global Heart LLM instance (initialized on startup)
heart_llm = None

@router.on_event("startup")
async def startup_event():
    """Initialize Heart LLM on server startup"""
    global heart_llm
    
    # Get API key from environment variable
    gemini_api_key = "AIzaSyBR-j_7vbLMCvE4yo4vqLqaLPWYKecqPuY"
    
    if not gemini_api_key:
        print("⚠️  WARNING: GEMINI_API_KEY not set in environment variables!")
        print("   Set it with: export GEMINI_API_KEY='your-api-key'")
    else:
        heart_llm = HeartLLM(gemini_api_key=gemini_api_key)
        print("✅ NNX Agent initialized successfully")


@router.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "NNX Agent API is running",
        "version": "1.0.0",
        "status": "active" if heart_llm else "not initialized"
    }


@router.post("/process-pdf")
async def process_pdf(
    file: UploadFile = File(...),
    document_type: str = "Unknown"
):
    """
    Main endpoint to process PDF files.
    
    Args:
        file: PDF file uploaded by user
        document_type: Type of document (optional)
    
    Returns:
        Complete JSON with all pages processed by all agents
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
        result = heart_llm.process_document(temp_pdf_path, document_type)
        
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


@router.post("/process-pdf-url")
async def process_pdf_url(
    pdf_url: str,
    document_type: str = "Unknown"
):
    """
    Process PDF from URL.
    
    Args:
        pdf_url: URL to PDF file
        document_type: Type of document (optional)
    
    Returns:
        Complete JSON with all pages processed
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
        result = heart_llm.process_document(temp_pdf_path, document_type)
        
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
            "image_agent": "active" if heart_llm else "inactive"
        }
    }


# Run with: uvicorn main:app --reload --host 0.0.0.0 --port 8000
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)