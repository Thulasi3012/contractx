"""
Document Summary Generation Service
- Generates detailed summaries using OpenAI GPT
- Extracts deadlines, obligations, alerts, parties, financial info
- Returns structured JSON summary
- Updates database with generated summary
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional
from app.database.database import get_db 

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from openai import OpenAI
from app.database.database import SessionLocal
from app.database.models import Document
from app.config.config import settings

# ---------------------
# Logging config
# ---------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("summary_service")


# ---------------------
# OpenAI Configuration
# ---------------------
# Get API key from environment variable
OPENAI_API_KEY = settings.OPENAI_API_KEY
if not OPENAI_API_KEY:
    logger.warning("OPENAI_API_KEY not found in environment variables!")

OPENAI_MODEL = settings.OPENAI_MODEL

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# ---------------------
# Request/Response Models
# ---------------------
class SummaryRequest(BaseModel):
    document_id: str

class SummaryResponse(BaseModel):
    document_id: str
    document_name: str
    summary_generated_at: str
    summary: Dict[str, Any]
    status: str

# ---------------------
# Router
# ---------------------
router = APIRouter(prefix="/summary", tags=["summary generation"])

# ---------------------
# OpenAI Prompt Builder
# ---------------------
def build_summary_prompt(document_data: Dict[str, Any]) -> str:
    """
    Builds a comprehensive prompt for OpenAI to extract structured information
    from the document.
    """
    prompt = """You are an expert legal document analyzer. Analyze the following contract/document and provide a comprehensive summary in JSON format.

Extract and organize the following information with high precision:

1. **Document Overview**
   - document_type: Type of document (e.g., "Purchase Agreement", "Service Contract", "NDA")
   - document_title: Full title of the document
   - start_date: Contract start date (format: YYYY-MM-DD or null if not found)
   - end_date: Contract end date (format: YYYY-MM-DD or null if not found)
   - duration: Contract duration in readable format (e.g., "12 months", "2 years")
   - parties: Object containing all parties involved
     - seller: Name and details of seller/service provider
     - buyer: Name and details of buyer/client
     - other_parties: Array of any other parties mentioned

2. **Key Dates & Deadlines**
   - Array of deadline objects, each containing:
     - description: What the deadline is for
     - date: Deadline date (YYYY-MM-DD)
     - priority: "high", "medium", or "low"
     - consequences: What happens if missed (if mentioned)

3. **Obligations**
   - seller_obligations: Array of seller's responsibilities
   - buyer_obligations: Array of buyer's responsibilities
   - mutual_obligations: Array of obligations for all parties
   - compliance_requirements: Any regulatory or legal compliance needs

4. **Alerts & Critical Points**
   - Array of alert objects, each containing:
     - type: "penalty", "termination", "liability", "confidentiality", "dispute_resolution", "force_majeure"
     - description: Detailed description of the clause
     - severity: "critical", "high", "medium", "low"
     - action_required: What action is needed (if any)

5. **Financial Information**
   - total_amount: Total contract value
   - currency: Currency code (USD, EUR, etc.)
   - payment_terms: Payment conditions and schedule
   - payment_schedule: Array of payment milestones
   - penalties_fees: Any penalty or late fee information
   - deposits: Deposit requirements

6. **Termination & Renewal**
   - termination_conditions: How the contract can be terminated
   - notice_period: Required notice period for termination
   - renewal_terms: Auto-renewal or renewal conditions
   - exit_clauses: Any exit strategy or buyout clauses

7. **Special Conditions & Risk Factors**
   - special_clauses: Any unusual or unique clauses
   - risk_factors: Potential risks or red flags
   - confidentiality: Confidentiality requirements
   - intellectual_property: IP-related clauses
   - liability_limits: Limitation of liability clauses
   - insurance_requirements: Required insurance coverage

8. **Dispute Resolution**
   - governing_law: Which jurisdiction governs the contract
   - dispute_resolution_method: Arbitration, mediation, litigation, etc.
   - venue: Where disputes will be resolved

**IMPORTANT INSTRUCTIONS:**
- Return ONLY valid JSON, no other text
- Use null for any information not found in the document
- Be precise with dates - use YYYY-MM-DD format
- Extract exact amounts and numbers
- Quote important phrases directly from the document
- If multiple interpretations exist, include both
- Flag ambiguous or unclear clauses in risk_factors

**Document Data:**
"""
    
    # Include the structured JSON if available, otherwise use cleaned text
    if document_data.get("text_as_json"):
        prompt += f"\n\nStructured Document JSON:\n{json.dumps(document_data['text_as_json'], indent=2)}"
    
    if document_data.get("cleaned_text"):
        # Truncate if too long (keep first 15000 chars to stay within token limits)
        cleaned_text = document_data["cleaned_text"]
        if len(cleaned_text) > 15000:
            cleaned_text = cleaned_text[:15000] + "\n\n[Document truncated for length...]"
        prompt += f"\n\nRaw Document Text:\n{cleaned_text}"
    
    prompt += "\n\nProvide the analysis in the JSON structure specified above."
    
    return prompt

# ---------------------
# OpenAI Summary Generation
# ---------------------
def generate_summary_with_openai(document_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calls OpenAI API to generate a structured summary of the document.
    """
    if not client:
        raise HTTPException(status_code=500, detail="OpenAI API key not configured")
    
    logger.info("Building prompt for OpenAI...")
    prompt = build_summary_prompt(document_data)
    
    logger.info(f"Calling OpenAI API with model: {OPENAI_MODEL}")
    
    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert legal document analyzer. You extract structured information from contracts and legal documents with high precision. Always return valid JSON."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            response_format={"type": "json_object"},  # Force JSON response
            # temperature=0.3,  # Lower temperature for more factual, consistent output
            max_completion_tokens=5000
        )
        
        # Extract the JSON response
        summary_text = response.choices[0].message.content
        logger.info("Received response from OpenAI")
        
        # Parse JSON
        summary_json = json.loads(summary_text)
        logger.info("Successfully parsed summary JSON")
        
        return summary_json
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse OpenAI response as JSON: {e}")
        raise HTTPException(status_code=500, detail="OpenAI returned invalid JSON")
    
    except Exception as e:
        logger.error(f"OpenAI API call failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate summary: {str(e)}")

# ---------------------
# Main Endpoint: Generate Summary
# ---------------------
@router.post("/generate-summary/", response_model=SummaryResponse)
async def generate_summary(request: SummaryRequest):
    """
    Generate a detailed summary for a document by its UUID.
    
    Input: { "document_id": "uuid-string" }
    
    Returns: Structured JSON summary with all extracted information
    """
    document_id = request.document_id
    logger.info(f"Received request to generate summary for document: {document_id}")
    
    # Get database session
    db = next(get_db())
    
    try:
        # 1. Query the document from database
        logger.info(f"Querying database for document: {document_id}")
        document = db.query(Document).filter(Document.id == document_id).first()
        
        if not document:
            logger.warning(f"Document not found: {document_id}")
            raise HTTPException(status_code=404, detail=f"Document with ID {document_id} not found")
        
        logger.info(f"Found document: {document.document_name}")
        
        # 2. Check if summary already exists
        if document.summary and document.summary.strip() not in ["null", "None", ""]:
            logger.info("Summary already exists, returning cached version")
            try:
                existing_summary = json.loads(document.summary) if isinstance(document.summary, str) else document.summary
                return SummaryResponse(
                    document_id=document.id,
                    document_name=document.document_name,
                    summary_generated_at=document.uploaded_on.isoformat(),
                    summary=existing_summary,
                    status="success_cached"
                )
            except:
                logger.info("Existing summary is invalid, regenerating...")
        
        # 3. Prepare document data for OpenAI
        document_data = {
            "document_name": document.document_name,
            "cleaned_text": document.cleaned_text,
            "text_as_json": document.text_as_json
        }
        
        # 4. Generate summary using OpenAI
        logger.info("Generating summary with OpenAI...")
        summary_json = generate_summary_with_openai(document_data)
        
        # 5. Update database with generated summary
        logger.info("Updating database with generated summary...")
        document.summary = json.dumps(summary_json)  # Store as JSON string
        db.commit()
        db.refresh(document)
        logger.info("Database updated successfully")
        
        # 6. Return response
        return SummaryResponse(
            document_id=document.id,
            document_name=document.document_name,
            summary_generated_at=datetime.utcnow().isoformat(),
            summary=summary_json,
            status="success"
        )
        
    except HTTPException:
        raise
    
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    
    finally:
        db.close()


# ---------------------
# Additional Endpoint: Get Existing Summary
# ---------------------
@router.get("/get-summary/{document_id}")
async def get_summary(document_id: str):
    """
    Retrieve existing summary for a document (doesn't generate new one).
    """
    db = next(get_db())
    
    try:
        document = db.query(Document).filter(Document.id == document_id).first()
        
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        if not document.summary or document.summary.strip() in ["null", "None", ""]:
            return {
                "document_id": document.id,
                "document_name": document.document_name,
                "summary": None,
                "status": "no_summary",
                "message": "Summary has not been generated yet. Use /generate-summary/ endpoint."
            } 
        try:
            summary_json = json.loads(document.summary) if isinstance(document.summary, str) else document.summary
        except:
            summary_json = document.summary
        
        return {
            "document_id": document.id,
            "document_name": document.document_name,
            "uploaded_on": document.uploaded_on.isoformat(),
            "summary": summary_json,
            "status": "success"
        }
        
    finally:
        db.close()


# ---------------------
# Health Check Endpoint
# ---------------------
@router.get("/health")
async def health_check():
    """Check if the summary service is operational"""
    return {
        "status": "healthy",
        "openai_configured": client is not None,
        "model": OPENAI_MODEL,
        "database": "connected"
    }