#!/usr/bin/env python3
"""
Enhanced Contract PDF Parser API with Gemini AI
Professional version with advanced clause detection and comprehensive analysis
"""

import os
import json
import logging
import tempfile
import requests
from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid
import re

import pdfplumber
import google.generativeai as genai
from fastapi import FastAPI, UploadFile, File, HTTPException, APIRouter
from fastapi.responses import JSONResponse
from sqlalchemy import Column, String, DateTime, Text, JSON, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session
from app.database.models import Document
from app.database.database import get_db, engine

# =====================================================
# HARDCODED API KEYS
# =====================================================
OCR_SPACE_API_KEY = "K86948634088957"
GEMINI_API_KEY = "AIzaSyBR-j_7vbLMCvE4yo4vqLqaLPWYKecqPuY"
OCR_SPACE_URL = "https://api.ocr.space/parse/image"
# =====================================================

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)

# Logging config
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("contract_parser")

# FastAPI app
app = FastAPI(title="ContractX PDF Parser API")
router = APIRouter(prefix="/Document_parser", tags=["Document Parser"])

logger.info("=== ContractX PDF Parser API Starting ===")
logger.info(f"OCR.space API Key: {'SET' if OCR_SPACE_API_KEY else 'NOT SET'}")
logger.info(f"Gemini API Key: {'SET' if GEMINI_API_KEY else 'NOT SET'}")


def _log_step(logs: List[str], level: str, message: str) -> None:
    """Record a processing step"""
    entry = f"{level.upper()} | {message}"
    logs.append(entry)
    getattr(logger, level.lower(), logger.info)(message)


def extract_with_ocrspace(pdf_path: str, logs: List[str]) -> tuple[str, int]:
    """Extract text using OCR.space API"""
    if not OCR_SPACE_API_KEY:
        _log_step(logs, "warning", "OCR.space key not set, skipping")
        return "", 0
    
    try:
        _log_step(logs, "info", "Starting OCR.space extraction")
        
        with open(pdf_path, 'rb') as f:
            response = requests.post(
                OCR_SPACE_URL,
                files={'file': f},
                data={
                    'apikey': OCR_SPACE_API_KEY,
                    'language': 'eng',
                    'isOverlayRequired': False,
                    'detectOrientation': True,
                    'scale': True,
                    'OCREngine': 2,
                    'filetype': 'PDF'
                },
                timeout=300
            )
            
            if response.status_code != 200:
                _log_step(logs, "error", f"OCR.space error: {response.status_code}")
                return "", 0
            
            result = response.json()
            
            if result.get('IsErroredOnProcessing'):
                error_msg = result.get('ErrorMessage', ['Unknown'])[0]
                _log_step(logs, "error", f"OCR.space error: {error_msg}")
                return "", 0
            
            text_parts = []
            page_count = 0
            for idx, page_result in enumerate(result.get('ParsedResults', []), 1):
                page_text = page_result.get('ParsedText', '')
                text_parts.append(page_text)
                page_count += 1
                _log_step(logs, "info", f"Extracted page {idx}: {len(page_text)} chars")
            
            text = "\n".join(text_parts)
            _log_step(logs, "info", f"OCR.space complete: {len(text)} total chars, {page_count} pages")
            return text, page_count
            
    except Exception as e:
        _log_step(logs, "error", f"OCR.space exception: {str(e)}")
        return "", 0


def extract_with_pdfplumber(pdf_path: str, logs: List[str]) -> tuple[str, int]:
    """Extract text using pdfplumber"""
    _log_step(logs, "info", "Starting pdfplumber extraction")
    
    try:
        text_parts = []
        with pdfplumber.open(pdf_path) as pdf:
            page_count = len(pdf.pages)
            _log_step(logs, "info", f"PDF has {page_count} pages")
            for pageno, page in enumerate(pdf.pages, 1):
                page_text = page.extract_text() or ""
                text_parts.append(page_text)
                _log_step(logs, "info", f"Extracted page {pageno}: {len(page_text)} chars")
        
        text = "\n".join(text_parts)
        _log_step(logs, "info", f"pdfplumber complete: {len(text)} total chars")
        return text, page_count
        
    except Exception as e:
        _log_step(logs, "error", f"pdfplumber exception: {str(e)}")
        return "", 0


def analyze_with_gemini(text: str, filename: str, logs: List[str]) -> Dict[str, Any]:
    """Comprehensive document analysis using Gemini AI with professional prompting"""
    _log_step(logs, "info", "Starting Gemini AI comprehensive analysis")
    
    if not GEMINI_API_KEY:
        _log_step(logs, "error", "Gemini key not set")
        return _empty_analysis(filename)
    
    try:
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
        # Truncate if needed
        max_chars = 1000000
        if len(text) > max_chars:
            _log_step(logs, "warning", f"Truncating text from {len(text)} to {max_chars} chars")
            text = text[:max_chars] + "\n[...truncated...]"
        
        prompt = f"""You are an expert legal document analyzer. Analyze this contract/document comprehensively and return a structured JSON response.

CONTRACT/DOCUMENT TEXT:
{text}

IMPORTANT INSTRUCTIONS:
1. **Clause ID Recognition** - Detect ALL these patterns:
   - 1.1, 1.1.1, 1.1.1.1 (decimal numbering)
   - (1), (2), (a), (b), (A), (B) (parenthetical)
   - i., ii., iii., iv., I., II., III. (roman numerals)
   - a), b), c), A), B), C) (letter with parenthesis)
   - •, -, *, ◦ (bullets)
   - "Clause 1.1", "Section 2.3" (explicit mentions)

2. **Heading Detection** - Recognize these patterns:
   - "1. Introduction", "1) Introduction", "1 Introduction"
   - "I. INTRODUCTION", "I INTRODUCTION"
   - "A. Background", "A) Background"
   - "INTRODUCTION" (all caps standalone)
   - "Section 1 – Definitions", "Article I – Definitions"
   - "Part 1 – Obligations", "Schedule 1", "Annex A"

3. **Content Analysis Requirements**:
   - Document summary (what it's about, main purpose)
   - Identify ALL parties (buyer, seller, contractor, client, etc.)
   - Extract ALL deadlines with dates and descriptions
   - Identify ALL alerts (payment terms, termination clauses, penalties)
   - List ALL obligations by party
   - Detect document version/revision if present
4. you are not allowed to miss any section or clause, you must extract everything, and if you unable to find the section or a clause but the content is present, then you are allowed to create a new section or clause with a name, based on the content you 

REQUIRED JSON OUTPUT FORMAT (respond ONLY with this JSON, no markdown):
{{
  "schedule_name": "{filename}",
  "document_type": "Contract|Agreement|MOU|Policy|Other",
  "document_version": "Version number or null",
  
  "summary": {{
    "main_purpose": "Brief description of what this document covers",
    "key_points": ["point 1", "point 2", "point 3"]
  }},
  
  "parties": {{
    "buyer": "Buyer name or null",
    "seller": "Seller name or null",
    "all_parties": [
      {{
        "role": "buyer|seller|contractor|client|party_name",
        "name": "Legal entity name",
        "details": "Additional identifying info if present"
      }}
    ]
  }},
  
  "deadlines": [
    {{
      "date": "YYYY-MM-DD or original format",
      "description": "What needs to be done",
      "party_responsible": "Who is responsible",
      "priority": "high|medium|low"
    }}
  ],
  
  "alerts": [
    {{
      "type": "payment|termination|penalty|compliance|other",
      "description": "Alert description",
      "severity": "critical|high|medium|low",
      "reference_clause": "Clause ID if applicable"
    }}
  ],
  
  "obligations": [
    {{
      "party": "Party name",
      "obligation": "Description of obligation",
      "clause_reference": "Clause ID",
      "type": "payment|delivery|compliance|reporting|other"
    }}
  ],
  
  "sections": [
    {{
      "section_name": "1. Introduction",
      "clauses": [
        {{
          "clause_id": "1.1",
          "content": "Full clause text content",
          "subclauses": [
            {{
              "clause_id": "1.1.1",
              "content": "Subclause content"
            }}
          ]
        }}
      ]
    }}
  ]
}}

CRITICAL RULES:
✓ Return ONLY valid JSON (no markdown, no backticks, no explanations)
✓ Extract ALL sections, clauses, and subclauses with proper hierarchy
✓ Preserve original numbering and structure exactly
✓ Be thorough - don't skip content
✓ Use null for missing information, never omit fields
✓ Detect nested clauses properly (e.g., 1.1.1 under 1.1)
✓ Identify all party roles accurately
✓ Extract specific dates in ISO format when possible
✓ Prioritize alerts and deadlines for business-critical items

Begin JSON analysis:"""

        _log_step(logs, "info", f"Sending {len(text)} chars to Gemini for comprehensive analysis")
        
        response = model.generate_content(
            prompt,
            generation_config={
                "temperature": 0.2,  # Slightly higher for better extraction
                "top_p": 0.95,
                "max_output_tokens": 15192,
            }
        )
        
        if not response or not response.text:
            _log_step(logs, "error", "Gemini returned empty response")
            return _empty_analysis(filename)
        
        _log_step(logs, "info", f"Received {len(response.text)} chars from Gemini")
        
        # Clean response
        response_text = response.text.strip()
        response_text = _clean_json_response(response_text)
        
        # Parse JSON
        try:
            data = json.loads(response_text)
            
            # Validate and normalize structure
            data = _validate_analysis_structure(data, filename)
            
            # Log statistics
            _log_analysis_stats(data, logs)
            
            return data
            
        except json.JSONDecodeError as e:
            _log_step(logs, "error", f"JSON parse error: {str(e)}")
            _log_step(logs, "error", f"Response preview: {response_text[:500]}")
            return _empty_analysis(filename)
        
    except Exception as e:
        _log_step(logs, "error", f"Gemini exception: {str(e)}")
        import traceback
        _log_step(logs, "error", traceback.format_exc())
        return _empty_analysis(filename)


def _clean_json_response(text: str) -> str:
    """Clean Gemini response to extract pure JSON"""
    # Remove markdown code blocks
    if text.startswith("```json"):
        text = text[7:].strip()
    elif text.startswith("```"):
        text = text[3:].strip()
    if text.endswith("```"):
        text = text[:-3].strip()
    
    # Extract JSON (find first { to last })
    first_brace = text.find('{')
    last_brace = text.rfind('}')
    if first_brace != -1 and last_brace != -1:
        text = text[first_brace:last_brace+1]
    
    return text


def _validate_analysis_structure(data: Dict[str, Any], filename: str) -> Dict[str, Any]:
    """Validate and normalize the analysis structure"""
    # Ensure all required fields exist
    if "schedule_name" not in data:
        data["schedule_name"] = filename
    
    if "document_type" not in data:
        data["document_type"] = "Unknown"
    
    if "document_version" not in data:
        data["document_version"] = None
    
    if "summary" not in data or not isinstance(data["summary"], dict):
        data["summary"] = {"main_purpose": "", "key_points": []}
    
    if "parties" not in data or not isinstance(data["parties"], dict):
        data["parties"] = {"buyer": None, "seller": None, "all_parties": []}
    
    if "deadlines" not in data or not isinstance(data["deadlines"], list):
        data["deadlines"] = []
    
    if "alerts" not in data or not isinstance(data["alerts"], list):
        data["alerts"] = []
    
    if "obligations" not in data or not isinstance(data["obligations"], list):
        data["obligations"] = []
    
    if "sections" not in data or not isinstance(data["sections"], list):
        data["sections"] = []
    
    return data


def _empty_analysis(filename: str) -> Dict[str, Any]:
    """Return empty analysis structure"""
    return {
        "schedule_name": filename,
        "document_type": "Unknown",
        "document_version": None,
        "summary": {"main_purpose": "", "key_points": []},
        "parties": {"buyer": None, "seller": None, "all_parties": []},
        "deadlines": [],
        "alerts": [],
        "obligations": [],
        "sections": []
    }


def _log_analysis_stats(data: Dict[str, Any], logs: List[str]) -> None:
    """Log statistics about the analysis"""
    section_count = len(data.get("sections", []))
    clause_count = sum(len(s.get("clauses", [])) for s in data.get("sections", []))
    deadline_count = len(data.get("deadlines", []))
    alert_count = len(data.get("alerts", []))
    obligation_count = len(data.get("obligations", []))
    party_count = len(data.get("parties", {}).get("all_parties", []))
    
    _log_step(logs, "info", f"Analysis complete:")
    _log_step(logs, "info", f"  - Sections: {section_count}")
    _log_step(logs, "info", f"  - Clauses: {clause_count}")
    _log_step(logs, "info", f"  - Parties: {party_count}")
    _log_step(logs, "info", f"  - Deadlines: {deadline_count}")
    _log_step(logs, "info", f"  - Alerts: {alert_count}")
    _log_step(logs, "info", f"  - Obligations: {obligation_count}")


@router.post("/parse-pdf/DOCUMENT_ANALYSIS")
async def parse_pdf(file: UploadFile = File(...)):
    """Parse PDF with comprehensive Gemini AI analysis"""
    logs = []
    _log_step(logs, "info", f"=== Upload: {file.filename} ===")

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files supported")

    doc_id = str(uuid.uuid4())
    _log_step(logs, "info", f"Document ID: {doc_id}")

    # Save temp file
    tmp_path = None
    try:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        tmp_path = tmp.name
        contents = await file.read()
        tmp.write(contents)
        tmp.close()
        _log_step(logs, "info", f"Saved to: {tmp_path} ({len(contents)} bytes)")
    except Exception as e:
        raise HTTPException(500, f"Failed to save file: {str(e)}")

    try:
        # Extract text
        _log_step(logs, "info", "=== TEXT EXTRACTION ===")
        extracted_text, page_count = extract_with_ocrspace(tmp_path, logs)
        
        if extracted_text and extracted_text.strip():
            used_model = "ocrspace"
        else:
            _log_step(logs, "info", "Falling back to pdfplumber")
            extracted_text, page_count = extract_with_pdfplumber(tmp_path, logs)
            used_model = "pdfplumber" if extracted_text.strip() else "none"

        if used_model == "none":
            raise HTTPException(500, "Text extraction failed")

        _log_step(logs, "info", f"Extraction method: {used_model}, Pages: {page_count}")

        # Comprehensive analysis with Gemini
        _log_step(logs, "info", "=== GEMINI COMPREHENSIVE ANALYSIS ===")
        analysis = analyze_with_gemini(extracted_text, file.filename, logs)

        # Prepare summary text
        summary_text = analysis.get("summary", {}).get("main_purpose", "")
        if analysis.get("parties", {}).get("buyer"):
            summary_text += f"\nBuyer: {analysis['parties']['buyer']}"
        if analysis.get("parties", {}).get("seller"):
            summary_text += f"\nSeller: {analysis['parties']['seller']}"

        # Store in database
        _log_step(logs, "info", "=== DATABASE STORAGE ===")
        db = next(get_db())
        try:
            doc = Document(
                id=doc_id,
                document_name=file.filename,
                uploaded_on=datetime.utcnow(),
                summary=summary_text,
                document_type=analysis.get("document_type"),
                document_version=analysis.get("document_version"),
                buyer=analysis.get("parties", {}).get("buyer"),
                seller=analysis.get("parties", {}).get("seller"),
                parties_json=analysis.get("parties"),
                deadlines=analysis.get("deadlines"),
                alerts=analysis.get("alerts"),
                obligations=analysis.get("obligations"),
                cleaned_text=extracted_text,
                text_as_json=analysis,
                page_count=page_count,
                extraction_method=used_model
            )
            db.add(doc)
            db.commit()
            db.refresh(doc)
            _log_step(logs, "info", f"Stored document: {doc_id}")
        finally:
            db.close()

    finally:
        # Cleanup
        if tmp_path:
            try:
                os.remove(tmp_path)
                _log_step(logs, "info", f"Removed temp file")
            except:
                pass

    _log_step(logs, "info", "=== COMPLETE ===")
    
    return JSONResponse({
        "document_id": doc_id,
        "processing_log": logs,
        "used_model": used_model,
        "page_count": page_count,
        "analysis": {
            "document_type": analysis.get("document_type"),
            "document_version": analysis.get("document_version"),
            "parties": analysis.get("parties"),
            "deadlines_count": len(analysis.get("deadlines", [])),
            "alerts_count": len(analysis.get("alerts", [])),
            "obligations_count": len(analysis.get("obligations", [])),
            "sections_count": len(analysis.get("sections", []))
        },
        "structured_json": analysis
    })


@router.get("/documents/{document_id}")
async def get_document(document_id: str):
    """Get document by ID with full analysis"""
    db = next(get_db())
    try:
        doc = db.query(Document).filter(Document.id == document_id).first()
        if not doc:
            raise HTTPException(404, "Document not found")
        
        return {
            "id": doc.id,
            "document_name": doc.document_name,
            "uploaded_on": doc.uploaded_on.isoformat(),
            "document_type": doc.document_type,
            "document_version": doc.document_version,
            "summary": doc.summary,
            "buyer": doc.buyer,
            "seller": doc.seller,
            "parties": doc.parties_json,
            "deadlines": doc.deadlines,
            "alerts": doc.alerts,
            "obligations": doc.obligations,
            "page_count": doc.page_count,
            "extraction_method": doc.extraction_method,
            "cleaned_text": doc.cleaned_text,
            "text_as_json": doc.text_as_json
        }
    finally:
        db.close()


@router.get("/documents/")
async def list_documents(skip: int = 0, limit: int = 100):
    """List all documents with summary info"""
    db = next(get_db())
    try:
        total = db.query(Document).count()
        docs = db.query(Document).offset(skip).limit(limit).all()
        
        return {
            "total": total,
            "documents": [
                {
                    "id": d.id,
                    "document_name": d.document_name,
                    "uploaded_on": d.uploaded_on.isoformat(),
                    "document_type": d.document_type,
                    "summary": d.summary,
                    "buyer": d.buyer,
                    "seller": d.seller,
                    "deadlines_count": len(d.deadlines) if d.deadlines else 0,
                    "alerts_count": len(d.alerts) if d.alerts else 0
                }
                for d in docs
            ]
        }
    finally:
        db.close()


@router.get("/documents/{document_id}/deadlines")
async def get_deadlines(document_id: str):
    """Get all deadlines for a document"""
    db = next(get_db())
    try:
        doc = db.query(Document).filter(Document.id == document_id).first()
        if not doc:
            raise HTTPException(404, "Document not found")
        
        return {
            "document_id": doc.id,
            "document_name": doc.document_name,
            "deadlines": doc.deadlines or []
        }
    finally:
        db.close()


@router.get("/documents/{document_id}/alerts")
async def get_alerts(document_id: str):
    """Get all alerts for a document"""
    db = next(get_db())
    try:
        doc = db.query(Document).filter(Document.id == document_id).first()
        if not doc:
            raise HTTPException(404, "Document not found")
        
        return {
            "document_id": doc.id,
            "document_name": doc.document_name,
            "alerts": doc.alerts or []
        }
    finally:
        db.close()


@router.get("/documents/{document_id}/obligations")
async def get_obligations(document_id: str):
    """Get all obligations for a document"""
    db = next(get_db())
    try:
        doc = db.query(Document).filter(Document.id == document_id).first()
        if not doc:
            raise HTTPException(404, "Document not found")
        
        return {
            "document_id": doc.id,
            "document_name": doc.document_name,
            "obligations": doc.obligations or []
        }
    finally:
        db.close()


app.include_router(router)


@app.get("/health")
async def health():
    """Health check with configuration status"""
    return {
        "status": "healthy",
        "ocr_configured": bool(OCR_SPACE_API_KEY),
        "gemini_configured": bool(GEMINI_API_KEY),
        "version": "2.0-enhanced"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)