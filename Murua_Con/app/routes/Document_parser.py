#!/usr/bin/env python3
"""
Production-Grade Contract PDF Parser with Intelligent Batching
Handles large PDFs by splitting into OCR batches and Gemini chunks
"""

import os
import json
import logging
import tempfile
import requests
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import uuid
import re
from pathlib import Path

import pdfplumber
from PyPDF2 import PdfReader, PdfWriter
import google.generativeai as genai
from fastapi import FastAPI, UploadFile, File, HTTPException, APIRouter, BackgroundTasks
from fastapi.responses import JSONResponse
from sqlalchemy import Column, String, DateTime, Text, JSON, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session
from app.database.models import Document
from app.database.database import get_db, engine
from app.config import config

# =====================================================
# CONFIGURATION
# =====================================================
OCR_SPACE_API_KEY = "K86948634088957"
GEMINI_API_KEY = "AIzaSyBR-j_7vbLMCvE4yo4vqLqaLPWYKecqPuY"
OCR_SPACE_URL = "https://api.ocr.space/parse/image"

# Batching Configuration
OCR_BATCH_SIZE = 3  # Pages per OCR batch
GEMINI_PAGES_PER_CHUNK = 15  # Pages per Gemini analysis (5 OCR batches)
MAX_CHARS_PER_CHUNK = 800000  # Safety limit for Gemini input
GEMINI_MAX_OUTPUT_TOKENS = 15192
# =====================================================

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)

# Logging config
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger("contract_parser")

# FastAPI app
app = FastAPI(title="ContractX PDF Parser API v3.0")
router = APIRouter(prefix="/Document_parser", tags=["AI Document praser"])

logger.info("=== ContractX PDF Parser API v3.0 Starting ===")
logger.info(f"OCR Batch Size: {OCR_BATCH_SIZE} pages")
logger.info(f"Gemini Chunk Size: {GEMINI_PAGES_PER_CHUNK} pages")


class ProcessingStats:
    """Track processing statistics"""
    def __init__(self):
        self.ocr_batches_processed = 0
        self.gemini_chunks_processed = 0
        self.total_pages = 0
        self.total_chars_extracted = 0
        self.failed_ocr_batches = 0
        self.failed_gemini_chunks = 0


def _log_step(logs: List[str], level: str, message: str) -> None:
    """Record a processing step"""
    entry = f"{datetime.now().strftime('%H:%M:%S')} | {level.upper()} | {message}"
    logs.append(entry)
    getattr(logger, level.lower(), logger.info)(message)


def split_pdf_into_batches(pdf_path: str, batch_size: int, logs: List[str]) -> List[str]:
    """
    Split PDF into smaller batch files for OCR processing
    Returns list of temporary batch file paths
    """
    _log_step(logs, "info", f"Splitting PDF into {batch_size}-page batches")
    
    try:
        reader = PdfReader(pdf_path)
        total_pages = len(reader.pages)
        _log_step(logs, "info", f"Total pages: {total_pages}")
        
        batch_paths = []
        batch_num = 0
        
        for start_page in range(0, total_pages, batch_size):
            end_page = min(start_page + batch_size, total_pages)
            batch_num += 1
            
            # Create batch PDF
            writer = PdfWriter()
            for page_num in range(start_page, end_page):
                writer.add_page(reader.pages[page_num])
            
            # Save to temp file
            batch_file = tempfile.NamedTemporaryFile(
                delete=False, 
                suffix=f"_batch{batch_num}.pdf"
            )
            writer.write(batch_file)
            batch_file.close()
            
            batch_paths.append(batch_file.name)
            _log_step(logs, "info", 
                     f"Created batch {batch_num}: pages {start_page+1}-{end_page} → {batch_file.name}")
        
        _log_step(logs, "info", f"Created {len(batch_paths)} batches")
        return batch_paths
        
    except Exception as e:
        _log_step(logs, "error", f"Failed to split PDF: {str(e)}")
        raise


def extract_batch_with_ocrspace(batch_path: str, batch_num: int, logs: List[str]) -> Tuple[str, bool]:
    """
    Extract text from a single batch using OCR.space
    Returns (text, success_flag)
    """
    if not OCR_SPACE_API_KEY:
        _log_step(logs, "warning", f"Batch {batch_num}: OCR.space key not set")
        return "", False
    
    try:
        _log_step(logs, "info", f"Batch {batch_num}: Starting OCR.space extraction")
        
        with open(batch_path, 'rb') as f:
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
                _log_step(logs, "error", 
                         f"Batch {batch_num}: OCR.space HTTP {response.status_code}")
                return "", False
            
            result = response.json()
            
            if result.get('IsErroredOnProcessing'):
                error_msg = result.get('ErrorMessage', ['Unknown'])[0]
                _log_step(logs, "error", 
                         f"Batch {batch_num}: OCR.space error: {error_msg}")
                return "", False
            
            # Combine all pages in this batch
            text_parts = []
            for page_result in result.get('ParsedResults', []):
                page_text = page_result.get('ParsedText', '')
                text_parts.append(page_text)
            
            batch_text = "\n".join(text_parts)
            _log_step(logs, "info", 
                     f"Batch {batch_num}: Extracted {len(batch_text)} chars")
            return batch_text, True
            
    except Exception as e:
        _log_step(logs, "error", f"Batch {batch_num}: Exception: {str(e)}")
        return "", False


def extract_batch_with_pdfplumber(batch_path: str, batch_num: int, logs: List[str]) -> Tuple[str, bool]:
    """
    Fallback: Extract text from batch using pdfplumber
    Returns (text, success_flag)
    """
    try:
        _log_step(logs, "info", f"Batch {batch_num}: Using pdfplumber fallback")
        
        text_parts = []
        with pdfplumber.open(batch_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                text_parts.append(page_text)
        
        batch_text = "\n".join(text_parts)
        _log_step(logs, "info", 
                 f"Batch {batch_num}: Extracted {len(batch_text)} chars (pdfplumber)")
        return batch_text, True
        
    except Exception as e:
        _log_step(logs, "error", f"Batch {batch_num}: pdfplumber failed: {str(e)}")
        return "", False


def process_ocr_batches(batch_paths: List[str], logs: List[str], stats: ProcessingStats) -> List[str]:
    """
    Process all OCR batches and return list of extracted texts
    Each element corresponds to one batch (OCR_BATCH_SIZE pages)
    """
    _log_step(logs, "info", f"=== PROCESSING {len(batch_paths)} OCR BATCHES ===")
    
    batch_texts = []
    
    for idx, batch_path in enumerate(batch_paths, 1):
        _log_step(logs, "info", f"Processing OCR batch {idx}/{len(batch_paths)}")
        
        # Try OCR.space first
        text, success = extract_batch_with_ocrspace(batch_path, idx, logs)
        
        # Fallback to pdfplumber if OCR.space fails
        if not success or not text.strip():
            _log_step(logs, "warning", f"Batch {idx}: OCR.space failed, using pdfplumber")
            text, success = extract_batch_with_pdfplumber(batch_path, idx, logs)
        
        if success and text.strip():
            batch_texts.append(text)
            stats.ocr_batches_processed += 1
            stats.total_chars_extracted += len(text)
        else:
            _log_step(logs, "error", f"Batch {idx}: Both extraction methods failed")
            batch_texts.append("")  # Keep position for page numbering
            stats.failed_ocr_batches += 1
        
        # Cleanup batch file
        try:
            os.remove(batch_path)
        except:
            pass
    
    stats.total_pages = len(batch_paths) * OCR_BATCH_SIZE
    _log_step(logs, "info", 
             f"OCR Complete: {stats.ocr_batches_processed}/{len(batch_paths)} successful")
    return batch_texts

@router.get("/documents/{document_id}")
async def get_document(document_id: str):
    """Get complete document analysis by ID"""
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
            "full_analysis": doc.text_as_json
        }
    finally:
        db.close()


@router.get("/documents/")
async def list_documents(skip: int = 0, limit: int = 100):
    """List all processed documents"""
    db = next(get_db())
    try:
        total = db.query(Document).count()
        docs = db.query(Document).offset(skip).limit(limit).all()
        
        return {
            "total": total,
            "skip": skip,
            "limit": limit,
            "documents": [
                {
                    "id": d.id,
                    "document_name": d.document_name,
                    "uploaded_on": d.uploaded_on.isoformat(),
                    "document_type": d.document_type,
                    "summary": d.summary,
                    "buyer": d.buyer,
                    "seller": d.seller,
                    "page_count": d.page_count,
                    "deadlines_count": len(d.deadlines) if d.deadlines else 0,
                    "alerts_count": len(d.alerts) if d.alerts else 0,
                    "extraction_method": d.extraction_method
                }
                for d in docs
            ]
        }
    finally:
        db.close()


@router.get("/documents/{document_id}/sections")
async def get_sections(document_id: str):
    """Get all sections and clauses for a document"""
    db = next(get_db())
    try:
        doc = db.query(Document).filter(Document.id == document_id).first()
        if not doc:
            raise HTTPException(404, "Document not found")
        
        analysis = doc.text_as_json or {}
        sections = analysis.get("sections", [])
        
        return {
            "document_id": doc.id,
            "document_name": doc.document_name,
            "sections_count": len(sections),
            "total_clauses": sum(len(s.get("clauses", [])) for s in sections),
            "sections": sections
        }
    finally:
        db.close()


@router.get("/documents/{document_id}/deadlines")
async def get_deadlines(document_id: str, priority: Optional[str] = None):
    """Get deadlines for a document, optionally filtered by priority"""
    db = next(get_db())
    try:
        doc = db.query(Document).filter(Document.id == document_id).first()
        if not doc:
            raise HTTPException(404, "Document not found")
        
        deadlines = doc.deadlines or []
        
        if priority:
            deadlines = [d for d in deadlines if d.get("priority") == priority.lower()]
        
        return {
            "document_id": doc.id,
            "document_name": doc.document_name,
            "deadlines_count": len(deadlines),
            "deadlines": deadlines
        }
    finally:
        db.close()


@router.get("/documents/{document_id}/alerts")
async def get_alerts(document_id: str, severity: Optional[str] = None):
    """Get alerts for a document, optionally filtered by severity"""
    db = next(get_db())
    try:
        doc = db.query(Document).filter(Document.id == document_id).first()
        if not doc:
            raise HTTPException(404, "Document not found")
        
        alerts = doc.alerts or []
        
        if severity:
            alerts = [a for a in alerts if a.get("severity") == severity.lower()]
        
        return {
            "document_id": doc.id,
            "document_name": doc.document_name,
            "alerts_count": len(alerts),
            "alerts": alerts
        }
    finally:
        db.close()


@router.get("/documents/{document_id}/obligations")
async def get_obligations(document_id: str, party: Optional[str] = None):
    """Get obligations for a document, optionally filtered by party"""
    db = next(get_db())
    try:
        doc = db.query(Document).filter(Document.id == document_id).first()
        if not doc:
            raise HTTPException(404, "Document not found")
        
        obligations = doc.obligations or []
        
        if party:
            obligations = [
                o for o in obligations 
                if party.lower() in o.get("party", "").lower()
            ]
        
        return {
            "document_id": doc.id,
            "document_name": doc.document_name,
            "obligations_count": len(obligations),
            "obligations": obligations
        }
    finally:
        db.close()


@router.get("/documents/{document_id}/parties")
async def get_parties(document_id: str):
    """Get all parties information for a document"""
    db = next(get_db())
    try:
        doc = db.query(Document).filter(Document.id == document_id).first()
        if not doc:
            raise HTTPException(404, "Document not found")
        
        return {
            "document_id": doc.id,
            "document_name": doc.document_name,
            "buyer": doc.buyer,
            "seller": doc.seller,
            "all_parties": doc.parties_json.get("all_parties", []) if doc.parties_json else []
        }
    finally:
        db.close()


@router.get("/documents/{document_id}/financial-terms")
async def get_financial_terms(document_id: str):
    """Get all financial terms from a document"""
    db = next(get_db())
    try:
        doc = db.query(Document).filter(Document.id == document_id).first()
        if not doc:
            raise HTTPException(404, "Document not found")
        
        analysis = doc.text_as_json or {}
        financial_terms = analysis.get("financial_terms", [])
        
        return {
            "document_id": doc.id,
            "document_name": doc.document_name,
            "financial_terms_count": len(financial_terms),
            "financial_terms": financial_terms,
            "total_value": sum(
                float(ft.get("amount", 0)) 
                for ft in financial_terms 
                if ft.get("amount", "").replace(".", "").isdigit()
            )
        }
    finally:
        db.close()


@router.get("/documents/{document_id}/special-clauses")
async def get_special_clauses(document_id: str):
    """Get special clauses (indemnification, liability, etc.)"""
    db = next(get_db())
    try:
        doc = db.query(Document).filter(Document.id == document_id).first()
        if not doc:
            raise HTTPException(404, "Document not found")
        
        analysis = doc.text_as_json or {}
        special_clauses = analysis.get("special_clauses", [])
        
        return {
            "document_id": doc.id,
            "document_name": doc.document_name,
            "special_clauses_count": len(special_clauses),
            "special_clauses": special_clauses
        }
    finally:
        db.close()


@router.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    """Delete a document from the database"""
    db = next(get_db())
    try:
        doc = db.query(Document).filter(Document.id == document_id).first()
        if not doc:
            raise HTTPException(404, "Document not found")
        
        db.delete(doc)
        db.commit()
        
        return {
            "success": True,
            "message": f"Document {document_id} deleted successfully"
        }
    finally:
        db.close()


@router.get("/health")
async def health_check():
    """Detailed health check with system status"""
    return {
        "status": "healthy",
        "version": "3.0-production",
        "features": {
            "ocr_batching": True,
            "gemini_chunking": True,
            "large_pdf_support": True,
            "max_pages": "unlimited (batched processing)"
        },
        "configuration": {
            "ocr_configured": bool(OCR_SPACE_API_KEY),
            "gemini_configured": bool(GEMINI_API_KEY),
            "ocr_batch_size": OCR_BATCH_SIZE,
            "gemini_chunk_size": GEMINI_PAGES_PER_CHUNK,
            "max_chars_per_chunk": MAX_CHARS_PER_CHUNK
        },
        "endpoints": {
            "parse": "/Document_parser/parse-pdf/DOCUMENT_ANALYSIS",
            "list": "/Document_parser/documents/",
            "get": "/Document_parser/documents/{id}",
            "sections": "/Document_parser/documents/{id}/sections",
            "deadlines": "/Document_parser/documents/{id}/deadlines",
            "alerts": "/Document_parser/documents/{id}/alerts",
            "obligations": "/Document_parser/documents/{id}/obligations",
            "parties": "/Document_parser/documents/{id}/parties",
            "financial": "/Document_parser/documents/{id}/financial-terms",
            "special_clauses": "/Document_parser/documents/{id}/special-clauses"
        }
    }

def merge_batches_into_chunks(batch_texts: List[str], pages_per_chunk: int, 
                               batch_size: int, logs: List[str]) -> List[Dict[str, Any]]:
    """
    Merge OCR batch texts into larger chunks for Gemini processing
    Each chunk contains multiple batches (up to pages_per_chunk total pages)
    
    Returns list of dicts with:
    - text: merged text for this chunk
    - start_page: first page number (1-indexed)
    - end_page: last page number
    - batch_indices: which OCR batches are included
    """
    _log_step(logs, "info", f"=== MERGING INTO GEMINI CHUNKS ===")
    
    batches_per_chunk = pages_per_chunk // batch_size
    chunks = []
    
    for chunk_idx in range(0, len(batch_texts), batches_per_chunk):
        chunk_batches = batch_texts[chunk_idx:chunk_idx + batches_per_chunk]
        
        # Merge texts
        chunk_text = "\n\n".join(chunk_batches)
        
        # Calculate page numbers
        start_page = chunk_idx * batch_size + 1
        end_page = min((chunk_idx + len(chunk_batches)) * batch_size, 
                       len(batch_texts) * batch_size)
        
        # Truncate if too large
        if len(chunk_text) > MAX_CHARS_PER_CHUNK:
            _log_step(logs, "warning", 
                     f"Chunk {len(chunks)+1}: Truncating from {len(chunk_text)} to {MAX_CHARS_PER_CHUNK} chars")
            chunk_text = chunk_text[:MAX_CHARS_PER_CHUNK] + "\n[...truncated for size...]"
        
        chunks.append({
            "text": chunk_text,
            "start_page": start_page,
            "end_page": end_page,
            "batch_indices": list(range(chunk_idx, chunk_idx + len(chunk_batches))),
            "char_count": len(chunk_text)
        })
        
        _log_step(logs, "info", 
                 f"Chunk {len(chunks)}: Pages {start_page}-{end_page}, {len(chunk_text)} chars")
    
    _log_step(logs, "info", f"Created {len(chunks)} chunks for Gemini analysis")
    return chunks


def analyze_chunk_with_gemini(chunk: Dict[str, Any], chunk_num: int, 
                               total_chunks: int, logs: List[str]) -> Dict[str, Any]:
    """
    Analyze a single chunk with Gemini AI
    Returns structured JSON for this chunk
    """
    _log_step(logs, "info", 
             f"Analyzing chunk {chunk_num}/{total_chunks} (pages {chunk['start_page']}-{chunk['end_page']})")
    
    if not GEMINI_API_KEY:
        _log_step(logs, "error", "Gemini API key not configured")
        return _empty_chunk_analysis(chunk['start_page'], chunk['end_page'])
    
    try:
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
        # Enhanced prompt with context awareness
        is_first_chunk = chunk_num == 1
        is_last_chunk = chunk_num == total_chunks
        
        context_note = ""
        if total_chunks > 1:
            context_note = f"\n\nIMPORTANT CONTEXT:\n- This is chunk {chunk_num} of {total_chunks} total chunks\n"
            if is_first_chunk:
                context_note += "- This is the BEGINNING of the document. Focus on: title, parties, definitions, recitals\n"
            elif is_last_chunk:
                context_note += "- This is the END of the document. Focus on: signatures, schedules, annexes\n"
            else:
                context_note += f"- This is a MIDDLE section (pages {chunk['start_page']}-{chunk['end_page']}). Extract all clauses completely\n"
            context_note += "- Maintain clause numbering exactly as it appears\n"
            context_note += "- Do not assume missing context - analyze only what is present\n"
        
        prompt = f"""You are an expert legal document analyzer. Analyze this contract section and return structured JSON.

{context_note}

DOCUMENT SECTION (Pages {chunk['start_page']}-{chunk['end_page']}):
{chunk['text']}

ANALYSIS REQUIREMENTS:

1. **Clause Detection** - Recognize ALL numbering patterns:
   - Decimal: 1.1, 1.1.1, 1.1.1.1
   - Parenthetical: (1), (2), (a), (b), (A), (B)
   - Roman: i., ii., iii., I., II., III., IV.
   - Letter: a), b), c), A), B), C)
   - Bullets: •, -, *, ◦
   - Explicit: "Clause 1.1", "Section 2.3", "Article 5"

2. **Section Headings** - Identify patterns:
   - "1. INTRODUCTION", "2) DEFINITIONS", "Section 3 - Terms"
   - "ARTICLE I - GENERAL", "Part A - Obligations"
   - "SCHEDULE 1", "ANNEX A", "APPENDIX B"
   - ALL CAPS standalone headings
   - Underlined or bold headings (infer from context)

3. **Content Extraction**:
   - ALL parties mentioned (full legal names with addresses if present)
   - ALL dates (preserve original format, note if ambiguous)
   - Payment terms, amounts, currencies
   - Deadlines with responsible parties
   - Termination conditions
   - Penalty clauses
   - Compliance requirements
   - Definitions (if present)

4. **Critical Business Terms**:
   - Contract value/price
   - Payment schedules
   - Delivery dates
   - Warranties and representations
   - Indemnification clauses
   - Limitation of liability
   - Dispute resolution mechanisms
   - Governing law and jurisdiction

REQUIRED JSON OUTPUT (ONLY valid JSON, no markdown):
{{
  "chunk_metadata": {{
    "start_page": {chunk['start_page']},
    "end_page": {chunk['end_page']},
    "chunk_number": {chunk_num}
  }},
  
  "document_metadata": {{
    "title": "Full document title if found, else null",
    "date": "Document date if found, else null",
    "reference_number": "Any reference/contract number, else null"
  }},
  
  "parties": [
    {{
      "role": "buyer|seller|contractor|client|party_name",
      "full_legal_name": "Complete legal entity name",
      "short_name": "Common/short reference name",
      "address": "Full address if present",
      "contact": "Email/phone if present",
      "representative": "Authorized signatory if mentioned"
    }}
  ],
  
  "definitions": [
    {{
      "term": "Defined term",
      "definition": "Full definition text",
      "clause_reference": "Clause ID where defined"
    }}
  ],
  
  "sections": [
    {{
      "section_id": "Section number/identifier",
      "section_name": "Section title/heading",
      "page_start": {chunk['start_page']},
      "clauses": [
        {{
          "clause_id": "Full clause number (e.g., 5.3.2)",
          "clause_title": "Clause heading if present",
          "content": "Complete clause text",
          "clause_type": "definition|obligation|right|condition|term|other",
          "subclauses": [
            {{
              "clause_id": "Sub-clause number",
              "content": "Sub-clause text"
            }}
          ]
        }}
      ]
    }}
  ],
  
  "financial_terms": [
    {{
      "type": "price|fee|penalty|deposit|payment",
      "amount": "Numerical amount",
      "currency": "Currency code",
      "description": "What this amount is for",
      "due_date": "When payable",
      "clause_reference": "Where mentioned"
    }}
  ],
  
  "deadlines": [
    {{
      "date": "YYYY-MM-DD or original format",
      "description": "What must be done",
      "party_responsible": "Who must do it",
      "consequence": "What happens if missed",
      "priority": "critical|high|medium|low",
      "clause_reference": "Clause ID"
    }}
  ],
  
  "obligations": [
    {{
      "party": "Party name",
      "obligation": "What must be done",
      "type": "payment|delivery|reporting|compliance|other",
      "timing": "When it must be done",
      "conditions": "Any conditions or triggers",
      "clause_reference": "Clause ID"
    }}
  ],
  
  "alerts": [
    {{
      "type": "termination|penalty|breach|compliance|warranty|other",
      "description": "Alert description",
      "severity": "critical|high|medium|low",
      "trigger": "What causes this",
      "consequence": "What happens",
      "clause_reference": "Clause ID"
    }}
  ],
  
  "special_clauses": [
    {{
      "type": "indemnification|limitation_of_liability|warranty|confidentiality|ip_rights|dispute_resolution|force_majeure|other",
      "summary": "Brief summary",
      "key_terms": ["term1", "term2"],
      "clause_reference": "Clause ID"
    }}
  ]
}}

CRITICAL RULES:
✓ Return ONLY valid JSON (no markdown, no backticks, no explanations)
✓ Extract ALL content - do not skip or summarize
✓ Preserve exact clause numbering and hierarchy
✓ Use null for missing information, never omit fields
✓ For incomplete clauses (cut off at page boundary), include what's present
✓ Extract all numbers, dates, amounts with units
✓ Identify all cross-references to other clauses
✓ Note any ambiguities or unclear terms
✓ Keep clause content complete - do not truncate
✓ Maintain original legal language precision

Begin JSON:"""

        _log_step(logs, "info", f"Sending {len(chunk['text'])} chars to Gemini")
        
        response = model.generate_content(
            prompt,
            generation_config={
                "temperature": 0.1,  # Low for accuracy
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": GEMINI_MAX_OUTPUT_TOKENS,
            }
        )
        
        if not response or not response.text:
            _log_step(logs, "error", f"Chunk {chunk_num}: Empty Gemini response")
            return _empty_chunk_analysis(chunk['start_page'], chunk['end_page'])
        
        _log_step(logs, "info", f"Chunk {chunk_num}: Received {len(response.text)} chars from Gemini")
        
        # Clean and parse JSON
        response_text = _clean_json_response(response.text)
        
        try:
            data = json.loads(response_text)
            _log_chunk_stats(data, chunk_num, logs)
            return data
            
        except json.JSONDecodeError as e:
            _log_step(logs, "error", f"Chunk {chunk_num}: JSON parse error: {str(e)}")
            _log_step(logs, "error", f"Response preview: {response_text[:500]}")
            return _empty_chunk_analysis(chunk['start_page'], chunk['end_page'])
        
    except Exception as e:
        _log_step(logs, "error", f"Chunk {chunk_num}: Exception: {str(e)}")
        import traceback
        _log_step(logs, "error", traceback.format_exc())
        return _empty_chunk_analysis(chunk['start_page'], chunk['end_page'])


def merge_chunk_analyses(chunk_results: List[Dict[str, Any]], 
                         filename: str, logs: List[str]) -> Dict[str, Any]:
    """
    Merge all chunk analyses into final comprehensive document analysis
    Handles deduplication and proper ordering
    """
    _log_step(logs, "info", "=== MERGING CHUNK ANALYSES ===")
    
    # Initialize merged structure
    merged = {
        "schedule_name": filename,
        "document_type": "Contract",  # Will be refined
        "document_version": None,
        "document_metadata": {},
        "summary": {
            "main_purpose": "",
            "key_points": []
        },
        "parties": {"buyer": None, "seller": None, "all_parties": []},
        "definitions": [],
        "sections": [],
        "financial_terms": [],
        "deadlines": [],
        "obligations": [],
        "alerts": [],
        "special_clauses": []
    }
    
    # Track seen items to avoid duplicates
    seen_parties = set()
    seen_definitions = set()
    
    for idx, chunk_data in enumerate(chunk_results, 1):
        _log_step(logs, "info", f"Merging chunk {idx}/{len(chunk_results)}")
        
        # Document metadata (use first occurrence)
        if not merged["document_metadata"] and chunk_data.get("document_metadata"):
            merged["document_metadata"] = chunk_data["document_metadata"]
            if chunk_data["document_metadata"].get("title"):
                merged["summary"]["main_purpose"] = f"Analysis of: {chunk_data['document_metadata']['title']}"
        
        # Parties (deduplicate by full legal name)
        for party in chunk_data.get("parties", []):
            party_key = party.get("full_legal_name", "").lower()
            if party_key and party_key not in seen_parties:
                seen_parties.add(party_key)
                merged["parties"]["all_parties"].append(party)
                
                # Set buyer/seller if role matches
                role = party.get("role", "").lower()
                if "buyer" in role and not merged["parties"]["buyer"]:
                    merged["parties"]["buyer"] = party["full_legal_name"]
                elif "seller" in role and not merged["parties"]["seller"]:
                    merged["parties"]["seller"] = party["full_legal_name"]
        
        # Definitions (deduplicate by term)
        for defn in chunk_data.get("definitions", []):
            term_key = defn.get("term", "").lower()
            if term_key and term_key not in seen_definitions:
                seen_definitions.add(term_key)
                merged["definitions"].append(defn)
        
        # Sections (append all, maintain order)
        merged["sections"].extend(chunk_data.get("sections", []))
        
        # Financial terms, deadlines, obligations, alerts (append all)
        merged["financial_terms"].extend(chunk_data.get("financial_terms", []))
        merged["deadlines"].extend(chunk_data.get("deadlines", []))
        merged["obligations"].extend(chunk_data.get("obligations", []))
        merged["alerts"].extend(chunk_data.get("alerts", []))
        merged["special_clauses"].extend(chunk_data.get("special_clauses", []))
    
    # Sort by priority/severity
    merged["deadlines"].sort(key=lambda x: {
        "critical": 0, "high": 1, "medium": 2, "low": 3
    }.get(x.get("priority", "low"), 4))
    
    merged["alerts"].sort(key=lambda x: {
        "critical": 0, "high": 1, "medium": 2, "low": 3
    }.get(x.get("severity", "low"), 4))
    
    # Generate summary key points
    merged["summary"]["key_points"] = [
        f"{len(merged['parties']['all_parties'])} parties identified",
        f"{len(merged['sections'])} sections with {sum(len(s.get('clauses', [])) for s in merged['sections'])} clauses",
        f"{len(merged['deadlines'])} deadlines",
        f"{len(merged['alerts'])} alerts",
        f"{len(merged['obligations'])} obligations"
    ]
    
    _log_final_stats(merged, logs)
    return merged


def _clean_json_response(text: str) -> str:
    """Clean Gemini response to extract pure JSON"""
    # Remove markdown code blocks
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()
    
    # Extract JSON object
    first_brace = text.find('{')
    last_brace = text.rfind('}')
    if first_brace != -1 and last_brace != -1:
        text = text[first_brace:last_brace+1]
    
    return text


def _empty_chunk_analysis(start_page: int, end_page: int) -> Dict[str, Any]:
    """Return empty structure for failed chunk"""
    return {
        "chunk_metadata": {
            "start_page": start_page,
            "end_page": end_page,
            "chunk_number": 0
        },
        "document_metadata": {},
        "parties": [],
        "definitions": [],
        "sections": [],
        "financial_terms": [],
        "deadlines": [],
        "obligations": [],
        "alerts": [],
        "special_clauses": []
    }


def _log_chunk_stats(data: Dict[str, Any], chunk_num: int, logs: List[str]) -> None:
    """Log statistics for a chunk analysis"""
    stats = {
        "parties": len(data.get("parties", [])),
        "sections": len(data.get("sections", [])),
        "clauses": sum(len(s.get("clauses", [])) for s in data.get("sections", [])),
        "deadlines": len(data.get("deadlines", [])),
        "alerts": len(data.get("alerts", [])),
        "obligations": len(data.get("obligations", []))
    }
    _log_step(logs, "info", f"Chunk {chunk_num} stats: {stats}")


def _log_final_stats(merged: Dict[str, Any], logs: List[str]) -> None:
    """Log final merged statistics"""
    _log_step(logs, "info", "=== FINAL ANALYSIS STATISTICS ===")
    _log_step(logs, "info", f"Parties: {len(merged['parties']['all_parties'])}")
    _log_step(logs, "info", f"Definitions: {len(merged['definitions'])}")
    _log_step(logs, "info", f"Sections: {len(merged['sections'])}")
    _log_step(logs, "info", f"Total Clauses: {sum(len(s.get('clauses', [])) for s in merged['sections'])}")
    _log_step(logs, "info", f"Financial Terms: {len(merged['financial_terms'])}")
    _log_step(logs, "info", f"Deadlines: {len(merged['deadlines'])}")
    _log_step(logs, "info", f"Obligations: {len(merged['obligations'])}")
    _log_step(logs, "info", f"Alerts: {len(merged['alerts'])}")
    _log_step(logs, "info", f"Special Clauses: {len(merged['special_clauses'])}")


#!/usr/bin/env python3
"""
Fixed sections of the Contract PDF Parser
"""

@router.post("/parse-pdf/DOCUMENT_ANALYSIS_MAX_PAGE")
async def parse_pdf(file: UploadFile = File(...)):
    """
    Parse PDF with intelligent batching for large documents
    Handles 100+ page PDFs efficiently
    """
    logs = []
    stats = ProcessingStats()
    
    _log_step(logs, "info", f"=== NEW UPLOAD: {file.filename} ===")
    
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files supported")
    
    doc_id = str(uuid.uuid4())
    _log_step(logs, "info", f"Document ID: {doc_id}")
    
    # Save uploaded file
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
        # STEP 1: Split into OCR batches
        _log_step(logs, "info", "=== STEP 1: SPLIT INTO OCR BATCHES ===")
        batch_paths = split_pdf_into_batches(tmp_path, OCR_BATCH_SIZE, logs)
        stats.total_pages = len(batch_paths) * OCR_BATCH_SIZE
        
        # STEP 2: Process OCR batches
        _log_step(logs, "info", "=== STEP 2: OCR EXTRACTION ===")
        batch_texts = process_ocr_batches(batch_paths, logs, stats)
        
        # STEP 3: Merge into Gemini chunks
        _log_step(logs, "info", "=== STEP 3: CREATE GEMINI CHUNKS ===")
        chunks = merge_batches_into_chunks(
            batch_texts, 
            GEMINI_PAGES_PER_CHUNK, 
            OCR_BATCH_SIZE, 
            logs
        )
        
        # STEP 4: Analyze chunks with Gemini
        _log_step(logs, "info", "=== STEP 4: GEMINI ANALYSIS ===")
        chunk_results = []
        for chunk_num, chunk in enumerate(chunks, 1):
            result = analyze_chunk_with_gemini(
                chunk, 
                chunk_num, 
                len(chunks), 
                logs
            )
            chunk_results.append(result)
            stats.gemini_chunks_processed += 1
        
        # STEP 5: Merge all analyses
        _log_step(logs, "info", "=== STEP 5: MERGE RESULTS ===")
        final_analysis = merge_chunk_analyses(chunk_results, file.filename, logs)
        
        # STEP 6: Save to database
        _log_step(logs, "info", "=== STEP 6: SAVE TO DATABASE ===")
        db = next(get_db())
        try:
            # Combine all batch texts for full document text
            full_text = "\n\n".join(batch_texts)
            
            document = Document(
                id=doc_id,
                document_name=file.filename,
                uploaded_on=datetime.now(),
                document_type=final_analysis.get("document_type", "Contract"),
                document_version=final_analysis.get("document_version"),
                summary=final_analysis.get("summary", {}).get("main_purpose"),
                buyer=final_analysis.get("parties", {}).get("buyer"),
                seller=final_analysis.get("parties", {}).get("seller"),
                parties_json=final_analysis.get("parties"),
                deadlines=final_analysis.get("deadlines", []),
                alerts=final_analysis.get("alerts", []),
                obligations=final_analysis.get("obligations", []),
                page_count=stats.total_pages,
                extraction_method=f"OCR.space + Gemini ({stats.ocr_batches_processed} batches, {stats.gemini_chunks_processed} chunks)",
                cleaned_text=full_text[:100000],  # Store first 100K chars
                text_as_json=final_analysis
            )
            
            db.add(document)
            db.commit()
            db.refresh(document)
            
            _log_step(logs, "info", f"Document saved to database: {doc_id}")
            
        except Exception as e:
            db.rollback()
            _log_step(logs, "error", f"Database error: {str(e)}")
            raise HTTPException(500, f"Database save failed: {str(e)}")
        finally:
            db.close()
        
        # STEP 7: Final statistics
        _log_step(logs, "info", "=== PROCESSING COMPLETE ===")
        _log_step(logs, "info", f"Total Pages: {stats.total_pages}")
        _log_step(logs, "info", f"OCR Batches: {stats.ocr_batches_processed}/{len(batch_paths)}")
        _log_step(logs, "info", f"Gemini Chunks: {stats.gemini_chunks_processed}/{len(chunks)}")
        _log_step(logs, "info", f"Total Characters: {stats.total_chars_extracted:,}")
        _log_step(logs, "info", f"Failed OCR Batches: {stats.failed_ocr_batches}")
        
        return JSONResponse({
            "success": True,
            "document_id": doc_id,
            "filename": file.filename,
            "processing_stats": {
                "total_pages": stats.total_pages,
                "ocr_batches_processed": stats.ocr_batches_processed,
                "gemini_chunks_processed": stats.gemini_chunks_processed,
                "total_chars_extracted": stats.total_chars_extracted,
                "failed_ocr_batches": stats.failed_ocr_batches,
                "failed_gemini_chunks": stats.failed_gemini_chunks
            },
            "analysis_summary": {
                "parties": len(final_analysis.get("parties", {}).get("all_parties", [])),
                "sections": len(final_analysis.get("sections", [])),
                "deadlines": len(final_analysis.get("deadlines", [])),
                "alerts": len(final_analysis.get("alerts", [])),
                "obligations": len(final_analysis.get("obligations", [])),
                "financial_terms": len(final_analysis.get("financial_terms", []))
            },
            "document": {
                "id": doc_id,
                "name": file.filename,
                "type": final_analysis.get("document_type"),
                "buyer": final_analysis.get("parties", {}).get("buyer"),
                "seller": final_analysis.get("parties", {}).get("seller"),
                "summary": final_analysis.get("summary")
            },
            "logs": logs
        })
        
    except HTTPException:
        raise
    except Exception as e:
        _log_step(logs, "error", f"Fatal error: {str(e)}")
        import traceback
        _log_step(logs, "error", traceback.format_exc())
        raise HTTPException(500, f"Processing failed: {str(e)}")
    finally:
        # Cleanup main temp file
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
                _log_step(logs, "info", f"Cleaned up: {tmp_path}")
            except OSError as e:
                _log_step(logs, "warning", f"Failed to remove {tmp_path}: {e}")


# =====================================================
# FIXED: Main entry point
# =====================================================
app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    
    logger.info("=" * 60)
    logger.info("Starting ContractX PDF Parser API v3.0")
    logger.info("=" * 60)
    logger.info(f"OCR Batch Size: {OCR_BATCH_SIZE} pages")
    logger.info(f"Gemini Chunk Size: {GEMINI_PAGES_PER_CHUNK} pages")
    logger.info(f"Max Chars Per Chunk: {MAX_CHARS_PER_CHUNK:,}")
    logger.info(f"Gemini Max Output Tokens: {GEMINI_MAX_OUTPUT_TOKENS:,}")
    logger.info(f"OCR Configured: {bool(OCR_SPACE_API_KEY)}")
    logger.info(f"Gemini Configured: {bool(GEMINI_API_KEY)}")
    logger.info("=" * 60)
