#!/usr/bin/env python3
"""
Contract PDF Parser API (full pipeline)
- Primary extractor: Doctr OCR (ocr_predictor pretrained)
- Fallback extractor: pdfplumber
- Classification into JSON schema: schedule_name -> sections -> clauses
- Handles multi-level numbering (1, 1.1, 1.1.1...), bullets, tables, appendices
- Normalizes section names (strips leading numbering)
- Detailed step-by-step logging returned with the response
- FastAPI app; run with: uvicorn app:app --reload
"""

import os
import re
import json
import logging
import tempfile
from typing import List, Dict, Any

import pdfplumber
from fastapi import FastAPI, UploadFile, File, HTTPException,APIRouter
from fastapi.responses import JSONResponse
from doctr.models import ocr_predictor
from doctr.io import DocumentFile

# ---------------------
# Logging config
# ---------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("contract_parser")

# ---------------------
# FastAPI app
# ---------------------
app = FastAPI(title="Contract PDF Parser API (Doctr primary, pdfplumber fallback)")
router = APIRouter(prefix="/Document_praser", tags=["process document"]) 
# ---------------------
# Load Doctr model once at startup
# ---------------------
logger.info("Loading Doctr OCR model (ocr_predictor pretrained). This may take a moment...")
try:
    doctr_model = ocr_predictor(pretrained=True)
    logger.info("Doctr OCR model loaded successfully.")
except Exception as e:
    # If model load fails at startup, keep reference None and attempt on-demand;
    # but we still report the failure in logs and to client if used.
    doctr_model = None
    logger.error(f"Failed to load Doctr model at startup: {e}")


# ---------------------
# Helper: record a log step (keeps logs to return to client)
# ---------------------
def _log_step(logs: List[str], level: str, message: str) -> None:
    ts = f"{logging.Formatter().formatTime(logging.LogRecord('', '', '', '', '', None, None))}"
    entry = f"{level.upper()} | {message}"
    logs.append(entry)
    if level.lower() == "info":
        logger.info(message)
    elif level.lower() == "warning":
        logger.warning(message)
    elif level.lower() == "error":
        logger.error(message)
    else:
        logger.debug(message)


# ---------------------
# Extraction functions
# ---------------------
def extract_with_doctr(pdf_path: str, logs: List[str]) -> str:
    """Try to extract text using Doctr OCR (primary). Returns extracted text (possibly empty)."""
    if doctr_model is None:
        _log_step(logs, "warning", "Doctr model not loaded (None). Skipping Doctr extraction.")
        return ""
    try:
        _log_step(logs, "info", "Starting extraction with Doctr OCR (primary).")
        doc = DocumentFile.from_pdf(pdf_path)
        result = doctr_model(doc)  # performs OCR
        text = result.render()  # get plain text string
        if text and text.strip():
            _log_step(logs, "info", "Doctr OCR extraction succeeded and returned non-empty text.")
        else:
            _log_step(logs, "warning", "Doctr OCR extraction returned empty text.")
        return text or ""
    except Exception as e:
        _log_step(logs, "error", f"Doctr OCR extraction failed with exception: {e}")
        return ""


def extract_with_pdfplumber(pdf_path: str, logs: List[str]) -> str:
    """Fallback extraction using pdfplumber. Returns extracted text (possibly empty)."""
    _log_step(logs, "info", "Starting fallback extraction with pdfplumber.")
    text_parts: List[str] = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for pageno, page in enumerate(pdf.pages, start=1):
                page_text = page.extract_text() or ""
                text_parts.append(page_text)
                _log_step(logs, "info", f"pdfplumber: extracted page {pageno} (length {len(page_text)}).")
        joined = "\n".join(text_parts)
        if joined.strip():
            _log_step(logs, "info", "pdfplumber extraction completed and returned non-empty text.")
        else:
            _log_step(logs, "warning", "pdfplumber extraction returned empty text.")
        return joined
    except Exception as e:
        _log_step(logs, "error", f"pdfplumber extraction failed with exception: {e}")
        return ""


# ---------------------
# Classification / parsing
# ---------------------
def normalize_section_name(section_line: str) -> str:
    """Keep numbering in the section header, just normalize spaces/punctuation"""
    return section_line.strip()

def classify_text_to_json(text: str, schedule_name: str, logs: List[str]) -> Dict[str, Any]:
    """
    Converts extracted text into the JSON schema:
      { "schedule_name": str,
        "sections": [ { "section_name": str, "clauses": [ { "clause_id": str, "content": str } ] } ]
      }
    Handles multi-level numbering, bullets, tables, appendices.
    """
    _log_step(logs, "info", "Starting classification of extracted text into JSON structure.")

    output = {"schedule_name": schedule_name, "sections": []}

    current_section = None
    current_clauses: List[Dict[str, str]] = []

    # Pre-split lines and normalize whitespace
    lines = [ln.strip() for ln in text.splitlines()]

    # Useful regexes
    section_re = re.compile(r"^\d+\.\s+")              # e.g., "1. "
    clause_re = re.compile(r"^\d+(?:\.\d+)+")         # e.g., "1.1" or "3.2.1"
    bullet_re = re.compile(r"^[-•\u2022]\s+")         # hyphen or bullet char
    appendix_re = re.compile(r"\b(Appendix|Appendices|Table|Annex)\b", re.IGNORECASE)

    for idx, line in enumerate(lines):
        if not line:
            continue

        # Detect a Section heading like "1. Glossary" or "10. APPLICATIONS"
        if section_re.match(line):
            _log_step(logs, "info", f"Detected SECTION line (idx={idx}): '{line[:80]}'")
            # push previous section
            if current_section is not None:
                output["sections"].append({
                    "section_name": normalize_section_name(current_section),
                    "clauses": current_clauses
                })
            current_section = line
            current_clauses = []
            continue

        # Detect a clause with multi-level numbering e.g., "3.1", "3.2.1"
        m_clause = clause_re.match(line)
        if m_clause:
            clause_id = m_clause.group(0)
            content = line[len(clause_id):].strip(" .:-—–")  # remove trailing punctuation
            _log_step(logs, "info", f"Detected CLAUSE '{clause_id}' (idx={idx})")
            current_clauses.append({"clause_id": clause_id, "content": content})
            continue

        # Detect bullet points
        if bullet_re.match(line):
            _log_step(logs, "info", f"Detected BULLET (idx={idx})")
            if current_clauses:
                # append bullet text into last clause content, mark as newline bullet
                current_clauses[-1]["content"] += "\n" + line
            else:
                # if no clause exists, create an anonymous bullet clause
                current_clauses.append({"clause_id": "bullet", "content": line})
            continue

        # Detect Table / Appendix mentions - add as note clause
        if appendix_re.search(line):
            _log_step(logs, "info", f"Detected APPENDIX/TABLE mention (idx={idx})")
            current_clauses.append({"clause_id": "note", "content": line})
            continue

        # Otherwise treat as continuation of previous clause content (if any)
        if current_clauses:
            current_clauses[-1]["content"] += " " + line
        else:
            # If we have no section recognized yet, treat this as a top-level unnamed section
            if current_section is None:
                current_section = "1. Untitled Section"
                current_clauses = []
                _log_step(logs, "warning", "No section header detected yet — creating 'Untitled Section' to hold content.")
            current_clauses.append({"clause_id": "p", "content": line})

    # push last section if exists
    if current_section is not None:
        output["sections"].append({
            "section_name": normalize_section_name(current_section),
            "clauses": current_clauses
        })

    _log_step(logs, "info", "Classification complete.")
    return output


# ---------------------
# API endpoint
# ---------------------
@router.post("/parse-pdf_thulasi/")
async def parse_pdf(file: UploadFile = File(...)):
    """
    Parse uploaded PDF and return:
      {
        "processing_log": [ ... ],
        "used_model": "doctr" | "pdfplumber" | "none",
        "structured_json": { ... }
      }
    """
    processing_logs: List[str] = []
    _log_step(processing_logs, "info", f"Received upload: {file.filename}")

    if not file.filename.lower().endswith(".pdf"):
        _log_step(processing_logs, "error", "Uploaded file is not a PDF.")
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    # Save to a secure temp file
    try:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        tmp_path = tmp.name
        contents = await file.read()
        tmp.write(contents)
        tmp.flush()
        tmp.close()
        _log_step(processing_logs, "info", f"Saved uploaded PDF to temporary path: {tmp_path}")
    except Exception as e:
        _log_step(processing_logs, "error", f"Failed to save uploaded file: {e}")
        raise HTTPException(status_code=500, detail="Failed to save uploaded file.")

    used_model = "none"
    extracted_text = ""

    # 1) Try Doctr first
    extracted_text = extract_with_doctr(tmp_path, processing_logs)
    if extracted_text and extracted_text.strip():
        used_model = "doctr"
    else:
        # 2) Fallback to pdfplumber
        _log_step(processing_logs, "info", "Doctr produced no usable text; falling back to pdfplumber.")
        extracted_text = extract_with_pdfplumber(tmp_path, processing_logs)
        if extracted_text and extracted_text.strip():
            used_model = "pdfplumber"
        else:
            used_model = "none"

    if used_model == "none":
        _log_step(processing_logs, "error", "Both Doctr and pdfplumber failed to extract text.")
        # clean up temp file before raising
        try:
            os.remove(tmp_path)
        except Exception:
            pass
        raise HTTPException(status_code=500, detail="Failed to extract text from PDF using both Doctr and pdfplumber.")

    _log_step(processing_logs, "info", f"Text extraction used: {used_model}")

    # Classify the extracted text
    structured = classify_text_to_json(extracted_text, schedule_name=file.filename, logs=processing_logs)

    # Cleanup temp
    try:
        os.remove(tmp_path)
        _log_step(processing_logs, "info", f"Removed temporary file: {tmp_path}")
    except Exception as e:
        _log_step(processing_logs, "warning", f"Failed to remove temp file: {e}")

    # Return both the structured JSON and the logs so user can audit steps
    response = {
        "processing_log": processing_logs,
        "used_model": used_model,
        "structured_json": structured
    }
    return JSONResponse(content=response)
