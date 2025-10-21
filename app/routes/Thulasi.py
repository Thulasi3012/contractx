import asyncio
import json
import logging
import sys
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
from enum import Enum
import time
from collections import deque
from io import BytesIO
from datetime import datetime
import os
from pathlib import Path
import uuid
import re
import base64

# FastAPI imports
from fastapi import FastAPI, File, UploadFile, HTTPException, APIRouter
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# External libs
import PyPDF2
import google.generativeai as genai

# ==================== CONFIGURATION ====================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyDPCHgs42vHw477AHrWbw4ZQ-OEKCxgjvQ")
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# ==================== LOGGING SETUP ====================

def setup_logger(name: str = "DocumentPipeline") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    file_handler = logging.FileHandler(f'pipeline_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    file_handler.setLevel(logging.DEBUG)
    
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

logger = setup_logger()

# ==================== DATA MODELS ====================

class ContentType(Enum):
    TEXT = "text"
    TABLE = "table"
    CHART = "chart"
    FORM = "form"

class ChunkPosition(Enum):
    BEGINNING = "beginning"
    MIDDLE = "middle"
    END = "end"

class Priority(Enum):
    HIGH = 1
    MEDIUM = 2
    LOW = 3

@dataclass
class ChunkMetadata:
    chunk_id: str
    page_range: tuple
    section_id: str
    position: ChunkPosition
    prev_chunk: Optional[str]
    next_chunk: Optional[str]
    content_types: List[ContentType]

@dataclass
class DocumentChunk:
    metadata: ChunkMetadata
    content: bytes
    overlap_pages: List[int]
    priority: Priority

@dataclass
class MasterSchema:
    document_type: str
    major_sections: List[Dict[str, Any]]
    expected_structure: Dict[str, Any]
    extraction_rules: Dict[str, Any]

@dataclass
class ProcessingResult:
    chunk_id: str
    extracted_data: Dict[str, Any]
    summary: str
    confidence: float
    issues: List[str]

# ==================== DOCUMENT CHUNKER ====================

class DocumentChunker:
    def __init__(self, chunk_size: int = 3, overlap_pages: int = 0):
        self.chunk_size = min(chunk_size, 3)
        self.overlap_pages = overlap_pages
        logger.info(f"DocumentChunker initialized | Chunk size: {self.chunk_size}, Overlap: {overlap_pages}")

    async def create_chunks(self, document_path: str, total_pages: int, gemini_model) -> List[DocumentChunk]:
        logger.info(f"PHASE 1: Creating chunks | Total pages: {total_pages}")
        start_time = time.time()
        
        chunks = []
        chunk_idx = 0
        current_page = 0
        
        while current_page < total_pages:
            start = current_page
            end = min(current_page + self.chunk_size - 1, total_pages - 1)
            
            position = ChunkPosition.BEGINNING if chunk_idx == 0 else (
                ChunkPosition.END if end == total_pages - 1 else ChunkPosition.MIDDLE
            )

            metadata = ChunkMetadata(
                chunk_id=f"chunk_{chunk_idx}",
                page_range=(start, end),
                section_id=f"Section_{chunk_idx}",
                position=position,
                prev_chunk=f"chunk_{chunk_idx-1}" if chunk_idx > 0 else None,
                next_chunk=f"chunk_{chunk_idx+1}" if end < total_pages - 1 else None,
                content_types=[ContentType.TEXT]
            )

            overlap = []
            priority = Priority.HIGH
            chunk_bytes = self._extract_pages(document_path, start, end)

            chunk = DocumentChunk(
                metadata=metadata,
                content=chunk_bytes,
                overlap_pages=overlap,
                priority=priority
            )
            chunks.append(chunk)
            logger.info(f"Created chunk_{chunk_idx} | Pages: {start}-{end}")

            chunk_idx += 1
            current_page = end + 1

        elapsed = time.time() - start_time
        logger.info(f"PHASE 1 COMPLETE | {len(chunks)} chunks covering all {total_pages} pages in {elapsed:.2f}s")
        return chunks

    def _extract_pages(self, document_path: str, start: int, end: int) -> bytes:
        reader = PyPDF2.PdfReader(document_path)
        writer = PyPDF2.PdfWriter()
        total = len(reader.pages)
        s = max(0, start)
        e = min(end, total - 1)
        for p in range(s, e + 1):
            writer.add_page(reader.pages[p])
        out = BytesIO()
        writer.write(out)
        out.seek(0)
        return out.read()

# ==================== GEMINI PROVIDER ====================

class GeminiProvider:
    def __init__(self, api_key: str, model_name: str = "gemini-2.0-flash"):
        self.api_key = api_key
        self.model_name = model_name
        self.rpm_limit = 15
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        
        # Check if file upload is available
        self.has_file_api = hasattr(genai, 'upload_file')
        
        if self.has_file_api:
            logger.info(f"GeminiProvider initialized | Model: {model_name} | File API: Available (Vision Mode)")
        else:
            logger.warning(f"GeminiProvider initialized | Model: {model_name} | File API: Not available (Text-only Mode)")

    async def process_full_extraction(self, chunk: DocumentChunk, chunk_pdf_path: str) -> Dict:
        chunk_id = chunk.metadata.chunk_id
        page_range = chunk.metadata.page_range
        logger.info(f"Extracting {chunk_id} | Pages: {page_range[0]}-{page_range[1]} | Mode: {'Vision' if self.has_file_api else 'Text-only'}")
        
        try:
            # METHOD 1: Try using File API if available (BEST - includes visual detection)
            if self.has_file_api:
                try:
                    uploaded_file = genai.upload_file(chunk_pdf_path)
                    logger.info(f"{chunk_id}: Using File API (Vision Mode) - Full visual element detection")
                    result = await self._extract_with_file_api(uploaded_file, page_range, chunk_id)
                    
                    # Clean up uploaded file
                    try:
                        genai.delete_file(uploaded_file.name)
                    except:
                        pass
                    
                    return result
                except Exception as e:
                    logger.warning(f"File API failed for {chunk_id}: {e}. Falling back to text extraction")
            
            # METHOD 2: Fallback - Extract text from PDF and process (LIMITED visual detection)
            logger.warning(f"{chunk_id}: Using Text-only Mode - Visual elements may be incomplete")
            return await self._extract_with_text(chunk_pdf_path, page_range, chunk_id)
            
        except Exception as e:
            logger.error(f"Extraction error for {chunk_id}: {e}", exc_info=True)
            return {"pages": [], "confidence": 0.0, "issues": [str(e)]}

    async def _extract_with_file_api(self, uploaded_file, page_range: tuple, chunk_id: str) -> Dict:
        """Extract using Gemini File API with FULL visual element detection"""
        prompt = self._create_extraction_prompt(page_range)
        
        start_time = time.time()
        response = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.model.generate_content([uploaded_file, prompt])
        )
        
        elapsed = time.time() - start_time
        txt = response.text.strip()
        logger.debug(f"Vision extraction in {elapsed:.2f}s | Length: {len(txt)} chars")
        
        return self._parse_extraction_response(txt, page_range, chunk_id)

    async def _extract_with_text(self, chunk_pdf_path: str, page_range: tuple, chunk_id: str) -> Dict:
        """Extract using text-based approach (fallback method) - LIMITED visual detection"""
        try:
            # Extract text from PDF pages
            reader = PyPDF2.PdfReader(chunk_pdf_path)
            extracted_text = []
            
            for i, page in enumerate(reader.pages):
                page_num = page_range[0] + i
                text = page.extract_text()
                
                # Try to detect tables from text patterns
                has_table_indicators = any([
                    '\t' in text,  # Tab-separated data
                    '|' in text,   # Pipe-separated data
                    re.search(r'\n\s*[\w\s]+\s+[\w\s]+\s+[\w\s]+\s*\n', text),  # Multiple columns
                ])
                
                extracted_text.append({
                    "page_number": page_num,
                    "text": text,
                    "has_table_indicators": has_table_indicators
                })
            
            # Create enhanced prompt
            prompt = f"""⚠️ TEXT-ONLY MODE: Analyze this text-extracted document content.

PAGES: {page_range[0]} to {page_range[1]}

TEXT CONTENT:
{json.dumps(extracted_text, indent=2)[:15000]}

🔍 EXTRACTION REQUIREMENTS:

1. **TEXT EXTRACTION**
   - Extract all text content
   - Identify section headings and clauses
   - Preserve structure and numbering

2. **TABLE DETECTION** (from text patterns)
   - Look for aligned columns in text
   - Look for tab-separated or pipe-separated data
   - Look for repeated row patterns
   - Extract as structured table if found
   
3. **VISUAL ELEMENT INDICATORS**
   - Look for text like "Figure 1:", "Table 1:", "Chart:", "Diagram:"
   - Look for "[IMAGE]", "[LOGO]", "[CHART]" placeholders
   - Note any references to visual elements

EXPECTED OUTPUT:

{{
  "pages": [
    {{
      "page_number": {page_range[0]},
      "has_tables": true/false,
      "has_visual_references": true/false,
      "sections": [
        {{
          "section_name": "Section heading",
          "clauses": [
            {{
              "clause_id": "1.1",
              "content": "Full text",
              "tables": [
                {{
                  "table_id": "T1",
                  "table_title": "Title if found",
                  "headers": ["Col1", "Col2"],
                  "rows": [["val1", "val2"]],
                  "note": "Extracted from text alignment"
                }}
              ],
              "visual_references": [
                {{
                  "type": "image/chart/table",
                  "reference": "See Figure 1 or Table 1",
                  "note": "Visual element mentioned but not visible in text"
                }}
              ]
            }}
          ]
        }}
      ]
    }}
  ],
  "confidence": 0.70,
  "issues": ["Text-only extraction - visual elements may not be fully captured"]
}}

⚠️ NOTE: This is text-only mode. Extract what you can see in the text, but note that images, charts, and visual tables may not be fully captured.

Return ONLY valid JSON."""

            start_time = time.time()
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.model.generate_content(prompt)
            )
            
            elapsed = time.time() - start_time
            txt = response.text.strip()
            logger.debug(f"Text-based extraction in {elapsed:.2f}s | Length: {len(txt)} chars")
            logger.warning(f"{chunk_id}: Using text-only mode - visual elements may be incomplete")
            
            return self._parse_extraction_response(txt, page_range, chunk_id)
            
        except Exception as e:
            logger.error(f"Text extraction error for {chunk_id}: {e}", exc_info=True)
            return {"pages": [], "confidence": 0.0, "issues": [f"Text extraction failed: {str(e)}"]}

    def _create_extraction_prompt(self, page_range: tuple) -> str:
        """Create comprehensive extraction prompt with FULL visual element detection"""
        return f"""🔍 COMPREHENSIVE DOCUMENT ANALYSIS - Pages {page_range[0]} to {page_range[1]}

CRITICAL MISSION:
You are analyzing a PDF document with FULL VISUAL ACCESS. You must extract EVERYTHING - text, tables, images, charts, barcodes, and all visual elements with complete accuracy.

📋 EXTRACTION REQUIREMENTS:

1️⃣ **TEXT EXTRACTION**
   - Extract EVERY word on each page
   - Preserve formatting (bold, italic, underline)
   - Identify headings vs body text
   - Capture numbered/bulleted lists
   - Note any handwritten text or annotations

2️⃣ **TABLE DETECTION & EXTRACTION** ⚠️ CRITICAL
   - Scan EVERY page for tables (even small ones)
   - For EACH table found:
     * Identify table boundaries
     * Extract column headers (first row typically)
     * Extract ALL rows with exact values
     * Note merged cells or special formatting
     * Preserve data types (numbers, dates, text)
     * Include table title/caption if present
   - Common table indicators: grid lines, aligned columns, repeated headers

3️⃣ **IMAGE ANALYSIS**
   - Detect ALL images, photos, diagrams, illustrations
   - For each image provide:
     * Detailed description (what's shown)
     * Image type (photo, diagram, illustration, logo, icon)
     * Location on page (top, middle, bottom, left, right)
     * Size indication (small, medium, large, full-page)
     * Any text within the image (OCR)
     * Color scheme (color, grayscale, black & white)

4️⃣ **CHART & GRAPH DETECTION**
   - Identify: pie charts, bar charts, line graphs, scatter plots, flowcharts
   - For each chart extract:
     * Chart type
     * Title/label
     * Axis labels (x and y)
     * Data values/legend
     * Key insights visible in the chart
     * Colors used

5️⃣ **SPECIAL VISUAL ELEMENTS**
   - Barcodes (1D barcodes, UPC, EAN)
   - QR codes
   - Stamps or seals (official stamps, company seals)
   - Signatures (handwritten or digital)
   - Watermarks
   - Logos and branding elements
   - Vector graphics or icons

6️⃣ **STRUCTURAL ELEMENTS**
   - Headers and footers
   - Page numbers
   - Margin notes or sidebars
   - Text boxes or callouts
   - Colored backgrounds or highlighting

🎯 OUTPUT FORMAT (JSON):

{{
  "pages": [
    {{
      "page_number": {page_range[0]},
      "has_tables": true,
      "has_images": true,
      "has_charts": false,
      "sections": [
        {{
          "section_name": "Exact heading from document",
          "section_type": "heading/body/footer",
          "clauses": [
            {{
              "clause_id": "1.1",
              "content": "COMPLETE clause text with all formatting",
              "formatting": ["bold", "italic"],
              "sub_clauses": [
                {{
                  "clause_id": "1.1.1",
                  "content": "Complete sub-clause"
                }}
              ],
              "tables": [
                {{
                  "table_id": "T1",
                  "table_title": "Revenue Breakdown 2024",
                  "position": "middle-center",
                  "size": "medium",
                  "headers": ["Quarter", "Revenue ($M)", "Growth %", "Notes"],
                  "rows": [
                    ["Q1 2024", "125.5", "15.2", "Strong performance"],
                    ["Q2 2024", "142.8", "13.8", "New contracts"],
                    ["Q3 2024", "156.2", "9.4", "Steady growth"]
                  ],
                  "total_rows": 3,
                  "total_columns": 4,
                  "has_merged_cells": false,
                  "summary": "Quarterly revenue showing consistent growth"
                }}
              ],
              "images": [
                {{
                  "image_id": "IMG1",
                  "type": "logo",
                  "description": "Company logo showing blue eagle with text 'Acme Corp'",
                  "position": "top-left",
                  "size": "small",
                  "contains_text": "Acme Corp",
                  "color_scheme": "color",
                  "purpose": "branding"
                }},
                {{
                  "image_id": "IMG2",
                  "type": "diagram",
                  "description": "Network topology diagram showing servers, routers, and connections. Shows main server connected to 3 regional nodes with bidirectional arrows. Labels indicate bandwidth: 10Gbps main link, 1Gbps regional links.",
                  "position": "middle-center",
                  "size": "large",
                  "contains_text": "Main Server, Node A, Node B, Node C, 10Gbps, 1Gbps",
                  "color_scheme": "color",
                  "purpose": "technical illustration"
                }}
              ],
              "charts": [
                {{
                  "chart_id": "CHART1",
                  "type": "pie_chart",
                  "title": "Market Share Distribution",
                  "description": "Pie chart showing market share: Company A 45%, Company B 30%, Company C 15%, Others 10%. Colors: blue, green, red, gray.",
                  "position": "bottom-right",
                  "data_values": {{"Company A": "45%", "Company B": "30%", "Company C": "15%", "Others": "10%"}},
                  "legend": ["Company A", "Company B", "Company C", "Others"],
                  "insights": "Company A leads with 45% market share"
                }}
              ],
              "special_elements": [
                {{
                  "element_id": "BARCODE1",
                  "type": "barcode",
                  "format": "QR_code",
                  "content": "URL or encoded data if readable",
                  "position": "bottom-right",
                  "purpose": "document tracking"
                }},
                {{
                  "element_id": "STAMP1",
                  "type": "stamp",
                  "description": "Circular red stamp reading 'APPROVED' with date '2024-10-15'",
                  "position": "bottom-center",
                  "color": "red"
                }}
              ]
            }}
          ]
        }}
      ],
      "header": "Text in header area",
      "footer": "Page 1 of 25 | Document ID: ABC123"
    }}
  ],
  "confidence": 0.95,
  "issues": []
}}

⚡ CRITICAL REMINDERS:
- Create a page object for EVERY page from {page_range[0]} to {page_range[1]}
- If you see a TABLE (rows and columns), you MUST extract it with headers and ALL rows
- If you see an IMAGE, you MUST describe it in detail
- If you see a CHART, you MUST identify its type and extract data
- Do NOT skip any visual elements
- Do NOT truncate table data
- Do NOT summarize - extract COMPLETE content

Include page objects for pages: {', '.join(str(p) for p in range(page_range[0], page_range[1] + 1))}"""

    def _parse_extraction_response(self, response_text: str, page_range: tuple, chunk_id: str) -> Dict:
        """Parse and validate extraction response"""
        try:
            # Extract JSON from response
            start = response_text.find("{")
            end = response_text.rfind("}") + 1
            
            if start == -1 or end <= start:
                logger.error(f"No JSON found in response for {chunk_id}")
                return {"pages": [], "confidence": 0.0, "issues": ["No JSON in response"]}
            
            json_str = response_text[start:end]
            parsed = json.loads(json_str)
            
            # Validate all pages present
            expected_pages = list(range(page_range[0], page_range[1] + 1))
            extracted_pages = [p.get("page_number") for p in parsed.get("pages", [])]
            missing_pages = set(expected_pages) - set(extracted_pages)
            
            if missing_pages:
                logger.warning(f"{chunk_id}: Missing pages {missing_pages}")
                parsed.setdefault("issues", []).append(f"Missing pages: {sorted(missing_pages)}")
            
            result = {
                "pages": parsed.get("pages", []),
                "confidence": float(parsed.get("confidence", 0.85)),
                "issues": parsed.get("issues", [])
            }
            
            # Log visual element counts
            total_tables = sum(
                len(clause.get("tables", [])) 
                for page in result["pages"] 
                for section in page.get("sections", []) 
                for clause in section.get("clauses", [])
            )
            total_images = sum(
                len(clause.get("images", [])) 
                for page in result["pages"] 
                for section in page.get("sections", []) 
                for clause in section.get("clauses", [])
            )
            total_charts = sum(
                len(clause.get("charts", [])) 
                for page in result["pages"] 
                for section in page.get("sections", []) 
                for clause in section.get("clauses", [])
            )
            
            logger.info(f"{chunk_id}: Extracted {len(result['pages'])} pages | "
                       f"Tables: {total_tables} | Images: {total_images} | Charts: {total_charts}")
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error for {chunk_id}: {e}")
            return {"pages": [], "confidence": 0.0, "issues": [f"JSON parse error: {str(e)}"]}
        except Exception as e:
            logger.error(f"Parse error for {chunk_id}: {e}", exc_info=True)
            return {"pages": [], "confidence": 0.0, "issues": [f"Parse error: {str(e)}"]}

    async def extract_parties_from_full_text(self, all_text: str) -> List[str]:
        """Extract party names from document text"""
        try:
            prompt = f"""Extract the party names from this document excerpt.

Look for:
- "This Agreement is made between [Party A] and [Party B]"
- "PARTIES: [Party A] and [Party B]"
- Company names in headers/footers
- Signature blocks

Document text:
{all_text[:5000]}

Return ONLY JSON:
{{
  "parties": ["Exact Party Name 1", "Exact Party Name 2"]
}}"""

            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.model.generate_content(prompt)
            )
            
            txt = response.text.strip()
            start = txt.find("{")
            end = txt.rfind("}") + 1
            if start != -1 and end > start:
                parsed = json.loads(txt[start:end])
                parties = parsed.get("parties", [])
                if parties and len(parties) > 0:
                    logger.info(f"Detected parties: {parties}")
                    return parties[:2]
        except Exception as e:
            logger.error(f"Party detection error: {e}")
        
        return ["Party A", "Party B"]

    async def extract_summary_data(self, all_pages: List[Dict], parties: List[str]) -> Dict:
        logger.info("Extracting summary data")
        
        try:
            # Compile comprehensive text
            all_text = []
            for page in all_pages:
                for section in page.get("sections", []):
                    section_name = section.get("section_name", "")
                    all_text.append(f"## {section_name}")
                    
                    for clause in section.get("clauses", []):
                        clause_text = f"{clause.get('clause_id', '')}: {clause.get('content', '')}"
                        all_text.append(clause_text)
                        
                        for sub in clause.get("sub_clauses", []):
                            sub_text = f"  {sub.get('clause_id', '')}: {sub.get('content', '')}"
                            all_text.append(sub_text)
            
            combined_text = "\n".join(all_text[:100])
            
            prompt = f"""Analyze this document and extract obligations, deadlines, and alerts.

**PARTIES:** {', '.join(parties)}

**DOCUMENT CONTENT:**
{combined_text[:10000]}

Extract:

1. **OBLIGATIONS** - What each party must do
2. **DEADLINES** - Time-based requirements
3. **ALERTS** - Important conditions or triggers
4. **BUYER/SELLER** - Identify relationship

Return JSON:
{{
  "obligations": [
    {{
      "party": "Party name",
      "type": "Category",
      "description": "Detailed description"
    }}
  ],
  "deadlines": [
    {{
      "description": "What must happen",
      "time_period": "Timeframe",
      "trigger_event": "Trigger"
    }}
  ],
  "alerts": [
    {{
      "type": "Alert type",
      "trigger": "Trigger",
      "responsible_party": "Who acts"
    }}
  ],
  "buyer": "Buyer name or null",
  "seller": "Seller name or null"
}}"""
            
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.model.generate_content(prompt)
            )
            
            txt = response.text.strip()
            start = txt.find("{")
            end = txt.rfind("}") + 1
            if start != -1 and end > start:
                summary = json.loads(txt[start:end])
            else:
                summary = {}
            
            summary.setdefault("obligations", [])
            summary.setdefault("deadlines", [])
            summary.setdefault("alerts", [])
            summary.setdefault("buyer", None)
            summary.setdefault("seller", None)
            
            logger.info(f"Summary: {len(summary['obligations'])} obligations, "
                       f"{len(summary['deadlines'])} deadlines, "
                       f"{len(summary['alerts'])} alerts")
            return summary
            
        except Exception as e:
            logger.error(f"Summary extraction error: {e}", exc_info=True)
            return {
                "obligations": [],
                "deadlines": [],
                "alerts": [],
                "buyer": None,
                "seller": None
            }

# ==================== PROCESSING COMPONENTS ====================

class HierarchicalProcessor:
    def __init__(self, gemini_provider: GeminiProvider):
        self.gemini = gemini_provider
        self.master_schema: Optional[MasterSchema] = None

    async def create_master_schema(self, document_path: str, total_pages: int) -> MasterSchema:
        logger.info("Creating master schema")
        
        try:
            reader = PyPDF2.PdfReader(document_path)
            sample_text = ""
            
            for i in range(min(3, total_pages)):
                sample_text += reader.pages[i].extract_text()
            
            prompt = f"""Analyze this document and identify:
1. Document type (contract, agreement, SLA, exhibit, etc.)
2. Main purpose

Return JSON:
{{
  "document_type": "Service Level Agreement",
  "purpose": "Brief description"
}}

Text: {sample_text[:3000]}"""
            
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.gemini.model.generate_content(prompt)
            )
            
            txt = response.text.strip()
            start = txt.find("{")
            end = txt.rfind("}") + 1
            if start != -1 and end > start:
                parsed = json.loads(txt[start:end])
            else:
                parsed = {}
            
            self.master_schema = MasterSchema(
                document_type=parsed.get('document_type', 'Document'),
                major_sections=[],
                expected_structure={},
                extraction_rules={}
            )
            
            logger.info(f"Master schema: {self.master_schema.document_type}")
            
        except Exception as e:
            logger.error(f"Schema creation error: {e}", exc_info=True)
            self.master_schema = MasterSchema(
                document_type="Document",
                major_sections=[],
                expected_structure={},
                extraction_rules={}
            )
        
        return self.master_schema

    async def process_chunk(self, chunk: DocumentChunk, chunk_pdf_path: str) -> ProcessingResult:
        result = await self.gemini.process_full_extraction(chunk, chunk_pdf_path)
        
        pages = result.get("pages", [])
        summary = f"Pages {chunk.metadata.page_range[0]}-{chunk.metadata.page_range[1]}: {len(pages)} pages extracted"
        
        return ProcessingResult(
            chunk_id=chunk.metadata.chunk_id,
            extracted_data=pages,
            summary=summary,
            confidence=result.get("confidence", 0.85),
            issues=result.get("issues", [])
        )

class RateLimiter:
    def __init__(self, rpm_limit: int = 15):
        self.rpm_limit = rpm_limit
        self.request_times = deque()
        self.lock = asyncio.Lock()

    async def acquire(self):
        async with self.lock:
            now = time.time()
            while self.request_times and now - self.request_times[0] > 60:
                self.request_times.popleft()
            if len(self.request_times) >= self.rpm_limit:
                sleep_time = 60 - (now - self.request_times[0]) + 1
                if sleep_time > 0:
                    logger.warning(f"Rate limit, sleeping {sleep_time:.2f}s")
                    await asyncio.sleep(sleep_time)
                    return await self.acquire()
            self.request_times.append(now)

class SequentialProcessor:
    """Sequential processor that handles chunks one by one"""
    def __init__(self, gemini_provider: GeminiProvider):
        self.gemini = gemini_provider
        self.rate_limiter = RateLimiter(gemini_provider.rpm_limit)
        self.results: Dict[str, ProcessingResult] = {}

    async def process_documents(self, chunks: List[DocumentChunk], document_path: str, total_pages: int) -> Dict[str, ProcessingResult]:
        logger.info(f"PHASE 2: Processing {len(chunks)} chunks sequentially")
        start_time = time.time()
        
        processor = HierarchicalProcessor(self.gemini)
        await processor.create_master_schema(document_path, total_pages)
        
        # Process each chunk sequentially
        for idx, chunk in enumerate(chunks, 1):
            # Save chunk as temporary PDF
            chunk_path = UPLOAD_DIR / f"temp_{chunk.metadata.chunk_id}.pdf"
            with open(chunk_path, "wb") as f:
                f.write(chunk.content)
            
            # Apply rate limiting
            await self.rate_limiter.acquire()
            
            logger.info(f"Processing chunk {idx}/{len(chunks)}: {chunk.metadata.chunk_id}")
            
            # Process the chunk
            result = await processor.process_chunk(chunk, str(chunk_path))
            self.results[chunk.metadata.chunk_id] = result
            
            logger.info(f"Completed {chunk.metadata.chunk_id}: {result.summary}")
            
            # Clean up temp file
            try:
                os.remove(chunk_path)
            except Exception as e:
                logger.warning(f"Failed to remove temp file {chunk_path}: {e}")
        
        elapsed = time.time() - start_time
        logger.info(f"PHASE 2 COMPLETE | {len(self.results)} results in {elapsed:.2f}s")
        return self.results

class JSONMerger:
    def __init__(self, master_schema: MasterSchema):
        self.master_schema = master_schema

    def merge(self, results: Dict[str, ProcessingResult]) -> List[Dict]:
        logger.info(f"PHASE 3: Merging {len(results)} chunks")
        
        sorted_results = sorted(
            results.values(),
            key=lambda r: int(r.chunk_id.split("_")[1]) if "_" in r.chunk_id else 0
        )
        
        all_pages = []
        for result in sorted_results:
            pages = result.extracted_data
            if isinstance(pages, list):
                all_pages.extend(pages)
        
        # Better duplicate removal
        seen_pages = {}
        for page in all_pages:
            page_num = page.get("page_number", -1)
            if page_num not in seen_pages:
                seen_pages[page_num] = page
            else:
                # Merge sections if same page appears twice
                existing_sections = seen_pages[page_num].get("sections", [])
                new_sections = page.get("sections", [])
                existing_sections.extend(new_sections)
        
        unique_pages = list(seen_pages.values())
        unique_pages.sort(key=lambda p: p.get("page_number", 0))
        
        logger.info(f"PHASE 3 COMPLETE | {len(unique_pages)} unique pages")
        return unique_pages

class DocumentProcessingPipeline:
    def __init__(self, gemini_provider: GeminiProvider, chunk_size: int = 3, overlap_pages: int = 0):
        self.chunker = DocumentChunker(chunk_size=chunk_size, overlap_pages=overlap_pages)
        self.processor = SequentialProcessor(gemini_provider)
        self.gemini = gemini_provider
        logger.info("DocumentProcessingPipeline initialized (Sequential Mode with Visual Detection)")

    async def process(self, document_path: str, document_name: str, total_pages: int) -> Dict[str, Any]:
        pipeline_start = time.time()
        logger.info(f"Starting pipeline | Document: {document_name} | Pages: {total_pages}")
        
        # Phase 1: Chunking
        chunks = await self.chunker.create_chunks(document_path, total_pages, self.gemini.model)
        logger.info(f"Created {len(chunks)} chunks")
        
        # Phase 2: Sequential Processing with Visual Detection
        results = await self.processor.process_documents(chunks, document_path, total_pages)
        logger.info(f"Processed {len(results)} chunks")
        
        # Phase 3: Merging
        processor = HierarchicalProcessor(self.gemini)
        master_schema = await processor.create_master_schema(document_path, total_pages)
        merger = JSONMerger(master_schema)
        all_pages = merger.merge(results)
        
        logger.info(f"Merged into {len(all_pages)} pages")
        
        # Extract full text for party detection
        full_text = ""
        for page in all_pages:
            for section in page.get("sections", []):
                for clause in section.get("clauses", []):
                    full_text += str(clause.get("content") or "") + " "
        
        # Detect parties from full text
        parties = await self.gemini.extract_parties_from_full_text(full_text[:8000])
        
        # Extract summary
        summary_data = await self.gemini.extract_summary_data(all_pages, parties)
        
        total_obligations = len(summary_data.get("obligations", []))
        total_deadlines = len(summary_data.get("deadlines", []))
        total_alerts = len(summary_data.get("alerts", []))
        
        pipeline_elapsed = time.time() - pipeline_start
        
        # Build final output
        final_output = {
            "metadata": {
                "document_name": document_name,
                "document_type": master_schema.document_type,
                "uploaded_on": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
                "total_pages": total_pages,
                "pages_extracted": len(all_pages),
                "processing_time_seconds": round(pipeline_elapsed, 2),
                "processing_mode": "sequential_with_vision",
                "vision_mode_enabled": self.gemini.has_file_api,
                "language": "English",
                "total_obligations": total_obligations,
                "total_deadlines": total_deadlines,
                "total_alerts": total_alerts,
                "detected_parties": parties
            },
            "pages": all_pages,
            "overall_summary": summary_data
        }
        
        logger.info(f"PIPELINE COMPLETE | Extracted {len(all_pages)}/{total_pages} pages in {pipeline_elapsed:.2f}s")
        return final_output

# ==================== FASTAPI APPLICATION ====================

app = FastAPI(
    title="Document Extraction API v2.4",
    description="AI-powered document extraction with comprehensive visual element detection",
    version="2.4.0"
)
router = APIRouter()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@router.on_event("startup")
async def startup_event():
    logger.info("=" * 80)
    logger.info("🚀 Starting Document Extraction API v2.4")
    logger.info("=" * 80)
    logger.info("✨ NEW FEATURES:")
    logger.info("  📊 Full table extraction with headers and rows")
    logger.info("  🖼️  Complete image analysis and descriptions")
    logger.info("  📈 Chart detection (pie, bar, line, scatter)")
    logger.info("  🔲 Barcode and QR code identification")
    logger.info("  ✅ Stamp, signature, and logo detection")
    logger.info("=" * 80)
    logger.info(f"🔑 Gemini API Key: {GEMINI_API_KEY[:20]}...")
    
    # Test vision capabilities
    test_provider = GeminiProvider(GEMINI_API_KEY)
    if test_provider.has_file_api:
        logger.info("✅ Vision Mode: ENABLED (Full visual element detection)")
    else:
        logger.warning("⚠️  Vision Mode: DISABLED (Text-only mode - limited visual detection)")

@router.get("/")
async def root():
    return {
        "message": "Document Extraction API with Visual Element Detection",
        "version": "2.4.0",
        "endpoint": "/extract",
        "processing_mode": "sequential_with_vision",
        "features": [
            "Complete text extraction",
            "Table detection and structure extraction (headers + rows)",
            "Image analysis and description",
            "Chart detection (pie, bar, line, scatter)",
            "Barcode and QR code detection",
            "Stamp, signature, and logo identification",
            "Sequential processing for accuracy",
            "Party detection and obligation extraction"
        ],
        "visual_elements_supported": {
            "tables": "Headers, rows, columns, merged cells",
            "images": "Logos, diagrams, photos with descriptions",
            "charts": "Pie, bar, line, scatter plots with data",
            "special": "Barcodes, QR codes, stamps, signatures, watermarks"
        }
    }

@router.get("/health")
async def health_check():
    """Health check endpoint with vision capability detection"""
    try:
        # Test Gemini API connection
        test_provider = GeminiProvider(GEMINI_API_KEY)
        
        return {
            "status": "healthy",
            "gemini_api": "connected",
            "file_api_available": test_provider.has_file_api,
            "vision_mode": "enabled" if test_provider.has_file_api else "disabled",
            "processing_mode": "sequential_with_vision",
            "capabilities": {
                "text_extraction": True,
                "table_detection": True,
                "image_analysis": test_provider.has_file_api,
                "chart_detection": test_provider.has_file_api,
                "barcode_detection": test_provider.has_file_api,
                "full_visual_detection": test_provider.has_file_api
            },
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

@router.post("/extract_thulasi")
async def extract_document(
    file: UploadFile = File(...),
    chunk_size: int = 3,
    overlap_pages: int = 0
):
    """
    🚀 Extract complete document structure with FULL visual element detection
    
    **Parameters:**
    - **file**: PDF file to process
    - **chunk_size**: Pages per chunk (2-3 recommended for visual accuracy, max 5)
    - **overlap_pages**: Overlap between chunks (0-1)
    
    **Returns:** Complete structured JSON with:
    - ✅ ALL text content (no truncation)
    - 📊 Tables with headers and rows
    - 🖼️ Images with detailed descriptions
    - 📈 Charts with data extraction
    - 🔲 Barcodes, QR codes, stamps
    - 📝 Obligations, deadlines, and alerts
    
    **Visual Elements Detected:**
    - **Tables**: Complete structure with headers, rows, columns
    - **Images**: Logos, diagrams, photos with descriptions
    - **Charts**: Pie, bar, line, scatter plots with data values
    - **Barcodes**: QR codes, 1D barcodes, UPC codes
    - **Special**: Stamps, signatures, watermarks, seals
    
    **Note:** 
    - Vision Mode: Requires Gemini File API for full visual detection
    - Text-only Mode: Limited visual element detection (fallback)
    """
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    document_uuid = None
    file_path = None
    
    try:
        # Generate UUID for document
        document_uuid = str(uuid.uuid4())
        
        # Save uploaded file
        file_path = UPLOAD_DIR / f"{document_uuid}_{file.filename}"
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)
        
        logger.info(f"File uploaded: {file.filename} | UUID: {document_uuid}")
        
        # Get total pages
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            total_pages = len(reader.pages)
        
        logger.info(f"Document has {total_pages} pages")
        
        # Validate parameters
        if chunk_size < 1 or chunk_size > 10:
            raise HTTPException(status_code=400, detail="chunk_size must be between 1 and 10")
        
        if overlap_pages < 0 or overlap_pages > 2:
            raise HTTPException(status_code=400, detail="overlap_pages must be between 0 and 2")
        
        # Cap chunk size at 3 for complete extraction
        effective_chunk_size = min(chunk_size, 3)
        
        # Initialize Gemini provider
        gemini_provider = GeminiProvider(GEMINI_API_KEY)
        
        # Log processing mode
        if gemini_provider.has_file_api:
            logger.info("🎯 Processing Mode: VISION ENABLED - Full visual element detection")
        else:
            logger.warning("⚠️  Processing Mode: TEXT-ONLY - Limited visual detection")
        
        # Initialize pipeline with sequential processing
        pipeline = DocumentProcessingPipeline(
            gemini_provider=gemini_provider,
            chunk_size=effective_chunk_size,
            overlap_pages=overlap_pages
        )
        
        # Process document
        logger.info(f"Starting extraction for {file.filename}")
        result = await pipeline.process(str(file_path), file.filename, total_pages)
        
        # Add UUID to metadata
        result["metadata"]["document_uuid"] = document_uuid
        result["metadata"]["file_path"] = str(file_path)
        
        # Validation check
        pages_extracted = len(result["pages"])
        if pages_extracted < total_pages:
            logger.warning(f"Only extracted {pages_extracted}/{total_pages} pages - some pages may be missing")
            result["metadata"]["extraction_warning"] = f"Extracted {pages_extracted}/{total_pages} pages"
            result["metadata"]["extraction_status"] = "incomplete"
        else:
            result["metadata"]["extraction_status"] = "complete"
        
        # Add extraction statistics
        result["metadata"]["extraction_stats"] = {
            "total_sections": sum(len(page.get("sections", [])) for page in result["pages"]),
            "total_clauses": sum(
                len(section.get("clauses", [])) 
                for page in result["pages"] 
                for section in page.get("sections", [])
            ),
            "total_tables": sum(
                len(clause.get("tables", [])) 
                for page in result["pages"] 
                for section in page.get("sections", []) 
                for clause in section.get("clauses", [])
            ),
            "total_images": sum(
                len(clause.get("images", [])) 
                for page in result["pages"] 
                for section in page.get("sections", []) 
                for clause in section.get("clauses", [])
            ),
            "total_charts": sum(
                len(clause.get("charts", [])) 
                for page in result["pages"] 
                for section in page.get("sections", []) 
                for clause in section.get("clauses", [])
            ),
            "total_special_elements": sum(
                len(clause.get("special_elements", [])) 
                for page in result["pages"] 
                for section in page.get("sections", []) 
                for clause in section.get("clauses", [])
            ),
            "pages_with_tables": sum(
                1 for page in result["pages"] if page.get("has_tables", False)
            ),
            "pages_with_images": sum(
                1 for page in result["pages"] if page.get("has_images", False)
            ),
            "pages_with_charts": sum(
                1 for page in result["pages"] if page.get("has_charts", False)
            )
        }
        
        logger.info(f"Processing complete | UUID: {document_uuid} | Pages: {pages_extracted}/{total_pages}")
        logger.info(f"📊 Visual Stats: Tables: {result['metadata']['extraction_stats']['total_tables']} | "
                   f"Images: {result['metadata']['extraction_stats']['total_images']} | "
                   f"Charts: {result['metadata']['extraction_stats']['total_charts']}")
        
        # Return JSON response
        return JSONResponse(
            content=result,
            status_code=200,
            headers={
                "X-Document-UUID": document_uuid,
                "X-Pages-Extracted": str(pages_extracted),
                "X-Total-Pages": str(total_pages),
                "X-Processing-Mode": "sequential_with_vision",
                "X-Vision-Enabled": str(gemini_provider.has_file_api),
                "X-Tables-Found": str(result['metadata']['extraction_stats']['total_tables']),
                "X-Images-Found": str(result['metadata']['extraction_stats']['total_images']),
                "X-Charts-Found": str(result['metadata']['extraction_stats']['total_charts'])
            }
        )
        
    except HTTPException:
        raise
        
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail=f"Error parsing extraction results: {str(e)}"
        )
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}", exc_info=True)
        raise HTTPException(
            status_code=404, 
            detail=f"File not found: {str(e)}"
        )
        
    except Exception as e:
        logger.error(f"Processing error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail=f"Internal server error: {str(e)}"
        )

app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    logger.info("=" * 80)
    logger.info("🚀 Starting Document Extraction API v2.4 with Visual Detection")
    logger.info("=" * 80)
    uvicorn.run(app, host="0.0.0.0", port=8000)