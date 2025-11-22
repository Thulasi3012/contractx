# table_extract.py
"""
Table extraction FastAPI app (Table-Transformer detection + Gemini OCR/Structure)
 - Table detection: microsoft/table-transformer-detection
 - OCR + structure extraction: gemini-2.5-flash
 - Output includes complete table structure with merged cells handling
 - Saves cropped table images to extracted_tables/
 - Requires: .env with GEMINI_API_KEY
"""
import os
import json
import base64
import math
import traceback
from io import BytesIO
from typing import List, Dict, Any, Optional, Tuple

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pdf2image import convert_from_bytes
from PIL import Image
import torch
import requests

from transformers import AutoImageProcessor, TableTransformerForObjectDetection

# -------------------------
# CONFIG
# -------------------------
OUTPUT_DIR = "extracted_tables"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TABLE_DET_MODEL = "microsoft/table-transformer-detection"
GEMINI_MODEL = "gemini-2.5-flash"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
if not GEMINI_API_KEY:
    print("⚠ WARNING: GEMINI_API_KEY missing — Gemini calls will fail.")

GEMINI_ENDPOINT = (
    f"https://generativelanguage.googleapis.com/v1beta/models/"
    f"{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------
# MODEL LOADING
# -------------------------
models_status = {
    "table_detection": False,
    "gemini_api_key_present": bool(GEMINI_API_KEY),
    "errors": []
}

_det_processor = None
_det_model = None


def try_load_models():
    global _det_processor, _det_model
    try:
        print("Loading table detection model:", TABLE_DET_MODEL)
        _det_processor = AutoImageProcessor.from_pretrained(TABLE_DET_MODEL)
        _det_model = TableTransformerForObjectDetection.from_pretrained(TABLE_DET_MODEL)
        _det_model.to(DEVICE)
        models_status["table_detection"] = True
        print("Loaded table detection model.")
    except Exception as e:
        models_status["errors"].append(f"table_detection_load_error:{str(e)}")
        print("Failed to load table detection model:", e)


try_load_models()

# -------------------------
# UTILITIES
# -------------------------
def int_box(bbox: List[float]) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    return (
        max(0, int(math.floor(x1))),
        max(0, int(math.floor(y1))),
        max(0, int(math.ceil(x2))),
        max(0, int(math.ceil(y2))),
    )


def pil_image_to_base64(img: Image.Image, fmt: str = "PNG") -> str:
    buf = BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def safe_json_loads(s: str) -> Optional[Any]:
    try:
        return json.loads(s)
    except Exception:
        start = s.find("{")
        end = s.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(s[start:end + 1])
            except Exception:
                pass
        return None


# -------------------------
# TABLE DETECTION
# -------------------------
def detect_tables(page_image: Image.Image, confidence_threshold: float = 0.95) -> List[Dict[str, Any]]:
    """
    Detect table bounding boxes on a PIL page image using the Table Transformer detection model.
    Returns list of {"bbox": [x1,y1,x2,y2], "confidence": float}
    """
    if not models_status["table_detection"]:
        return []

    inputs = _det_processor(images=page_image, return_tensors="pt")
    inputs = {k: v.to(_det_model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = _det_model(**inputs)

    target_sizes = torch.tensor([page_image.size[::-1]]).to(_det_model.device)
    results = _det_processor.post_process_object_detection(
        outputs, threshold=confidence_threshold, target_sizes=target_sizes
    )[0]

    tables = []
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        # label 0 corresponds to table class in the microsoft model
        if int(label.item()) == 0:
            tables.append({"bbox": [float(x) for x in box.tolist()], "confidence": float(score.item())})

    # sort top-to-bottom
    tables.sort(key=lambda t: t["bbox"][1])
    return tables


# -------------------------
# GEMINI TABLE PARSE
# -------------------------
def gemini_extract_table_from_image(img: Image.Image, page_num: int, table_idx: int, timeout: int = 90) -> Optional[Dict[str, Any]]:
    """
    Send a cropped table image to Gemini and return parsed JSON following the enhanced schema.
    """
    if not GEMINI_API_KEY:
        models_status["errors"].append("gemini_api_key_missing")
        return None

    try:
        b64 = pil_image_to_base64(img)

        instruction_text = """📊 SECTION 2: TABLE EXTRACTION (COMPLETE STRUCTURE)
═══════════════════════════════════════════════════════════════════════════════
DETECTION STRATEGY:
🔍 Scan ENTIRE page for:
• Explicit grid lines (bordered tables)
• Implicit tables (aligned columns, no borders)
• Financial tables (numbers, currency, percentages)
• Schedule tables (dates, timelines, deliverables)
• Comparison tables (features, specifications)
• Nested tables (tables within tables)
• Tables in margins, headers, footers
• Tables embedded in screenshots

FOR EVERY TABLE:
1. STRUCTURE EXTRACTION
   • table_id: Unique identifier (e.g., T1, T2, T3...)
   • table_title: Title or caption (if present, otherwise empty string)
   • position: Location on page (e.g., "top", "center", "bottom")
   • size: Coverage area (small/medium/large/full-width)
   • table_type: "financial", "schedule", "comparison", "data", "specification"

2. HEADERS
   • Extract ALL header rows (may be multiple rows)
   • Preserve column header hierarchy
   • Note header formatting (bold, background color)
   • Example: ["Deliverables", "Estimated Sign-off", "Status"]

3. DATA ROWS - COMPLETE EXTRACTION
   • Extract EVERY row
   • Extract EVERY cell value
   • NO truncation, NO "..." placeholders
   • Preserve cell data types (text, number, currency, date, percentage)
   • Note cell formatting (bold, color, alignment)

4. MERGED CELLS DETECTION & HANDLING ⚠ CRITICAL
   
   VISUAL INDICATORS OF MERGED CELLS:
   • Cell borders that span multiple rows/columns
   • Single text value positioned over multiple row/column spaces
   • Large whitespace in bordered areas
   • Content logically shared across rows
   
   MERGED CELL EXTRACTION RULES:
   • If cell spans multiple rows → REPEAT value in each row
   • If cell spans multiple columns → REPEAT value in each column
   • Mark merged regions in "merged_cells" field
   • NEVER output "(blank)" where merged cells exist
   
   EXAMPLE INPUT (Visual):
   ┌────────────────────────────┬──────────────┐
   │ Task 1                     │              │
   ├────────────────────────────┤  Aug 2024    │
   │ Task 2                     │              │
   ├────────────────────────────┼──────────────┤
   
   REQUIRED OUTPUT:
   {
     "rows": [
       ["Task 1", "Aug 2024"],
       ["Task 2", "Aug 2024"]
     ],
     "has_merged_cells": true,
     "merged_cells": "Column 2: 'Aug 2024' spans rows 1-2"
   }

5. TABLE METADATA
   • total_rows: Row count (excluding headers)
   • total_columns: Column count
   • has_merged_cells: true/false
   • merged_cells: Description of merged regions (string or null)
   • data_types: Type of each column (optional)
   • has_subtotals: true/false (optional)
   • has_totals: true/false (optional)
   • notes: Any footnotes or table notes (optional)

REQUIRED JSON OUTPUT SCHEMA:
{
  "table_id": "T<idx>",
  "table_title": "<title or empty string>",
  "position": "<position on page>",
  "size": "<small/medium/large/full-width>",
  "table_type": "<financial/schedule/comparison/data/specification>",
  "headers": ["Column1", "Column2", "Column3"],
  "rows": [
    ["Value1", "Value2", "Value3"],
    ["Value4", "Value5", "Value6"]
  ],
  "total_rows": <number>,
  "total_columns": <number>,
  "has_merged_cells": <true/false>,
  "merged_cells": "<description or null>",
  "data_types": ["text", "number", "date"],
  "has_subtotals": <true/false>,
  "has_totals": <true/false>,
  "notes": "<any footnotes or empty string>"
}

IMPORTANT RULES:
- Return ONLY valid JSON with no commentary or markdown
- Extract ALL rows completely, no truncation
- Handle merged cells by repeating values as shown above
- If a field is optional and not applicable, you may omit it or set to null/empty
- Ensure headers are distinct from data rows
"""

        payload = {
            "contents": [
                {"role": "user", "parts": [{"text": instruction_text},
                                           {"inlineData": {"mimeType": "image/png", "data": b64}}]}
            ],
            "generationConfig": {
                "maxOutputTokens": 8192,
                "temperature": 0.0,
                "candidateCount": 1
            }
        }

        headers = {"Content-Type": "application/json"}

        resp = requests.post(GEMINI_ENDPOINT, headers=headers, json=payload, timeout=timeout)
        if resp.status_code != 200:
            models_status["errors"].append(f"gemini_http_error:{resp.status_code}:{resp.text[:500]}")
            return None

        resp_json = resp.json()

        # Attempt to extract textual JSON output from common Gemini response shapes
        text_output = None
        try:
            cand = resp_json.get("candidates", [None])[0]
            if cand:
                content = cand.get("content", {})
                parts = content.get("parts")
                if isinstance(parts, list):
                    texts = []
                    for p in parts:
                        if isinstance(p, dict) and "text" in p:
                            texts.append(p["text"])
                        elif isinstance(p, str):
                            texts.append(p)
                    if texts:
                        text_output = "\n".join(texts).strip()
                if text_output is None and isinstance(content.get("text"), str):
                    text_output = content.get("text").strip()
        except Exception:
            pass

        if text_output is None:
            # fallback: entire response
            text_output = json.dumps(resp_json)

        # Strip code fences if present
        t = text_output.strip()
        if t.startswith(""):
            first = t.find("")
            last = t.rfind("```")
            if last > first:
                t = t[first + 3:last].strip()
                # Remove language identifier if present
                if t.startswith("json"):
                    t = t[4:].strip()

        parsed = safe_json_loads(t)
        if parsed is None:
            parsed = safe_json_loads(text_output)
        if parsed is None:
            models_status["errors"].append("gemini_unparseable_output")
            return None

        # Validate required fields
        if isinstance(parsed, dict):
            required_fields = ["table_id", "headers", "rows", "total_rows", "total_columns", "has_merged_cells"]
            if all(field in parsed for field in required_fields):
                return parsed

        models_status["errors"].append("gemini_invalid_schema")
        return None

    except Exception as e:
        models_status["errors"].append(f"gemini_exception:{str(e)}")
        models_status["errors"].append(traceback.format_exc()[:1000])
        return None


# -------------------------
# CROP + PROCESS TABLE
# -------------------------
def crop_table_and_save(page_image: Image.Image, bbox: List[float], page_num: int, table_idx: int) -> Tuple[str, Image.Image]:
    x1, y1, x2, y2 = int_box(bbox)
    pad = 6
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(page_image.width, x2 + pad)
    y2 = min(page_image.height, y2 + pad)
    crop = page_image.crop((x1, y1, x2, y2))
    fname = os.path.join(OUTPUT_DIR, f"page_{page_num}_T{table_idx}.png")
    crop.save(fname)
    return fname, crop


def process_table_with_gemini(page_image: Image.Image, bbox: List[float], page_num: int, table_idx: int) -> Dict[str, Any]:
    fname, table_img = crop_table_and_save(page_image, bbox, page_num, table_idx)
    gemini_result = gemini_extract_table_from_image(table_img, page_num, table_idx)
    
    if gemini_result:
        try:
            # Add additional metadata
            gemini_result["page_number"] = page_num
            gemini_result["bbox"] = bbox
            gemini_result["image_file"] = fname
            gemini_result["source"] = GEMINI_MODEL
            
            # Ensure merged_cells is null if not present
            if "merged_cells" not in gemini_result:
                gemini_result["merged_cells"] = None
            
            return gemini_result
            
        except Exception as e:
            models_status["errors"].append(f"gemini_postprocess_error:{str(e)}")
            models_status["errors"].append(traceback.format_exc()[:1000])

    # Fallback structure if Gemini fails
    models_status["errors"].append(f"gemini_failed_table_page{page_num}_t{table_idx}")
    return {
        "table_id": f"T{table_idx}",
        "table_title": "",
        "position": "unknown",
        "size": "unknown",
        "table_type": "unknown",
        "headers": [],
        "rows": [],
        "total_rows": 0,
        "total_columns": 0,
        "has_merged_cells": False,
        "merged_cells": None,
        "page_number": page_num,
        "bbox": bbox,
        "image_file": fname,
        "source": "gemini-failed"
    }


# -------------------------
# PDF / PAGE PROCESSING
# -------------------------
def extract_from_page(page_image: Image.Image, page_num: int) -> Dict[str, Any]:
    detected = detect_tables(page_image)
    if not detected:
        return {"tables": []}
    
    tables_list = []
    for idx, t in enumerate(detected, 1):
        try:
            table_obj = process_table_with_gemini(page_image, t["bbox"], page_num, idx)
            tables_list.append(table_obj)
        except Exception as e:
            models_status["errors"].append(f"process_table_error_page{page_num}_t{idx}:{str(e)}")
            tables_list.append({
                "table_id": f"T{idx}",
                "table_title": "",
                "position": "unknown",
                "size": "error",
                "table_type": "error",
                "headers": [],
                "rows": [],
                "total_rows": 0,
                "total_columns": 0,
                "has_merged_cells": False,
                "merged_cells": None,
                "page_number": page_num,
                "bbox": t.get("bbox", []),
                "image_file": "",
                "source": "error"
            })
    
    return {"tables": tables_list}


def extract_from_pdf_bytes(pdf_bytes: bytes, dpi: int = 300) -> Tuple[Dict[str, Any], int]:
    pages = convert_from_bytes(pdf_bytes, dpi=dpi)
    num_pages = len(pages)
    all_tables = []
    
    for i, pg in enumerate(pages, 1):
        page_result = extract_from_page(pg, i)
        tables = page_result.get("tables", [])
        if tables:
            all_tables.extend(tables)
    
    return {"tables": all_tables}, num_pages

# # -------------------------
# # FASTAPI APP
# # -------------------------
# app = FastAPI(title="Table extraction", version="2.0.0")
# app.add_middleware(CORSMiddleware, allow_origins=[""], allow_credentials=True, allow_methods=[""], allow_headers=["*"])


# @app.get("/health")
# async def health():
#     return {
#         "status": "ok",
#         "cuda_available": torch.cuda.is_available(),
#         "table_detection_loaded": models_status["table_detection"],
#         "gemini_api_key_present": models_status["gemini_api_key_present"],
#         "errors": models_status["errors"],
#         "device": ("cuda" if torch.cuda.is_available() else "cpu")
#     }


# @app.get("/")
# async def root():
#     return {
#         "service": "Table extraction",
#         "models": {
#             "detection": TABLE_DET_MODEL,
#             "ocr_and_structure": GEMINI_MODEL
#         },
#         "version": "2.0.0"
#     }


# @app.post("/extract-tables/")
# async def extract_tables(file: UploadFile = File(...), dpi: int = Query(300, ge=100, le=600), include_metadata: bool = Query(True)):
#     if not file.filename.lower().endswith(".pdf"):
#         raise HTTPException(status_code=400, detail="Only PDF files are supported (.pdf)")
#     if not models_status["table_detection"]:
#         raise HTTPException(status_code=500, detail="Table detection model not loaded; check /health for details")
#     if not GEMINI_API_KEY:
#         raise HTTPException(status_code=500, detail="GEMINI_API_KEY not set in environment (.env) - check /health")

#     try:
#         pdf_bytes = await file.read()
#         result, num_pages = extract_from_pdf_bytes(pdf_bytes, dpi=dpi)
        
#         if include_metadata:
#             total_tables = len(result.get("tables", []))
#             result["metadata"] = {
#                 "total_pages": num_pages,
#                 "total_tables": total_tables,
#                 "dpi": dpi,
#                 "filename": file.filename,
#                 "extraction_method": "tabletransformer_detection+gemini-2.5-flash",
#                 "version": "2.0.0"
#             }
        
#         return JSONResponse(content=result)
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")
