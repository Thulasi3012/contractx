"""
Full detection + Gemini summarization (Option 1) — updated
- Added permissive fallback detector per-page to catch low-edge visuals (Gantt)
- If no visuals found in the entire PDF -> prints warning and writes message to JSON
- All other features remain (text masking, logo/watermark removal, Gemini retry)
"""

import os
import cv2
import json
import fitz
# import time
import asyncio
import numpy as np
import pytesseract
from pathlib import Path
from dotenv import load_dotenv
import google.generativeai as genai

# ---------------------------
# CONFIG
# ---------------------------
DPI = 150
MIN_AREA = 20000
MIN_W = 120
MIN_H = 120
EDGE_DENSITY_THRESHOLD = 0.003
ENTROPY_THRESHOLD = 3.0
MAX_PARALLEL = 2
SAVE_CROPS = False
GEMINI_RETRY_SLEEP = 25
LLM_MODEL = "gemini-2.5-flash"

# Permissive fallback thresholds (more lenient)
MIN_AREA_PERMISSIVE = 8000
EDGE_DENSITY_THRESHOLD_PERMISSIVE = 0.0015
ENTROPY_THRESHOLD_PERMISSIVE = 2.5
MIN_W_PERMISSIVE = 80
MIN_H_PERMISSIVE = 60

# ---------------------------
# GEMINI INIT
# ---------------------------
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("Missing GEMINI_API_KEY in .env")

genai.configure(api_key=API_KEY)
LLM = genai.GenerativeModel(LLM_MODEL)

# ---------------------------
# LLM PROMPT
# ---------------------------
LLM_PROMPT = """
You are an expert PDF visual analyzer.

Your task:
- Detect ALL types of non-text visuals in the image.
- This includes ANY visual object such as diagrams, illustrations, graphics,
  charts, shapes, icons, screenshots, process flows, infographics, dashboards,
  UI mockups, engineering drawings, network diagrams, floor plans, timelines,
  Gantt charts, flow diagrams, and any other type of visual content.
- The detection must be fully dynamic. Do NOT limit to predefined categories.

Ignore and exclude:
- tables of any form
- paragraphs or blocks of text
- headers, footers, page numbers, decorative lines

For each visual you detect, return:
{
  "visual_type": "<dynamic classification>",
  "description": "<what the visual represents>",
  "summary": "<2-3 sentence summary of the visual>",
  "bbox": [x1, y1, x2, y2]
}

Rules:
- Respond ONLY with valid JSON.
- If no visuals are present, return: { "visuals": [] }
- NEVER classify tables as visuals.
- If something is ambiguous, classify with your best guess and describe what you see.
"""

# ---------------------------
# Helpers
# ---------------------------


def shannon_entropy_gray(gray):
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    probs = hist / (hist.sum() + 1e-12)
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs))


def to_native(obj):
    """Recursively convert numpy types to native Python types for JSON."""
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, (np.ndarray,)):
        return [to_native(x) for x in obj.tolist()]
    if isinstance(obj, list):
        return [to_native(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): to_native(v) for k, v in obj.items()}
    return obj

# ---------------------------
# PDF rendering
# ---------------------------


def render_pdf(pdf_path, dpi=DPI):
    doc = fitz.open(pdf_path)
    pages = []
    for i in range(len(doc)):
        pix = doc.load_page(i).get_pixmap(dpi=dpi)
        pages.append(pix.tobytes("png"))
    doc.close()
    return pages

# ---------------------------
# Get text boxes (PyMuPDF) scaled to DPI
# ---------------------------


def get_text_boxes_for_page(pdf_path, page_index, dpi=DPI):
    doc = fitz.open(pdf_path)
    page = doc.load_page(page_index)
    raw_blocks = page.get_text("blocks")  # coordinates in PDF points (72 dpi)
    zoom = dpi / 72.0
    boxes = []
    for b in raw_blocks:
        x1, y1, x2, y2, text = b[0], b[1], b[2], b[3], b[4]
        if text and str(text).strip():
            boxes.append([int(x1 * zoom), int(y1 * zoom),
                         int(x2 * zoom), int(y2 * zoom)])
    doc.close()
    return boxes

# ---------------------------
# Page text-heavy check
# ---------------------------


def page_is_text_heavy(page_bgr, text_boxes, threshold=0.75):
    h, w = page_bgr.shape[:2]
    page_area = h * w
    text_area = 0
    for (x1, y1, x2, y2) in text_boxes:
        text_area += max(0, (x2 - x1) * (y2 - y1))
    return (text_area / (page_area + 1e-12)) > threshold

# ---------------------------
# Mask text boxes
# ---------------------------


def apply_text_mask(page_bgr, text_boxes):
    mask = np.ones(page_bgr.shape[:2], dtype=np.uint8) * 255
    for (x1, y1, x2, y2) in text_boxes:
        pad_x = int((x2 - x1) * 0.02) + 2
        pad_y = int((y2 - y1) * 0.02) + 2
        xa, ya = max(0, x1 - pad_x), max(0, y1 - pad_y)
        xb, yb = min(page_bgr.shape[1], x2 +
                     pad_x), min(page_bgr.shape[0], y2 + pad_y)
        cv2.rectangle(mask, (xa, ya), (xb, yb), 0, -1)
    gray = cv2.cvtColor(page_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.bitwise_and(gray, gray, mask=mask), mask

# ---------------------------
# Heuristics: logo / watermark / table / visual
# ---------------------------


def is_logo(crop):
    h, w = crop.shape[:2]
    if h == 0 or w == 0:
        return False
    ar = w / h
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    entropy = shannon_entropy_gray(gray)
    if entropy < 3.0 and (0.5 < ar < 1.8) and (w < 360 and h < 360):
        return True
    return False


def is_watermark(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 160)
    area = crop.shape[0] * crop.shape[1]
    if area == 0:
        return False
    edge_density = np.sum(edges > 0) / (area + 1e-12)
    if edge_density < 0.002 and area > 50000:
        return True
    return False


def detect_if_table(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 180)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180,
                            threshold=80, minLineLength=80, maxLineGap=8)
    if lines is None:
        return False
    # tables have many lines; require high count to avoid flagging Gantt bars
    return len(lines) > 25


def is_real_visual(crop, edge_thresh=EDGE_DENSITY_THRESHOLD, entropy_thresh=ENTROPY_THRESHOLD, area_thresh=150000):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 160)
    area = crop.shape[0] * crop.shape[1]
    if area == 0:
        return False
    edge_density = np.sum(edges > 0) / (area + 1e-12)
    entropy = shannon_entropy_gray(gray)
    if (edge_density > edge_thresh) or (entropy > entropy_thresh) or (area > area_thresh):
        return True
    return False

# ---------------------------
# Merge helper
# ---------------------------


def merge_boxes(boxes, iou_thresh=0.15):
    if not boxes:
        return []
    boxes_np = np.array(boxes)
    x1 = boxes_np[:, 0]
    y1 = boxes_np[:, 1]
    x2 = boxes_np[:, 2]
    y2 = boxes_np[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    idxs = list(range(len(boxes)))
    keep = []
    while idxs:
        i = idxs.pop(0)
        bx = [int(x1[i]), int(y1[i]), int(x2[i]), int(y2[i])]
        keep.append(bx)
        remove = []
        for j in idxs:
            xx1 = max(x1[i], x1[j])
            yy1 = max(y1[i], y1[j])
            xx2 = min(x2[i], x2[j])
            yy2 = min(y2[i], y2[j])
            w = max(0, xx2 - xx1)
            h = max(0, yy2 - yy1)
            inter = w * h
            union = areas[i] + areas[j] - inter
            iou = inter / (union + 1e-12)
            if iou > iou_thresh:
                remove.append(j)
                keep[-1][0] = min(keep[-1][0], int(x1[j]))
                keep[-1][1] = min(keep[-1][1], int(y1[j]))
                keep[-1][2] = max(keep[-1][2], int(x2[j]))
                keep[-1][3] = max(keep[-1][3], int(y2[j]))
        idxs = [k for k in idxs if k not in remove]
    return keep

# ---------------------------
# Strict detector (original behavior)
# ---------------------------


def detect_visuals_strict(page_bgr, mask=None):
    gray = cv2.cvtColor(page_bgr, cv2.COLOR_BGR2GRAY)
    if mask is not None:
        gray = cv2.bitwise_and(gray, gray, mask=mask)
    th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 51, 9)
    merge_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 30))
    merged = cv2.dilate(th, merge_kernel, iterations=2)
    contours, _ = cv2.findContours(
        merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        area = bw * bh
        if area < MIN_AREA:
            continue
        if bw < MIN_W or bh < MIN_H:
            continue
        crop = page_bgr[y:y+bh, x:x+bw]
        if is_logo(crop):
            continue
        if is_watermark(crop):
            continue
        if is_real_visual(crop) or detect_if_table(crop):
            boxes.append([int(x), int(y), int(x + bw), int(y + bh)])
    boxes = merge_boxes(boxes)
    boxes = sorted(boxes, key=lambda b: (b[1], b[0]))
    return boxes

# ---------------------------
# Permissive detector (fallback per page)
# ---------------------------


def detect_visuals_permissive(page_bgr, mask=None):
    """
    More lenient detection to catch soft images (like Gantt timelines).
    Runs only when strict detector finds nothing on a page.
    """
    gray = cv2.cvtColor(page_bgr, cv2.COLOR_BGR2GRAY)
    # apply mask but be less aggressive: erode mask slightly so we still inspect near-text regions
    if mask is not None:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        relaxed_mask = cv2.erode(mask, kernel, iterations=1)
        gray = cv2.bitwise_and(gray, gray, mask=relaxed_mask)

    # Use a smaller morph kernel to preserve smaller shapes
    th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                               cv2.THRESH_BINARY_INV, 41, 7)
    merge_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    merged = cv2.dilate(th, merge_kernel, iterations=1)

    contours, _ = cv2.findContours(
        merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        area = bw * bh
        if area < MIN_AREA_PERMISSIVE:
            continue
        if bw < MIN_W_PERMISSIVE or bh < MIN_H_PERMISSIVE:
            continue
        crop = page_bgr[y:y+bh, x:x+bw]

        # relax logo/watermark checks slightly: still skip tiny logos, keep large faint images
        if is_logo(crop) and (bw < 220 and bh < 220):
            continue
        if is_watermark(crop) and area > 200000:
            # if area is very large and extremely faint, treat as watermark
            continue

        # permissive visual test
        if is_real_visual(crop, edge_thresh=EDGE_DENSITY_THRESHOLD_PERMISSIVE,
                          entropy_thresh=ENTROPY_THRESHOLD_PERMISSIVE, area_thresh=100000) or detect_if_table(crop):
            boxes.append([int(x), int(y), int(x + bw), int(y + bh)])

    boxes = merge_boxes(boxes, iou_thresh=0.1)
    boxes = sorted(boxes, key=lambda b: (b[1], b[0]))
    return boxes

# ---------------------------
# OCR title extraction
# ---------------------------


def extract_title(page_bgr, bbox, top_pad=150):
    return None


def crop_to_bytes(page_bgr, bbox):
    x1, y1, x2, y2 = bbox
    crop = page_bgr[y1:y2, x1:x2]
    _, buf = cv2.imencode(".png", crop)
    return buf.tobytes(), crop

# ---------------------------
# Safe Gemini call
# ---------------------------


async def safe_generate_content_async(contents):
    while True:
        try:
            response = await LLM.generate_content_async(contents=contents)
            return response
        except Exception as e:
            s = str(e).lower()
            if "quota" in s or "rate limit" in s or "exceeded" in s or "429" in s:
                print("[gemini] quota/rate limit hit, sleeping",
                      GEMINI_RETRY_SLEEP, "s")
                await asyncio.sleep(GEMINI_RETRY_SLEEP)
                continue
            else:
                raise


async def call_gemini_async(image_bytes, page_num, bbox, title, vtype):
    prompt = LLM_PROMPT + "\n" + json.dumps({
        "page_number": int(page_num),
        "bbox": [int(x) for x in bbox],
        "visual_type_cv": vtype
    })
    contents = [
        {"mime_type": "image/png", "data": image_bytes},
        {"text": prompt}
    ]
    response = await safe_generate_content_async(contents)
    raw = response.text or ""
    start = raw.find("{")
    end = raw.rfind("}")
    try:
        data = json.loads(raw[start:end+1])
    except Exception:
        data = {"summary": "Invalid JSON", "unreadable": True}
    # enrich and cast to native types
    data["page_number"] = int(page_num)
    data["bbox"] = [int(x) for x in bbox]
    data["visual_title"] = title if title else None
    data["type"] = str(vtype)
    try:
        usage = response.usage_metadata
        data["tokens_in"] = int(getattr(usage, "prompt_token_count", 0) or 0)
        data["tokens_out"] = int(
            getattr(usage, "candidates_token_count", 0) or 0)
    except Exception:
        data["tokens_in"] = 0
        data["tokens_out"] = 0
    data = to_native(data)
    return data


def classify_visual(crop):
    if detect_if_table(crop):
        return "table"
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100,
                            minLineLength=100, maxLineGap=20)
    if lines is not None:
        horiz = sum(1 for l in lines if abs(l[0][1]-l[0][3]) < 5)
        vert = sum(1 for l in lines if abs(l[0][0]-l[0][2]) < 5)
        if horiz and vert:
            return "chart"
        return "diagram"
    return "image"

# ---------------------------
# Main processor
# ---------------------------


async def process_pdf_async(pdf_path):
    pages = render_pdf(pdf_path)
    tasks = []
    sem = asyncio.Semaphore(MAX_PARALLEL)

    async def process_visual(page_idx, page_bgr, bbox):
        async with sem:
            crop_bytes, crop_img = crop_to_bytes(page_bgr, bbox)
            title = extract_title(page_bgr, bbox)
            vtype = classify_visual(crop_img)
            return await call_gemini_async(crop_bytes, page_idx, bbox, title, vtype)

    # track total candidates (post-processing) for the entire doc
    page_count = len(pages)
    total_candidates = 0

    for page_idx, pg_bytes in enumerate(pages, start=1):
        print(f"Processing page {page_idx}/{page_count}...")
        page_bgr = cv2.imdecode(np.frombuffer(
            pg_bytes, np.uint8), cv2.IMREAD_COLOR)
        text_boxes = get_text_boxes_for_page(pdf_path, page_idx - 1, dpi=DPI)
        masked_gray, mask = apply_text_mask(page_bgr, text_boxes)

        # If page is extremely text heavy skip strict detect (but still try permissive if needed)
        if page_is_text_heavy(page_bgr, text_boxes, threshold=0.90):
            print(
                f"[Page {page_idx}] page is >90% text, skipping strict detector")
            boxes = []
        else:
            boxes = detect_visuals_strict(page_bgr, mask=mask)

        # If strict detector found nothing, run permissive detector for this page
        if not boxes:
            print(
                f"[Page {page_idx}] no strict candidates — running permissive detector")
            boxes = detect_visuals_permissive(page_bgr, mask=mask)
            if boxes:
                print(
                    f"[Page {page_idx}] permissive detector found {len(boxes)} candidates")

        print(f"[Page {page_idx}] {len(boxes)} candidate visuals found")
        total_candidates += len(boxes)

        if SAVE_CROPS and boxes:
            outdir = Path("debug_crops")
            outdir.mkdir(parents=True, exist_ok=True)
            for i, b in enumerate(boxes):
                _, crop = crop_to_bytes(page_bgr, b)
                cv2.imwrite(str(outdir / f"p{page_idx}_v{i}.png"), crop)

        for bbox in boxes:
            tasks.append(process_visual(page_idx, page_bgr, bbox))

    results = []
    if tasks:
        print(
            f"Sending {len(tasks)} visuals to Gemini (parallel={MAX_PARALLEL})...")
        gathered = await asyncio.gather(*tasks, return_exceptions=True)
        for idx, r in enumerate(gathered, start=1):
            if isinstance(r, Exception):
                print(f"[ERROR] task {idx} failed:", r)
            else:
                results.append(r)

    out = {
        "document": os.path.basename(pdf_path),
        "visual_count": len(results),
        "total_input_tokens": sum((r.get("tokens_in") or 0) for r in results),
        "total_output_tokens": sum((r.get("tokens_out") or 0) for r in results),
        "visuals": results
    }

    # If no visuals across the document, add message and print warning
    if out["visual_count"] == 0:
        msg = "No visuals found in this PDF."
        print("⚠", msg)
        out["message"] = msg

    # output filename: same base name as input (stem.json)
    outfile = Path(pdf_path).with_suffix(".json")
    out = to_native(out)
    with open(outfile, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=4)

    print("\n✔ DONE!")
    print(" Visuals:", out["visual_count"])
    print(" Saved:", outfile)
    return out