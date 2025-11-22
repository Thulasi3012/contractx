# app/routes/documents.py
import os
import time
import logging
import tempfile
from io import BytesIO

from fastapi import APIRouter, UploadFile, File, HTTPException, Form

from app.service.Document_chunker import DocumentChunker
from app.service.document_parser import DocumentParser
from app.service.Tables import extract_from_pdf_bytes
from app.service.image import process_pdf_async

import PyPDF2

router = APIRouter(
    prefix="/api/documents",
    tags=["Document Retrieval"]
)

# -----------------------------------------------------
# LOGGER
# -----------------------------------------------------
logger = logging.getLogger("DocumentProcessor")
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    ))
    logger.addHandler(handler)

# -----------------------------------------------------
# PAGE COUNTER
# -----------------------------------------------------
def get_total_pages_from_bytes(pdf_bytes: bytes) -> int:
    try:
        reader = PyPDF2.PdfReader(BytesIO(pdf_bytes))
        return len(reader.pages)
    except Exception as e:
        logger.error(f"❌ PDF page count failed: {e}")
        raise HTTPException(status_code=400, detail="Invalid PDF file")

# -----------------------------------------------------
# MAIN PIPELINE
# -----------------------------------------------------
@router.post("/process-document")
async def process_document(
    file: UploadFile = File(...),
    chunk_size: int = Form(5),
    overlap_pages: int = Form(1)
):
    """
    Full pipeline:
    1) Chunk PDF
    2) Extract text
    3) Extract tables
    4) Extract visual formats (images/charts/forms)
    5) Return JSON output
    """

    pipeline_start = time.time()
    logger.info("📥 Upload request received")

    pdf_bytes = await file.read()
    logger.info(f"📄 Uploaded: {file.filename} ({len(pdf_bytes)} bytes)")

    total_pages = get_total_pages_from_bytes(pdf_bytes)
    logger.info(f"📘 Total PDF pages: {total_pages}")

    # -----------------------
    # Save temp PDF
    # -----------------------
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".pdf")
    os.close(tmp_fd)
    with open(tmp_path, "wb") as f:
        f.write(pdf_bytes)

    logger.info(f"📂 Temp PDF stored at: {tmp_path}")

    # -----------------------
    # Extract visual formats
    # -----------------------
    logger.info("🖼 Extracting visual formats (images/charts/diagrams)...")
    image_results = await process_pdf_async(tmp_path)

    # -----------------------
    # Chunking
    # -----------------------
    logger.info(f"✂️ Starting chunking | size={chunk_size}, overlap={overlap_pages}")
    chunker = DocumentChunker(
        chunk_size=chunk_size,
        overlap_pages=overlap_pages
    )

    chunks = await chunker.create_chunks(
        document_path=tmp_path,
        total_pages=total_pages
    )
    logger.info(f"📦 Total chunks created: {len(chunks)}")

    # Delete temp file after chunking
    os.remove(tmp_path)
    logger.info("🗑 Temp file deleted")

    parser = DocumentParser()
    final_chunks = []
    global_tables = []
    global_text = ""

    logger.info("🔍 Starting per-chunk extraction...")

    # -----------------------
    # Per-chunk processing
    # -----------------------
    for chunk in chunks:
        cid = chunk.metadata.chunk_id
        page_start, page_end = chunk.metadata.page_range

        logger.info(f"➡️ Processing {cid} | pages {page_start}-{page_end}")

        text = parser.extract_text(chunk.content)
        global_text += f"\n\n--- CHUNK {cid} ---\n{text}"

        tables_result, _ = extract_from_pdf_bytes(chunk.content)
        tables = tables_result.get("tables", [])

        if tables:
            logger.info(f"   🟦 Tables found: {len(tables)}")
            global_tables.extend(tables)
        else:
            logger.info(f"   ⚪ No tables found")

        final_chunks.append({
            "chunk_id": cid,
            "page_range": chunk.metadata.page_range,
            "section_id": chunk.metadata.section_id,
            "position": chunk.metadata.position.value,
            "prev_chunk": chunk.metadata.prev_chunk,
            "next_chunk": chunk.metadata.next_chunk,
            "overlap_pages": chunk.overlap_pages
        })

    logger.info("🟢 Chunk extraction completed")

    total_time = round(time.time() - pipeline_start, 2)
    logger.info(f"🏁 Done in {total_time}s")

    return {
        "status": "success",
        "processing_time_sec": total_time,
        "pages": total_pages,
        "chunks": final_chunks,
        "parsed_text": global_text.strip(),
        "tables": global_tables,
        "images": image_results
    }
