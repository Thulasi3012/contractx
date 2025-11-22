# app/service/Document_chunker.py
import os
import time
from io import BytesIO
from dataclasses import dataclass
from typing import List, Optional
from enum import Enum
import logging
import PyPDF2

logger = logging.getLogger("DocumentChunker")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    logger.addHandler(ch)


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


@dataclass
class DocumentChunk:
    metadata: ChunkMetadata
    content: bytes
    overlap_pages: List[int]
    priority: Priority

class DocumentChunker:
    """
    DocumentChunker with async create_chunks to match pipeline usage.

    Rules:
      - chunk_size: number of pages in each chunk (last chunk may be smaller)
      - overlap_pages: number of pages to overlap with previous chunk
      - step = chunk_size - overlap_pages  (min 1)
    """

    def __init__(self, chunk_size: int = 3, overlap_pages: int = 0):
        if chunk_size < 1:
            raise ValueError("chunk_size must be >= 1")
        if overlap_pages < 0:
            raise ValueError("overlap_pages must be >= 0")
        if overlap_pages >= chunk_size:
            logger.warning("overlap_pages >= chunk_size — treating step as 1")
        self.chunk_size = chunk_size
        self.overlap_pages = overlap_pages
        logger.info(f"Initialized DocumentChunker | chunk_size={self.chunk_size} overlap_pages={self.overlap_pages}")

    async def create_chunks(self, document_path: str, total_pages: int) -> List[DocumentChunk]:
        """
        Create chunks for a PDF at document_path covering total_pages pages.
        Returns list[DocumentChunk].
        """
        logger.info(f"create_chunks called | path={document_path} total_pages={total_pages}")
        start_time = time.time()

        chunks: List[DocumentChunk] = []
        chunk_idx = 0

        # step determines how much we advance for the next chunk
        step = max(1, self.chunk_size - self.overlap_pages)

        # iterate start indices
        start = 0
        while start < total_pages:
            end = min(start + self.chunk_size - 1, total_pages - 1)  # inclusive end index

            # position of chunk
            position = (
                ChunkPosition.BEGINNING if chunk_idx == 0 else
                ChunkPosition.END if end == total_pages - 1 else
                ChunkPosition.MIDDLE
            )

            # compute prev/next chunk ids
            prev_chunk = f"chunk_{chunk_idx - 1}" if chunk_idx > 0 else None
            next_chunk = None
            # we can detect if there will be a next chunk by checking if end < last page
            if end < total_pages - 1:
                next_chunk = f"chunk_{chunk_idx + 1}"

            metadata = ChunkMetadata(
                chunk_id=f"chunk_{chunk_idx}",
                page_range=(start, end),
                section_id=f"Section_{chunk_idx}",
                position=position,
                prev_chunk=prev_chunk,
                next_chunk=next_chunk
            )

            # overlap pages relative to this chunk: pages that were included in previous chunk(s)
            overlap_pages_list: List[int] = []
            if self.overlap_pages > 0 and start > 0:
                overlap_start = max(0, start - self.overlap_pages)
                overlap_end = start - 1
                overlap_pages_list = list(range(overlap_start, overlap_end + 1)) if overlap_end >= overlap_start else []

            # extract bytes for this chunk
            chunk_bytes = self._extract_pages(document_path, start, end)

            chunk = DocumentChunk(
                metadata=metadata,
                content=chunk_bytes,
                overlap_pages=overlap_pages_list,
                priority=Priority.HIGH
            )

            chunks.append(chunk)
            logger.info(f"Created {metadata.chunk_id} pages {start}-{end} overlap={overlap_pages_list}")

            chunk_idx += 1
            start += step  # advance by step (accounts for overlap)

        elapsed = time.time() - start_time
        logger.info(f"create_chunks complete | total_chunks={len(chunks)} elapsed={elapsed:.2f}s")
        return chunks

    def _extract_pages(self, document_path: str, start: int, end: int) -> bytes:
        """
        Extract pages [start..end] (inclusive) from PDF and return bytes of a PDF containing only those pages.
        """
        reader = PyPDF2.PdfReader(document_path)
        total = len(reader.pages)
        s = max(0, start)
        e = min(end, total - 1)

        writer = PyPDF2.PdfWriter()
        for p in range(s, e + 1):
            writer.add_page(reader.pages[p])

        out = BytesIO()
        writer.write(out)
        out.seek(0)
        return out.read()
