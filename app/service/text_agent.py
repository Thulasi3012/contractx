"""
Text Agent - Extracts text content from PDF pages using Gemini Vision LLM
"""

import fitz  # PyMuPDF
import google.generativeai as genai
from PIL import Image
import io
from typing import Dict, Any
from dataclasses import dataclass, asdict


@dataclass
class TextContent:
    """Structured text output for a page"""
    page_number: int
    raw_text: str
    word_count: int
    char_count: int
    status: str
    error_message: str = None


class TextAgent:
    """
    Agent responsible for extracting text content from PDF pages using Gemini Vision.
    Focuses ONLY on non-table text extraction (paragraphs, headings, standalone text).
    """
    
    def __init__(self, api_key: str, model_name: str = "gemini-2.0-flash-exp"):
        """
        Initialize Text Agent with Gemini API.
        
        Args:
            api_key: Google Gemini API key
            model_name: Gemini model to use
        """
        self.name = "TextAgent"
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        
        # Specialized prompt for text-only extraction
        self.extraction_prompt = """You are a document text extraction agent specialized in extracting ONLY plain text content.

CRITICAL INSTRUCTION: Extract ONLY text that is NOT part of any table structure.

What TO extract:
- Headings and titles
- Paragraphs and body text
- Standalone sentences
- Addresses and contact information
- Any text that appears outside of tables

What NOT to extract (IGNORE COMPLETELY):
- ANY content inside tables (rows, columns, cells)
- Tabular data with multiple columns
- Grids with headers like "Product Name", "Qty", "Price", "Material #"
- Any structured data in table format
- Table headers and table content
- Lists that are formatted as tables

Rules:
1. Skip ALL tables - do not extract any text from table cells
2. Only extract free-flowing text paragraphs
3. Maintain paragraph breaks and natural text flow
4. Do NOT add explanations like "Here is the text:" or comments
5. If the entire page is just a table, return empty string or minimal text
6. Return clean plain text only

Extract the non-table text now:"""
    
    def     pdf_page_to_image(self, page: fitz.Page, dpi: int = 250) -> Image.Image:
        """
        Convert a PDF page to a PIL Image for LLM processing.
        
        Args:
            page: PyMuPDF Page object
            dpi: Resolution for image conversion (default: 250)
            
        Returns:
            PIL Image object
        """
        try:
            pix = page.get_pixmap(dpi=dpi)
            img_bytes = pix.tobytes("png")
            image = Image.open(io.BytesIO(img_bytes))
            return image
        except Exception as e:
            raise Exception(f"Failed to convert page to image: {str(e)}")
    
    def extract_text_with_gemini(self, image: Image.Image) -> str:
        """
        Extract text from a page image using Gemini vision model.
        
        Args:
            image: PIL Image of the PDF page
            
        Returns:
            Extracted text as a string
        """
        try:
            response = self.model.generate_content(
                [self.extraction_prompt, image],
                generation_config={
                    "temperature": 0,
                    "top_p": 0.95,
                    "top_k": 40,
                    "max_output_tokens": 8192,
                }
            )
            return response.text.strip()
        except Exception as e:
            raise Exception(f"Gemini API error: {str(e)}")
    
    def calculate_stats(self, text: str) -> Dict[str, int]:
        """Calculate text statistics"""
        words = text.split()
        return {
            "word_count": len(words),
            "char_count": len(text)
        }
    
    def process_page(self, page: fitz.Page, page_number: int = None) -> TextContent:
        """
        Main method to process a page and extract text content.
        This is called by Heart LLM for each page.
        
        Args:
            page: PyMuPDF Page object (passed from Heart LLM)
            page_number: Page number (1-indexed)
        
        Returns:
            TextContent with structured text data
        """
        # Determine page number
        if page_number is None:
            page_number = page.number + 1
        
        try:
            # Step 1: Convert page to image for LLM processing
            image = self.pdf_page_to_image(page)
            
            # Step 2: Extract text using Gemini Vision
            raw_text = self.extract_text_with_gemini(image)
            
            # Step 3: Calculate statistics
            stats = self.calculate_stats(raw_text)
            
            # Step 4: Create structured output
            text_content = TextContent(
                page_number=page_number,
                raw_text=raw_text,
                word_count=stats["word_count"],
                char_count=stats["char_count"],
                status="success"
            )
            
            return text_content
            
        except Exception as e:
            # Handle errors gracefully
            print(f"⚠️  Text Agent error on page {page_number}: {str(e)}")
            
            return TextContent(
                page_number=page_number,
                raw_text="",
                word_count=0,
                char_count=0,
                status="failed",
                error_message=str(e)
            )
    
    def to_dict(self, text_content: TextContent) -> Dict[str, Any]:
        """Convert TextContent to dictionary for Heart LLM"""
        return asdict(text_content)
