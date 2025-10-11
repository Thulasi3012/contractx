"""
Table Agent - Detects and extracts tables from PDF pages using Gemini Vision LLM
"""

import fitz  # PyMuPDF
import google.generativeai as genai
from PIL import Image
import io
import json
from typing import Dict, Any, List
from dataclasses import dataclass, asdict


@dataclass
class TableContent:
    """Structured table output for a page"""
    page_number: int
    tables: List[Dict[str, Any]]
    table_count: int
    status: str
    error_message: str = None


class TableAgent:
    """
    Agent responsible for detecting and extracting tables from PDF pages using Gemini Vision.
    Returns structured JSON format with table data.
    """
    
    def __init__(self, api_key: str, model_name: str = "gemini-2.0-flash-exp"):
        """
        Initialize Table Agent with Gemini API.
        
        Args:
            api_key: Google Gemini API key
            model_name: Gemini model to use
        """
        self.name = "TableAgent"
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        
        # Specialized prompt for table extraction
        self.extraction_prompt = """You are a table extraction specialist. Your ONLY job is to detect and extract tables from documents.

CRITICAL INSTRUCTIONS:
1. Extract ONLY table data - ignore all other text
2. Return tables in valid JSON format
3. Each table should have: table_id, headers, rows, row_count, col_count

JSON FORMAT (STRICT):
{
  "tables": [
    {
      "table_id": 1,
      "headers": ["Column1", "Column2", "Column3"],
      "rows": [
        ["cell1", "cell2", "cell3"],
        ["cell4", "cell5", "cell6"]
      ],
      "row_count": 2,
      "col_count": 3
    }
  ]
}

RULES:
- If NO tables exist, return: {"tables": []}
- Extract ALL tables on the page
- Maintain exact cell values (don't summarize)
- Headers must be in the "headers" array
- Data rows in the "rows" array
- Count rows excluding headers
- Return ONLY valid JSON, no explanations

Extract tables now:"""
    
    def pdf_page_to_image(self, page: fitz.Page, dpi: int = 250) -> Image.Image:
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
    
    def extract_tables_with_gemini(self, image: Image.Image) -> List[Dict[str, Any]]:
        """
        Extract tables from a page image using Gemini vision model.
        
        Args:
            image: PIL Image of the PDF page
            
        Returns:
            List of table dictionaries
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
            
            # Parse JSON response
            response_text = response.text.strip()
            
            # Clean markdown code blocks if present
            if response_text.startswith("```json"):
                response_text = response_text.split("```json")[1]
            if response_text.startswith("```"):
                response_text = response_text.split("```")[1]
            if response_text.endswith("```"):
                response_text = response_text.rsplit("```", 1)[0]
            
            response_text = response_text.strip()
            
            # Parse JSON
            data = json.loads(response_text)
            tables = data.get("tables", [])
            
            return tables
            
        except json.JSONDecodeError as e:
            print(f"⚠️  JSON parsing error: {str(e)}")
            print(f"   Raw response: {response_text[:200]}...")
            return []
        except Exception as e:
            raise Exception(f"Gemini API error: {str(e)}")
    
    def process_page(self, page: fitz.Page, page_number: int = None) -> TableContent:
        """
        Main method to process a page and extract table content.
        This is called by Heart LLM for each page.
        
        Args:
            page: PyMuPDF Page object (passed from Heart LLM)
            page_number: Page number (1-indexed)
        
        Returns:
            TableContent with structured table data
        """
        # Determine page number
        if page_number is None:
            page_number = page.number + 1
        
        try:
            # Step 1: Convert page to image for LLM processing
            image = self.pdf_page_to_image(page)
            
            # Step 2: Extract tables using Gemini Vision
            tables = self.extract_tables_with_gemini(image)
            
            # Step 3: Create structured output
            table_content = TableContent(
                page_number=page_number,
                tables=tables,
                table_count=len(tables),
                status="success"
            )
            
            return table_content
            
        except Exception as e:
            # Handle errors gracefully
            print(f"⚠️  Table Agent error on page {page_number}: {str(e)}")
            
            return TableContent(
                page_number=page_number,
                tables=[],
                table_count=0,
                status="failed",
                error_message=str(e)
            )
    
    def to_dict(self, table_content: TableContent) -> Dict[str, Any]:
        """Convert TableContent to dictionary for Heart LLM"""
        return asdict(table_content)