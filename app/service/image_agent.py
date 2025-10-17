"""
Enhanced Image Agent - Comprehensive detection, extraction, and analysis of all visual content
from PDFs including images, tables, graphs, charts, diagrams, and scanned documents.
"""

import fitz  # PyMuPDF
import base64
import hashlib
import json
import io
import os
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from dotenv import load_dotenv

import google.generativeai as genai
from PIL import Image

# Load environment variables
load_dotenv()

# --- COMPREHENSIVE ANALYSIS PROMPT ---
COMPREHENSIVE_IMAGE_ANALYSIS_PROMPT = """
You are an advanced AI specialist in visual content analysis, OCR, and data interpretation. 
Analyze the provided image with extreme precision and extract ALL information present.

**Your Task:**
Perform a comprehensive analysis of the image, identifying its type and extracting all relevant information.

**Image Types to Identify:**
1. **Tables** - Data organized in rows and columns
2. **Bar Charts/Graphs** - Comparative data visualization
3. **Line Graphs** - Trend analysis over time or continuous data
4. **Pie Charts** - Proportional/percentage breakdowns
5. **Scatter Plots** - Correlation between variables
6. **Flowcharts** - Process flows, decision trees, workflows
7. **Diagrams** - Technical diagrams, architectural drawings, schematics
8. **Timelines** - Chronological events or milestones
9. **Infographics** - Mixed visual information designs
10. **Vector Graphics** - Logos, icons, illustrations
11. **Photographs** - Real-world images
12. **Screenshots** - UI/Application captures
13. **Maps** - Geographical representations
14. **Organizational Charts** - Hierarchical structures
15. **Gantt Charts** - Project timelines
16. **Venn Diagrams** - Set relationships
17. **Network Diagrams** - Connections and relationships
18. **Mathematical Equations/Formulas** - Scientific notation
19. **Mixed Content** - Combination of text, images, and graphics

**Analysis Requirements:**

1. **Text Extraction (OCR):**
   - Extract ALL visible text with 100% accuracy
   - Include headers, titles, labels, legends, axis labels
   - Capture table headers, row labels, and cell values
   - Extract annotations, footnotes, and captions
   - Preserve numerical data exactly as shown
   - Maintain logical reading order
   - Handle rotated or skewed text
   - Process text in scanned/low-quality images

2. **Visual Content Analysis:**
   - Identify the primary type of visual (table, chart, diagram, etc.)
   - Describe layout and structure
   - Identify colors, patterns, and visual elements
   - Note any legends, keys, or scales
   - Describe spatial relationships

3. **Data Extraction (for charts/graphs/tables):**
   - Extract all numerical values precisely
   - Identify axis labels and units of measurement
   - List all data series and categories
   - Extract percentages, ratios, or statistics
   - Note trends, patterns, or outliers
   - Identify max/min values if relevant

4. **Structural Analysis:**
   - For Tables: Row and column count, headers, data types
   - For Flowcharts: Steps, decision points, flow direction
   - For Timelines: Events, dates, sequences
   - For Diagrams: Components, connections, labels
   - For Organizational Charts: Hierarchy levels, relationships

5. **Contextual Understanding:**
   - What is the main message or purpose?
   - What insights can be derived?
   - What conclusions does the visual support?
   - What is the intended audience?

6. **Quality Assessment:**
   - Note if image is scanned or digitally created
   - Mention any clarity issues, artifacts, or distortions
   - Flag any text that's unclear or illegible

**Output Format:**
Provide your analysis in the following JSON structure:

{
  "visual_type": "<Primary type: table, bar_chart, line_graph, pie_chart, flowchart, diagram, timeline, infographic, photograph, screenshot, map, org_chart, gantt_chart, venn_diagram, network_diagram, equation, mixed, or other>",
  "visual_subtype": "<More specific classification if applicable>",
  "title": "<Title or heading of the visual, if present>",
  "transcribed_text": "<Complete OCR extraction of ALL text in logical order>",
  "data_extracted": {
    "tables": [
      {
        "headers": ["column1", "column2", "..."],
        "rows": [["value1", "value2", "..."], ["...", "...", "..."]]
      }
    ],
    "numerical_data": {
      "data_series": [
        {
          "name": "<series name>",
          "values": [value1, value2, ...]
        }
      ],
      "key_statistics": {
        "maximum": "<value>",
        "minimum": "<value>",
        "average": "<value>",
        "total": "<value>"
      }
    },
    "timeline_events": [
      {
        "date": "<date/time>",
        "event": "<description>"
      }
    ],
    "flowchart_steps": [
      {
        "step_number": 1,
        "type": "<process/decision/start/end>",
        "content": "<step description>",
        "connections": ["connects to step X"]
      }
    ]
  },
  "visual_elements": {
    "colors_used": ["<list of prominent colors>"],
    "layout_description": "<Description of visual layout>",
    "legend_items": ["<legend entries if present>"],
    "axes_labels": {
      "x_axis": "<label and unit>",
      "y_axis": "<label and unit>"
    }
  },
  "detailed_summary": "<Comprehensive 3-5 sentence summary covering: 1) What type of visual this is, 2) What data/information it presents, 3) Key findings or trends, 4) The main message or conclusion>",
  "key_insights": [
    "<Insight 1: notable pattern, trend, or finding>",
    "<Insight 2: comparative observation or significant data point>",
    "<Insight 3: conclusion or implication>"
  ],
  "context_and_purpose": "<What is this visual trying to communicate and to whom?>",
  "quality_notes": "<Notes on image quality, scanned vs digital, any issues>",
  "contains_embedded_images": "<yes/no - if this visual contains other images within it>",
  "technical_details": "<Any formulas, equations, technical specifications, or specialized information>"
}

**Important Guidelines:**
- Be exhaustive in text extraction - miss nothing
- For tables, preserve exact structure and all cell values
- For charts, extract ALL data points, not just samples
- For scanned images, apply advanced OCR techniques
- If multiple elements exist (e.g., table with embedded chart), analyze both
- If text is unclear, note it but provide best interpretation
- Always provide detailed_summary even if other fields are minimal
- Prioritize accuracy over brevity
"""


# --- DATA STRUCTURES ---
@dataclass
class ImageData:
    """Represents a single extracted and comprehensively analyzed image"""
    image_id: int
    bbox: tuple
    width: int
    height: int
    format: str
    size_kb: float
    image_hash: str
    visual_type: str
    visual_subtype: str
    title: str
    transcribed_text: str
    data_extracted: Dict[str, Any]
    visual_elements: Dict[str, Any]
    detailed_summary: str
    key_insights: List[str]
    context_and_purpose: str
    quality_notes: str
    contains_embedded_images: str
    technical_details: str
    base64_data: str
    analysis_status: str = "success"
    analysis_error: Optional[str] = None


@dataclass
class ImageContent:
    """Structured image output for a single page"""
    page_number: int
    images: List[Dict[str, Any]]
    image_count: int
    total_size_kb: float
    status: str
    error_message: Optional[str] = None


# --- AGENT CLASS ---
class ImageAgent:
    """
    Enhanced agent for comprehensive detection, extraction, and analysis of all visual content
    from PDFs, including scanned documents, tables, charts, diagrams, and complex graphics.
    """

    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None):
        """
        Initialize the Image Agent with Gemini Vision API.
        
        Args:
            api_key: Google AI API key. If None, reads from GEMINI_API_KEY env variable
            model_name: Specific model name. If None, reads from GEMINI_MODEL env variable or auto-detects
        """
        self.name = "ImageAgent_Enhanced_Vision"
        
        # Get API key from parameter or environment
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Gemini API key not found. Please provide it as a parameter or set GEMINI_API_KEY in your .env file"
            )
        
        # Get model name from parameter or environment
        self.model_name = model_name or os.getenv("GEMINI_MODEL")
        
        self.model = self._initialize_model()
        self.seen_hashes = set()
        print(f"✅ {self.name} initialized successfully.")

    def _initialize_model(self):
        """
        Initializes the Gemini vision model from environment configuration or auto-detection.
        """
        try:
            genai.configure(api_key=self.api_key)
            
            # If specific model name is provided, use it
            if self.model_name:
                print(f"🔧 Using specified model: {self.model_name}")
                model = genai.GenerativeModel(self.model_name)
                print(f"✅ Model '{self.model_name}' initialized successfully.")
                return model
            
            # Otherwise, auto-detect the best available vision model
            print("🔎 Auto-detecting best available Gemini vision model...")
            available_models = []
            
            for m in genai.list_models():
                if 'generateContent' in m.supported_generation_methods:
                    available_models.append(m.name)
            
            if not available_models:
                raise RuntimeError(
                    "No suitable Gemini vision models found. Please check your API key and permissions."
                )
            
            # Prefer newer/better models (gemini-pro-vision, gemini-1.5-pro, etc.)
            # Sort to prioritize models with higher version numbers
            available_models.sort(reverse=True)
            selected_model = available_models[0]
            
            print(f"✅ Auto-selected model: {selected_model}")
            print(f"   Available models: {', '.join(available_models)}")
            
            model = genai.GenerativeModel(selected_model)
            print("✅ Gemini Vision model initialized successfully.")
            return model

        except Exception as e:
            print(f"❌ Failed to initialize Gemini Vision model: {e}")
            print("   Make sure GEMINI_API_KEY is set correctly in your .env file")
            raise

    def _get_comprehensive_analysis(self, image_bytes: bytes) -> Dict[str, Any]:
        """
        Sends image to Gemini for comprehensive analysis including OCR, structure detection,
        and detailed summarization of tables, charts, diagrams, and all visual content.
        """
        if not self.model:
            return self._create_error_response("Model not initialized")

        try:
            # Convert bytes to PIL Image for better handling
            pil_image = Image.open(io.BytesIO(image_bytes))
            
            # Prepare image for Gemini
            image_parts = [{"mime_type": "image/png", "data": image_bytes}]
            
            print(f"      📤 Sending image to Gemini for comprehensive analysis...")
            
            # Generate content with comprehensive prompt
            response = self.model.generate_content(
                [COMPREHENSIVE_IMAGE_ANALYSIS_PROMPT, *image_parts],
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,  # Low temperature for factual accuracy
                    top_p=0.95,
                    top_k=40,
                )
            )
            
            # Parse response
            response_text = response.text.strip()
            
            # Clean up markdown code blocks if present
            response_text = response_text.replace("```json", "").replace("```", "").strip()
            
            # Parse JSON response
            analysis_data = json.loads(response_text)
            
            print(f"      ✅ Analysis complete - Type: {analysis_data.get('visual_type', 'unknown')}")
            
            return analysis_data

        except json.JSONDecodeError as e:
            error_msg = f"Failed to parse Gemini response as JSON: {str(e)}"
            print(f"      ⚠️ {error_msg}")
            print(f"      Raw response: {response_text[:500]}...")
            return self._create_error_response(error_msg, response_text)
            
        except Exception as e:
            error_msg = f"Gemini API Error: {str(e)}"
            print(f"      ⚠️ {error_msg}")
            return self._create_error_response(error_msg)

    def _create_error_response(self, error_message: str, raw_response: str = "") -> Dict[str, Any]:
        """Creates a standardized error response structure"""
        return {
            "visual_type": "error",
            "visual_subtype": "",
            "title": "",
            "transcribed_text": raw_response if raw_response else "Error during analysis",
            "data_extracted": {},
            "visual_elements": {},
            "detailed_summary": f"Analysis failed: {error_message}",
            "key_insights": [],
            "context_and_purpose": "",
            "quality_notes": error_message,
            "contains_embedded_images": "unknown",
            "technical_details": "",
            "analysis_status": "error",
            "analysis_error": error_message
        }

    def extract_images_from_page(self, page: fitz.Page) -> List[ImageData]:
        """
        Extracts, analyzes, and deduplicates all images from a PDF page.
        Handles scanned PDFs, embedded images, and images within tables.
        """
        images = []
        image_list = page.get_images(full=True)
        
        if not image_list:
            print(f"  ℹ️  No images found on page {page.number + 1}")
            return images

        print(f"  🔍 Found {len(image_list)} image(s) on page {page.number + 1}")

        for img_index, img_info in enumerate(image_list):
            try:
                xref = img_info[0]
                base_image = page.parent.extract_image(xref)
                image_bytes = base_image["image"]
                
                # Calculate hash for deduplication
                img_hash = hashlib.sha256(image_bytes).hexdigest()

                # Skip duplicates
                if img_hash in self.seen_hashes:
                    print(f"      ⏭️  Skipping duplicate image (hash: {img_hash[:8]}...)")
                    continue
                
                self.seen_hashes.add(img_hash)

                # Get image dimensions
                pil_image = Image.open(io.BytesIO(image_bytes))
                
                print(f"      🤖 Analyzing image {img_index + 1}/{len(image_list)} ({pil_image.width}x{pil_image.height})...")
                
                # Get comprehensive analysis from Gemini
                analysis = self._get_comprehensive_analysis(image_bytes)
                
                # Create ImageData object with all extracted information
                image_data = ImageData(
                    image_id=img_index + 1,
                    bbox=tuple(page.get_image_bbox(img_info)),
                    # width=pil_image.width,
                    # height=pil_image.height,
                    format=base_image["ext"],
                    # size_kb=round(len(image_bytes) / 1024, 2),
                    # image_hash=img_hash,
                    visual_type=analysis.get("visual_type", "unknown"),
                    visual_subtype=analysis.get("visual_subtype", ""),
                    title=analysis.get("title", ""),
                    transcribed_text=analysis.get("transcribed_text", ""),
                    data_extracted=analysis.get("data_extracted", {}),
                    visual_elements=analysis.get("visual_elements", {}),
                    detailed_summary=analysis.get("detailed_summary", ""),
                    key_insights=analysis.get("key_insights", []),
                    context_and_purpose=analysis.get("context_and_purpose", ""),
                    quality_notes=analysis.get("quality_notes", ""),
                    contains_embedded_images=analysis.get("contains_embedded_images", "no"),
                    technical_details=analysis.get("technical_details", ""),
                    # base64_data=base64.b64encode(image_bytes).decode('utf-8'),
                    analysis_status=analysis.get("analysis_status", "success"),
                    analysis_error=analysis.get("analysis_error")
                )
                
                images.append(image_data)
                print(f"      ✅ Image {img_index + 1} analyzed successfully")

            except Exception as e:
                print(f"      ❌ Error processing image {img_index + 1}: {str(e)}")
                continue
                
        return images

    def reset(self):
        """Clears the deduplication cache for processing a new document"""
        print("🧹 Clearing Image Agent's deduplication cache")
        self.seen_hashes.clear()

    def process_page(self, page: fitz.Page, page_number: Optional[int] = None) -> ImageContent:
        """
        Main entry point: processes a PDF page and extracts all visual content with analysis.
        
        Args:
            page: PyMuPDF page object
            page_number: Optional page number (defaults to page.number + 1)
            
        Returns:
            ImageContent object with all extracted and analyzed images
        """
        if page_number is None:
            page_number = page.number + 1

        print(f"\n📄 Processing page {page_number}...")

        try:
            # Extract and analyze all images
            image_data_list = self.extract_images_from_page(page)
            
            # Convert to dictionaries
            images_dict = [asdict(img) for img in image_data_list]
            
            # Calculate total size
            total_size = sum(img.size_kb for img in image_data_list)
            
            print(f"✅ Page {page_number} complete: {len(images_dict)} image(s) processed")
            
            return ImageContent(
                page_number=page_number,
                images=images_dict,
                image_count=len(images_dict),
                total_size_kb=round(total_size, 2),
                status="success"
            )
            
        except Exception as e:
            error_msg = f"Failed to process page {page_number}: {str(e)}"
            print(f"❌ {error_msg}")
            
            return ImageContent(
                page_number=page_number,
                images=[],
                image_count=0,
                total_size_kb=0.0,
                status="failed",
                error_message=error_msg
            )

    def to_dict(self, image_content: ImageContent) -> Dict[str, Any]:
        """Converts ImageContent to dictionary"""
        return asdict(image_content)
