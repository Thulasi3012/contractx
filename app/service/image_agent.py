"""
Image Agent - Detects, extracts, and analyzes images from PDF pages using Gemini Vision.
"""

import fitz  # PyMuPDF
import base64
import hashlib
import json
import io
from typing import Dict, Any, List
from dataclasses import dataclass, asdict

import google.generativeai as genai
from PIL import Image

# --- CONSTANTS ---
IMAGE_DESCRIPTION_PROMPT = """
You are an expert Optical Character Recognition (OCR) and document comprehension system.

For the provided image, perform the following tasks:

1.  **Extract All Text:** Transcribe every piece of visible text with high accuracy. This includes headers, paragraphs, tables, captions, and any text inside diagrams or charts. Maintain the logical reading order.
2.  **Comprehend and Summarize:** After transcription, provide a concise but comprehensive summary. The summary should capture the image's core message, key data points, and overall purpose.

Return the output in a clean JSON format as shown below:
{
  "transcribed_text": "<The complete and accurate text transcription from the image>",
  "summary": "<A clear and comprehensive summary of the image's content and purpose>"
}
"""


# --- DATA STRUCTURES ---
@dataclass
class ImageData:
    """Represents a single extracted and analyzed image"""
    image_id: int
    bbox: tuple
    width: int
    height: int
    format: str
    size_kb: float
    image_hash: str
    transcribed_text: str
    summary: str
    base64_data: str


@dataclass
class ImageContent:
    """Structured image output for a single page"""
    page_number: int
    images: List[Dict[str, Any]]
    image_count: int
    total_size_kb: float
    status: str
    error_message: str = None


# --- AGENT CLASS ---
class ImageAgent:
    """
    Agent responsible for detecting, extracting, and analyzing images from PDF pages.
    It uses PyMuPDF for extraction and Google Gemini for OCR and summarization.
    """

    def __init__(self, api_key: str):
        self.name = "ImageAgent_Vision"
        self.model = self._initialize_model(api_key)
        self.seen_hashes = set()

    def _initialize_model(self, api_key: str):
        """
        Dynamically finds and initializes the first available Gemini vision model.
        """
        try:
            genai.configure(api_key=api_key)
            
            # --- DYNAMIC MODEL FINDER LOGIC ---
            print("🔎 Searching for an available Gemini vision model...")
            for m in genai.list_models():
                if 'generateContent' in m.supported_generation_methods:
                    # This check ensures the model can handle the type of requests we will make.
                    # Vision models are typically the ones supporting this method.
                    model_name = m.name
                    print(f"✅ Found vision model: {model_name}")
                    model = genai.GenerativeModel(model_name)
                    print("✅ Gemini Vision model initialized successfully for Image Agent.")
                    return model
            
            # If no model is found after checking all of them
            raise RuntimeError("No suitable Gemini vision model found. Please check your API key and permissions.")

        except Exception as e:
            print(f"❌ Failed to initialize Gemini Vision model: {e}")
            return None

    def _get_image_description(self, image_bytes: bytes) -> Dict[str, str]:
        """Sends image bytes to the Gemini model for OCR and summarization."""
        if not self.model:
            return {
                "transcribed_text": "Model not initialized.",
                "summary": "Could not analyze image because the Gemini model failed to initialize."
            }

        try:
            image_parts = [{"mime_type": "image/png", "data": image_bytes}]
            response = self.model.generate_content([IMAGE_DESCRIPTION_PROMPT, *image_parts])
            response_text = response.text.strip().replace("```json", "").replace("```", "")
            data = json.loads(response_text)
            return {
                "transcribed_text": data.get("transcribed_text", ""),
                "summary": data.get("summary", "")
            }
        except Exception as e:
            # Provide the full error from the API for better debugging
            full_error_message = f"Gemini API Error: {str(e)}"
            print(f"  - ⚠️ {full_error_message}")
            return {
                "transcribed_text": "Error during analysis.",
                "summary": full_error_message
            }

    def extract_images_from_page(self, page: fitz.Page) -> List[ImageData]:
        """Extracts, analyzes, and deduplicates all images from a given PDF page."""
        images = []
        image_list = page.get_images(full=True)

        for img_index, img_info in enumerate(image_list):
            try:
                xref = img_info[0]
                base_image = page.parent.extract_image(xref)
                image_bytes = base_image["image"]
                img_hash = hashlib.sha256(image_bytes).hexdigest()

                if img_hash in self.seen_hashes:
                    print(f"  - Skipping duplicate image on page {page.number + 1} (hash: {img_hash[:8]}...).")
                    continue
                self.seen_hashes.add(img_hash)

                pil_image = Image.open(io.BytesIO(image_bytes))
                print(f"  - 🤖 Analyzing image {img_index + 1} on page {page.number + 1} with Gemini...")
                description_data = self._get_image_description(image_bytes)
                
                image_data = ImageData(
                    image_id=img_index + 1,
                    bbox=tuple(page.get_image_bbox(img_info)),
                    width=pil_image.width,
                    height=pil_image.height,
                    format=base_image["ext"],
                    size_kb=round(len(image_bytes) / 1024, 2),
                    image_hash=img_hash,
                    transcribed_text=description_data["transcribed_text"],
                    summary=description_data["summary"],
                    base64_data=base64.b64encode(image_bytes).decode('utf-8')
                )
                images.append(image_data)

            except Exception as e:
                print(f"⚠️ Error processing image {img_index + 1} on page {page.number + 1}: {str(e)}")
                continue
        return images
        
    def reset(self):
        """Clears the set of seen image hashes for a new document."""
        print("🧹 Clearing Image Agent's deduplication cache.")
        self.seen_hashes.clear()

    def process_page(self, page: fitz.Page, page_number: int = None) -> ImageContent:
        """Main method to process a page and extract image content."""
        if page_number is None:
            page_number = page.number + 1

        try:
            image_data_list = self.extract_images_from_page(page)
            images_dict = [asdict(img) for img in image_data_list]
            return ImageContent(
                page_number=page_number,
                images=images_dict,
                image_count=len(images_dict),
                total_size_kb=round(sum(img.size_kb for img in image_data_list), 2),
                status="success"
            )
        except Exception as e:
            print(f"⚠️ Image Agent failed on page {page_number}: {str(e)}")
            return ImageContent(
                page_number=page_number,
                images=[],
                image_count=0,
                total_size_kb=0.0,
                status="failed",
                error_message=str(e)
            )

    def to_dict(self, image_content: ImageContent) -> Dict[str, Any]:
        """Converts ImageContent dataclass to a dictionary."""
        return asdict(image_content)