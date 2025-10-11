"""
Image Agent - Detects and extracts images from PDF pages
"""

import fitz  # PyMuPDF
import base64
from typing import Dict, Any, List
from dataclasses import dataclass, asdict
from PIL import Image
import io


@dataclass
class ImageData:
    """Represents a single extracted image"""
    image_id: int
    bbox: tuple  # (x0, y0, x1, y1)
    width: int
    height: int
    format: str  # jpeg, png, etc.
    size_kb: float
    base64_data: str  # Base64 encoded image


@dataclass
class ImageContent:
    """Structured image output for a page"""
    page_number: int
    images: List[Dict[str, Any]]
    image_count: int
    total_size_kb: float
    status: str
    error_message: str = None


class ImageAgent:
    """
    Agent responsible for detecting and extracting images from PDF pages.
    Returns images as base64-encoded strings with metadata.
    """
    
    def __init__(self):
        """Initialize Image Agent"""
        self.name = "ImageAgent"
    
    def extract_images_from_page(self, page: fitz.Page) -> List[ImageData]:
        """
        Extract all images from a PDF page.
        
        Args:
            page: PyMuPDF Page object
            
        Returns:
            List of ImageData objects
        """
        images = []
        image_list = page.get_images(full=True)
        
        for img_index, img_info in enumerate(image_list):
            try:
                # Get image reference
                xref = img_info[0]
                
                # Extract image bytes
                base_image = page.parent.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                
                # Get image dimensions and metadata
                pil_image = Image.open(io.BytesIO(image_bytes))
                width, height = pil_image.size
                
                # Calculate size in KB
                size_kb = len(image_bytes) / 1024
                
                # Get bounding box (position on page)
                bbox = self._get_image_bbox(page, xref)
                
                # Convert to base64 for JSON storage
                base64_data = base64.b64encode(image_bytes).decode('utf-8')
                
                # Create ImageData object
                image_data = ImageData(
                    image_id=img_index + 1,
                    bbox=bbox,
                    width=width,
                    height=height,
                    format=image_ext,
                    size_kb=round(size_kb, 2),
                    base64_data=base64_data
                )
                
                images.append(image_data)
                
            except Exception as e:
                print(f"⚠️  Error extracting image {img_index + 1}: {str(e)}")
                continue
        
        return images
    
    def _get_image_bbox(self, page: fitz.Page, xref: int) -> tuple:
        """
        Get the bounding box (position) of an image on the page.
        
        Args:
            page: PyMuPDF Page object
            xref: Image reference number
            
        Returns:
            Tuple (x0, y0, x1, y1) or (0, 0, 0, 0) if not found
        """
        try:
            # Get all image instances on the page
            image_instances = page.get_image_bbox(xref)
            if image_instances:
                # Return the bounding box of the first instance
                bbox = image_instances
                return tuple(bbox)
        except:
            pass
        
        return (0, 0, 0, 0)
    
    def process_page(self, page: fitz.Page, page_number: int = None) -> ImageContent:
        """
        Main method to process a page and extract image content.
        This is called by Heart LLM for each page.
        
        Args:
            page: PyMuPDF Page object (passed from Heart LLM)
            page_number: Page number (1-indexed)
        
        Returns:
            ImageContent with structured image data
        """
        # Determine page number
        if page_number is None:
            page_number = page.number + 1
        
        try:
            # Step 1: Extract images from page
            image_data_list = self.extract_images_from_page(page)
            
            # Step 2: Calculate total size
            total_size = sum(img.size_kb for img in image_data_list)
            
            # Step 3: Convert to dictionaries
            images_dict = [asdict(img) for img in image_data_list]
            
            # Step 4: Create structured output
            image_content = ImageContent(
                page_number=page_number,
                images=images_dict,
                image_count=len(images_dict),
                total_size_kb=round(total_size, 2),
                status="success"
            )
            
            return image_content
            
        except Exception as e:
            # Handle errors gracefully
            print(f"⚠️  Image Agent error on page {page_number}: {str(e)}")
            
            return ImageContent(
                page_number=page_number,
                images=[],
                image_count=0,
                total_size_kb=0.0,
                status="failed",
                error_message=str(e)
            )
    
    def to_dict(self, image_content: ImageContent) -> Dict[str, Any]:
        """Convert ImageContent to dictionary for Heart LLM"""
        return asdict(image_content)
    
    def save_image_to_file(self, base64_data: str, output_path: str):
        """
        Save a base64 encoded image to a file.
        
        Args:
            base64_data: Base64 encoded image string
            output_path: Output file path
        """
        image_bytes = base64.b64decode(base64_data)
        with open(output_path, 'wb') as f:
            f.write(image_bytes)

