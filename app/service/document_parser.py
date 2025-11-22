import PyPDF2
from io import BytesIO

class DocumentParser:
    """
    Extracts clean text from PDF bytes.
    Works with FastAPI upload system.
    """

    def extract_text(self, pdf_bytes: bytes) -> str:
        """
        Convert PDF bytes → readable text
        """
        try:
            pdf_reader = PyPDF2.PdfReader(BytesIO(pdf_bytes))
            full_text = ""

            for page in pdf_reader.pages:
                extracted = page.extract_text()
                if extracted:
                    # Clean spacing and newlines
                    cleaned = " ".join(extracted.split())
                    full_text += cleaned + "\n\n"

            return full_text.strip()

        except Exception as e:
            print("PDF parsing error:", str(e))
            return ""
