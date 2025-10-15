"""
Text Agent - Extracts text content and contract information from PDF pages using Gemini Vision LLM
"""

import fitz  # PyMuPDF
import google.generativeai as genai
from PIL import Image
import io
import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict, field


@dataclass
class ContactInformation:
    """Contact details extracted from the document"""
    addresses: List[str] = field(default_factory=list)
    emails: List[str] = field(default_factory=list)
    phone_numbers: List[str] = field(default_factory=list)


@dataclass
class Party:
    """Party information (Buyer/Seller)"""
    name: str
    role: str  # "buyer" or "seller"
    address: Optional[str] = None
    contact_info: Optional[ContactInformation] = None


@dataclass
class Deadline:
    """Deadline or important date"""
    description: str
    date: str
    type: str  # "payment", "delivery", "milestone", "termination", etc.
    associated_clause: Optional[str] = None


@dataclass
class Obligation:
    """Contractual obligation"""
    party: str  # Which party has this obligation
    description: str
    type: str  # "payment", "delivery", "performance", "reporting", etc.
    deadline: Optional[str] = None
    associated_clause: Optional[str] = None


@dataclass
class Alert:
    """Important alerts or notices"""
    type: str  # "penalty", "termination_clause", "liability", "warranty", etc.
    description: str
    severity: str  # "high", "medium", "low"
    associated_clause: Optional[str] = None


@dataclass
class SubClause:
    """Sub-clause structure"""
    sub_clause_id: str
    sub_clause_number: Optional[str] = None
    content: str = ""
    level: int = 2  # Nesting level
    pattern_type: Optional[str] = None  # Pattern type detected


@dataclass
class Clause:
    """Clause structure with hierarchical sub-clauses"""
    clause_id: str
    clause_number: Optional[str] = None
    section_name: str = ""
    content: str = ""
    pattern_type: Optional[str] = None  # Pattern type detected
    sub_clauses: List[SubClause] = field(default_factory=list)


@dataclass
class PageSummary:
    """Summary of the page content"""
    main_topics: List[str] = field(default_factory=list)
    key_points: List[str] = field(default_factory=list)
    summary_text: str = ""


@dataclass
class TextContent:
    """Structured text output for a page - BACKWARD COMPATIBLE"""
    page_number: int
    raw_text: str
    word_count: int
    char_count: int
    status: str
    error_message: Optional[str] = None
    
    # NEW: Extended contract information
    buyer: Optional[Party] = None
    seller: Optional[Party] = None
    other_parties: List[Party] = field(default_factory=list)
    contact_information: ContactInformation = field(default_factory=ContactInformation)
    sections: List[Clause] = field(default_factory=list)
    obligations: List[Obligation] = field(default_factory=list)
    deadlines: List[Deadline] = field(default_factory=list)
    alerts: List[Alert] = field(default_factory=list)
    page_summary: PageSummary = field(default_factory=PageSummary)


class TextAgent:
    """
    Agent responsible for extracting text content and contract information from PDF pages.
    Maintains backward compatibility while adding contract extraction features.
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
        
        # Specialized prompt for contract information extraction
        self.extraction_prompt = """You are a specialized contract analysis AI. Extract structured information from this contract page and return it as a valid JSON object.

CRITICAL: Your response must be ONLY a valid JSON object, no additional text or markdown.

SECTION/CLAUSE DETECTION PATTERNS:
Identify headings using these patterns:
- Number followed by dot/bracket/space: "1. Introduction", "1) Introduction", "1 Introduction"
- Roman numerals: "I. INTRODUCTION", "I INTRODUCTION"
- Capital letters: "A. Background", "A) Background"
- Uppercase lines: "INTRODUCTION", "DEFINITIONS"
- Section/Article/Schedule keywords: "Section 1 – Definitions", "Article I – Definitions", "Schedule 1"

Identify clauses/subclauses using these patterns:
- Decimal numbering: "1.1 Clause", "1.1.1 Subclause", "1.1.1.1 Sub-subclause"
- Parenthesized numbers: "(1) Clause", "(2) Clause"
- Parenthesized letters: "(a) Clause", "(A) Clause"
- Roman numerals with dot: "i. Clause", "I. Clause"
- Bullets/dashes: "• Clause", "– Clause", "- Clause"
- Clause/Part keywords: "Clause 2.1", "Part 1 – Obligations"

HIERARCHY RULES:
- Main sections: 1, 2, 3 or I, II, III or A, B, C (level 1)
- Subclauses: 1.1, 1.2, 2.1 (level 2)
- Sub-subclauses: 1.1.1, 1.1.2, 2.1.1 (level 3)
- Further nesting: 1.1.1.1, (a), (i) (level 4+)

Extract the following information and return as JSON with this exact structure:

{
  "buyer": {
    "name": "Buyer company/person name or null",
    "role": "buyer",
    "address": "Full address if present or null",
    "contact_info": {
      "emails": ["email@example.com"],
      "phone_numbers": ["+1234567890"],
      "addresses": ["Address line"]
    }
  },
  "seller": {
    "name": "Seller company/person name or null",
    "role": "seller",
    "address": "Full address if present or null",
    "contact_info": {
      "emails": [],
      "phone_numbers": [],
      "addresses": []
    }
  },
  "other_parties": [],
  "contact_information": {
    "addresses": ["All addresses found"],
    "emails": ["All emails found"],
    "phone_numbers": ["All phone numbers found"]
  },
  "sections": [
    {
      "clause_id": "section_1",
      "clause_number": "1",
      "section_name": "Definitions",
      "content": "Full content of this section/clause",
      "pattern_type": "numbered_dot",
      "sub_clauses": [
        {
          "sub_clause_id": "section_1_1",
          "sub_clause_number": "1.1",
          "content": "Sub-clause content",
          "level": 2,
          "pattern_type": "decimal"
        },
        {
          "sub_clause_id": "section_1_1_1",
          "sub_clause_number": "1.1.1",
          "content": "Sub-sub-clause content",
          "level": 3,
          "pattern_type": "decimal"
        },
        {
          "sub_clause_id": "section_1_1_1_a",
          "sub_clause_number": "(a)",
          "content": "Further nested content",
          "level": 4,
          "pattern_type": "parenthesized_letter"
        }
      ]
    },
    {
      "clause_id": "section_A",
      "clause_number": "A",
      "section_name": "Background",
      "content": "Section content",
      "pattern_type": "letter_dot",
      "sub_clauses": []
    },
    {
      "clause_id": "article_I",
      "clause_number": "I",
      "section_name": "GENERAL PROVISIONS",
      "content": "Article content",
      "pattern_type": "roman_numeral",
      "sub_clauses": []
    }
  ],
  "obligations": [
    {
      "party": "buyer",
      "description": "Description of what the party must do",
      "type": "payment|delivery|performance|reporting",
      "deadline": "Specific date or timeframe",
      "associated_clause": "Clause reference"
    }
  ],
  "deadlines": [
    {
      "description": "What needs to be done by this date",
      "date": "Specific date or timeframe",
      "type": "payment|delivery|milestone|termination",
      "associated_clause": "Clause reference"
    }
  ],
  "alerts": [
    {
      "type": "penalty|termination_clause|liability|warranty|indemnification",
      "description": "Important notice or warning",
      "severity": "high|medium|low",
      "associated_clause": "Clause reference"
    }
  ],
  "page_summary": {
    "main_topics": ["Topic 1", "Topic 2"],
    "key_points": ["Key point 1", "Key point 2"],
    "summary_text": "Brief summary of the page content"
  },
  "raw_text": "All non-table text from the page"
}

IMPORTANT INSTRUCTIONS:
1. IGNORE all table content - do not extract text from tables
2. Extract only free-flowing text, headings, and paragraphs
3. Identify clause hierarchies using the patterns above
4. For each section/clause, capture:
   - clause_id: unique identifier (e.g., "section_1", "clause_1_1", "article_I")
   - clause_number: the visible number/letter (e.g., "1", "1.1", "A", "I", "(a)")
   - section_name: the heading text
   - content: full text content
   - pattern_type: which pattern was matched (e.g., "numbered_dot", "decimal", "roman_numeral", "uppercase_line", "parenthesized_letter")
   - For subclauses, also include level (2, 3, 4, etc.)
5. Maintain proper hierarchy in sub_clauses array
6. Identify buyer and seller parties clearly
7. Extract ALL contact information (addresses, emails, phones)
8. Identify obligations with responsible party and deadlines
9. Flag important alerts (penalties, termination clauses, liabilities)
10. Extract all important dates and deadlines
11. Return ONLY valid JSON - no markdown, no extra text
12. If certain information is not present, use null or empty arrays

PATTERN TYPE VALUES (use these exact strings):
- "numbered_dot" (1. Text)
- "numbered_bracket" (1) Text)
- "numbered_space" (1 Text)
- "roman_numeral_dot" (I. Text)
- "roman_numeral_space" (I Text)
- "letter_dot" (A. Text)
- "letter_bracket" (A) Text)
- "uppercase_line" (UPPERCASE TEXT)
- "section_keyword" (Section 1 – Text)
- "article_keyword" (Article I – Text)
- "schedule_keyword" (Schedule 1)
- "decimal" (1.1 Text, 1.1.1 Text)
- "parenthesized_number" ((1) Text)
- "parenthesized_letter_lower" ((a) Text)
- "parenthesized_letter_upper" ((A) Text)
- "roman_numeral_clause" (i. Text, I. Text)
- "bullet" (• Text)
- "dash" (– Text, - Text)
- "clause_keyword" (Clause 2.1)
- "part_keyword" (Part 1 – Text)

Extract the information now:"""
    
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
    
    def extract_text_with_gemini(self, image: Image.Image) -> str:
        """
        Extract text from a page image using Gemini vision model.
        DEPRECATED: Use extract_contract_data_with_gemini for full features.
        Kept for backward compatibility.
        
        Args:
            image: PIL Image of the PDF page
            
        Returns:
            Extracted text as a string
        """
        try:
            # Use the full extraction and just return raw_text
            data = self._extract_gemini_response(image)
            return data.get("raw_text", "")
        except Exception as e:
            raise Exception(f"Gemini API error: {str(e)}")
    
    def _extract_gemini_response(self, image: Image.Image) -> Dict[str, Any]:
        """
        Internal method to extract structured contract data from Gemini.
        
        Args:
            image: PIL Image of the PDF page
            
        Returns:
            Dictionary with structured contract data
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
            
            # Remove markdown code blocks if present
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.startswith("```"):
                response_text = response_text[3:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            
            response_text = response_text.strip()
            
            # Parse JSON
            data = json.loads(response_text)
            return data
            
        except json.JSONDecodeError as e:
            raise Exception(f"Failed to parse Gemini JSON response: {str(e)}")
        except Exception as e:
            raise Exception(f"Gemini API error: {str(e)}")
    
    def calculate_stats(self, text: str) -> Dict[str, int]:
        """Calculate text statistics"""
        words = text.split()
        return {
            "word_count": len(words),
            "char_count": len(text)
        }
    
    def _parse_contract_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse the extracted JSON data into structured objects.
        
        Args:
            data: Dictionary from Gemini response
            
        Returns:
            Dictionary with structured contract components
        """
        result = {}
        
        # Parse buyer
        if data.get("buyer") and data["buyer"].get("name"):
            buyer_data = data["buyer"]
            result["buyer"] = Party(
                name=buyer_data.get("name", ""),
                role="buyer",
                address=buyer_data.get("address"),
                contact_info=ContactInformation(
                    addresses=buyer_data.get("contact_info", {}).get("addresses", []),
                    emails=buyer_data.get("contact_info", {}).get("emails", []),
                    phone_numbers=buyer_data.get("contact_info", {}).get("phone_numbers", [])
                ) if buyer_data.get("contact_info") else None
            )
        
        # Parse seller
        if data.get("seller") and data["seller"].get("name"):
            seller_data = data["seller"]
            result["seller"] = Party(
                name=seller_data.get("name", ""),
                role="seller",
                address=seller_data.get("address"),
                contact_info=ContactInformation(
                    addresses=seller_data.get("contact_info", {}).get("addresses", []),
                    emails=seller_data.get("contact_info", {}).get("emails", []),
                    phone_numbers=seller_data.get("contact_info", {}).get("phone_numbers", [])
                ) if seller_data.get("contact_info") else None
            )
        
        # Parse other parties
        result["other_parties"] = []
        for party_data in data.get("other_parties", []):
            if party_data.get("name"):
                party = Party(
                    name=party_data.get("name", ""),
                    role=party_data.get("role", "other"),
                    address=party_data.get("address"),
                    contact_info=ContactInformation(
                        addresses=party_data.get("contact_info", {}).get("addresses", []),
                        emails=party_data.get("contact_info", {}).get("emails", []),
                        phone_numbers=party_data.get("contact_info", {}).get("phone_numbers", [])
                    ) if party_data.get("contact_info") else None
                )
                result["other_parties"].append(party)
        
        # Parse contact information
        contact_info_data = data.get("contact_information", {})
        result["contact_information"] = ContactInformation(
            addresses=contact_info_data.get("addresses", []),
            emails=contact_info_data.get("emails", []),
            phone_numbers=contact_info_data.get("phone_numbers", [])
        )
        
        # Parse sections/clauses
        result["sections"] = []
        for section_data in data.get("sections", []):
            sub_clauses = []
            for sub_data in section_data.get("sub_clauses", []):
                sub_clause = SubClause(
                    sub_clause_id=sub_data.get("sub_clause_id", ""),
                    sub_clause_number=sub_data.get("sub_clause_number"),
                    content=sub_data.get("content", ""),
                    level=sub_data.get("level", 2),
                    pattern_type=sub_data.get("pattern_type")
                )
                sub_clauses.append(sub_clause)
            
            clause = Clause(
                clause_id=section_data.get("clause_id", ""),
                clause_number=section_data.get("clause_number"),
                section_name=section_data.get("section_name", ""),
                content=section_data.get("content", ""),
                pattern_type=section_data.get("pattern_type"),
                sub_clauses=sub_clauses
            )
            result["sections"].append(clause)
        
        # Parse obligations
        result["obligations"] = []
        for obl_data in data.get("obligations", []):
            obligation = Obligation(
                party=obl_data.get("party", ""),
                description=obl_data.get("description", ""),
                type=obl_data.get("type", ""),
                deadline=obl_data.get("deadline"),
                associated_clause=obl_data.get("associated_clause")
            )
            result["obligations"].append(obligation)
        
        # Parse deadlines
        result["deadlines"] = []
        for dl_data in data.get("deadlines", []):
            deadline = Deadline(
                description=dl_data.get("description", ""),
                date=dl_data.get("date", ""),
                type=dl_data.get("type", ""),
                associated_clause=dl_data.get("associated_clause")
            )
            result["deadlines"].append(deadline)
        
        # Parse alerts
        result["alerts"] = []
        for alert_data in data.get("alerts", []):
            alert = Alert(
                type=alert_data.get("type", ""),
                description=alert_data.get("description", ""),
                severity=alert_data.get("severity", "medium"),
                associated_clause=alert_data.get("associated_clause")
            )
            result["alerts"].append(alert)
        
        # Parse page summary
        summary_data = data.get("page_summary", {})
        result["page_summary"] = PageSummary(
            main_topics=summary_data.get("main_topics", []),
            key_points=summary_data.get("key_points", []),
            summary_text=summary_data.get("summary_text", "")
        )
        
        return result
    
    def process_page(self, page: fitz.Page, page_number: int = None) -> TextContent:
        """
        Main method to process a page and extract text content.
        This is called by Heart LLM for each page.
        BACKWARD COMPATIBLE with original TextAgent interface.
        
        Args:
            page: PyMuPDF Page object (passed from Heart LLM)
            page_number: Page number (1-indexed)
        
        Returns:
            TextContent with structured text data and contract information
        """
        # Determine page number
        if page_number is None:
            page_number = page.number + 1
        
        try:
            # Step 1: Convert page to image for LLM processing
            image = self.pdf_page_to_image(page)
            
            # Step 2: Extract structured data using Gemini Vision
            data = self._extract_gemini_response(image)
            
            # Step 3: Parse contract data
            parsed_data = self._parse_contract_data(data)
            
            # Step 4: Get raw text and calculate statistics
            raw_text = data.get("raw_text", "")
            stats = self.calculate_stats(raw_text)
            
            # Step 5: Create structured output (BACKWARD COMPATIBLE)
            text_content = TextContent(
                page_number=page_number,
                raw_text=raw_text,
                word_count=stats["word_count"],
                char_count=stats["char_count"],
                status="success",
                # NEW FIELDS
                buyer=parsed_data.get("buyer"),
                seller=parsed_data.get("seller"),
                other_parties=parsed_data.get("other_parties", []),
                contact_information=parsed_data.get("contact_information", ContactInformation()),
                sections=parsed_data.get("sections", []),
                obligations=parsed_data.get("obligations", []),
                deadlines=parsed_data.get("deadlines", []),
                alerts=parsed_data.get("alerts", []),
                page_summary=parsed_data.get("page_summary", PageSummary())
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
    
    def to_json(self, text_content: TextContent, indent: int = 2) -> str:
        """
        Convert TextContent to formatted JSON string.
        
        Args:
            text_content: TextContent object
            indent: JSON indentation level
            
        Returns:
            JSON string
        """
        return json.dumps(self.to_dict(text_content), indent=indent)
