"""
JSON Formatter Agent - Structures raw agent outputs into clean, formatted JSON
"""

import google.generativeai as genai
import json
from typing import Dict, Any, List
from dataclasses import dataclass, asdict


@dataclass
class FormattedOutput:
    """Structured formatted JSON output"""
    document_name: str
    document_type: str
    formatted_data: Dict[str, Any]
    status: str
    error_message: str = None


class JSONFormatterAgent:
    """
    Agent responsible for taking raw combined outputs from all agents
    and formatting them into a clean, structured JSON format using LLM intelligence.
    """
    
    def __init__(self, api_key: str, model_name: str = "gemini-2.0-flash-exp"):
        """
        Initialize JSON Formatter Agent with Gemini API.
        
        Args:
            api_key: Google Gemini API key
            model_name: Gemini model to use
        """
        self.name = "JSONFormatterAgent"
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        
        # Specialized prompt for JSON formatting
        self.formatting_prompt = """You are a document structuring specialist. Your job is to analyze raw extracted data and format it into a clean, logical JSON structure.

INPUT: You will receive raw data extracted from a PDF document containing:
- Text content from multiple pages
- Tables with structured data
- Image metadata

YOUR TASK:
1. Analyze all the content intelligently
2. Identify logical sections, headers, and structure
3. Group related content together
4. Create a well-organized JSON output

OUTPUT FORMAT (adapt based on document type):

For CONTRACTS/LEGAL DOCUMENTS:
{
  "document_type": "Contract",
  "metadata": {
    "parties": ["Party A", "Party B"],
    "date": "2024-01-01",
    "reference_number": "..."
  },
  "sections": [
    {
      "section_name": "Agreement Terms",
      "content": "...",
      "clauses": [
        {"clause_id": "1.1", "title": "...", "content": "..."}
      ]
    }
  ],
  "tables": [...],
  "signatures": [...]
}

For INVOICES/BILLS:
{
  "document_type": "Invoice",
  "invoice_details": {
    "invoice_number": "...",
    "date": "...",
    "due_date": "...",
    "vendor": {...},
    "customer": {...}
  },
  "line_items": [
    {"item": "...", "quantity": "...", "price": "...", "total": "..."}
  ],
  "summary": {
    "subtotal": "...",
    "tax": "...",
    "total": "..."
  }
}

For REPORTS/DOCUMENTS:
{
  "document_type": "Report",
  "title": "...",
  "author": "...",
  "date": "...",
  "sections": [
    {
      "section_title": "...",
      "subsections": [
        {"title": "...", "content": "..."}
      ],
      "tables": [...],
      "figures": [...]
    }
  ],
  "conclusion": "..."
}

RULES:
1. Analyze the content to determine document type
2. Extract key metadata (dates, names, IDs, amounts)
3. Organize content into logical sections
4. Preserve all tables in structured format
5. Include image metadata where relevant
6. Return ONLY valid JSON
7. Be intelligent - adapt structure to document type
8. Don't lose any important information

IMPORTANT:
- Return ONLY the JSON, no explanations
- Ensure proper JSON formatting (valid quotes, commas, brackets)
- Use consistent field names
- Group related information together

Now analyze the following document data and create a structured JSON:"""
    
    def format_with_gemini(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Use Gemini LLM to intelligently format raw data into structured JSON.
        
        Args:
            raw_data: Raw combined output from all agents
            
        Returns:
            Formatted JSON structure
        """
        try:
            # Convert raw data to string for LLM
            raw_data_str = json.dumps(raw_data, indent=2)
            
            # Create full prompt
            full_prompt = f"{self.formatting_prompt}\n\n```json\n{raw_data_str}\n```"
            
            # Call Gemini
            response = self.model.generate_content(
                full_prompt,
                generation_config={
                    "temperature": 0.1,  # Low temperature for consistent formatting
                    "top_p": 0.95,
                    "top_k": 40,
                    "max_output_tokens": 8192,
                }
            )
            
            # Parse response
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
            formatted_json = json.loads(response_text)
            
            return formatted_json
            
        except json.JSONDecodeError as e:
            print(f"⚠️  JSON parsing error in formatter: {str(e)}")
            print(f"   Raw response: {response_text[:300]}...")
            # Return raw data if formatting fails
            return {
                "formatting_error": "Failed to parse LLM response",
                "raw_data": raw_data
            }
        except Exception as e:
            raise Exception(f"Gemini formatting error: {str(e)}")
    
    def process_document(self, 
                        raw_combined_data: Dict[str, Any],
                        document_name: str = "document.pdf",
                        document_type: str = "Unknown") -> FormattedOutput:
        """
        Main method to process and format the complete document data.
        This is called by Heart LLM after all pages are processed.
        
        Args:
            raw_combined_data: Combined output from all agents (all pages)
            document_name: Name of the document
            document_type: Type of document
        
        Returns:
            FormattedOutput with clean, structured JSON
        """
        try:
            print(f"\n🎨 JSON Formatter Agent - Formatting document...")
            
            # Step 1: Use Gemini to intelligently format the data
            formatted_data = self.format_with_gemini(raw_combined_data)
            
            # Step 2: Add document metadata if not present
            if "document_name" not in formatted_data:
                formatted_data["document_name"] = document_name
            if "document_type" not in formatted_data:
                formatted_data["document_type"] = document_type
            
            # Step 3: Create structured output
            formatted_output = FormattedOutput(
                document_name=document_name,
                document_type=document_type,
                formatted_data=formatted_data,
                status="success"
            )
            
            print(f"   ✓ Formatting completed successfully")
            
            return formatted_output
            
        except Exception as e:
            # Handle errors gracefully
            print(f"⚠️  JSON Formatter Agent error: {str(e)}")
            
            # Return raw data with error status
            return FormattedOutput(
                document_name=document_name,
                document_type=document_type,
                formatted_data=raw_combined_data,
                status="failed",
                error_message=str(e)
            )
    
    def to_dict(self, formatted_output: FormattedOutput) -> Dict[str, Any]:
        """Convert FormattedOutput to dictionary"""
        return asdict(formatted_output)
    
    def save_to_file(self, formatted_output: FormattedOutput, output_path: str):
        """
        Save formatted JSON to a file.
        
        Args:
            formatted_output: FormattedOutput object
            output_path: Output file path
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(formatted_output.formatted_data, f, indent=2, ensure_ascii=False)
        print(f"💾 Formatted JSON saved to: {output_path}")


# Example Usage (Testing individually)
if __name__ == "__main__":
    
    # Configuration
    GEMINI_API_KEY = "YOUR_API_KEY_HERE"  # Replace with your API key
    
    # Sample raw data (simulating output from Heart LLM)
    raw_data = {
        "pages": [
            {
                "page": 1,
                "text": {
                    "raw_text": "SERVICE AGREEMENT\n\nThis agreement is made between ABC Corp and XYZ Ltd on January 15, 2024.\n\nTerms and Conditions:\n1. Service Period: 12 months\n2. Payment: Monthly installments",
                    "word_count": 28,
                    "status": "success"
                },
                "tables": {
                    "tables": [
                        {
                            "table_id": 1,
                            "headers": ["Service", "Cost", "Duration"],
                            "rows": [
                                ["Consulting", "$5000", "6 months"],
                                ["Support", "$2000", "12 months"]
                            ],
                            "row_count": 2,
                            "col_count": 3
                        }
                    ],
                    "table_count": 1,
                    "status": "success"
                },
                "images": {
                    "images": [],
                    "image_count": 0,
                    "status": "success"
                }
            }
        ]
    }
    
    # Initialize agent
    agent = JSONFormatterAgent(api_key=GEMINI_API_KEY)
    
    # Format the document
    print(f"🎨 Testing JSON Formatter Agent...")
    result = agent.process_document(
        raw_combined_data=raw_data,
        document_name="service_agreement.pdf",
        document_type="Contract"
    )
    # Save to file
    agent.save_to_file(result, "formatted_output.json")