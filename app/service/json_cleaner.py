import re
from copy import deepcopy

class UniversalJSONCleaner:
    def __init__(self):
        self.stats = {
            'sections_processed': 0,
            'sections_removed': 0,
            'clauses_processed': 0,
            'clauses_removed': 0
        }
        # Valid clause_id: numbers, sub-numbers, bullets, roman numerals
        self.valid_clause_id = re.compile(
            r'^(\d+(\.\d+)*|[ivxlcdmIVXLCDM]+|[a-zA-Z]\.|[-*•])$'
        )
        self.placeholder_patterns = [
            r'Confidential',
            r'Execution Version',
            r'INFOSYS',
            r'^(\d+)$'  # single numbers
        ]

    def basic_clean(self, text):
        if not isinstance(text, str) or not text.strip():
            return text
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n\s*\n+', '\n\n', text)
        text = re.sub(r'\s+([.,;:)\]])', r'\1', text)
        text = re.sub(r'([(\[])\s+', r'\1', text)
        return text.strip()

    def is_valid_clause(self, clause_id, content):
        if not clause_id or not content:
            return False
        content = content.strip()
        # Remove short content
        if len(content) < 20:
            return False
        # Remove placeholders
        for pattern in self.placeholder_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                return False
        # Validate clause_id
        if not self.valid_clause_id.match(clause_id.strip()):
            return False
        return True

    def clean_clause(self, clause):
        self.stats['clauses_processed'] += 1
        clause_id = clause.get('clause_id', '')
        content = clause.get('content', '')
        if self.is_valid_clause(clause_id, content):
            clause['content'] = self.basic_clean(content)
            return clause
        else:
            self.stats['clauses_removed'] += 1
            return None

    def clean_section(self, section):
        self.stats['sections_processed'] += 1
        section_name = section.get('section_name', '')
        clauses = section.get('clauses', [])
        cleaned_clauses = [self.clean_clause(c) for c in clauses]
        cleaned_clauses = [c for c in cleaned_clauses if c]  # remove None

        # Remove section if untitled or no valid clauses
        if not cleaned_clauses or "Untitled" in section_name:
            self.stats['sections_removed'] += 1
            return None

        section['clauses'] = cleaned_clauses
        section['section_name'] = self.basic_clean(section_name)
        return section

    def clean_json(self, data):
        data_copy = deepcopy(data)
        sections = data_copy.get('sections', [])
        cleaned_sections = [self.clean_section(s) for s in sections]
        cleaned_sections = [s for s in cleaned_sections if s]  # remove None
        data_copy['sections'] = cleaned_sections
        return data_copy

    def get_stats(self):
        return self.stats
