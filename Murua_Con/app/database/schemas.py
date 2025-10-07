from pydantic import BaseModel
from typing import Dict, Optional
from datetime import datetime

class TemplateBase(BaseModel):
    template_type: str
    sector: str
    keywords: Dict

class TemplateCreate(TemplateBase):
    pass

class TemplateUpdate(BaseModel):
    template_type: Optional[str] = None
    sector: Optional[str] = None
    keywords: Optional[Dict] = None

class TemplateResponse(TemplateBase):
    id: int
    created_on: datetime
    updated_on: datetime

    class Config:
        orm_mode = True
