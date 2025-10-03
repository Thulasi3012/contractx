from sqlalchemy.orm import Session
from fastapi import HTTPException
from app.service import crud
from app.database import database, models, schemas

def create_template_service(db: Session, template: schemas.TemplateCreate):
    return crud.create_template(db, template)

def get_template_service(db: Session, template_id: int):
    template = crud.get_template(db, template_id)
    if not template:
        raise HTTPException(status_code=404, detail="Template not found")
    return template

def get_all_templates_service(db: Session):
    return crud.get_templates(db)

def update_template_service(db: Session, template_id: int, update_data: schemas.TemplateUpdate):
    template = crud.update_template(db, template_id, update_data)
    if not template:
        raise HTTPException(status_code=404, detail="Template not found")
    return template

def delete_template_service(db: Session, template_id: int):
    template = crud.delete_template(db, template_id)
    if not template:
        raise HTTPException(status_code=404, detail="Template not found")
    return {"message": f"Template {template_id} deleted successfully"}
