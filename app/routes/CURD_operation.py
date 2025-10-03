from app.database import schemas,models,database
from app.service import services
from fastapi import FastAPI,Depends,APIRouter
from sqlalchemy.orm import Session

router = APIRouter(prefix="/templates", tags=["Templates"]) 

@router.post("/templates/", response_model=schemas.TemplateResponse)
def create_template(template: schemas.TemplateCreate, db: Session = Depends(database.get_db)):
    return services.create_template_service(db, template)

@router.get("/templates/{template_id}", response_model=schemas.TemplateResponse)
def read_template(template_id: int, db: Session = Depends(database.get_db)):
    return services.get_template_service(db, template_id)

@router.get("/templates/", response_model=list[schemas.TemplateResponse])
def read_all_templates(db: Session = Depends(database.get_db)):
    return services.get_all_templates_service(db)

@router.put("/templates/{template_id}", response_model=schemas.TemplateResponse)
def update_template(template_id: int, update_data: schemas.TemplateUpdate, db: Session = Depends(database.get_db)):
    return services.update_template_service(db, template_id, update_data)

@router.delete("/templates/{template_id}")
def delete_template(template_id: int, db: Session = Depends(database.get_db)):
    return services.delete_template_service(db, template_id)