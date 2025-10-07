from sqlalchemy.orm import Session
from app.database import models, schemas

# Create
def create_template(db: Session, template: schemas.TemplateCreate):
    db_template = models.Template(
        template_type=template.template_type,
        sector=template.sector,
        keywords=template.keywords
    )
    db.add(db_template)
    db.commit()
    db.refresh(db_template)
    return db_template

# Read (by ID)
def get_template(db: Session, template_id: int):
    return db.query(models.Template).filter(models.Template.id == template_id).first()

# Read all
def get_templates(db: Session):
    return db.query(models.Template).all()

# Update
def update_template(db: Session, template_id: int, update_data: schemas.TemplateUpdate):
    template = db.query(models.Template).filter(models.Template.id == template_id).first()
    if not template:
        return None

    if update_data.template_type is not None:
        template.template_type = update_data.template_type
    if update_data.sector is not None:
        template.sector = update_data.sector
    if update_data.keywords is not None:
        template.keywords = update_data.keywords

    db.commit()
    db.refresh(template)
    return template

# Delete
def delete_template(db: Session, template_id: int):
    template = db.query(models.Template).filter(models.Template.id == template_id).first()
    if not template:
        return None
    db.delete(template)
    db.commit()
    return template
