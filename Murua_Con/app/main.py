from fastapi import FastAPI
from app.database import database, models
from app.routes import CURD_operation,generate_summary,Thulasi_AI
import logging

# Initialize FastAPI app
app = FastAPI(
    title="API for analysing the Contract Document",
    description="Your smart AI Intelligent For Contract Document Analyser",
    version="1.0.0",
)
# Create tables
models.Base.metadata.create_all(bind=database.engine)

# Register routers
app.include_router(Thulasi_AI.router)
app.include_router(CURD_operation.router)
app.include_router(generate_summary.router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=1177, reload=True)
