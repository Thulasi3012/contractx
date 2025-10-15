from fastapi import FastAPI
from app.database import database, models
from app.routes import Document_praser
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
app.include_router(Document_praser.router)

if __name__ == "__main__":
    import logging
    import uvicorn

    # --- Patch logging before uvicorn starts ---
    logging.basicConfig(level=logging.INFO, force=True)
    for handler in logging.root.handlers:
        try:
            handler.stream = open(1, "w", encoding="utf-8", closefd=False)  # reset to stdout
        except Exception:
            pass

    # --- Run the app without the reload subprocess ---
    uvicorn.run(
        "main:app",             # change to your module path
        host="0.0.0.0",
        port=8000,
        reload=False,           # avoid multiprocessing reload
        log_config=None,        # prevent uvicorn from overriding logging
    )

