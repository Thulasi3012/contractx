from sqlalchemy import create_engine

DATABASE_URL = "postgresql+psycopg2://Thulasi:Thulasi%403012@localhost:5432/Contract"

engine = create_engine(DATABASE_URL)

# test connection
with engine.connect() as conn:
    print("✅ Connected to PostgreSQL:", conn)
