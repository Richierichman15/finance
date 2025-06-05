#!/usr/bin/env python3
from app.database import engine
from app.models.database_models import Base

def init_database():
    print("Creating database tables...")
    Base.metadata.create_all(bind=engine)
    print("Database tables created successfully!")

if __name__ == "__main__":
    init_database()