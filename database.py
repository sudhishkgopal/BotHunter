"""
Database initialisation utility for BotHunter.

Usage:
    from database import engine, SessionLocal, init_db

    # Create all tables (idempotent)
    init_db()

    # Obtain a session
    with SessionLocal() as session:
        ...
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from models import Base

import os
DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///bothunter.db")



engine = create_engine(
    DATABASE_URL,
    echo=False,
    connect_args={"check_same_thread": False},  # required for SQLite
)

SessionLocal = sessionmaker(bind=engine)


def init_db() -> None:
    """Create every table defined in models.py (safe to call repeatedly)."""
    Base.metadata.create_all(bind=engine)


if __name__ == "__main__":
    init_db()
    print(f"Database initialised at {DATABASE_URL}")