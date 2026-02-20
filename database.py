"""
Database initialisation utility for BotHunter.
Reads database_path from config.json, falls back to bothunter.db.
"""

import json
import os

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from models import Base


def _load_db_path() -> str:
    """Resolve DB path: env var > config.json > default."""
    env = os.environ.get("DATABASE_URL")
    if env:
        return env

    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            cfg = json.load(f)
        return f"sqlite:///{cfg.get('database_path', 'bothunter.db')}"

    return "sqlite:///bothunter.db"


DATABASE_URL = _load_db_path()

engine = create_engine(
    DATABASE_URL,
    echo=False,
    connect_args={"check_same_thread": False},
)

SessionLocal = sessionmaker(bind=engine)


def init_db() -> None:
    """Create every table defined in models.py (safe to call repeatedly)."""
    Base.metadata.create_all(bind=engine)


if __name__ == "__main__":
    init_db()
    print(f"Database initialised at {DATABASE_URL}")
