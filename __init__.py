"""
BotHunter — Graph-based social media bot detection.

Public API surface for library use:

    from processor import compute_features, classify_nodes, build_graph
    from database import SessionLocal, init_db
    from models import User, Relationship, AnalysisResult

Install:
    pip install .                  # core runtime
    pip install ".[dev]"           # + pytest, ruff, mypy
    pip install ".[ai]"            # + LLM provider SDKs
    pip install ".[gpu]"           # + CUDA-accelerated graph libs
    pip install ".[deploy]"        # + gunicorn, psycopg2

CLI:
    bothunter audit --threshold 20

API:
    uvicorn main:app --reload --port 8000

Dashboard:
    python -m streamlit run app.py
"""

__version__ = "1.0.0"
__author__ = "Sudhish Gopalakrishnan"
__license__ = "MIT"
