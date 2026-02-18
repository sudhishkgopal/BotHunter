"""
BotHunter CLI — professional audit interface.

Usage:
    python cli.py audit
    python cli.py audit --threshold 20
    python cli.py audit --threshold 8 --label "nightly scan"
"""

import json
from datetime import datetime, timezone

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

import networkx as nx

from database import SessionLocal, init_db
from models import AnalysisResult, Relationship, User

app = typer.Typer(
    name="bothunter",
    help="BotHunter — graph-based bot detection engine.",
    no_args_is_help=True,
)
console = Console()

