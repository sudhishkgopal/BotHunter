"""
BotHunter CLI — professional audit interface.
"""

import json
from datetime import datetime, timezone

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table


from database import SessionLocal, init_db
from models import AnalysisResult, Relationship, User
from processor import build_graph, compute_features, classify_nodes

app = typer.Typer(
    name="bothunter",
    help="BotHunter — graph-based bot detection engine.",
    no_args_is_help=True,
)
console = Console()

