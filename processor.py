"""
Multi-feature bot detection engine for BotHunter.

Pipeline:
  1. Load all users and edges from SQLite into a NetworkX DiGraph
  2. Compute per-node features:
       - K-Core number (structural density)
       - Local clustering coefficient (neighbourhood interconnectedness)
       - In-degree / out-degree (follower asymmetry)
  3. Apply a weighted risk scoring formula to classify each node
  4. Persist results to the analysis_results table
"""
import argparse
import json
import logging
from datetime import datetime, timezone

import networkx as nx

from database import SessionLocal, init_db
from models import AnalysisResult, Relationship, User

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Creating graph from database
# Using a directed graph to mimic social media relationships (person A follows person B != person B follows person A)

def build_graph(session) -> tuple[nx.DiGraph, dict[int, User]]:
    """
    Load every User and Relationship from SQLite into a NetworkX DiGraph.

    Returns:
        G         — the directed graph
        user_map  — {user.id: {"platform_id": ..., "username": ..., "is_bot": ...}}
    """
    users = session.query(User).all()
    edges = session.query(Relationship).all()

    user_map = {
        u.id: {
            "platform_id": u.platform_id,
            "username": u.username,
            "is_bot": u.is_bot,
        }
        for u in users
    }

    G = nx.DiGraph()
    G.add_nodes_from(user_map.keys())

    for e in edges:
        # If multiple edge types exist (follow & like), we only care about the "follow" edges for graph analysis.
        # Likes/comments add noise but don't represent connections.
        if e.relation_type == "follow":
            G.add_edge(e.source_user_id, e.target_user_id)

    log.info("Graph loaded: %d nodes, %d edges", G.number_of_nodes(), G.number_of_edges())
    return G, user_map