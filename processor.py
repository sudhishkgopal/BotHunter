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

def compute_features(G: nx.DiGraph) -> dict[int, dict]:
    """
    Compute three graph-theoretic features for every node.

    Returns:
        {node_id: {"k_core": int, "clustering": float, "in_deg": int, "out_deg": int}}

    Notes:
        Engagement pods form dense connected cluster resulting in high k-core
        Human users have lower k-core usually
        Star bots have large degree but low core due to unidirectional connections

        Using undirected k-core. Now, mutual follows become an edge
      
        Bots who all know each other might have higher cluster coefficients
        Popular accounts & star bots have low clustering since their followers don't follow each other
        
        In-Degree(followers) / Out-Degree(following)

        Expected stereotypes:
            Healthy accounts: in =(approx.) out  (roughly balanced)
            Influencers:      in >> out (many followers, follow few)
            Star bots:        out >> in (mass-follow, almost no followers)
            Pod bots:         in =(approx.) out  but both are high AND mutual
    """

    # K-core on the undirected projection (mutual connections)
    G_undirected = G.to_undirected()
    core_numbers = nx.core_number(G_undirected)

    # Clustering on undirected graph 
    clustering = nx.clustering(G_undirected)

    features: dict[int, dict] = {}
    for node in G.nodes():
        features[node] = {
            "k_core": core_numbers.get(node, 0),
            "clustering": clustering.get(node, 0.0),
            "in_deg": G.in_degree(node),
            "out_deg": G.out_degree(node),
        }

    return features