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

# Risk calculation and classification
#
#  The risk score combines multiple weak signals into one strong classifier.
#  No single metric is sufficient:
#
#    - High k-core alone? Could be a tight friend group.
#    - High out-degree alone? Could be a social butterfly.
#    - Low clustering alone? Could be a new user.
#
#  HIGH out-degree + LOW clustering + LOW in-degree together = strong bot signal.

CLASSIFICATION_BOT = "bot"
CLASSIFICATION_POD = "engagement_pod"
CLASSIFICATION_INFLUENCER = "influencer"
CLASSIFICATION_NORMAL = "normal"


def classify_nodes(
    features: dict[int, dict],
    k_threshold: int,
    ) -> dict[int, dict]:
    """
    Assign a risk score (0.0 - 1.0) and a label to every node.

    Scoring weights:
      - Degree asymmetry  (0.40) — out/in ratio; bots skew heavily outward
      - Clustering inverse (0.35) — low clustering in a high-degree node is suspicious
      - K-core density    (0.25) — high core number indicates pod membership

    The final label is decided by combining the score with feature context.
    """
    if not features:
        return {}

    # Normalisation bounds to consider outliers
    max_out = max((f["out_deg"] for f in features.values()), default=1) or 1 
    max_in = max((f["in_deg"] for f in features.values()), default=1) or 1 
    max_core = max((f["k_core"] for f in features.values()), default=1) or 1

    results: dict[int, dict] = {}

    for node, f in features.items():
        in_deg = f["in_deg"]
        out_deg = f["out_deg"]
        clust = f["clustering"]
        k_core = f["k_core"]

        # DEGREE ASYMMETRY 
        # A balanced account scores ~0.  A star bot (out >> in) scores ~1.
        # Formula: out_ratio - in_ratio, clamped to [0, 1]
        out_ratio = out_deg / max_out
        in_ratio = in_deg / max_in
        asymmetry = max(0.0, min(1.0, out_ratio - in_ratio + 0.5)) #round up to give buffer for normal accounts
        

        # CLUSTERING INVERSE
        # High clustering -> low suspicion. Low clustering -> high suspicion.
        # BUT only if the node has connections (isolated nodes get 0).
        if in_deg + out_deg > 0:
            clust_inv = 1.0 - clust
        else:
            clust_inv = 0.0

        # K-CORE DENSITY
        core_norm = k_core / max_core

        # RISK SCORE
        W_ASYMMETRY = 0.40
        W_CLUSTERING = 0.35
        W_KCORE = 0.25

        risk = (
            W_ASYMMETRY * asymmetry
            + W_CLUSTERING * clust_inv
            + W_KCORE * core_norm
        )

        # ASSIGN LABEL
        # Scores consider suspicion level
        # The feature context determines the type of suspect.

        if k_core >= k_threshold and clust > 0.6:
            # Dense mutual cluster + high clustering = engagement pod
            label = CLASSIFICATION_POD
        elif out_deg > in_deg * 5 and clust < 0.2:
            # Massive outgoing, almost no incoming, no clustering = star bot
            label = CLASSIFICATION_BOT
        elif risk > 0.7 and clust < 0.3:
            # High risk + low clustering = likely bot (catches edge cases)
            label = CLASSIFICATION_BOT
        elif in_deg > out_deg * 3 and clust < 0.3:
            # High in-degree, low clustering = celebrity / influencer
            label = CLASSIFICATION_INFLUENCER
        elif in_deg > out_deg * 3 and clust >= 0.3:
            # High in-degree WITH high clustering = authority in a community
            label = CLASSIFICATION_INFLUENCER
        else:
            label = CLASSIFICATION_NORMAL

        results[node] = {
            "risk_score": round(risk, 4),
            "label": label,
            **f,  # include raw features for transparency
        }

    return results

