"""
Unit tests for the BotHunter detection engine.

Tests the four classification patterns at the extremes so any
change to weights or thresholds is immediately caught.
"""

import networkx as nx
import pytest

# Adjust path for direct test runs
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from processor import compute_features, classify_nodes

# ─── Helpers ──────────────────────────────────────────────────────────────────

def _build_star_bot_graph(num_targets: int = 200) -> tuple[nx.DiGraph, int]:
    """Star bot: one node follows many, barely anyone follows back."""
    G = nx.DiGraph()
    bot = 0
    G.add_node(bot)
    for i in range(1, num_targets + 1):
        G.add_edge(bot, i)   # bot follows everyone
    for i in range(1, 4):
        G.add_edge(i, bot)   # only 3 follow back
    return G, bot


def _build_engagement_pod_graph(size: int = 20) -> tuple[nx.DiGraph, list[int]]:
    """Engagement pod: near-complete mutual follow clique."""
    import random
    random.seed(42)
    G = nx.DiGraph()
    members = list(range(size))
    G.add_nodes_from(members)
    for a in members:
        for b in members:
            if a != b and random.random() < 0.95:
                G.add_edge(a, b)
    return G, members


def _build_influencer_graph(followers: int = 200) -> tuple[nx.DiGraph, int]:
    """
    Influencer: high in-degree, low out-degree.
    Add a few mutual connections so clustering is not zero,
    preventing the catch-all bot rule (risk > 0.7 AND clustering < 0.3) from firing first.
    """
    G = nx.DiGraph()
    celebrity = 0
    G.add_node(celebrity)

    # Many followers who don't know each other (strangers)
    for i in range(1, followers + 1):
        G.add_edge(i, celebrity)

    # Celebrity follows a small circle of friends (creates mutual edges)
    for i in range(1, 6):
        G.add_edge(celebrity, i)   # celebrity follows
        # also make those 5 follow each other a bit to give celebrity nonzero clustering
        for j in range(1, 6):
            if i != j:
                G.add_edge(i, j)

    return G, celebrity


def _build_organic_graph(size: int = 30) -> tuple[nx.DiGraph, int]:
    """Small friend group — balanced follows, high clustering."""
    import random
    random.seed(7)
    G = nx.DiGraph()
    members = list(range(size))
    G.add_nodes_from(members)
    for a in members:
        for b in random.sample([m for m in members if m != a], k=min(5, size - 1)):
            G.add_edge(a, b)
            G.add_edge(b, a)
    return G, members[0]


# ─── Feature computation tests ────────────────────────────────────────────────

class TestComputeFeatures:
    def test_returns_all_nodes(self):
        G = nx.DiGraph([(0, 1), (1, 2), (2, 0)])
        features = compute_features(G)
        assert set(features.keys()) == set(G.nodes())

    def test_feature_keys(self):
        G = nx.DiGraph([(0, 1), (1, 0)])
        features = compute_features(G)
        for f in features.values():
            assert {"k_core", "clustering", "in_deg", "out_deg"} <= f.keys()

    def test_empty_graph(self):
        features = compute_features(nx.DiGraph())
        assert features == {}

    def test_single_node(self):
        G = nx.DiGraph()
        G.add_node(42)
        features = compute_features(G)
        assert features[42]["k_core"] == 0
        assert features[42]["in_deg"] == 0
        assert features[42]["out_deg"] == 0


# ─── Classification tests ─────────────────────────────────────────────────────

class TestClassifyNodes:
    def test_star_bot_detected(self):
        G, bot = _build_star_bot_graph()
        features = compute_features(G)
        results  = classify_nodes(features, k_threshold=10)
        assert results[bot]["label"] == "bot", (
            f"Star bot not detected — got {results[bot]['label']} "
            f"(out={results[bot]['out_deg']}, in={results[bot]['in_deg']})"
        )

    def test_star_bot_high_risk(self):
        G, bot = _build_star_bot_graph()
        features = compute_features(G)
        results  = classify_nodes(features, k_threshold=10)
        assert results[bot]["risk_score"] > 0.5

    def test_engagement_pod_detected(self):
        G, members = _build_engagement_pod_graph()
        features = compute_features(G)
        results  = classify_nodes(features, k_threshold=5)
        pod_labels = [results[m]["label"] for m in members]
        # Majority of pod members should be classified as engagement_pod
        pod_count = pod_labels.count("engagement_pod")
        assert pod_count > len(members) * 0.5, (
            f"Only {pod_count}/{len(members)} pod members detected"
        )

    def test_influencer_detected(self):
        G, celeb = _build_influencer_graph(followers=200)
        features = compute_features(G)
        results  = classify_nodes(features, k_threshold=10)
        assert results[celeb]["label"] == "influencer", (
            f"Influencer not detected — got {results[celeb]['label']} "
            f"(in={results[celeb]['in_deg']}, out={results[celeb]['out_deg']})"
        )

    def test_organic_not_flagged(self):
        G, node = _build_organic_graph()
        features = compute_features(G)
        results  = classify_nodes(features, k_threshold=10)
        assert results[node]["label"] in ("organic", "normal"), (
            f"Organic account wrongly flagged as {results[node]['label']}"
        )

    def test_empty_features_returns_empty(self):
        assert classify_nodes({}, k_threshold=10) == {}

    def test_risk_score_bounded(self):
        G, _ = _build_star_bot_graph()
        features = compute_features(G)
        results  = classify_nodes(features, k_threshold=10)
        for r in results.values():
            assert 0.0 <= r["risk_score"] <= 1.0, f"Risk score out of bounds: {r['risk_score']}"
