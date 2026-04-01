"""
Unit tests for the BotHunter detection engine (processor.py).

Covers:
  - build_graph   — DB → NetworkX loading
  - compute_features — per-node feature extraction
  - classify_nodes   — risk scoring and label assignment
"""

import networkx as nx

from processor import build_graph, classify_nodes, compute_features
from tests.conftest import HUMAN_IDS, POD_IDS, STAR_BOT_ID

# ─── Helpers (large graphs purpose-built for classification correctness) ──────

def _star_bot_graph(num_targets: int = 200) -> tuple[nx.DiGraph, int]:
    """
    Star bot: one node follows num_targets others; only 3 follow back.
    out_deg >> in_deg guarantees the bot rule fires regardless of graph size.
    """
    G = nx.DiGraph()
    bot = 0
    for i in range(1, num_targets + 1):
        G.add_edge(bot, i)
    for i in range(1, 4):
        G.add_edge(i, bot)
    return G, bot


def _engagement_pod_graph(size: int = 20) -> tuple[nx.DiGraph, list[int]]:
    """Near-complete mutual-follow clique (deterministic 95 % edges)."""
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


def _influencer_graph(followers: int = 200) -> tuple[nx.DiGraph, int]:
    """
    Pure in-hub: many followers, celebrity follows nobody (out_deg = 0).

    With out_deg = 0 the degree-asymmetry term is 0, capping risk at 0.60
    (= 0.35 × clust_inv + 0.25 × core_norm ≤ 0.35 + 0.25).  This keeps the
    high-risk-bot catch-all rule (risk > 0.7) from firing, so the influencer
    rule (in > out × 3 AND in ≥ 50) is reached and fires correctly.
    """
    G = nx.DiGraph()
    celebrity = 0
    for i in range(1, followers + 1):
        G.add_edge(i, celebrity)
    return G, celebrity


def _organic_graph(size: int = 30) -> tuple[nx.DiGraph, int]:
    """Small friend group with balanced follows and reasonable clustering."""
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


# ─── build_graph ─────────────────────────────────────────────────────────────

class TestBuildGraph:
    def test_build_graph_loads_all_nodes_and_edges(self, seeded_db_session):
        """
        build_graph should load all 3 users as nodes and only the 2 follow
        edges — the like edge must be filtered out.
        """
        G, user_map = build_graph(seeded_db_session)
        assert G.number_of_nodes() == 3, (
            f"Expected 3 nodes, got {G.number_of_nodes()}"
        )
        assert G.number_of_edges() == 2, (
            f"Expected 2 edges (follow-only), got {G.number_of_edges()}"
        )
        assert len(user_map) == 3

    def test_build_graph_user_map_has_expected_keys(self, seeded_db_session):
        _, user_map = build_graph(seeded_db_session)
        for entry in user_map.values():
            assert {"platform_id", "username", "is_bot"} <= entry.keys()

    def test_build_graph_empty_db_returns_empty_graph(self, db_session):
        G, user_map = build_graph(db_session)
        assert G.number_of_nodes() == 0
        assert user_map == {}


# ─── compute_features ────────────────────────────────────────────────────────

class TestComputeFeatures:
    def test_compute_features_returns_expected_keys(self, graph_features):
        """Every node entry must expose the four feature keys."""
        expected = {"k_core", "clustering", "in_deg", "out_deg"}
        for node_id, f in graph_features.items():
            assert expected <= f.keys(), (
                f"Node {node_id} missing keys: {expected - f.keys()}"
            )

    def test_all_nodes_present_in_features(self, small_graph, graph_features):
        assert set(graph_features.keys()) == set(small_graph.nodes())

    def test_empty_graph_doesnt_crash(self):
        features = compute_features(nx.DiGraph())
        assert features == {}

    def test_single_node_graph(self):
        G = nx.DiGraph()
        G.add_node(42)
        features = compute_features(G)
        assert features[42]["k_core"]   == 0
        assert features[42]["in_deg"]   == 0
        assert features[42]["out_deg"]  == 0
        assert features[42]["clustering"] == 0.0

    def test_feature_values_are_non_negative(self, graph_features):
        for f in graph_features.values():
            assert f["k_core"]     >= 0
            assert f["in_deg"]     >= 0
            assert f["out_deg"]    >= 0
            assert 0.0 <= f["clustering"] <= 1.0

    def test_directed_degrees_match_graph(self, small_graph, graph_features):
        for node in small_graph.nodes():
            assert graph_features[node]["in_deg"]  == small_graph.in_degree(node)
            assert graph_features[node]["out_deg"] == small_graph.out_degree(node)


# ─── classify_nodes ──────────────────────────────────────────────────────────

class TestClassifyNodes:

    # — small_graph fixture-based tests (structural assertions) ——————————————

    def test_star_bot_detected(self, small_graph):
        """
        The conftest star bot (ID=15) follows 10 humans, 1 follows back.
        out_deg / in_deg = 10 — well above the × 5 threshold.
        """
        features = compute_features(small_graph)
        results  = classify_nodes(features, k_threshold=3)
        assert results[STAR_BOT_ID]["label"] == "bot", (
            f"Star bot misclassified as {results[STAR_BOT_ID]['label']} "
            f"(out={results[STAR_BOT_ID]['out_deg']}, in={results[STAR_BOT_ID]['in_deg']})"
        )

    def test_engagement_pod_detected(self, small_graph):
        """
        Pod members (IDs 10-14) form a complete mutual clique.
        With k_threshold=3, k_core=4 ≥ 3 and clustering=1.0 > 0.6 → engagement_pod.
        """
        features = compute_features(small_graph)
        results  = classify_nodes(features, k_threshold=3)
        pod_labels = [results[m]["label"] for m in POD_IDS]
        pod_count  = pod_labels.count("engagement_pod")
        assert pod_count == len(POD_IDS), (
            f"Only {pod_count}/{len(POD_IDS)} pod members detected: {pod_labels}"
        )

    def test_human_not_flagged(self, small_graph):
        """
        Human nodes have balanced in/out-degree and are not in a dense core.
        At least the majority should be labelled 'normal'.
        """
        features = compute_features(small_graph)
        results  = classify_nodes(features, k_threshold=3)
        human_labels = [results[h]["label"] for h in HUMAN_IDS]
        normal_count = sum(1 for lbl in human_labels if lbl in ("normal", "organic"))
        assert normal_count >= len(HUMAN_IDS) // 2, (
            f"Too many humans flagged as bots: {human_labels}"
        )

    # — large dedicated graphs (classification correctness) —————————————————

    def test_star_bot_detected_large(self):
        G, bot = _star_bot_graph()
        results = classify_nodes(compute_features(G), k_threshold=10)
        assert results[bot]["label"] == "bot"

    def test_star_bot_risk_above_half(self):
        G, bot = _star_bot_graph()
        results = classify_nodes(compute_features(G), k_threshold=10)
        assert results[bot]["risk_score"] > 0.5

    def test_engagement_pod_majority_detected(self):
        G, members = _engagement_pod_graph()
        results    = classify_nodes(compute_features(G), k_threshold=5)
        pod_count  = sum(1 for m in members if results[m]["label"] == "engagement_pod")
        assert pod_count > len(members) * 0.5, (
            f"Only {pod_count}/{len(members)} pod members detected"
        )

    def test_influencer_detected(self):
        G, celeb = _influencer_graph(followers=200)
        results  = classify_nodes(compute_features(G), k_threshold=10)
        assert results[celeb]["label"] == "influencer", (
            f"Influencer misclassified as {results[celeb]['label']} "
            f"(in={results[celeb]['in_deg']}, out={results[celeb]['out_deg']})"
        )

    def test_organic_not_flagged(self):
        G, node = _organic_graph()
        results = classify_nodes(compute_features(G), k_threshold=10)
        assert results[node]["label"] in ("organic", "normal"), (
            f"Organic account wrongly flagged as {results[node]['label']}"
        )

    # — edge cases ————————————————————————————————————————————————————————————

    def test_empty_graph_doesnt_crash(self):
        assert classify_nodes({}, k_threshold=10) == {}

    def test_single_node_graph(self):
        G = nx.DiGraph()
        G.add_node(99)
        results = classify_nodes(compute_features(G), k_threshold=5)
        assert results[99]["label"] in ("normal", "organic")

    def test_risk_score_bounded(self):
        G, _ = _star_bot_graph()
        results = classify_nodes(compute_features(G), k_threshold=10)
        for r in results.values():
            assert 0.0 <= r["risk_score"] <= 1.0, (
                f"Risk score out of [0, 1]: {r['risk_score']}"
            )

    def test_result_contains_raw_features(self):
        G = nx.DiGraph([(0, 1), (1, 0)])
        results = classify_nodes(compute_features(G), k_threshold=5)
        for r in results.values():
            assert {"risk_score", "label", "k_core", "clustering", "in_deg", "out_deg"} <= r.keys()
