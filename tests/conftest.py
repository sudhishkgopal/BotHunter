"""
Shared pytest fixtures for BotHunter tests.

Fixtures:
  small_graph       — nx.DiGraph with known bot patterns (pod-5, star-bot, 10 humans)
  graph_features    — pre-computed feature dict for small_graph
  db_session        — clean in-memory SQLite session, torn down after each test
  seeded_db_session — db_session pre-loaded with 3 users + 3 edges (2 follow, 1 like)
"""

import pytest
import networkx as nx
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from models import Base, User, Relationship
from processor import compute_features

# ─── Node-ID layout for small_graph ──────────────────────────────────────────
#   0  – 9  : humans (two balanced cliques of 5)
#  10  – 14 : engagement pod (near-complete mutual follow)
#  15       : star bot (follows all 10 humans, only 1 follows back)

HUMAN_IDS   = list(range(10))
POD_IDS     = list(range(10, 15))
STAR_BOT_ID = 15


@pytest.fixture(scope="session")
def small_graph() -> nx.DiGraph:
    """
    Minimal DiGraph with three distinct network patterns:

    Humans (0-9):
        Isolated — no edges between them.  Each receives one follow from the
        star bot (in_deg = 1, out_deg = 0), giving a low risk score and a
        "normal" label regardless of k_threshold.  Using isolated nodes keeps
        the star bot's neighbor-clustering at 0, which is required for the
        first bot rule (out > in × 5 AND clust < 0.2) to fire.

    Engagement pod (10-14):
        Complete mutual follow clique (deterministic — all 20 directed edges).
        Undirected k-core = 4, clustering = 1.0.
        Detected as engagement_pod with k_threshold ≤ 4.

    Star bot (15):
        Follows all 10 humans (out_deg = 10), only human-0 follows back
        (in_deg = 1).  Because the humans are isolated from each other the
        star bot's neighbour-clustering = 0 < 0.2, so the first bot rule fires:
        out_deg > in_deg × 5 AND clust < 0.2 → label = "bot".
    """
    G = nx.DiGraph()

    # Humans — isolated (no intra-human edges)
    G.add_nodes_from(HUMAN_IDS)

    # Engagement pod — complete mutual follow (deterministic, no randomness)
    for a in POD_IDS:
        for b in POD_IDS:
            if a != b:
                G.add_edge(a, b)

    # Star bot — mass outgoing, single return follow
    for h in HUMAN_IDS:
        G.add_edge(STAR_BOT_ID, h)   # bot → human
    G.add_edge(HUMAN_IDS[0], STAR_BOT_ID)  # one follower back

    return G


@pytest.fixture(scope="session")
def graph_features(small_graph: nx.DiGraph) -> dict:
    """Pre-computed feature dict for small_graph (reused across multiple tests)."""
    return compute_features(small_graph)


# ─── Database fixtures ────────────────────────────────────────────────────────

@pytest.fixture()
def db_session():
    """
    Clean in-memory SQLite session.  Tables are created fresh and dropped after
    every test so tests never share state.
    """
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()
    Base.metadata.drop_all(engine)


@pytest.fixture()
def seeded_db_session(db_session):
    """
    db_session pre-loaded with:
      - 3 User rows  (u1, u2 — human; u3 — bot)
      - 2 follow edges: u1→u2, u2→u3
      - 1 like edge:    u1→u3  (must NOT appear in the graph built by build_graph)
    """
    u1 = User(platform_id="seed_u1", username="alice", is_bot=False)
    u2 = User(platform_id="seed_u2", username="bob",   is_bot=False)
    u3 = User(platform_id="seed_u3", username="bot_x", is_bot=True)
    db_session.add_all([u1, u2, u3])
    db_session.flush()

    db_session.add_all([
        Relationship(source_user_id=u1.id, target_user_id=u2.id, relation_type="follow"),
        Relationship(source_user_id=u2.id, target_user_id=u3.id, relation_type="follow"),
        Relationship(source_user_id=u1.id, target_user_id=u3.id, relation_type="like"),
    ])
    db_session.flush()

    return db_session
