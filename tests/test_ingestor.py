"""
Unit tests for the BotHunter synthetic data generator (ingestor.py).

Each test uses an in-memory SQLite session (db_session fixture from conftest)
so the real bothunter.db is never touched and tests never share state.
"""

import pytest

from models import Relationship, User
from ingestor import create_humans, create_engagement_pod, create_star_bot


# ─── create_humans ────────────────────────────────────────────────────────────

class TestCreateHumans:
    def test_create_humans_produces_correct_count(self, db_session):
        """Exactly `total` human User rows should be inserted, all is_bot=False."""
        humans = create_humans(db_session, total=10, group_count=2)
        db_session.flush()

        users = db_session.query(User).all()
        assert len(users) == 10, f"Expected 10 users, got {len(users)}"
        assert all(not u.is_bot for u in users), "Human users must have is_bot=False"

    def test_create_humans_platform_ids_are_unique(self, db_session):
        create_humans(db_session, total=15, group_count=3)
        db_session.flush()

        ids = [u.platform_id for u in db_session.query(User).all()]
        assert len(ids) == len(set(ids)), "platform_ids must be unique"

    def test_create_humans_inserts_follow_edges(self, db_session):
        """
        Intra-group follow density is 60–90 %, so a group of 10 split into
        2 groups must produce well more than 10 follow edges.
        """
        create_humans(db_session, total=10, group_count=2)
        db_session.flush()

        follow_count = (
            db_session.query(Relationship)
            .filter(Relationship.relation_type == "follow")
            .count()
        )
        assert follow_count > 10, (
            f"Expected >10 follow edges for 10 humans, got {follow_count}"
        )

    def test_create_humans_returns_list_of_user_objects(self, db_session):
        humans = create_humans(db_session, total=8, group_count=2)
        assert isinstance(humans, list)
        assert len(humans) == 8
        assert all(isinstance(u, User) for u in humans)


# ─── create_engagement_pod ────────────────────────────────────────────────────

class TestCreateEngagementPod:
    def test_engagement_pod_creates_correct_user_count(self, db_session):
        pod = create_engagement_pod(db_session, pod_size=5)
        db_session.flush()

        users = db_session.query(User).all()
        assert len(users) == 5

    def test_engagement_pod_users_are_flagged_as_bots(self, db_session):
        pod = create_engagement_pod(db_session, pod_size=6)
        db_session.flush()

        users = db_session.query(User).all()
        assert all(u.is_bot for u in users), "All pod members must have is_bot=True"

    def test_engagement_pod_creates_mutual_edges(self, db_session):
        """
        With 90 % follow probability per pair, a pod of 8 should contain
        mutual A↔B edges for the majority of the C(8,2)=28 unordered pairs.
        Asserting ≥ 1 mutual pair is a near-certain lower bound.
        """
        pod = create_engagement_pod(db_session, pod_size=8)
        db_session.flush()

        pod_ids = {u.id for u in pod}
        follow_edges = {
            (r.source_user_id, r.target_user_id)
            for r in db_session.query(Relationship)
            .filter(
                Relationship.relation_type == "follow",
                Relationship.source_user_id.in_(pod_ids),
                Relationship.target_user_id.in_(pod_ids),
            )
            .all()
        }

        # Count unordered pairs where both directions exist
        mutual_pairs = sum(
            1 for (a, b) in follow_edges if (b, a) in follow_edges and a < b
        )
        assert mutual_pairs >= 1, (
            f"Expected mutual follow pairs in pod, found {mutual_pairs}"
        )

    def test_engagement_pod_has_dense_follow_coverage(self, db_session):
        """
        With pod_size=6, there are 30 possible directed follow edges.
        At 90 % probability the expected count is 27; assert ≥ 15 (50 %).
        """
        pod = create_engagement_pod(db_session, pod_size=6)
        db_session.flush()

        pod_ids = {u.id for u in pod}
        follow_count = (
            db_session.query(Relationship)
            .filter(
                Relationship.relation_type == "follow",
                Relationship.source_user_id.in_(pod_ids),
                Relationship.target_user_id.in_(pod_ids),
            )
            .count()
        )
        max_possible = 6 * 5  # 30
        assert follow_count >= max_possible // 2, (
            f"Pod follow density too low: {follow_count}/{max_possible}"
        )


# ─── create_star_bot ──────────────────────────────────────────────────────────

class TestCreateStarBot:
    def test_star_bot_has_extreme_degree_ratio(self, db_session):
        """
        Star bot follows 10 humans (outgoing=10) but only 1 follows back
        (incoming=1).  Degree ratio = 10 → must satisfy out > in * 5.
        """
        humans = create_humans(db_session, total=10, group_count=2)
        db_session.flush()

        star = create_star_bot(db_session, humans, outgoing=10, incoming=1)
        db_session.flush()

        out_deg = (
            db_session.query(Relationship)
            .filter(
                Relationship.source_user_id == star.id,
                Relationship.relation_type == "follow",
            )
            .count()
        )
        in_deg = (
            db_session.query(Relationship)
            .filter(
                Relationship.target_user_id == star.id,
                Relationship.relation_type == "follow",
            )
            .count()
        )

        assert out_deg > in_deg * 5, (
            f"Star bot degree ratio not extreme enough: out={out_deg}, in={in_deg}"
        )

    def test_star_bot_is_flagged_as_bot(self, db_session):
        humans = create_humans(db_session, total=10, group_count=2)
        db_session.flush()
        star = create_star_bot(db_session, humans, outgoing=10, incoming=1)
        db_session.flush()

        assert star.is_bot is True

    def test_star_bot_has_distinct_platform_id(self, db_session):
        humans = create_humans(db_session, total=10, group_count=2)
        db_session.flush()
        star = create_star_bot(db_session, humans, outgoing=5, incoming=1)
        db_session.flush()

        assert star.platform_id == "star_bot_0"

    def test_star_bot_outgoing_edges_are_follow_type(self, db_session):
        humans = create_humans(db_session, total=10, group_count=2)
        db_session.flush()
        star = create_star_bot(db_session, humans, outgoing=8, incoming=1)
        db_session.flush()

        non_follow = (
            db_session.query(Relationship)
            .filter(
                Relationship.source_user_id == star.id,
                Relationship.relation_type != "follow",
            )
            .count()
        )
        assert non_follow == 0, "Star bot should only create follow-type edges"
