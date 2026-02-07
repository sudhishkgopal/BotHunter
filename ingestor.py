"""
Synthetic data ingestor for BotHunter.

Generates three distinct network patterns for bot-detection testing:

  1. Humans  — small friend-group clusters (5-12 people), spread
              cross-group links, steady activity increasing over time.

  2. Engagement Pod  — 50 accounts that mutually follow AND like each other,
                      forming a near-perfect clique with bursts of timestamps.

  3. Star Bot — one account with 5,000 outgoing follows but only
                10 incoming follows - common bot farm pattern.

"""

import argparse
import logging
import random
from datetime import datetime, timedelta, timezone

from database import SessionLocal, init_db
from models import Relationship, User

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

NOW = datetime.now(timezone.utc)


# Timestamp helpers 

def random_past(days_back_min: int, days_back_max: int) -> datetime:
    """Return a random UTC datetime between min and max params."""
    delta = random.randint(days_back_min, days_back_max)
    jitter = random.randint(0, 86_399)  # second-level jitter within the day
    return NOW - timedelta(days=delta, seconds=jitter)


def steady_activity(account_created: datetime) -> datetime:
    """last active within the past 0-14 days (regular usage)."""
    return NOW - timedelta(
        days=random.randint(0, 14),
        hours=random.randint(0, 23),
        minutes=random.randint(0, 59),
    )


def bursty_activity() -> datetime:
    """Pod: last active within the past 0-2 days (coordinated burst)."""
    return NOW - timedelta(
        hours=random.randint(0, 48),
        minutes=random.randint(0, 59),
    )


def dormant_activity() -> datetime:
    """last active 30-180 days ago (sleeper account)."""
    return NOW - timedelta(
        days=random.randint(30, 180),
        hours=random.randint(0, 23),
    )


#  Humans network
def create_humans(session, total: int, group_count: int) -> list[User]:
    """
    Create human users arranged in small friend-group clusters.
    Each cluster is an internal near-clique scattered cross-group links.
    """
    humans: list[User] = []
    for i in range(total):
        created = random_past(90, 730)  # account age: 3 months – 2 years
        user = User(
            platform_id=f"human_{i}",
            username=f"user_{i}",
            is_bot=False,
            created_at=steady_activity(created),
        )
        humans.append(user)

    session.add_all(humans)
    session.flush()
    log.info("Inserted %d human users", len(humans))

    # split into random friend groups
    random.shuffle(humans)
    groups: list[list[User]] = []
    base_size = total // group_count
    remainder = total % group_count
    idx = 0
    for g in range(group_count):
        size = base_size + (1 if g < remainder else 0)
        groups.append(humans[idx : idx + size])
        idx += size

    # intra-group edges: each member follows 60-90 % of the group (realistic rather than perfect network)
    edge_count = 0
    seen: set[tuple[int, int]] = set()
    for group in groups:
        for member in group:
            follow_pct = random.uniform(0.6, 0.9)
            targets = random.sample(
                [u for u in group if u.id != member.id],
                k=max(1, int(len(group) * follow_pct)),
            )
            for t in targets:
                pair = (member.id, t.id)
                if pair not in seen:
                    seen.add(pair)
                    session.add(Relationship(
                        source_user_id=member.id,
                        target_user_id=t.id,
                        relation_type="follow",
                        created_at=random_past(1, 365),
                    ))
                    edge_count += 1

    # scattered cross-group links: ~3 % of humans follow someone outside (this might include artists, large group pages, celebrities, etc.)
    for human in humans:
        if random.random() < 0.03:
            target = random.choice(humans)
            pair = (human.id, target.id)
            if target.id != human.id and pair not in seen:
                seen.add(pair)
                session.add(Relationship(
                    source_user_id=human.id,
                    target_user_id=target.id,
                    relation_type="follow",
                    created_at=random_past(1, 180),
                ))
                edge_count += 1

    session.flush()
    log.info("  Follow edges: %d", edge_count)
    log.info("  Friend groups: %d  (sizes %d – %d)",
             len(groups), min(len(g) for g in groups), max(len(g) for g in groups))
    return humans


# Engagement Pod 
def create_engagement_pod(session, pod_size: int) -> list[User]:
    """
    Create an engagement pod: accounts that mutually follow AND like
    each other, forming a near-complete clique with bursty timestamps.
    """
    bots: list[User] = []
    for i in range(pod_size):
        # accounts created 6-18 months ago (sleeper-aged)
        created = random_past(180, 540)
        bot = User(
            platform_id=f"pod_{i}",
            username=f"engagepod_{i}",
            is_bot=True,
            created_at=bursty_activity(),
        )
        bots.append(bot)

    session.add_all(bots)
    session.flush()
    log.info("Inserted engagement pod  (%d accounts)", len(bots))

    follow_edges = 0
    like_edges = 0
    for a in bots:
        for b in bots:
            if a.id == b.id:
                continue

            # follow: 90 % chance (near-perfect, not exactly 100 %)
            if random.random() < 0.90:
                session.add(Relationship(
                    source_user_id=a.id,
                    target_user_id=b.id,
                    relation_type="follow",
                    created_at=bursty_activity(),
                ))
                follow_edges += 1

            # like: 85 % chance
            if random.random() < 0.85:
                session.add(Relationship(
                    source_user_id=a.id,
                    target_user_id=b.id,
                    relation_type="like",
                    created_at=bursty_activity(),
                ))
                like_edges += 1

    session.flush()
    log.info("  Pod follow edges: %d", follow_edges)
    log.info("  Pod like edges:   %d", like_edges)
    return bots

