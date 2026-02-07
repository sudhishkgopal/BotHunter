"""
Synthetic data ingestor for BotHunter.

Generates a social network with:
  - Normal human users connected via sparse, random follows
  - Bot farm clusters where every bot follows every other bot (dense cliques)
  - Cross-links where bots also follow some humans to look legitimate

Usage:
    python ingestor.py
    python ingestor.py --humans 500 --bot-farms 3 --bots-per-farm 20
"""

import argparse
import random
import logging

from database import SessionLocal, init_db
from models import Relationship, User

logging.basicConfig(
  level=logging.INFO,
  format="%(asctime)s [%(levelname)s] %(message)s",
  datefmt="%Y-%m-%d %H:%M:%S",
)

log = logging.getLogger(__name__)

def create_humans(session, count:int) -> List[User]:
  """Insert human users"""
  humans = [
    User(platform_id=f"human_{i}", username=f"human_{i}", is_bot=False)
    for i in range(count)
  ]
  session.add_all(humans)
  session.flush() # assigns IDs without commiting
  log.info("Created %d human users", len(humans))
  return humans

def create_bot_farm(session, farm_id:int, bot_count:int) -> List[User]:
  """Insert a bot farm (clique of bots)"""
  bots = [
    User(platform_id=f"bot_{farm_id}_{i}", username=f"bot_{farm_id}_{i}", is_bot=True)
    for i in range(bot_count)
  ]
  session.add_all(bots)
  session.flush()
  log.info("Created bot farm %d with %d bots", farm_id, len(bots))
  return bots
