"""
SQLAlchemy ORM models for BotHunter.

Tables:
  - users: social-media nodes (human or bot)
  - relationships: directed edges between users (follower → followed)
  - analysis_results: persisted output of each bot-detection run
"""

from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    """Shared declarative base for all models."""


# ── Users ────────────────────────────────────────────────────────────────────

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    platform_id = Column(String(128), nullable=False, unique=True, index=True)
    username = Column(String(256), nullable=True)
    is_bot = Column(Boolean, nullable=False, default=False)
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    # back-references
    outgoing = relationship(
        "Relationship",
        foreign_keys="Relationship.source_user_id",
        back_populates="source",
        cascade="all, delete-orphan",
    )
    incoming = relationship(
        "Relationship",
        foreign_keys="Relationship.target_user_id",
        back_populates="target",
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        return f"<User id={self.id} platform_id={self.platform_id!r} is_bot={self.is_bot}>"


# ── Relationships (edges) ────────────────────────────────────────────────────

class Relationship(Base):
    __tablename__ = "relationships"
    __table_args__ = (
        UniqueConstraint("source_user_id", "target_user_id", name="uq_edge"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    source_user_id = Column(
        Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    target_user_id = Column(
        Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    relation_type = Column(String(64), nullable=False, default="follows")
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    source = relationship("User", foreign_keys=[source_user_id], back_populates="outgoing")
    target = relationship("User", foreign_keys=[target_user_id], back_populates="incoming")

    def __repr__(self) -> str:
        return (
            f"<Relationship {self.source_user_id} -[{self.relation_type}]-> "
            f"{self.target_user_id}>"
        )


# ── Analysis Results ─────────────────────────────────────────────────────────

class AnalysisResult(Base):
    __tablename__ = "analysis_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_label = Column(String(256), nullable=True)
    k_core_threshold = Column(Integer, nullable=False)
    total_nodes = Column(Integer, nullable=False)
    total_edges = Column(Integer, nullable=False)
    bots_detected = Column(Integer, nullable=False)
    bot_ids_json = Column(Text, nullable=False)
    detection_accuracy = Column(Float, nullable=True)
    ran_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    def __repr__(self) -> str:
        return (
            f"<AnalysisResult id={self.id} k={self.k_core_threshold} "
            f"bots={self.bots_detected}>"
        )