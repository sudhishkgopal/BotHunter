"""
SQLAlchemy ORM models for BotHunter (Instagram-style bot detection).

Tables:
  - users: social-media accounts with activity timestamps for sleep detection
  - relationships: directed, typed interactions (follow, like, comment)
  - analysis_results: persisted output of each bot-detection run
"""
from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Enum,
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

# User (nodes)
class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    platform_id = Column(String(128), nullable=False, unique=True, index=True)
    username = Column(String(256), nullable=True)
    is_bot = Column(Boolean, nullable=False, default=False)

    # account registeration
    account_created = Column(DateTime(timezone=True), nullable=True)

    #last observed activity (post, like, comment, login, etc.)
    last_active = Column(DateTime(timezone=True), nullable=True)

    # row-level bookkeeping (when we ingested this record)
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

    @property
    def is_sleeper(self) -> bool:
        """True when the account was created long ago but has no recent activity."""
        if self.account_created is None or self.last_active is None:
            return False
        dormant_days = (self.last_active - self.account_created).days
        idle_days = (datetime.now(timezone.utc) - self.last_active).days
        return dormant_days > 180 and idle_days > 90

    def __repr__(self) -> str:
        return (
            f"<User id={self.id} platform_id={self.platform_id!r} "
            f"is_bot={self.is_bot}>"
        )


#Relationships (edges)

INTERACTION_TYPES = ("follow", "like", "comment")

class Relationship(Base):
    __tablename__ = "relationships"
    __table_args__ = (
        # A user can follow someone AND like their post
        UniqueConstraint(
            "source_user_id", "target_user_id", "interaction_type",
            name="uq_edge_by_type",
        ),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    source_user_id = Column(
        Integer, ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    target_user_id = Column(
        Integer, ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )

    # updated — replaces the old `relation_type` freeform string
    interaction_type = Column(
        Enum(*INTERACTION_TYPES, name="interaction_type_enum"),
        nullable=False,
        default="follow",
    )

    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    source = relationship(
        "User", foreign_keys=[source_user_id], back_populates="outgoing",
    )
    target = relationship(
        "User", foreign_keys=[target_user_id], back_populates="incoming",
    )

    def __repr__(self) -> str:
        return (
            f"<Relationship {self.source_user_id} -[{self.interaction_type}]-> "
            f"{self.target_user_id}>"
        )


# Analysis Results 

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
