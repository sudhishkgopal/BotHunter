"""
SQLAlchemy ORM models for BotHunter (Instagram-style bot detection).

Tables:
  - users: social-media accounts with activity timestamps for sleep detection
  - relationships: directed, typed interactions (follow, like, comment)
  - analysis_results: persisted output of each bot-detection run
"""
from datetime import UTC, datetime

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
    account_created = Column(DateTime(timezone=True), nullable=True)
    last_active = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(UTC),
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


#Relationships (edges)

RELATION_TYPES = ("follow", "like", "comment")

class Relationship(Base):
    __tablename__ = "relationships"
    __table_args__ = (
        # A user can follow someone AND like their post
        UniqueConstraint(
            "source_user_id", "target_user_id", "relation_type",
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
    relation_type = Column(
        Enum(*RELATION_TYPES, name="relation_type_enum"),
        nullable=False,
        default="follow",
    )

    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(UTC),
    )

    source = relationship(
        "User", foreign_keys=[source_user_id], back_populates="outgoing",
    )
    target = relationship(
        "User", foreign_keys=[target_user_id], back_populates="incoming",
    )

    def __repr__(self) -> str:
        return (
            f"<Relationship {self.source_user_id} -[{self.relation_type}]-> "
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
        default=lambda: datetime.now(UTC),
    )

    def __repr__(self) -> str:
        return (
            f"<AnalysisResult id={self.id} k={self.k_core_threshold} "
            f"bots={self.bots_detected}>"
        )
