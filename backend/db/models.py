from sqlalchemy import (
    Column, String, Integer, Boolean, Text, DateTime, ForeignKey
)
from sqlalchemy.orm import declarative_base, relationship
from datetime import datetime
import uuid

Base = declarative_base()


def gen_id():
    return str(uuid.uuid4())


# ── User ────────────────────────────────────────────────────────────────────
class User(Base):
    __tablename__ = "users"

    id         = Column(String(36), primary_key=True, default=gen_id)
    name       = Column(String(120), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    sessions  = relationship("UserSession", back_populates="user", cascade="all, delete")
    roadmaps  = relationship("Roadmap",     back_populates="user", cascade="all, delete")


# ── Roadmap ──────────────────────────────────────────────────────────────────
class Roadmap(Base):
    __tablename__ = "roadmaps"

    id         = Column(String(36), primary_key=True, default=gen_id)
    user_id    = Column(String(36), ForeignKey("users.id"), nullable=False)
    goal       = Column(Text,       nullable=False)
    status     = Column(String(20), default="active")   # active | completed
    created_at = Column(DateTime,   default=datetime.utcnow)

    user       = relationship("User",      back_populates="roadmaps")
    milestones = relationship("Milestone", back_populates="roadmap",
                              cascade="all, delete", order_by="Milestone.order_index")


# ── Milestone ────────────────────────────────────────────────────────────────
class Milestone(Base):
    __tablename__ = "milestones"

    id          = Column(String(36),  primary_key=True, default=gen_id)
    roadmap_id  = Column(String(36),  ForeignKey("roadmaps.id"), nullable=False)
    title       = Column(String(255), nullable=False)
    description = Column(Text,        default="")
    phase       = Column(String(30),  default="Foundational")  # Foundational | Intermediate | Advanced
    order_index = Column(Integer,     default=0)
    completed   = Column(Boolean,     default=False)
    news_tags   = Column(Text,        default="")   # comma-separated keyword tags

    roadmap     = relationship("Roadmap", back_populates="milestones")


# ── UserSession ──────────────────────────────────────────────────────────────
class UserSession(Base):
    __tablename__ = "user_sessions"

    id                 = Column(String(36),  primary_key=True, default=gen_id)
    user_id            = Column(String(36),  ForeignKey("users.id"), nullable=False)
    active_roadmap_id  = Column(String(36),  ForeignKey("roadmaps.id"), nullable=True)
    active_node_id     = Column(String(36),  ForeignKey("milestones.id"), nullable=True)
    current_day        = Column(Integer,     default=1)
    day_completed      = Column(Boolean,     default=False)
    current_tech_stack = Column(String(255), default="")
    roadmap_status     = Column(String(20),  default="none")  # none | negotiating | active
    updated_at         = Column(DateTime,    default=datetime.utcnow, onupdate=datetime.utcnow)

    user               = relationship("User", back_populates="sessions")