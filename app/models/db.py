import uuid
from datetime import datetime

from pgvector.sqlalchemy import Vector
from app.core.config import settings
from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    Integer,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Source(Base):
    """A single ingested document or transcript."""

    __tablename__ = "sources"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    filename: Mapped[str] = mapped_column(Text, nullable=False)
    source_type: Mapped[str] = mapped_column(
        Text,
        CheckConstraint("source_type IN ('document', 'transcript')"),
        nullable=False,
    )
    title: Mapped[str | None] = mapped_column(Text, nullable=True)
    # MinIO object path, e.g. "documents/2026/05/guide.pdf". Null for
    # transcripts produced in-memory (audio not retained).
    storage_key: Mapped[str | None] = mapped_column(Text, nullable=True)
    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    # Flexible metadata: room, system, venue, author, date, topic, etc.
    metadata_: Mapped[dict] = mapped_column("metadata", JSONB, nullable=False, default=dict)

    chunks: Mapped[list["Chunk"]] = relationship(
        "Chunk", back_populates="source", cascade="all, delete-orphan"
    )


class Chunk(Base):
    """A chunk of text with its embedding vector."""

    __tablename__ = "chunks"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    source_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("sources.id", ondelete="CASCADE"), nullable=False
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    # Dimension set at runtime from config; migrations will specify the exact value.
    embedding: Mapped[list[float]] = mapped_column(Vector(settings.embedding_dim), nullable=True)
    metadata_: Mapped[dict] = mapped_column("metadata", JSONB, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    source: Mapped["Source"] = relationship("Source", back_populates="chunks")


class Asset(Base):
    """Structured registry of rooms, devices, systems, and people."""

    __tablename__ = "assets"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    name: Mapped[str] = mapped_column(Text, nullable=False)
    asset_type: Mapped[str] = mapped_column(Text, nullable=False)  # room, device, system, person
    location: Mapped[str | None] = mapped_column(Text, nullable=True)
    metadata_: Mapped[dict] = mapped_column("metadata", JSONB, nullable=False, default=dict)
