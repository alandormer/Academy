import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------

class IngestMetadata(BaseModel):
    """Optional metadata attached to a source at ingestion time."""

    title: str | None = None
    room: str | None = None
    venue: str | None = None
    system: str | None = None
    author: str | None = None
    topic: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)

    def to_jsonb(self) -> dict[str, Any]:
        data = self.model_dump(exclude_none=True, exclude={"extra"})
        data.update(self.extra)
        return data


class IngestResponse(BaseModel):
    source_id: uuid.UUID
    filename: str
    chunk_count: int


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------

class SourceOut(BaseModel):
    id: uuid.UUID
    filename: str
    source_type: str
    title: str | None
    storage_key: str | None
    ingested_at: datetime
    metadata: dict[str, Any]

    class Config:
        from_attributes = True


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=6, ge=1, le=20)
    metadata_filter: dict[str, Any] | None = Field(
        default=None,
        description="JSONB containment filter, e.g. {\"room\": \"Studio A\"}",
    )


class ChunkResult(BaseModel):
    chunk_id: uuid.UUID
    source_id: uuid.UUID
    filename: str
    chunk_index: int
    content: str
    score: float
    metadata: dict[str, Any]


class QueryMetadata(BaseModel):
    """Metadata about query execution."""
    elapsed_ms: float
    chunk_count: int
    room_filter_applied: str | None = None
    fallback_used: bool = False
    auto_detected_room: str | None = None


class QueryResponse(BaseModel):
    answer: str
    sources: list[ChunkResult]
    metadata: QueryMetadata


# ---------------------------------------------------------------------------
# Transcripts
# ---------------------------------------------------------------------------

class TranscriptUploadResponse(BaseModel):
    source_id: uuid.UUID
    filename: str
    chunk_count: int
    transcript_preview: str
