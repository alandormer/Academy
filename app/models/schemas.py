import uuid
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Metadata vocabulary
# ---------------------------------------------------------------------------

# Who can see this content (future access control + UI segmentation)
Visibility = Literal["public", "internal", "technical", "restricted"]

# Operational domain — drives UI sections and retrieval scoping
Domain = Literal[
    "audio",
    "av",
    "theatre",
    "mac_lab",
    "studio",
    "health_safety",
    "admin",
    "events",
    "general",
]

# Nature of the document — affects how answers should be framed
DocumentType = Literal[
    "technical_specification",
    "procedure",
    "manual",
    "schedule",
    "transcript",
    "policy",
    "reference",
]

# How trustworthy / complete the content is
Confidence = Literal["official", "draft", "transcript", "generated"]


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------

class IngestMetadata(BaseModel):
    """
    Metadata attached to a source at ingestion time.

    Core typed fields (visibility, domain, document_type, confidence) are
    first-class: they are filterable, validated, and carry defaults.

    Freeform fields (room/location, system, author, topic) remain for
    operational specificity.  Any additional keys go into `extra`.
    """

    title: str | None = None

    # Access / UI segmentation
    visibility: Visibility = "internal"

    # Operational domain (maps to future UI sections)
    domain: Domain = "general"

    # Document character
    document_type: DocumentType | None = None
    confidence: Confidence = "official"

    # Location / physical context
    location: str | None = None   # e.g. "Theatre 2", "Studio A", "Mac Lab 1"
    room: str | None = None       # legacy alias — synced to location on export

    # Descriptive
    system: str | None = None
    author: str | None = None
    topic: str | None = None

    extra: dict[str, Any] = Field(default_factory=dict)

    def to_jsonb(self) -> dict[str, Any]:
        # Always emit the four typed fields so JSONB filters are reliable
        data: dict[str, Any] = {
            "visibility": self.visibility,
            "domain": self.domain,
            "confidence": self.confidence,
        }
        if self.document_type:
            data["document_type"] = self.document_type
        # Normalise location/room — keep both keys in sync for backward compat
        location = self.location or self.room
        if location:
            data["location"] = location
            data["room"] = location          # legacy key kept for existing filters
        for field in ("title", "system", "author", "topic"):
            val = getattr(self, field)
            if val is not None:
                data[field] = val
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

class QueryFilter(BaseModel):
    """
    Typed, optional pre-filters applied as JSONB containment during retrieval.

    All fields are optional — only supplied fields are added to the filter.
    Multiple fields are ANDed together (JSONB @> containment semantics).

    Maps to future UI sections:
      visibility  → public/internal/technical/restricted views
      domain      → Audio | AV | Theatre | Mac Lab | Studio | H&S | Admin | Events
      location    → room / space scoping
      document_type → Procedures | Manuals | Specs | Schedules …
    """

    visibility: Visibility | None = None
    domain: Domain | None = None
    location: str | None = None     # also checked against legacy "room" key
    document_type: DocumentType | None = None

    def to_jsonb(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        if self.visibility:
            out["visibility"] = self.visibility
        if self.domain:
            out["domain"] = self.domain
        if self.location:
            # match against either normalised "location" or legacy "room" key
            out["location"] = self.location
        if self.document_type:
            out["document_type"] = self.document_type
        return out


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=6, ge=1, le=20)

    # Typed filter — preferred way to scope queries
    filter: QueryFilter | None = Field(
        default=None,
        description="Typed metadata filter: visibility, domain, location, document_type",
    )

    # Raw JSONB escape hatch — merged with `filter` if both supplied
    metadata_filter: dict[str, Any] | None = Field(
        default=None,
        description="Raw JSONB containment filter (advanced). Merged with `filter`.",
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
    """Metadata about query execution — surfaced to UI for transparency."""
    elapsed_ms: float
    chunk_count: int
    # Filter that was actually applied (merged typed + raw)
    applied_filter: dict[str, Any] | None = None
    fallback_used: bool = False
    # Legacy fields kept for UI compat
    room_filter_applied: str | None = None
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
