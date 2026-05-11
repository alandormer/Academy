"""
Ingestion pipeline.

Flow:
  1. Upload original file bytes to MinIO (academy-raw bucket) → get storage_key
  2. Parse text from the same in-memory bytes (no re-read)
  3. Split into chunks
  4. Generate embeddings in batch
  5. Persist source + chunks in a single PostgreSQL transaction

Audio files follow a different path (transcripts/router.py):
  audio upload → Whisper → transcript text → ingest_document(source_type="transcript")
  Audio bytes are NOT passed here and NOT stored in MinIO.
"""
from __future__ import annotations

import logging
import re
import uuid
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.embed.embedder import Embedder
from app.ingest.chunker import chunk_text
from app.ingest.parsers import extract_text
from app.models.db import Chunk, Source
from app.storage.minio import MinIOClient, make_storage_key

logger = logging.getLogger(__name__)


async def ingest_document(
    *,
    db: AsyncSession,
    embedder: Embedder,
    minio: MinIOClient,
    filename: str,
    content: bytes,
    source_type: str = "document",
    metadata: dict[str, Any] | None = None,
) -> tuple[uuid.UUID, int]:
    """
    Full ingestion pipeline for a single file.

    Returns (source_id, chunk_count).

    *minio* is the shared MinIOClient injected at the router level.
    Transcripts produced from Whisper in-memory have no original file to
    store — callers set source_type="transcript" and MinIO upload is skipped.
    """
    metadata = dict(metadata or {})

    # --- Infer location from filename if not supplied ---
    inferred_location = _infer_location(filename)
    if inferred_location:
        metadata.setdefault("location", inferred_location)
        metadata.setdefault("room", inferred_location)   # legacy compat

    # --- Apply typed metadata defaults ---
    metadata.setdefault("visibility", "internal")
    metadata.setdefault("domain", _infer_domain(filename, metadata))
    metadata.setdefault(
        "confidence",
        "transcript" if source_type == "transcript" else "official",
    )

    # 1. Upload raw file to MinIO (documents only — transcripts have no original)
    storage_key: str | None = None
    if source_type == "document":
        key = make_storage_key(filename, source_type="documents")
        storage_key = minio.upload(
            bucket=settings.minio_bucket_raw,
            key=key,
            data=content,
            content_type=_content_type(filename),
        )
        logger.info("Stored raw file at %s/%s", settings.minio_bucket_raw, storage_key)

    # 2. Parse text (from the same in-memory bytes — no re-read needed)
    logger.info("Parsing %s", filename)
    text = extract_text(filename, content)

    if not text.strip():
        raise ValueError(f"No text could be extracted from {filename!r}")

    # 3. Chunk
    chunks = chunk_text(text, chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)
    logger.info("Created %d chunks from %s", len(chunks), filename)

    # 4. Embed in batch
    logger.info("Embedding %d chunks…", len(chunks))
    vectors = embedder.embed_batch(chunks)

    # 5. Persist — single transaction
    async with db.begin():
        existing_result = await db.execute(
            select(Source).where(
                Source.filename == filename,
                Source.source_type == source_type,
            )
        )
        existing_sources = existing_result.scalars().all()
        for existing_source in existing_sources:
            await db.delete(existing_source)

        if existing_sources:
            logger.info(
                "Replacing %d existing source(s) for %s",
                len(existing_sources),
                filename,
            )

        source = Source(
            filename=filename,
            source_type=source_type,
            title=metadata.get("title"),
            storage_key=storage_key,
            metadata_=metadata,
        )
        db.add(source)
        await db.flush()  # Populate source.id before inserting chunks

        chunk_rows = [
            Chunk(
                source_id=source.id,
                chunk_index=idx,
                content=chunk_text_,
                embedding=vector,
                metadata_=metadata,
            )
            for idx, (chunk_text_, vector) in enumerate(zip(chunks, vectors))
        ]
        db.add_all(chunk_rows)

    logger.info(
        "Ingested source %s (%d chunks, storage_key=%s)",
        source.id, len(chunks), storage_key,
    )
    return source.id, len(chunks)


def _content_type(filename: str) -> str:
    from pathlib import Path
    _map = {
        ".pdf":  "application/pdf",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ".xls":  "application/vnd.ms-excel",
        ".txt":  "text/plain",
        ".md":   "text/markdown",
        ".rst":  "text/x-rst",
    }
    return _map.get(Path(filename).suffix.lower(), "application/octet-stream")


def _infer_location(filename: str) -> str | None:
    """Infer a physical location from filename patterns."""
    # Theatre N
    m = re.search(r"theatre\s*[-_ ]*(\d+)", filename, re.IGNORECASE)
    if m:
        return f"Theatre {m.group(1)}"
    # Studio A / Studio 1
    m = re.search(r"studio\s*[-_ ]*([a-z0-9]+)", filename, re.IGNORECASE)
    if m:
        return f"Studio {m.group(1).upper()}"
    # Mac Lab N
    m = re.search(r"mac\s*lab\s*[-_ ]*(\d+)", filename, re.IGNORECASE)
    if m:
        return f"Mac Lab {m.group(1)}"
    return None


# Legacy alias kept so any remaining callers don't break
_infer_room = _infer_location


_DOMAIN_PATTERNS: list[tuple[str, str]] = [
    (r"theatre",        "theatre"),
    (r"studio",         "audio"),
    (r"mac.?lab",       "mac_lab"),
    (r"av[-_ ]",        "av"),
    (r"health.?safety|h.s", "health_safety"),
    (r"procedure|policy",   "admin"),
    (r"schedule|event",     "events"),
]


def _infer_domain(filename: str, metadata: dict) -> str:
    """Infer operational domain from filename or existing metadata keys."""
    # If location was already set, use that to drive domain
    location = metadata.get("location") or metadata.get("room", "")
    combined = (filename + " " + location).lower()
    for pattern, domain in _DOMAIN_PATTERNS:
        if re.search(pattern, combined, re.IGNORECASE):
            return domain
    return "general"
