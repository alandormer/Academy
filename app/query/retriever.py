"""
Vector similarity retrieval via pgvector.

Executes a single parameterised SQL query against the chunks table.
Supports optional JSONB containment pre-filtering on metadata.
"""
from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.schemas import ChunkResult


async def retrieve(
    *,
    db: AsyncSession,
    query_vector: list[float],
    top_k: int = 6,
    metadata_filter: dict[str, Any] | None = None,
) -> list[ChunkResult]:
    """
    Return the top-k most similar chunks to *query_vector*.

    *metadata_filter* is applied as a JSONB containment check (@>)
    on the chunks.metadata column, e.g. {"room": "Studio A"}.
    """
    filter_clause = ""
    params: dict[str, Any] = {
        "query_vec": str(query_vector),
        "top_k": top_k,
    }

    if metadata_filter:
        import json
        filter_clause = "AND c.metadata @> CAST(:meta_filter AS jsonb)"
        params["meta_filter"] = json.dumps(metadata_filter)

    sql = text(
        f"""
        SELECT
            c.id                                                        AS chunk_id,
            c.source_id,
            s.filename,
            c.chunk_index,
            c.content,
            1 - (c.embedding <=> CAST(:query_vec AS vector))           AS score,
            c.metadata
        FROM chunks c
        JOIN sources s ON s.id = c.source_id
        WHERE c.embedding IS NOT NULL
        {filter_clause}
        ORDER BY c.embedding <=> CAST(:query_vec AS vector)
        LIMIT :top_k
        """
    )

    # Ensure IVFFlat scans all lists on small datasets
    await db.execute(text("SET ivfflat.probes = 50"))

    result = await db.execute(sql, params)
    rows = result.mappings().all()

    return [
        ChunkResult(
            chunk_id=row["chunk_id"],
            source_id=row["source_id"],
            filename=row["filename"],
            chunk_index=row["chunk_index"],
            content=row["content"],
            score=float(row["score"]),
            metadata=row["metadata"] or {},
        )
        for row in rows
    ]
