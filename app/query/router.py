import logging
import re
import time

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import get_db, get_embedder
from app.embed.embedder import Embedder
from app.query.generator import generate_answer
from app.models.schemas import QueryRequest, QueryResponse, QueryMetadata
from app.query.retriever import retrieve

logger = logging.getLogger(__name__)

router = APIRouter()


def _detect_room(query: str) -> str | None:
    """
    Extract a single 'Theatre N' from a free-text query for metadata pre-filtering.
    Returns None if zero or multiple theatres are mentioned (cross-venue query).
    """
    matches = re.findall(r'theatre\s+(\d+)', query, re.IGNORECASE)
    unique = set(matches)
    if len(unique) == 1:
        return f"Theatre {unique.pop()}"
    return None  # 0 or 2+ theatres → no filter


@router.post(
    "",
    response_model=QueryResponse,
    summary="Semantic query with grounded LLM response",
)
async def query_endpoint(
    request: QueryRequest,
    db: AsyncSession = Depends(get_db),
    embedder: Embedder = Depends(get_embedder),
) -> QueryResponse:
    start_time = time.time()
    
    # 1. Embed the query
    try:
        query_vector = embedder.embed(request.query)
    except Exception as exc:
        logger.exception("Embedding failed for query")
        raise HTTPException(status_code=500, detail="Embedding failed.") from exc

    # 2. Auto-detect room filter if caller didn't supply one
    metadata_filter = request.metadata_filter
    auto_metadata_filter: dict[str, str] | None = None
    auto_detected_room: str | None = None
    fallback_used = False
    
    if not metadata_filter:
        room = _detect_room(request.query)
        if room:
            logger.debug("Auto-detected room filter: %s", room)
            auto_detected_room = room
            auto_metadata_filter = {"room": room}
            metadata_filter = auto_metadata_filter

    # 3. Retrieve relevant chunks
    try:
        chunks = await retrieve(
            db=db,
            query_vector=query_vector,
            top_k=request.top_k,
            metadata_filter=metadata_filter,
        )
        if not chunks and auto_metadata_filter is not None:
            logger.info(
                "No chunks found for auto-detected filter %s; retrying without metadata filter",
                auto_metadata_filter,
            )
            fallback_used = True
            chunks = await retrieve(
                db=db,
                query_vector=query_vector,
                top_k=request.top_k,
                metadata_filter=None,
            )
    except Exception as exc:
        logger.exception("Retrieval failed")
        raise HTTPException(status_code=500, detail="Retrieval failed.") from exc

    # 4. Generate grounded answer
    try:
        answer = await generate_answer(query=request.query, chunks=chunks)
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    # 5. Compute metadata
    elapsed_ms = (time.time() - start_time) * 1000
    query_metadata = QueryMetadata(
        elapsed_ms=round(elapsed_ms, 2),
        chunk_count=len(chunks),
        room_filter_applied=list(metadata_filter.keys())[0] if metadata_filter else None,
        fallback_used=fallback_used,
        auto_detected_room=auto_detected_room,
    )

    return QueryResponse(answer=answer, sources=chunks, metadata=query_metadata)
