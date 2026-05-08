import logging
import re

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import get_db, get_embedder, get_llm
from app.embed.embedder import Embedder
from app.llm.ollama import generate_answer
from app.models.schemas import QueryRequest, QueryResponse
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
    llm=Depends(get_llm),
) -> QueryResponse:
    # 1. Embed the query
    try:
        query_vector = embedder.embed(request.query)
    except Exception as exc:
        logger.exception("Embedding failed for query")
        raise HTTPException(status_code=500, detail="Embedding failed.") from exc

    # 2. Auto-detect room filter if caller didn't supply one
    metadata_filter = request.metadata_filter
    if not metadata_filter:
        room = _detect_room(request.query)
        if room:
            logger.debug("Auto-detected room filter: %s", room)
            metadata_filter = {"room": room}

    # 3. Retrieve relevant chunks
    try:
        chunks = await retrieve(
            db=db,
            query_vector=query_vector,
            top_k=request.top_k,
            metadata_filter=metadata_filter,
        )
    except Exception as exc:
        logger.exception("Retrieval failed")
        raise HTTPException(status_code=500, detail="Retrieval failed.") from exc

    # 4. Generate grounded answer
    try:
        answer = await generate_answer(query=request.query, chunks=chunks, llm=llm)
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return QueryResponse(answer=answer, sources=chunks)
