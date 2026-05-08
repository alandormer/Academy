import json
import logging

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import get_db, get_embedder, get_minio
from app.embed.embedder import Embedder
from app.ingest.pipeline import ingest_document
from app.models.schemas import IngestMetadata, IngestResponse
from app.storage.minio import MinIOClient

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/document",
    response_model=IngestResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Ingest a document (PDF, DOCX, TXT)",
)
async def ingest_document_endpoint(
    file: UploadFile = File(..., description="PDF, DOCX, or TXT file"),
    metadata: str = Form(
        default="{}",
        description="JSON string of metadata: title, room, venue, system, author, topic",
    ),
    db: AsyncSession = Depends(get_db),
    embedder: Embedder = Depends(get_embedder),
    minio: MinIOClient = Depends(get_minio),
) -> IngestResponse:
    try:
        meta_dict = IngestMetadata(**json.loads(metadata)).to_jsonb()
    except (json.JSONDecodeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=f"Invalid metadata JSON: {exc}") from exc

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        source_id, chunk_count = await ingest_document(
            db=db,
            embedder=embedder,
            minio=minio,
            filename=file.filename or "unknown",
            content=content,
            source_type="document",
            metadata=meta_dict,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Ingestion failed for %s", file.filename)
        raise HTTPException(status_code=500, detail="Ingestion failed.") from exc

    return IngestResponse(
        source_id=source_id,
        filename=file.filename or "unknown",
        chunk_count=chunk_count,
    )


@router.post(
    "/transcript",
    response_model=IngestResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Ingest a plain-text transcript",
)
async def ingest_transcript_endpoint(
    file: UploadFile = File(..., description="Plain text transcript (.txt)"),
    metadata: str = Form(default="{}"),
    db: AsyncSession = Depends(get_db),
    embedder: Embedder = Depends(get_embedder),
    minio: MinIOClient = Depends(get_minio),
) -> IngestResponse:
    try:
        meta_dict = IngestMetadata(**json.loads(metadata)).to_jsonb()
    except (json.JSONDecodeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=f"Invalid metadata JSON: {exc}") from exc

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        source_id, chunk_count = await ingest_document(
            db=db,
            embedder=embedder,
            minio=minio,
            filename=file.filename or "transcript.txt",
            content=content,
            source_type="transcript",
            metadata=meta_dict,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Transcript ingestion failed for %s", file.filename)
        raise HTTPException(status_code=500, detail="Ingestion failed.") from exc

    return IngestResponse(
        source_id=source_id,
        filename=file.filename or "transcript.txt",
        chunk_count=chunk_count,
    )
