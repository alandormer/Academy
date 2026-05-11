import urllib.parse
import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.dependencies import get_db, get_minio
from app.models.db import Source
from app.models.schemas import SourceOut
from app.storage.minio import MinIOClient

router = APIRouter()


@router.get(
    "",
    response_model=list[SourceOut],
    summary="List all ingested sources",
)
async def list_sources(
    db: AsyncSession = Depends(get_db),
) -> list[SourceOut]:
    result = await db.execute(select(Source).order_by(Source.ingested_at.desc()))
    sources = result.scalars().all()
    return [
        SourceOut(
            id=s.id,
            filename=s.filename,
            source_type=s.source_type,
            title=s.title,
            storage_key=s.storage_key,
            ingested_at=s.ingested_at,
            metadata=s.metadata_,
        )
        for s in sources
    ]


@router.get(
    "/{source_id}",
    response_model=SourceOut,
    summary="Get a single source by ID",
)
async def get_source(
    source_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
) -> SourceOut:
    result = await db.execute(select(Source).where(Source.id == source_id))
    source = result.scalar_one_or_none()
    if not source:
        raise HTTPException(status_code=404, detail="Source not found.")
    return SourceOut(
        id=source.id,
        filename=source.filename,
        source_type=source.source_type,
        title=source.title,
        storage_key=source.storage_key,
        ingested_at=source.ingested_at,
        metadata=source.metadata_,
    )


@router.delete(
    "/{source_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a source and all its chunks",
)
async def delete_source(
    source_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
) -> None:
    async with db.begin():
        result = await db.execute(select(Source).where(Source.id == source_id))
        source = result.scalar_one_or_none()
        if not source:
            raise HTTPException(status_code=404, detail="Source not found.")
        await db.delete(source)  # CASCADE deletes chunks via FK


@router.get(
    "/{source_id}/download",
    summary="Download the original file for a source",
)
async def download_source(
    source_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    minio: MinIOClient = Depends(get_minio),
) -> StreamingResponse:
    result = await db.execute(select(Source).where(Source.id == source_id))
    source = result.scalar_one_or_none()
    if not source:
        raise HTTPException(status_code=404, detail="Source not found.")

    if not source.storage_key:
        raise HTTPException(
            status_code=404,
            detail="Original file unavailable — document was ingested before raw file storage was enabled.",
        )

    # All document originals live in academy-raw; audio transcripts have no
    # stored original (storage_key is NULL for those).
    bucket = settings.minio_bucket_raw

    if not minio.exists(bucket=bucket, key=source.storage_key):
        raise HTTPException(
            status_code=404,
            detail="Raw file missing from object storage.",
        )

    try:
        data = minio.get_object(bucket=bucket, key=source.storage_key)
    except Exception as exc:
        raise HTTPException(status_code=502, detail="Object storage error.") from exc

    # Infer content type from filename extension
    filename = source.filename or source.storage_key.split("/")[-1]
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    content_types = {
        "pdf": "application/pdf",
        "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "xls": "application/vnd.ms-excel",
        "txt": "text/plain",
        "md": "text/markdown",
        "mp3": "audio/mpeg",
        "wav": "audio/wav",
        "m4a": "audio/mp4",
    }
    media_type = content_types.get(ext, "application/octet-stream")

    safe_name = urllib.parse.quote(filename)

    return StreamingResponse(
        data.stream(32 * 1024),
        media_type=media_type,
        headers={"Content-Disposition": f"inline; filename*=UTF-8''{safe_name}"},
    )
