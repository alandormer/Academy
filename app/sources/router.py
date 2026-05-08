import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import get_db
from app.models.db import Source
from app.models.schemas import SourceOut

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
