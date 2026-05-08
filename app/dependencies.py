from typing import AsyncGenerator

from fastapi import Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import async_session_factory
from app.embed.embedder import Embedder
from app.storage.minio import MinIOClient


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with async_session_factory() as session:
        yield session


def get_embedder(request: Request) -> Embedder:
    """Return the embedder singleton loaded at startup."""
    return request.app.state.embedder


def get_minio(request: Request) -> MinIOClient:
    """Return the MinIO client singleton loaded at startup."""
    return request.app.state.minio


def get_llm(request: Request):
    """Return the llama-cpp-python Llama singleton loaded at startup."""
    return request.app.state.llm
