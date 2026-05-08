import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.core.config import settings
from app.core.logging import configure_logging
from app.db.session import engine
from app.embed.embedder import Embedder
from app.storage.minio import MinIOClient
from app.ingest.router import router as ingest_router
from app.query.router import router as query_router
from app.transcripts.router import router as transcript_router
from app.sources.router import router as sources_router

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()

    # Embedding model
    app.state.embedder = Embedder(
        model_name=settings.embedding_model,
        local_files_only=settings.embedding_local_files_only,
    )

    # MinIO client
    minio = MinIOClient()
    minio.ensure_buckets()
    app.state.minio = minio

    yield

    await engine.dispose()


app = FastAPI(
    title="Academy Knowledge System",
    version="0.1.0",
    description="Local-first operational knowledge and infrastructure memory system.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ingest_router,     prefix="/ingest",      tags=["Ingestion"])
app.include_router(query_router,      prefix="/query",       tags=["Query"])
app.include_router(transcript_router, prefix="/transcripts", tags=["Transcripts"])
app.include_router(sources_router,    prefix="/sources",     tags=["Sources"])


@app.get("/health", tags=["Health"])
async def health():
    return {"status": "ok"}


# Serve the chat UI — must come last so it doesn't shadow API routes
_static_dir = Path(__file__).parent / "static"
app.mount("/", StaticFiles(directory=str(_static_dir), html=True), name="static")
