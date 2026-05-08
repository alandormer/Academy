#!/usr/bin/env python
"""
One-time database initialisation script.

Run before starting the application:
    python scripts/init_db.py

Creates the pgvector extension, all tables, and the ANN index.
"""
import asyncio
import sys
from pathlib import Path

# Allow running from the project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from app.core.config import settings
from app.models.db import Base


async def init() -> None:
    engine = create_async_engine(settings.database_url, echo=True)

    async with engine.begin() as conn:
        # pgvector extension
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

        # Create all tables defined in models/db.py
        await conn.run_sync(Base.metadata.create_all)

        # IVFFlat index for approximate nearest-neighbour search.
        # lists=100 is a reasonable default; increase for larger datasets.
        await conn.execute(
            text(
                """
                CREATE INDEX IF NOT EXISTS chunks_embedding_idx
                    ON chunks
                    USING ivfflat (embedding vector_cosine_ops)
                    WITH (lists = 100)
                """
            )
        )

    await engine.dispose()
    print("Database initialised successfully.")


if __name__ == "__main__":
    asyncio.run(init())
