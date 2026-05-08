#!/usr/bin/env python
"""
Bulk ingest all supported documents from a directory.

Usage:
    python scripts/ingest_dir.py /path/to/documents \
        --metadata '{"room": "Studio A", "topic": "AV Infrastructure"}'

Walks the directory recursively and ingests every PDF, DOCX, and TXT file found.
"""
import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.config import settings
from app.db.session import async_session_factory
from app.embed.embedder import Embedder
from app.ingest.pipeline import ingest_document
from app.storage.minio import MinIOClient

SUPPORTED = {".pdf", ".docx", ".txt", ".md", ".rst"}


async def bulk_ingest(directory: Path, metadata: dict, skip_errors: bool = True) -> None:
    embedder = Embedder(model_name=settings.embedding_model)
    minio = MinIOClient()
    minio.ensure_buckets()
    files = [f for f in directory.rglob("*") if f.suffix.lower() in SUPPORTED]

    if not files:
        print(f"No supported files found in {directory}")
        return

    print(f"Found {len(files)} files to ingest.")
    success, failed = 0, 0

    for file_path in files:
        print(f"  Ingesting: {file_path.name} … ", end="", flush=True)
        try:
            content = file_path.read_bytes()
            async with async_session_factory() as db:
                source_id, chunk_count = await ingest_document(
                    db=db,
                    embedder=embedder,
                    minio=minio,
                    filename=file_path.name,
                    content=content,
                    source_type="document",
                    metadata={**metadata, "source_path": str(file_path)},
                )
            print(f"OK  ({chunk_count} chunks, id={source_id})")
            success += 1
        except Exception as exc:
            print(f"FAILED: {exc}")
            failed += 1
            if not skip_errors:
                raise

    print(f"\nDone. {success} ingested, {failed} failed.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Bulk ingest documents into Academy Knowledge.")
    parser.add_argument("directory", type=Path, help="Directory to scan for documents.")
    parser.add_argument(
        "--metadata",
        type=str,
        default="{}",
        help='JSON metadata to attach to all ingested sources, e.g. \'{"room": "Studio A"}\'',
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on first error instead of continuing.",
    )
    args = parser.parse_args()

    if not args.directory.is_dir():
        print(f"Error: {args.directory} is not a directory.")
        sys.exit(1)

    try:
        metadata = json.loads(args.metadata)
    except json.JSONDecodeError as exc:
        print(f"Error: Invalid metadata JSON: {exc}")
        sys.exit(1)

    asyncio.run(bulk_ingest(args.directory, metadata, skip_errors=not args.fail_fast))


if __name__ == "__main__":
    main()
