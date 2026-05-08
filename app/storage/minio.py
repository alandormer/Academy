"""
Minimal MinIO object storage wrapper.

Responsibilities:
- Upload raw file bytes to a bucket
- Return the storage key (object path) for DB storage
- Check whether an object already exists

MinIO stores immutable originals only. All structured data (text, embeddings,
metadata) lives in PostgreSQL. Audio files are transient and not archived.

Bucket layout:
  academy-raw/      Original PDFs, DOCX, TXT documents
  academy-exports/  Generated summaries or exports (future use)

Usage:
  client = MinIOClient()
  key = client.upload(bucket="academy-raw", key="2026/05/guide.pdf", data=bytes)
"""
from __future__ import annotations

import io
import logging
from datetime import datetime
from pathlib import Path

from minio import Minio
from minio.error import S3Error

from app.core.config import settings

logger = logging.getLogger(__name__)


class MinIOClient:
    """
    Thin wrapper around the MinIO Python client.

    Instantiated once at application startup and injected via dependency.
    """

    def __init__(self) -> None:
        self._client = Minio(
            endpoint=settings.minio_endpoint,
            access_key=settings.minio_access_key,
            secret_key=settings.minio_secret_key,
            secure=settings.minio_secure,
        )
        logger.info("MinIO client initialised (endpoint=%s)", settings.minio_endpoint)

    # ------------------------------------------------------------------
    # Bucket management
    # ------------------------------------------------------------------

    def ensure_buckets(self) -> None:
        """Create configured buckets if they do not already exist."""
        for bucket in (
            settings.minio_bucket_raw,
            settings.minio_bucket_exports,
        ):
            if not self._client.bucket_exists(bucket):
                self._client.make_bucket(bucket)
                logger.info("Created MinIO bucket: %s", bucket)

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def upload(self, *, bucket: str, key: str, data: bytes, content_type: str = "application/octet-stream") -> str:
        """
        Upload *data* to *bucket* under *key*.

        Returns the key (object path) for storage in the DB.
        """
        self._client.put_object(
            bucket_name=bucket,
            object_name=key,
            data=io.BytesIO(data),
            length=len(data),
            content_type=content_type,
        )
        logger.debug("Uploaded %s/%s (%d bytes)", bucket, key, len(data))
        return key

    def exists(self, *, bucket: str, key: str) -> bool:
        """Return True if the object exists in the bucket."""
        try:
            self._client.stat_object(bucket, key)
            return True
        except S3Error as exc:
            if exc.code == "NoSuchKey":
                return False
            raise

    def object_path(self, *, bucket: str, key: str) -> str:
        """Return a human-readable path string: bucket/key."""
        return f"{bucket}/{key}"


# ------------------------------------------------------------------
# Key generation helpers
# ------------------------------------------------------------------

def make_storage_key(filename: str, source_type: str = "documents") -> str:
    """
    Generate a dated storage key for an uploaded file.

    Example:  documents/2026/05/av-rack-guide.pdf
    """
    now = datetime.utcnow()
    safe_name = Path(filename).name  # Strip any path traversal
    return f"{source_type}/{now.year}/{now.month:02d}/{safe_name}"
