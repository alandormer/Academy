import json
import logging

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import get_db, get_embedder, get_minio
from app.embed.embedder import Embedder
from app.ingest.pipeline import ingest_document
from app.models.schemas import IngestMetadata, TranscriptUploadResponse
from app.storage.minio import MinIOClient
from app.transcripts.whisper import transcribe_audio

logger = logging.getLogger(__name__)

router = APIRouter()

_AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".mp4", ".webm"}


@router.post(
    "/upload",
    response_model=TranscriptUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload audio → Whisper transcription → ingest",
)
async def upload_audio(
    file: UploadFile = File(..., description="Audio file (mp3, wav, m4a, ogg, flac)"),
    metadata: str = Form(
        default="{}",
        description="JSON string of metadata: title, room, venue, system, author, topic",
    ),
    db: AsyncSession = Depends(get_db),
    embedder: Embedder = Depends(get_embedder),
    minio: MinIOClient = Depends(get_minio),
) -> TranscriptUploadResponse:
    from pathlib import Path

    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in _AUDIO_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported audio format {suffix!r}. Supported: {sorted(_AUDIO_EXTENSIONS)}",
        )

    try:
        meta_dict = IngestMetadata(**json.loads(metadata)).to_jsonb()
    except (json.JSONDecodeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=f"Invalid metadata JSON: {exc}") from exc

    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Uploaded audio file is empty.")

    # Transcribe
    try:
        transcript = transcribe_audio(audio_bytes, file.filename or "audio")
    except Exception as exc:
        logger.exception("Whisper transcription failed for %s", file.filename)
        raise HTTPException(status_code=500, detail="Transcription failed.") from exc

    if not transcript.strip():
        raise HTTPException(status_code=422, detail="Transcription produced no text.")

    # Ingest the transcript text.
    # Audio bytes are discarded after transcription — not archived.
    txt_filename = (file.filename or "transcript") + ".txt"
    try:
        source_id, chunk_count = await ingest_document(
            db=db,
            embedder=embedder,
            minio=minio,
            filename=txt_filename,
            content=transcript.encode("utf-8"),
            source_type="transcript",
            metadata=meta_dict,
        )
    except Exception as exc:
        logger.exception("Ingestion failed after transcription of %s", file.filename)
        raise HTTPException(status_code=500, detail="Ingestion failed.") from exc

    return TranscriptUploadResponse(
        source_id=source_id,
        filename=txt_filename,
        chunk_count=chunk_count,
        transcript_preview=transcript[:500],
    )
