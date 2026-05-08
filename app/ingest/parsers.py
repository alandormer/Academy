"""
Document text extraction.

Supports: PDF (via PyMuPDF), DOCX (via python-docx), plain TXT.
Each parser returns a single cleaned string of extracted text.
"""
from __future__ import annotations

import io
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def extract_text(filename: str, content: bytes) -> str:
    """
    Dispatch to the correct parser based on file extension.
    Returns extracted text as a single string.
    """
    suffix = Path(filename).suffix.lower()

    if suffix == ".pdf":
        return _extract_pdf(content)
    elif suffix == ".docx":
        return _extract_docx(content)
    elif suffix in {".txt", ".md", ".rst"}:
        return content.decode("utf-8", errors="replace")
    else:
        raise ValueError(f"Unsupported file type: {suffix!r}")


def _extract_pdf(content: bytes) -> str:
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:
        raise RuntimeError("pymupdf is required for PDF parsing. pip install pymupdf") from exc

    text_parts: list[str] = []
    with fitz.open(stream=content, filetype="pdf") as doc:
        for page in doc:
            text_parts.append(page.get_text())

    return "\n".join(text_parts)


def _extract_docx(content: bytes) -> str:
    try:
        from docx import Document
    except ImportError as exc:
        raise RuntimeError("python-docx is required for DOCX parsing. pip install python-docx") from exc

    doc = Document(io.BytesIO(content))
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n".join(paragraphs)
