import pytest


def test_chunker_basic():
    from app.ingest.chunker import chunk_text

    text = "This is a sentence. " * 100
    chunks = chunk_text(text, chunk_size=200, chunk_overlap=20)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk) > 0


def test_chunker_short_text():
    from app.ingest.chunker import chunk_text

    text = "Short text."
    chunks = chunk_text(text, chunk_size=500, chunk_overlap=50)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_parsers_txt():
    from app.ingest.parsers import extract_text

    content = b"Hello, world.\nThis is a test."
    result = extract_text("test.txt", content)
    assert "Hello, world." in result


def test_parsers_unsupported():
    from app.ingest.parsers import extract_text

    with pytest.raises(ValueError, match="Unsupported"):
        extract_text("test.xyz", b"data")


def test_infer_room_from_filename():
    from app.ingest.pipeline import _infer_room

    assert _infer_room("Theatre 2 Tech Spec.pdf") == "Theatre 2"
    assert _infer_room("theatre-1-notes.docx") == "Theatre 1"
    assert _infer_room("Studio lighting plot.pdf") is None
