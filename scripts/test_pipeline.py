#!/usr/bin/env python3
"""
Pipeline validation script.

Runs entirely standalone — no FastAPI server needed.
Tests: DB setup → PDF ingest → MinIO upload → embedding → retrieval

LLM generation step is skipped (Ollama not installed).
Retrieval results are printed directly so quality can be assessed.

Usage:
    .venv/bin/python3 scripts/test_pipeline.py
"""
import asyncio
import json
import sys
import textwrap
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

# ---------------------------------------------------------------------------
# Config overrides for this test — use contextengine DB credentials
# ---------------------------------------------------------------------------
import os
os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://contextengine:contextengine_dev@localhost:5432/academy_knowledge")
os.environ.setdefault("MINIO_ENDPOINT", "localhost:9100")
os.environ.setdefault("MINIO_ACCESS_KEY", "contextengine")
os.environ.setdefault("MINIO_SECRET_KEY", "contextengine123")
os.environ.setdefault("MINIO_SECURE", "false")
os.environ.setdefault("EMBEDDING_MODEL", "BAAI/bge-m3")
os.environ.setdefault("EMBEDDING_DIM", "1024")
os.environ.setdefault("CHUNK_SIZE", "400")
os.environ.setdefault("CHUNK_OVERLAP", "60")

GGUF_PATH = Path("/Users/alandormer/Projects/contextengine/models/mistral_models/mistral-7b-instruct-v0.3.Q6_K.gguf")

DOCS_DIR = Path(__file__).parent.parent / "files" / "documents"

# Allow overriding the target PDF via CLI: python test_pipeline.py "Theatre 1 – Technical Specification.pdf"
_cli_pdf = sys.argv[1] if len(sys.argv) > 1 else None

if _cli_pdf:
    PDF_PATH = DOCS_DIR / _cli_pdf
else:
    PDF_PATH = DOCS_DIR / "Theatre 2 Tech Spec.pdf"

# Auto-detect metadata + queries from filename
_fname = PDF_PATH.name.lower()
if "theatre 1" in _fname or "theater 1" in _fname:
    METADATA = {
        "venue": "Irish World Academy",
        "room": "Theatre 1",
        "document_type": "technical_specification",
        "systems": ["audio", "lighting", "video", "projection"],
        "confidence": "official",
    }
    RETRIEVAL_QUERIES = [
        "What FOH equipment exists in Theatre 1?",
        "What projector is installed in Theatre 1?",
        "How many stage boxes are available in Theatre 1?",
        "Does Theatre 1 have comms installed?",
        "What audio console is used in Theatre 1?",
    ]
else:
    METADATA = {
        "venue": "Irish World Academy",
        "room": "Theatre 2",
        "document_type": "technical_specification",
        "systems": ["audio", "lighting", "video", "projection"],
        "confidence": "official",
    }
    RETRIEVAL_QUERIES = [
        "What FOH equipment exists in Theatre 2?",
        "What projector is installed in Theatre 2?",
        "How many stage boxes are available?",
        "Does Theatre 2 have comms installed?",
        "What audio console is used in Theatre 2?",
    ]

SEP = "─" * 72


# ---------------------------------------------------------------------------
# Step 1: Create database and schema
# ---------------------------------------------------------------------------
async def setup_database():
    import asyncpg

    print(f"\n{SEP}")
    print("STEP 1 — Database setup")
    print(SEP)

    # Connect to postgres (default) DB to create academy_knowledge
    conn = await asyncpg.connect(
        "postgresql://contextengine:contextengine_dev@localhost:5432/postgres"
    )
    exists = await conn.fetchval(
        "SELECT 1 FROM pg_database WHERE datname = 'academy_knowledge'"
    )
    if not exists:
        await conn.execute("CREATE DATABASE academy_knowledge")
        print("  ✓ Created database: academy_knowledge")
    else:
        print("  ✓ Database already exists: academy_knowledge")
    await conn.close()

    # Connect to academy_knowledge and set up schema
    conn = await asyncpg.connect(
        "postgresql://contextengine:contextengine_dev@localhost:5432/academy_knowledge"
    )
    await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    print("  ✓ pgvector extension enabled")

    await conn.execute("""
        CREATE TABLE IF NOT EXISTS sources (
            id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            filename    TEXT NOT NULL,
            source_type TEXT NOT NULL,
            title       TEXT,
            storage_key TEXT,
            ingested_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            metadata    JSONB NOT NULL DEFAULT '{}'
        )
    """)

    # Check whether the chunks table exists with the right vector dimension.
    # If the embedding model has changed, drop and recreate.
    target_dim = int(os.environ["EMBEDDING_DIM"])
    existing_dim = await conn.fetchval("""
        SELECT atttypmod
        FROM pg_attribute
        JOIN pg_class ON pg_class.oid = pg_attribute.attrelid
        WHERE pg_class.relname = 'chunks'
          AND pg_attribute.attname = 'embedding'
          AND pg_attribute.attnum > 0
    """)
    if existing_dim is not None and existing_dim != target_dim:
        print(f"  ⚠  Vector dimension mismatch (DB={existing_dim}, model={target_dim}) — rebuilding chunks table")
        await conn.execute("DROP INDEX IF EXISTS chunks_embedding_idx")
        await conn.execute("DROP TABLE IF EXISTS chunks CASCADE")

    await conn.execute(f"""
        CREATE TABLE IF NOT EXISTS chunks (
            id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            source_id   UUID NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
            chunk_index INTEGER NOT NULL,
            content     TEXT NOT NULL,
            embedding   vector({target_dim}),
            metadata    JSONB NOT NULL DEFAULT '{{}}',
            created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
        )
    """)
    await conn.execute("""
        CREATE INDEX IF NOT EXISTS chunks_embedding_idx
            ON chunks USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 50)
    """)
    print(f"  ✓ Tables ready (embedding dim={target_dim})")
    await conn.close()


# ---------------------------------------------------------------------------
# Step 2: MinIO bucket setup
# ---------------------------------------------------------------------------
def setup_minio():
    from minio import Minio
    from minio.error import S3Error

    print(f"\n{SEP}")
    print("STEP 2 — MinIO setup")
    print(SEP)

    client = Minio("localhost:9100", access_key="contextengine", secret_key="contextengine123", secure=False)
    for bucket in ("academy-raw", "academy-exports"):
        if not client.bucket_exists(bucket):
            client.make_bucket(bucket)
            print(f"  ✓ Created bucket: {bucket}")
        else:
            print(f"  ✓ Bucket exists: {bucket}")
    return client


# ---------------------------------------------------------------------------
# LLM: load Mistral 7B GGUF directly via llama-cpp-python
# ---------------------------------------------------------------------------
_llm = None

def load_llm():
    global _llm
    from llama_cpp import Llama

    print(f"\n{SEP}")
    print("STEP 3b — Loading Mistral 7B GGUF")
    print(SEP)

    if not GGUF_PATH.exists():
        print(f"  ✗ GGUF not found at {GGUF_PATH}")
        return None

    print(f"  Model : {GGUF_PATH.name}")
    print(f"  Loading… (Metal GPU acceleration on Apple Silicon)")
    _llm = Llama(
        model_path=str(GGUF_PATH),
        n_ctx=4096,
        n_gpu_layers=-1,   # offload all layers to Metal
        verbose=False,
    )
    print(f"  ✓ Model loaded")
    return _llm


def llm_generate(query: str, chunks: list) -> str:
    if _llm is None:
        return "[LLM not loaded]"

    context_parts = []
    for i, row in enumerate(chunks, 1):
        context_parts.append(f"[{i}] {row['filename']} chunk #{row['chunk_index']}\n{row['content']}")
    context_block = "\n\n---\n\n".join(context_parts)

    prompt = f"""[INST] You are a technical assistant for a university technical department.
Answer ONLY using the CONTEXT below. Cite sources by filename and chunk number.
If the context does not contain the answer, say so clearly. Do not speculate.

CONTEXT:
{context_block}

QUESTION:
{query} [/INST]"""

    output = _llm(prompt, max_tokens=512, temperature=0.1, stop=["</s>"])
    return output["choices"][0]["text"].strip()



def load_embedder():
    from sentence_transformers import SentenceTransformer

    print(f"\n{SEP}")
    print("STEP 3 — Loading embedding model")
    print(SEP)

    model_name = os.environ["EMBEDDING_MODEL"]
    print(f"  Loading: {model_name}  (uses ~/.cache/huggingface if already downloaded)")
    model = SentenceTransformer(model_name)
    dim = model.get_embedding_dimension() if hasattr(model, 'get_embedding_dimension') else model.get_sentence_embedding_dimension()
    print(f"  ✓ Model loaded  |  embedding dimension: {dim}")
    return model


# ---------------------------------------------------------------------------
# Step 4: Parse PDF
# ---------------------------------------------------------------------------
def parse_pdf(path: Path) -> str:
    import fitz

    print(f"\n{SEP}")
    print("STEP 4 — Parsing PDF")
    print(SEP)

    content = path.read_bytes()
    text_parts = []
    with fitz.open(stream=content, filetype="pdf") as doc:
        print(f"  File     : {path.name}")
        print(f"  Size     : {len(content):,} bytes")
        print(f"  Pages    : {len(doc)}")
        for page in doc:
            text_parts.append(page.get_text())

    full_text = "\n".join(text_parts)
    print(f"  Extracted: {len(full_text):,} characters")
    print(f"\n  --- First 400 chars of extracted text ---")
    print(textwrap.indent(full_text[:400], "  "))
    return full_text, content


# ---------------------------------------------------------------------------
# Step 5: Chunk
# ---------------------------------------------------------------------------
def chunk_document(text: str) -> list[str]:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from app.ingest.chunker import chunk_text

    print(f"\n{SEP}")
    print("STEP 5 — Chunking")
    print(SEP)

    chunk_size = int(os.environ["CHUNK_SIZE"])
    overlap = int(os.environ["CHUNK_OVERLAP"])
    chunks = chunk_text(text, chunk_size=chunk_size, chunk_overlap=overlap)

    print(f"  chunk_size={chunk_size}, overlap={overlap}")
    print(f"  ✓ Total chunks: {len(chunks)}")
    print(f"  Avg chunk length: {sum(len(c) for c in chunks) // len(chunks)} chars")
    print(f"\n  --- Sample chunks ---")
    for i, c in enumerate(chunks[:3]):
        print(f"\n  [Chunk {i}] ({len(c)} chars)")
        print(textwrap.indent(c[:300], "    "))
    return chunks


# ---------------------------------------------------------------------------
# Step 6: Embed
# ---------------------------------------------------------------------------
def embed_chunks(model, chunks: list[str]) -> list[list[float]]:
    print(f"\n{SEP}")
    print("STEP 6 — Embedding")
    print(SEP)

    print(f"  Embedding {len(chunks)} chunks in batch…")
    import numpy as np
    vectors = model.encode(chunks, batch_size=32, normalize_embeddings=True, show_progress_bar=True)
    vectors_list = vectors.tolist()
    print(f"  ✓ Done")
    print(f"  Embedding dim : {len(vectors_list[0])}")
    print(f"  Sample (first 6 values of chunk 0): {[round(v,4) for v in vectors_list[0][:6]]}")
    return vectors_list


# ---------------------------------------------------------------------------
# Step 7: Upload to MinIO + store in PostgreSQL
# ---------------------------------------------------------------------------
async def store(minio_client, chunks: list[str], vectors: list[list[float]], raw_content: bytes):
    import asyncpg
    from datetime import datetime

    print(f"\n{SEP}")
    print("STEP 7 — Uploading to MinIO + storing in PostgreSQL")
    print(SEP)

    # MinIO upload
    import io
    now = datetime.utcnow()
    storage_key = f"documents/{now.year}/{now.month:02d}/{PDF_PATH.name}"
    minio_client.put_object(
        bucket_name="academy-raw",
        object_name=storage_key,
        data=io.BytesIO(raw_content),
        length=len(raw_content),
        content_type="application/pdf",
    )
    print(f"  ✓ MinIO upload: academy-raw/{storage_key}")

    # PostgreSQL
    conn = await asyncpg.connect(
        "postgresql://contextengine:contextengine_dev@localhost:5432/academy_knowledge"
    )

    # Check for existing source with same filename to avoid duplicates
    existing = await conn.fetchval(
        "SELECT id FROM sources WHERE filename = $1", PDF_PATH.name
    )
    if existing:
        print(f"  ℹ Source already exists ({existing}) — deleting for fresh ingest")
        await conn.execute("DELETE FROM sources WHERE id = $1", existing)

    # Derive a clean title from the filename (strip extension, replace separators)
    _title = PDF_PATH.stem.replace("–", "-").replace("—", "-").strip()

    # Insert source
    source_id = await conn.fetchval("""
        INSERT INTO sources (filename, source_type, title, storage_key, metadata)
        VALUES ($1, $2, $3, $4, $5)
        RETURNING id
    """,
        PDF_PATH.name,
        "document",
        _title,
        storage_key,
        json.dumps(METADATA),
    )
    print(f"  ✓ Source row created: {source_id}")

    # Insert chunks in batch
    chunk_records = [
        (source_id, idx, text, json.dumps(vector), json.dumps(METADATA))
        for idx, (text, vector) in enumerate(zip(chunks, vectors))
    ]
    await conn.executemany("""
        INSERT INTO chunks (source_id, chunk_index, content, embedding, metadata)
        VALUES ($1, $2, $3, $4::vector, $5)
    """, chunk_records)
    print(f"  ✓ Inserted {len(chunks)} chunks with embeddings")

    # Verify
    count = await conn.fetchval("SELECT COUNT(*) FROM chunks WHERE source_id = $1", source_id)
    sample = await conn.fetchrow("SELECT chunk_index, content FROM chunks WHERE source_id = $1 ORDER BY chunk_index LIMIT 1", source_id)
    meta_back = await conn.fetchval("SELECT metadata FROM sources WHERE id = $1", source_id)

    print(f"\n  --- Verification ---")
    print(f"  Chunk count in DB : {count}")
    print(f"  Storage key       : {storage_key}")
    print(f"  Metadata stored   : {json.dumps(json.loads(meta_back), indent=2)}")
    print(f"\n  First chunk preview:")
    print(textwrap.indent(sample['content'][:300], "    "))

    await conn.close()
    return source_id


# ---------------------------------------------------------------------------
# Step 8: Retrieval tests
# ---------------------------------------------------------------------------
async def run_retrieval(model, queries: list[str]):
    import asyncpg

    print(f"\n{SEP}")
    print("STEP 8 — Retrieval tests")
    print(SEP)

    conn = await asyncpg.connect(
        "postgresql://contextengine:contextengine_dev@localhost:5432/academy_knowledge"
    )

    for query in queries:
        print(f"\n  ┌─ QUERY: {query}")
        qvec = model.encode(query, normalize_embeddings=True).tolist()
        qvec_str = json.dumps(qvec)

        # SET ivfflat.probes ensures all lists are scanned — critical for
        # small datasets where default probes=1 misses most rows.
        await conn.execute("SET ivfflat.probes = 50")
        rows = await conn.fetch("""
            SELECT
                c.chunk_index,
                c.content,
                s.filename,
                1 - (c.embedding <=> $1::vector) AS score
            FROM chunks c
            JOIN sources s ON s.id = c.source_id
            ORDER BY c.embedding <=> $1::vector
            LIMIT 4
        """, qvec_str)

        if not rows:
            print("  │  No results found.")
        else:
            for i, row in enumerate(rows):
                score = float(row['score'])
                bar = "█" * int(score * 20)
                print(f"  │")
                print(f"  │  [{i+1}] score={score:.4f}  {bar}")
                print(f"  │      source: {row['filename']}  chunk #{row['chunk_index']}")
                preview = row['content'][:280].replace('\n', ' ')
                print(f"  │      {textwrap.fill(preview, width=65, subsequent_indent='  │      ')}")

        print(f"  │")
        if rows and _llm is not None:
            print(f"  │  Generating grounded answer (Mistral 7B)…")
            answer = llm_generate(query, rows)
            print(f"  │")
            for line in textwrap.wrap(answer, width=66):
                print(f"  │  {line}")
        else:
            print(f"  │  [LLM not loaded — skipping generation]")
        print(f"  └{'─'*68}")

    await conn.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def main():
    print(f"\n{'═'*72}")
    print("  ACADEMY KNOWLEDGE — Pipeline Validation")
    print(f"  Document: Theatre 2 Tech Spec.pdf")
    print(f"{'═'*72}")

    if not PDF_PATH.exists():
        print(f"ERROR: PDF not found at {PDF_PATH}")
        sys.exit(1)

    await setup_database()
    minio_client = setup_minio()
    model = load_embedder()
    load_llm()
    text, raw_content = parse_pdf(PDF_PATH)
    chunks = chunk_document(text)
    vectors = embed_chunks(model, chunks)
    source_id = await store(minio_client, chunks, vectors, raw_content)
    await run_retrieval(model, RETRIEVAL_QUERIES)

    print(f"\n{'═'*72}")
    print("  ✓ Pipeline validation complete")
    print(f"  source_id : {source_id}")
    print(f"  chunks    : {len(chunks)}")
    print(f"  embed dim : {len(vectors[0])}")
    print(f"  MinIO     : academy-raw/documents/")
    print(f"  LLM       : Mistral 7B Instruct (GGUF, Metal)")
    print(f"{'═'*72}\n")


if __name__ == "__main__":
    asyncio.run(main())
