"""
Local LLM inference via llama-cpp-python (GGUF, Metal GPU).

Handles prompt construction and grounded answer generation.
The Llama instance is loaded once at startup and injected via app.state.llm.

No retrieval logic lives here — chunks arrive pre-ranked from the query layer.
"""
from __future__ import annotations

import logging

from app.core.config import settings
from app.models.schemas import ChunkResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt — intentionally strict and grounded
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a technical assistant for a university technical department.
You answer questions about infrastructure, equipment, rooms, schedules, and operational procedures.

RULES:
- Answer ONLY using the information provided in the CONTEXT section below.
- If the context does not contain enough information to answer, say so clearly.
- Do not speculate or add information not present in the context.
- Always cite the source document(s) you used, by filename and chunk number.
- Be concise and precise.
- Do not introduce information from your training data.
"""


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

def build_context_block(chunks: list[ChunkResult]) -> str:
    """Format retrieved chunks into a numbered context block for the prompt."""
    parts: list[str] = []
    for i, chunk in enumerate(chunks, start=1):
        meta_parts = [f"{k}={v}" for k, v in chunk.metadata.items() if v and k != "source_path"]
        header = f"[{i}] {chunk.filename} (chunk {chunk.chunk_index})"
        if meta_parts:
            header += f"  |  {', '.join(meta_parts)}"
        parts.append(f"{header}\n{chunk.content}")
    return "\n\n---\n\n".join(parts)


def build_prompt(query: str, chunks: list[ChunkResult]) -> str:
    """
    Build a Mistral-instruct formatted prompt string.

    Uses the [INST] format expected by Mistral GGUF models.
    """
    context_block = build_context_block(chunks)
    body = (
        f"{_SYSTEM_PROMPT}\n\n"
        f"CONTEXT:\n{context_block}\n\n"
        f"QUESTION:\n{query}"
    )
    return f"[INST] {body} [/INST]"


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

async def generate_answer(*, query: str, chunks: list[ChunkResult], llm) -> str:
    """
    Generate a grounded answer using a llama-cpp-python Llama instance.

    *llm* is the loaded Llama object from app.state.llm.
    Returns a plain string. Raises RuntimeError on failure.
    """
    if not chunks:
        return "No relevant information was found in the knowledge base for your query."

    prompt = build_prompt(query, chunks)

    try:
        output = llm(prompt, max_tokens=512, temperature=0.1, stop=["[INST]"])
        return output["choices"][0]["text"].strip()
    except Exception as exc:
        logger.exception("GGUF generation failed")
        raise RuntimeError(f"LLM generation failed: {exc}") from exc
