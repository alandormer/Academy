"""
Grounded response generation via a local LLM through Ollama.

The LLM is given only the retrieved context chunks. It is explicitly
instructed to cite sources and to refuse speculation beyond what is
present in the context.
"""
from __future__ import annotations

import logging

import ollama

from app.core.config import settings
from app.models.schemas import ChunkResult

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a technical assistant for a university technical department.
You answer questions about infrastructure, equipment, rooms, schedules, and operational procedures.

RULES:
- Answer ONLY using the information provided in the CONTEXT section below.
- If the context does not contain enough information to answer, say so clearly.
- Do not speculate or add information not present in the context.
- Always cite the source document(s) you used, by filename and chunk index.
- Be concise and precise.
"""


def build_context_block(chunks: list[ChunkResult]) -> str:
    parts: list[str] = []
    for i, chunk in enumerate(chunks, start=1):
        meta_summary = ", ".join(f"{k}={v}" for k, v in chunk.metadata.items() if v)
        header = f"[{i}] {chunk.filename} (chunk {chunk.chunk_index})"
        if meta_summary:
            header += f" | {meta_summary}"
        parts.append(f"{header}\n{chunk.content}")
    return "\n\n---\n\n".join(parts)


async def generate_answer(
    *,
    query: str,
    chunks: list[ChunkResult],
) -> str:
    """
    Call the local Ollama LLM and return the grounded answer string.
    """
    if not chunks:
        return "No relevant information was found in the knowledge base for your query."

    context_block = build_context_block(chunks)
    user_message = f"CONTEXT:\n{context_block}\n\nQUESTION:\n{query}"

    try:
        response = ollama.chat(
            model=settings.ollama_model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": user_message},
            ],
            options={"temperature": 0.1},  # Low temperature for factual retrieval tasks
        )
        return response["message"]["content"]
    except Exception as exc:
        logger.exception("LLM generation failed")
        raise RuntimeError(f"LLM generation failed: {exc}") from exc
