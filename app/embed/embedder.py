from __future__ import annotations

import logging
from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class Embedder:
    """
    Thin wrapper around a sentence-transformers model.

    Loaded once at application startup and injected via FastAPI's
    dependency injection. The model runs entirely locally — no network
    calls after initial model download.
    """

    def __init__(self, model_name: str) -> None:
        logger.info("Loading embedding model: %s", model_name)
        self._model = SentenceTransformer(model_name)
        self._model_name = model_name
        logger.info("Embedding model loaded.")

    @property
    def model_name(self) -> str:
        return self._model_name

    def embed(self, text: str) -> list[float]:
        """Embed a single string. Returns a flat list of floats."""
        vector: np.ndarray = self._model.encode(text, normalize_embeddings=True)
        return vector.tolist()

    def embed_batch(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        """
        Embed a list of strings in batches.
        More efficient than calling embed() in a loop for large ingestion jobs.
        """
        vectors: np.ndarray = self._model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 50,
        )
        return vectors.tolist()
