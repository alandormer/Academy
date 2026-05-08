"""
Structured logging configuration.

Call configure_logging() once at application startup (in main.py lifespan).
Uses Python's standard logging module — no external dependencies.

Log format:
  - Development: human-readable with colour via uvicorn defaults
  - Production:  structured text suitable for log aggregation
"""
import logging
import sys

from app.core.config import settings


def configure_logging() -> None:
    """Configure root logger based on application settings."""
    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)

    fmt = "%(asctime)s [%(levelname)-8s] %(name)s: %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))

    root = logging.getLogger()
    # Avoid adding duplicate handlers if called more than once
    if not root.handlers:
        root.addHandler(handler)
    root.setLevel(log_level)

    # Reduce noise from chatty third-party libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("multipart").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Convenience wrapper — use in place of logging.getLogger()."""
    return logging.getLogger(name)
