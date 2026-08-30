# backend/app/utils/logger.py

import logging
import sys
from app.config import settings


def get_logger(name: str) -> logging.Logger:
    """
    Creates a named logger with consistent formatting.
    Usage: logger = get_logger(__name__)
    __name__ gives the module name e.g. "app.core.rag.pipeline"
    """

    logger = logging.getLogger(name)

    # Only add handler if logger doesn't already have one
    # (prevents duplicate log lines when module is imported multiple times)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)

        # Log format: timestamp | level | module name | message
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    # Show DEBUG logs in development, only INFO+ in production
    if settings.app_env == "development":
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    return logger