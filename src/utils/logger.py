# ============================================================
# src/utils/logger.py
# ============================================================
# PURPOSE:
#   Centralized logging configuration.
#   All modules use get_logger(__name__) to get a logger.
# ============================================================

import logging
import os
from datetime import datetime


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Return a configured logger.

    Args:
        name: Module name (use __name__).
        level: Logging level (default INFO).

    Returns:
        Configured Logger instance.
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger  # Already configured

    logger.setLevel(level)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(level)
    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S"
    )
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # Optional file handler
    log_dir = "logs"
    try:
        os.makedirs(log_dir, exist_ok=True)
        date_str = datetime.now().strftime("%Y%m%d")
        fh = logging.FileHandler(
            os.path.join(log_dir, f"nav_{date_str}.log"),
            encoding="utf-8"
        )
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    except Exception:
        pass  # File logging failure is not critical

    return logger
