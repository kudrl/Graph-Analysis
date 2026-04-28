from __future__ import annotations

import logging
from pathlib import Path


def configure_logging() -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logger = logging.getLogger("kodik")

    if not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
        try:
            logdir = Path(__file__).resolve().parents[1] / "logs"
            logdir.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(logdir / "kodik.log", encoding="utf-8")
            file_handler.setFormatter(
                logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
            )
            logger.addHandler(file_handler)
        except Exception:
            pass

    return logger
