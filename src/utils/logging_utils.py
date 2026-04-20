"""Logging configuration — console + file handler with consistent formatting."""
import logging
import sys
from pathlib import Path
from datetime import datetime


_FMT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_DATE_FMT = "%Y-%m-%d %H:%M:%S"


def setup_logging(
    log_level: str = "INFO",
    log_dir: Path | None = None,
    experiment_name: str = "experiment",
) -> logging.Logger:
    """Set up root logger with console and optional file handler.

    Returns root logger.  All downstream loggers inherit this configuration.
    If *log_dir* is provided, also writes to
    ``log_dir/{experiment_name}_{timestamp}.log``.
    """
    level = getattr(logging, log_level.upper(), logging.INFO)

    root = logging.getLogger()
    root.setLevel(level)

    # Remove any handlers already attached (e.g. from a previous call)
    root.handlers.clear()

    formatter = logging.Formatter(_FMT, datefmt=_DATE_FMT)

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(formatter)
    root.addHandler(console)

    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"{experiment_name}_{ts}.log"
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(formatter)
        root.addHandler(fh)
        root.info(f"Logging to file: {log_file}")

    return root


def get_logger(name: str) -> logging.Logger:
    """Get a named logger (call after setup_logging)."""
    return logging.getLogger(name)
