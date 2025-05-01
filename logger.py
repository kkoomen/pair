import logging
import os
import time


def get_log_level(level: str) -> int:
    """Convert string log level to integer."""
    levels = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL,
    }
    return levels.get(level.lower(), logging.INFO)


def setup_logger(name: str, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level if isinstance(level, int) else get_log_level(level))

    if not logger.handlers:
        formatter = logging.Formatter("[%(asctime)s][%(levelname)s][%(name)s] %(message)s")

        log_file = os.path.join("logs", f"{time.strftime('%Y%m%d_%H%M%S')}_{name}.log")
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        logger.propagate = False

    return logger
