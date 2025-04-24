import logging


def setup_logger(name: str, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("[%(asctime)s][%(levelname)s][%(name)s] %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False  # prevent logs from bubbling to the root logger

    return logger
