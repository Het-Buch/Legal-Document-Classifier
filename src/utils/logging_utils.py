import logging

def setup_logger(name: str):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.handlers = []
    logger.addHandler(handler)
    return logger
