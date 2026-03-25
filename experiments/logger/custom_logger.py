import logging
from datetime import datetime
from pathlib import Path
import structlog


class CustomLogger:
    def __init__(self, log_dir="logs"):
        # Anchor logs to experiments/logs/ regardless of where the script is run from.
        # __file__ = experiments/logger/custom_logger.py
        # .parent   = experiments/logger/
        # .parent   = experiments/
        experiments_dir = Path(__file__).resolve().parent.parent
        self.logs_dir   = experiments_dir / log_dir
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        log_file          = datetime.now().strftime("%m_%d_%Y_%H_%M_%S") + ".log"
        self.log_file_path = self.logs_dir / log_file

    def get_logger(self, name=__file__):
        logger_name = Path(name).name

        file_handler = logging.FileHandler(self.log_file_path)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter("%(message)s"))

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(logging.Formatter("%(message)s"))

        logging.basicConfig(
            level=logging.INFO,
            format="%(message)s",
            handlers=[console_handler, file_handler],
        )

        structlog.configure(
            processors=[
                structlog.processors.TimeStamper(fmt="iso", utc=True, key="timestamp"),
                structlog.processors.add_log_level,
                structlog.processors.EventRenamer(to="event"),
                structlog.processors.JSONRenderer(),
            ],
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )

        return structlog.get_logger(logger_name)
