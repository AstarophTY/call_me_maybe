import logging
import sys
from datetime import datetime
from pathlib import Path

from .formatters import FILE_DATE, FILE_FORMAT, ColorFormatter


def setup_logging(
    level: str = "DEBUG",
    logs_dir: str = "logs",
) -> Path:
    """Configure global logging and return the created log file path.

    Two handlers are attached to the root logger:

    * a console handler writing colorized output to ``stdout``;
    * a file handler writing plain text to
      ``<logs_dir>/YYYY-MM-DD_HH-MM-SS.log``.

    :param level: Minimum level name (e.g. ``"DEBUG"``, ``"INFO"``).
    :param logs_dir: Directory for log files; created if missing.
    :returns: Path to the timestamped log file that was created.
    """
    log_folder = Path(logs_dir)
    log_folder.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = log_folder / f"{timestamp}.log"

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(ColorFormatter())

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(
        logging.Formatter(FILE_FORMAT, datefmt=FILE_DATE)
    )

    logging.basicConfig(
        level=level,
        handlers=[console_handler, file_handler],
    )

    logging.getLogger(__name__).info("Logging initialized -> %s", log_file)
    return log_file
