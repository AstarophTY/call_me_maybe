# ABOUTME: Logging formatters for colorized terminal and plain file output.
# ABOUTME: ColorFormatter targets the console; FILE_FORMAT targets log files.

import logging

from . import colors

FILE_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
FILE_DATE = "%Y-%m-%d %H:%M:%S"


class ColorFormatter(logging.Formatter):
    """Format log records with ANSI colors for terminal output.

    The color and label are selected from the record level using the
    tables in :mod:`llm_sdk.log.colors`. Unknown levels fall back to a
    plain white color and the record's default level name.
    """

    def format(self, record: logging.LogRecord) -> str:
        """Render *record* as a single colorized line.

        Exception information, when present, is appended on a new line
        without coloring so tracebacks stay readable.
        """
        color = colors.LEVEL_COLORS.get(record.levelno, colors.WHITE)
        label = colors.LEVEL_LABELS.get(record.levelno, record.levelname)

        time_str = self.formatTime(record, "%H:%M:%S")
        name_str = f"{colors.DIM}{record.name}{colors.RESET}"
        level_str = f"{color}{colors.BOLD}{label}{colors.RESET}"
        msg_str = f"{color}{record.getMessage()}{colors.RESET}"

        line = (
            f"{colors.DIM}{time_str}{colors.RESET}  "
            f"{level_str}  {name_str}  {msg_str}"
        )

        if record.exc_info:
            line += "\n" + self.formatException(record.exc_info)

        return line
