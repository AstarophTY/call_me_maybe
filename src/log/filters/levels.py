import logging


class MaxLevelFilter(logging.Filter):
    """Filter that suppresses records above a maximum level.

    Only log records with level number <= max_level are allowed
    through this filter.
    """

    def __init__(self, max_level: int) -> None:
        """Initialize the filter with a maximum level.

        :param max_level: Maximum log level number to allow.
        """
        super().__init__()
        self.max_level = max_level

    def filter(self, record: logging.LogRecord) -> bool:
        """Filter log records based on maximum level.

        :param record: The log record to filter.
        :returns: True if record level <= max_level, False otherwise.
        """
        return record.levelno <= self.max_level


class MinLevelFilter(logging.Filter):
    """Filter that suppresses records below a minimum level.

    Only log records with level number >= min_level are allowed
    through this filter.
    """

    def __init__(self, min_level: int) -> None:
        """Initialize the filter with a minimum level.

        :param min_level: Minimum log level number to allow.
        """
        super().__init__()
        self.min_level = min_level

    def filter(self, record: logging.LogRecord) -> bool:
        """Filter log records based on minimum level.

        :param record: The log record to filter.
        :returns: True if record level >= min_level, False otherwise.
        """
        return record.levelno >= self.min_level
