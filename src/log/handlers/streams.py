import logging
from typing import TextIO

from ..filters import MaxLevelFilter, MinLevelFilter
from ..formatters import ColorFormatter


def make_stream_handler(
    stream: TextIO,
    *,
    min_level: int | None = None,
    max_level: int | None = None,
) -> logging.StreamHandler:
    """Create a stream handler with optional level filtering.

    Configures a handler with a ColorFormatter and optional
    MinLevelFilter and/or MaxLevelFilter for level-based routing.

    :param stream: Output stream (e.g., sys.stdout, sys.stderr).
    :param min_level: Minimum log level to allow (optional).
    :param max_level: Maximum log level to allow (optional).
    :returns: Configured logging.StreamHandler instance.
    """
    handler = logging.StreamHandler(stream)
    handler.setFormatter(ColorFormatter())

    if min_level is not None:
        handler.addFilter(MinLevelFilter(min_level))
    if max_level is not None:
        handler.addFilter(MaxLevelFilter(max_level))

    return handler
