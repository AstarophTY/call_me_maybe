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
    handler = logging.StreamHandler(stream)
    handler.setFormatter(ColorFormatter())

    if min_level is not None:
        handler.addFilter(MinLevelFilter(min_level))
    if max_level is not None:
        handler.addFilter(MaxLevelFilter(max_level))

    return handler
