from __future__ import annotations

from json import JSONDecodeError, dumps
from logging import Logger
from pathlib import Path
from typing import Any

from pydantic import ValidationError
from typer import Exit


def validate_file_exists(file_path: Path, log: Logger) -> None:
    """Validate that a path exists and points to a regular file.

    :param file_path: Path to validate.
    :param log: Logger used to report validation errors.
    """
    if not file_path.exists():
        log.error(f"File not found: {file_path}")
        raise Exit(code=1) from None

    if not file_path.is_file():
        log.error(f"File is not a regular file: {file_path}")
        raise Exit(code=1) from None


def load_validated_json(
    file_path: Path,
    model: Any,
    log: Logger,
) -> Any:
    """Load JSON from disk and validate it with a Pydantic model.

    :param file_path: JSON file path.
    :param model: Pydantic model class used for validation.
    :param log: Logger used to report parsing errors.
    :returns: Validated Pydantic model instance.
    """
    try:
        with open(file_path) as file_handle:
            parsed = model.model_validate_json(file_handle.read())
            log.debug(dumps(parsed.model_dump(), indent=2))
            return parsed
    except JSONDecodeError as error:
        log.error(f"Error loading JSON from {file_path}: {error}")
        raise Exit(code=1) from None
    except ValidationError as error:
        for detail in error.errors():
            log.error(f"Invalid structure in {file_path}: {detail['msg']}")
        raise Exit(code=1) from None


def create_output_dir(output_file: Path, log: Logger) -> None:
    """Create the output directory when it is missing.

    :param output_file: Output directory path.
    :param log: Logger used to report filesystem errors.
    """
    try:
        if output_file.exists() and not output_file.is_dir():
            raise FileExistsError(
                f"Output path '{output_file}' exists but is not a directory."
            )
        output_file.mkdir(parents=True, exist_ok=True)
    except (PermissionError, FileExistsError, OSError) as error:
        log.error(f"Error creating output directory: {error}")
        raise Exit(code=1) from None


def validate_loaded_data(
    input_data: Any,
    functions_definition_data: Any,
    log: Logger,
) -> None:
    """Validate that both parsed root collections are present.

    :param input_data: Parsed function calling tests.
    :param functions_definition_data: Parsed function definitions.
    :param log: Logger used to report validation errors.
    """
    if not input_data or not input_data.root:
        log.error("No input data available.")
        raise Exit(code=1) from None

    if not functions_definition_data or not functions_definition_data.root:
        log.error("No function definitions available.")
        raise Exit(code=1) from None
