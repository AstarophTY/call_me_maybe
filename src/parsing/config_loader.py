from pathlib import Path
from logging import getLogger
from typing import Any, cast
from typer import Exit
from json import JSONDecodeError, dumps
from pydantic import ValidationError

from .validator import FunctionCallingTestsFile, FunctionDefinitionsFile


class ConfigLoader:
    def __init__(
        self,
        input_file: str,
        output_file: str,
        functions_definition_file: str,
    ) -> None:
        self.log = getLogger(__name__)
        self.input_file = Path(input_file)
        self.output_file = Path(output_file)
        self.functions_definition_file = Path(functions_definition_file)
        self.data: dict[str, Any] = {}

        self._load_required_files()
        self._create_output_dir()
        self._validate_loaded_data()

    def _load_required_files(self) -> None:
        """Load and validate input and functions definition files."""
        loaders = [
            ("input_file", self.input_file, FunctionCallingTestsFile),
            (
                "functions_definition_file",
                self.functions_definition_file,
                FunctionDefinitionsFile,
            ),
        ]

        for file_name, file_path, model in loaders:
            self._validate_file_exists(file_path)
            self.data[file_name] = self._load_json(file_path, model)
            self.log.debug(
                dumps(cast(Any, self.data[file_name]).model_dump(), indent=2)
            )

    def _validate_file_exists(self, file_path: Path) -> None:
        """Validate that file exists and is a regular file."""
        if not file_path.exists():
            self.log.error(f"File not found: {file_path}")
            raise Exit(code=1) from None

        if not file_path.is_file():
            self.log.error(f"File is not a regular file: {file_path}")
            raise Exit(code=1) from None

    def _validate_loaded_data(self) -> None:
        """Validate that required data is present."""
        input_data = cast(
            FunctionCallingTestsFile,
            self.data.get("input_file"),
        )
        functions_definition_data = cast(
            FunctionDefinitionsFile,
            self.data.get("functions_definition_file"),
        )

        if (
            not input_data or
            not input_data.root
        ):
            self.log.error("No input data available.")
            raise Exit(code=1) from None

        if (
            not functions_definition_data or
            not functions_definition_data.root
        ):
            self.log.error("No function definitions available.")
            raise Exit(code=1) from None

    def _create_output_dir(self) -> None:
        """Create output directory if it doesn't exist."""
        try:
            if self.output_file.exists() and not self.output_file.is_dir():
                raise FileExistsError(
                    f"Output path '{self.output_file}'"
                    " exists but is not a directory."
                )
            self.output_file.mkdir(parents=True, exist_ok=True)
        except (PermissionError, FileExistsError, OSError) as e:
            self.log.error(f"Error creating output directory: {e}")
            raise Exit(code=1) from None

    def _load_json(self, file_path: Path, model: Any) -> Any:
        """Load and parse JSON from file."""
        try:
            with open(file_path) as f:
                return model.model_validate_json(f.read())
        except JSONDecodeError as e:
            self.log.error(f"Error loading JSON from {file_path}: {e}")
            raise Exit(code=1) from None
        except ValidationError as e:
            self.log.error(f"Invalid structure in {file_path}: {e}")
            raise Exit(code=1) from None
