from pathlib import Path
from logging import getLogger
from typing import Any, cast
from typer import Exit
from json import JSONDecodeError, dumps
from pydantic import ValidationError

from .validator import FunctionCallingTests, FunctionDefinitions


class ConfigLoader:
    """Load and validate function definitions and test prompts
    from JSON files.

    This loader parses two JSON files against Pydantic schemas:

        * ``input_file``: List of function calling test prompts.
        * ``functions_definition_file``: List of function definitions
          with parameters and return types.

    Validates file existence, creates output directory, and stores
    parsed data for later use.
    """

    def __init__(
        self,
        input_file: str,
        output_file: str,
        functions_definition_file: str,
    ) -> None:
        """Initialize the ConfigLoader with input and output paths.

        :param input_file: Path to function calling tests JSON.
        :param output_file: Directory for output results.
        :param functions_definition_file: Path to function definitions
            JSON.
        """
        self.log = getLogger(__name__)
        self.input_file = Path(input_file)
        self.output_file = Path(output_file)
        self.functions_definition_file = Path(functions_definition_file)
        self.data: dict[str, Any] = {}

        self._load_required_files()
        self._create_output_dir()
        self._validate_loaded_data()

    def _load_required_files(self) -> None:
        """Load and validate input and functions definition files.

        Parses JSON files against their respective Pydantic models
        and stores the validated data.
        """
        loaders = [
            ("input_file", self.input_file, FunctionCallingTests),
            (
                "functions_definition_file",
                self.functions_definition_file,
                FunctionDefinitions,
            ),
        ]

        for file_name, file_path, model in loaders:
            self._validate_file_exists(file_path)
            self.data[file_name] = self._load_json(file_path, model)
            self.log.debug(
                dumps(cast(Any, self.data[file_name]).model_dump(), indent=2)
            )

    def _validate_file_exists(self, file_path: Path) -> None:
        """Validate that file exists and is a regular file.

        Raises an exit error if the file does not exist or is not a
        regular file.

        :param file_path: Path object to validate.
        """
        if not file_path.exists():
            self.log.error(f"File not found: {file_path}")
            raise Exit(code=1) from None

        if not file_path.is_file():
            self.log.error(f"File is not a regular file: {file_path}")
            raise Exit(code=1) from None

    def _validate_loaded_data(self) -> None:
        """Validate that required data is present.

        Ensures both input and function definition data have been
        loaded and contain non-empty root lists.
        """
        self.input_data = cast(
            FunctionCallingTests, self.data.get("input_file")
        )

        self.functions_definition_data = cast(
            FunctionDefinitions,
            self.data.get("functions_definition_file"),
        )

        if (
            not self.input_data or
            not self.input_data.root
        ):
            self.log.error("No input data available.")
            raise Exit(code=1) from None

        if (
            not self.functions_definition_data or
            not self.functions_definition_data.root
        ):
            self.log.error("No function definitions available.")
            raise Exit(code=1) from None

    def _create_output_dir(self) -> None:
        """Create output directory if it doesn't exist.

        Raises an exit error if the path exists but is not a
        directory, or if creation fails.
        """
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
        """Load and parse JSON from file using Pydantic model.

        Validates JSON structure against the provided model and
        raises exit errors on JSON or validation failures.

        :param file_path: Path to JSON file to load.
        :param model: Pydantic model class for validation.
        :returns: Validated model instance.
        """
        try:
            with open(file_path) as f:
                return model.model_validate_json(f.read())
        except JSONDecodeError as e:
            self.log.error(f"Error loading JSON from {file_path}: {e}")
            raise Exit(code=1) from None
        except ValidationError as e:
            self.log.error(f"Invalid structure in {file_path}: {e}")
            raise Exit(code=1) from None
