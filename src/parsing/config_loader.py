from logging import getLogger
from pathlib import Path
from typing import Any, cast

from .config_loader_helpers import (
    create_output_dir,
    load_validated_json,
    validate_file_exists,
    validate_loaded_data,
)
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

        loaders = [
            ("input_file", self.input_file, FunctionCallingTests),
            (
                "functions_definition_file",
                self.functions_definition_file,
                FunctionDefinitions,
            ),
        ]

        for file_name, file_path, model in loaders:
            validate_file_exists(file_path, self.log)
            self.data[file_name] = load_validated_json(
                file_path,
                cast(Any, model),
                self.log,
            )

        self.input_data = cast(
            FunctionCallingTests, self.data.get("input_file")
        )

        self.functions_definition_data = cast(
            FunctionDefinitions,
            self.data.get("functions_definition_file"),
        )

        validate_loaded_data(
            self.input_data,
            self.functions_definition_data,
            self.log,
        )
        create_output_dir(self.output_file, self.log)
