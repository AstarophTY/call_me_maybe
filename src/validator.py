import json
from pathlib import Path
from typing import List
from pydantic import BaseModel, FilePath, model_validator
from enum import Enum


class Types(str, Enum):
    """Types for function.

    Args:
        str (name): Name of type
        Enum (name): Name of type
    """
    number = "number"
    string = "string"


class FunctionValidation(BaseModel):
    """Check function if is good format."""
    name: str
    description: str = ""
    parameters: dict[str, dict[str, str]]
    return_type: dict[str, Types]


class PromptValidation(BaseModel):
    """Check prompt if is valid."""
    prompt: str


class ParsingValidation(BaseModel):
    """Valid and create parsing object.

    Returns:
        ParsingValidation: Parsing object
    """

    functions_definition: FilePath = Path(
        "data/input/functions_definition.json"
    )
    input: FilePath = Path("data/input/function_calling_tests.json")
    output: str = "data/output/function_calling_results.json"
    functions: List[FunctionValidation] = []
    prompts: List[PromptValidation] = []

    @model_validator(mode="after")
    def load_and_validate_contents(self) -> 'ParsingValidation':
        """Validate and build function and prompts.

        Returns:
            ParsingValidation: Parsing object with prompt functions
        """
        with open(self.functions_definition, "r") as f:
            data = json.load(f)
            functions = []
            for fn in data:
                functions.append(FunctionValidation(
                    name=fn["name"],
                    description=fn.get("description", ""),
                    parameters=fn["parameters"],
                    return_type=fn["returns"],
                ))
            self.functions = functions

        with open(self.input, "r") as f:
            data = json.load(f)
            self.prompts = [PromptValidation(**p) for p in data]

        return self
