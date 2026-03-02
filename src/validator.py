import json
from pathlib import Path
from typing import List
from pydantic import BaseModel, FilePath, model_validator
from enum import Enum


class Types(str, Enum):
    """Types for function

    Args:
        str (name): Name of type
        Enum (name): Name of type
    """
    float = "float"
    int = "int"
    str = "str"
    bool = "bool"


class FunctionValidation(BaseModel):
    """Check function if is good format
    """
    fn_name: str
    description: str = ""
    args_names: List[str]
    args_types: dict[str, Types]
    return_type: Types


class PromptValidation(BaseModel):
    """Check prompt if is valid
    """
    prompt: str


class ParsingValidation(BaseModel):
    """Valid and create parsing object

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
        """Validate and build function and prompts

        Returns:
            ParsingValidation: Parsing object with prompt functions
        """
        with open(self.functions_definition, "r") as f:
            data = json.load(f)
            functions = []
            type_mapping = {
                "number": Types.int,
                "string": Types.str,
                "boolean": Types.bool,
                "float": Types.float,
            }
            for fn in data:
                fn_name = fn["name"]
                args_names = list(fn["parameters"].keys())
                args_types = {
                    k: type_mapping.get(v["type"], Types.str)
                    for k, v in fn["parameters"].items()
                }
                return_type = type_mapping.get(
                    fn["returns"]["type"], Types.str
                )
                functions.append(FunctionValidation(
                    fn_name=fn_name,
                    description=fn.get("description", ""),
                    args_names=args_names,
                    args_types=args_types,
                    return_type=return_type,
                ))
            self.functions = functions

        with open(self.input, "r") as f:
            data = json.load(f)
            self.prompts = [PromptValidation(**p) for p in data]

        return self
