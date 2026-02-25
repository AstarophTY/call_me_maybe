import json
from pathlib import Path
from typing import List
from pydantic import BaseModel, FilePath, model_validator


class FunctionValidation(BaseModel):
    fn_name: str
    args_names: List[str]
    args_types: dict[str, str]
    return_type: str


class PromptValidation(BaseModel):
    prompt: str


class ParsingValidation(BaseModel):
    functions_definition: FilePath = Path(
        "data/input/functions_definition.json"
    )
    input: FilePath = Path("data/input/function_calling_tests.json")
    output: str = "data/output/function_calling_results.json"
    functions: List[FunctionValidation] = []
    prompts: List[PromptValidation] = []

    @model_validator(mode="after")
    def load_and_validate_contents(self) -> 'ParsingValidation':
        with open(self.functions_definition, "r") as f:
            data = json.load(f)
            self.functions = [FunctionValidation(**fn) for fn in data]

        with open(self.input, "r") as f:
            data = json.load(f)
            self.prompts = [PromptValidation(**p) for p in data]

        return self
