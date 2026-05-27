from pydantic import RootModel
from .funtion_definition import FunctionDefinition


class FunctionDefinitions(RootModel[list[FunctionDefinition]]):
    """Root model representing a list of function definitions.

    Wraps a list of FunctionDefinition objects for JSON schema
    validation at the root level.
    """
    pass
