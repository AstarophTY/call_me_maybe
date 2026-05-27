from pydantic import RootModel

from .function_calling_test import FunctionCallingTest


class FunctionCallingTests(RootModel[list[FunctionCallingTest]]):
    """Root model representing a list of function calling tests.

    Wraps a list of FunctionCallingTest objects for JSON schema
    validation at the root level.
    """
    pass
