from pydantic import field_validator
from typing import Literal

from .strict_base_model import StrictBaseModel


class FunctionReturn(StrictBaseModel):
    """Represents a function's return type specification.

    The type must be one of the supported Pydantic types:
    string, number, boolean, or integer.
    """
    type: Literal["string", "number", "boolean", "integer"]

    @field_validator("type")
    @classmethod
    def validate_type(cls, value: str) -> str:
        """Validate that the return type is not empty after stripping.

        :param value: The return type value to validate.
        :returns: Stripped return type value.
        :raises ValueError: If return type is empty after stripping.
        """
        value = value.strip()
        if not value:
            raise ValueError("Return type cannot be empty.")
        return value
