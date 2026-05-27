from pydantic import field_validator
from typing import Literal

from .strict_base_model import StrictBaseModel


class FunctionParameter(StrictBaseModel):
    """Represents a single function parameter with its type.

    The type must be one of the supported Pydantic types:
    string, number, boolean, or integer.
    """
    type: Literal["string", "number", "boolean", "integer"]

    @field_validator("type")
    @classmethod
    def validate_type(cls, value: str) -> str:
        """Validate that the type field is not empty after stripping.

        :param value: The type value to validate.
        :returns: Stripped type value.
        :raises ValueError: If type is empty after stripping.
        """
        value = value.strip()
        if not value:
            raise ValueError("Parameter type cannot be empty.")
        return value
