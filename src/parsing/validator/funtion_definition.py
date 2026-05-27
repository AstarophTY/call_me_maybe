from pydantic import field_validator

from .strict_base_model import StrictBaseModel
from .function_parameter import FunctionParameter
from .function_return import FunctionReturn


class FunctionDefinition(StrictBaseModel):
    """Represents a complete function definition.

    Includes the function name, description, parameters map, and
    return type specification.
    """
    name: str
    description: str
    parameters: dict[str, FunctionParameter]
    returns: FunctionReturn

    @field_validator("name", "description")
    @classmethod
    def validate_text_fields(cls, value: str) -> str:
        """Validate that name and description are not empty.

        :param value: The field value to validate.
        :returns: Stripped field value.
        :raises ValueError: If field is empty after stripping.
        """
        value = value.strip()
        if not value:
            raise ValueError("Text fields cannot be empty.")
        return value

    @field_validator("parameters")
    @classmethod
    def validate_parameters_fields(
        cls,
        value: dict[str, FunctionParameter]
    ) -> dict[str, FunctionParameter]:
        """Validate that parameters map contains non-empty keys.

        :param value: The parameters dictionary to validate.
        :returns: The validated parameters dictionary.
        :raises ValueError: If any parameter key is empty.
        """
        if any(len(param) == 0 for param in value.keys()):
            raise ValueError("Parameters cannot be empty.")
        return value
