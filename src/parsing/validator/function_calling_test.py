from pydantic import field_validator

from .strict_base_model import StrictBaseModel


class FunctionCallingTest(StrictBaseModel):
    """Represents a single function calling test case.

    Contains a user prompt that should trigger a specific function
    call from the available function definitions.
    """
    prompt: str

    @field_validator("prompt")
    @classmethod
    def validate_prompt(cls, value: str) -> str:
        """Validate that the prompt is not empty after stripping.

        :param value: The prompt value to validate.
        :returns: Stripped prompt value.
        :raises ValueError: If prompt is empty after stripping.
        """
        value = value.strip()
        if not value:
            raise ValueError("Prompt cannot be empty.")
        return value
