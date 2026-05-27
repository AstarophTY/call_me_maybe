from pydantic import BaseModel, ConfigDict


class StrictBaseModel(BaseModel):
    """Base model that forbids extra fields in validation.

    Ensures that only fields explicitly defined in child models are
    accepted during Pydantic validation.
    """
    model_config = ConfigDict(extra="forbid")
