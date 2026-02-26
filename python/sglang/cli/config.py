from pydantic import BaseModel, field_validator


class ServerConfig(BaseModel):
    model_path: str
    port: int
    host: str = "127.0.0.1"

    @field_validator("port")
    @classmethod
    def validate_port(cls, v):
        if v < 1024:
            raise ValueError("Port must be >= 1024")
        return v

    @field_validator("model_path")
    @classmethod
    def validate_model_path(cls, v):
        if not v:
            raise ValueError("Model path cannot be empty")
        return v
