import pytest
from pydantic import ValidationError

from sglang.cli.config import ServerConfig


def test_server_config():
    server_config = ServerConfig(
        model_path="/path/to/model",
        port=8000,
        host="localhost",
    )
    assert server_config.model_path == "/path/to/model"
    assert server_config.port == 8000
    assert server_config.host == "localhost"


def test_server_config_invalid_port():
    with pytest.raises(ValidationError):
        ServerConfig(
            model_path="/path/to/model",
            port=0,
            host="localhost",
        )


def test_server_config_empty_model_path():
    with pytest.raises(ValidationError):
        ServerConfig(
            model_path="",
            port=8000,
            host="localhost",
        )
