"""
Test script to verify SGLang config file integration.
"""

import os
import tempfile

import pytest
import yaml

from sglang.srt.server_args import prepare_server_args
from sglang.srt.server_args_config_parser import ConfigArgumentMerger


@pytest.fixture
def merger():
    """Fixture providing a ConfigArgumentMerger instance."""
    return ConfigArgumentMerger()


def test_server_args_config_parser(merger):
    """Test the config parser functionality."""
    # Create a temporary config file
    config_data = {
        "model-path": "microsoft/DialoGPT-medium",
        "host": "0.0.0.0",
        "port": 30000,
        "tensor-parallel-size": 2,
        "trust-remote-code": False,
        "enable-metrics": True,
        "stream-output": True,
        "skip-server-warmup": False,
        "log-requests": True,
        "show-time-cost": True,
        "is-embedding": False,
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config_data, f)
        config_file = f.name

    try:
        # Test config parser directly
        config_args = merger._parse_yaml_config(config_file)

        # Test merging with CLI args
        cli_args = ["--config", config_file, "--max-running-requests", "128"]
        merged_args = merger.merge_config_with_args(cli_args)

        # Verify the merged args contain both config and CLI values
        assert "--model-path" in merged_args
        assert "microsoft/DialoGPT-medium" in merged_args
        assert "--host" in merged_args
        assert "0.0.0.0" in merged_args
        assert "--port" in merged_args
        assert "30000" in merged_args
        assert "--tensor-parallel-size" in merged_args
        assert "2" in merged_args
        assert "--max-running-requests" in merged_args
        assert "128" in merged_args

        # Test boolean arguments
        assert "--enable-metrics" in merged_args  # True boolean
        assert "--stream-output" in merged_args  # True boolean
        assert "--log-requests" in merged_args  # True boolean
        assert "--show-time-cost" in merged_args  # True boolean
        # False booleans should not be present (only add flag if True)
        assert "--trust-remote-code" not in merged_args  # False boolean
        assert "--skip-server-warmup" not in merged_args  # False boolean
        assert "--is-embedding" not in merged_args  # False boolean

    finally:
        os.unlink(config_file)


def test_server_args_integration():
    """Test the integration with server args."""
    # Create a temporary config file
    config_data = {
        "model-path": "microsoft/DialoGPT-medium",
        "host": "0.0.0.0",
        "port": 30000,
        "tensor-parallel-size": 1,
        "max-running-requests": 256,
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config_data, f)
        config_file = f.name

    try:
        # Test with config file
        argv = ["--config", config_file]
        server_args = prepare_server_args(argv)

        # Verify that config values were loaded
        assert server_args.model_path == "microsoft/DialoGPT-medium"
        assert server_args.host == "0.0.0.0"
        assert server_args.port == 30000
        assert server_args.tp_size == 1
        assert server_args.max_running_requests == 256

    finally:
        os.unlink(config_file)


def test_cli_override():
    """Test that CLI arguments override config file values."""
    # Create a temporary config file
    config_data = {
        "model-path": "microsoft/DialoGPT-medium",
        "port": 30000,
        "tensor-parallel-size": 1,
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config_data, f)
        config_file = f.name

    try:
        # Test CLI override (CLI should take precedence)
        argv = [
            "--config",
            config_file,
            "--port",
            "40000",
            "--tensor-parallel-size",
            "2",
        ]
        server_args = prepare_server_args(argv)

        # Verify that CLI values override config values
        assert server_args.model_path == "microsoft/DialoGPT-medium"  # From config
        assert server_args.port == 40000  # From CLI (overrides config)
        assert server_args.tp_size == 2  # From CLI (overrides config)

    finally:
        os.unlink(config_file)


def test_error_handling():
    """Test error handling for invalid config files."""
    # Test non-existent config file
    with pytest.raises(ValueError, match="Config file not found"):
        argv = ["--config", "non-existent.yaml"]
        prepare_server_args(argv)

    # Test invalid YAML file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write("invalid: yaml: content: [")
        invalid_yaml_file = f.name

    try:
        with pytest.raises(Exception):
            argv = ["--config", invalid_yaml_file]
            prepare_server_args(argv)
    finally:
        os.unlink(invalid_yaml_file)
