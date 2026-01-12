"""
Test script to verify SGLang config file integration.

Tests the ConfigArgumentMerger and prepare_server_args functionality,
ensuring proper merging of YAML config files with CLI arguments.
"""

import argparse
import os
import tempfile

import pytest
import yaml

from sglang.srt.server_args import ServerArgs, prepare_server_args
from sglang.srt.server_args_config_parser import ConfigArgumentMerger


@pytest.fixture
def merger():
    """Fixture providing a ConfigArgumentMerger instance."""
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    return ConfigArgumentMerger(parser)


class TestConfigArgumentMerger:
    """Tests for ConfigArgumentMerger class."""

    def test_parse_yaml_config_basic(self, merger):
        """Test parsing a basic YAML config file."""
        config_data = {
            "model-path": "microsoft/DialoGPT-medium",
            "host": "0.0.0.0",
            "port": 30000,
            "tensor-parallel-size": 2,
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            config_file = f.name

        try:
            config_args = merger._parse_yaml_config(config_file)

            assert config_args["model-path"] == "microsoft/DialoGPT-medium"
            assert config_args["host"] == "0.0.0.0"
            assert config_args["port"] == 30000
            assert config_args["tensor-parallel-size"] == 2
        finally:
            os.unlink(config_file)

    def test_parse_config_returns_dict(self, merger):
        """Test that parse_config returns a dictionary with normalized keys."""
        config_data = {
            "model-path": "microsoft/DialoGPT-medium",
            "max-running-requests": 128,
            "tensor-parallel-size": 2,
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            config_file = f.name

        try:
            cli_args = ["--config", config_file]
            parsed_config = merger.parse_config(cli_args)

            # Keys should be normalized to use underscores (argparse dest format)
            assert "model_path" in parsed_config
            assert parsed_config["model_path"] == "microsoft/DialoGPT-medium"
            assert "max_running_requests" in parsed_config
            assert parsed_config["max_running_requests"] == 128
            assert "tensor_parallel_size" in parsed_config
            assert parsed_config["tensor_parallel_size"] == 2
        finally:
            os.unlink(config_file)

    def test_parse_config_with_boolean_values(self, merger):
        """Test parsing config with boolean values."""
        config_data = {
            "model-path": "microsoft/DialoGPT-medium",
            "trust-remote-code": True,
            "enable-metrics": True,
            "skip-server-warmup": False,
            "log-requests": True,
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            config_file = f.name

        try:
            cli_args = ["--config", config_file]
            parsed_config = merger.parse_config(cli_args)

            assert parsed_config["trust_remote_code"] is True
            assert parsed_config["enable_metrics"] is True
            assert parsed_config["skip_server_warmup"] is False
            assert parsed_config["log_requests"] is True
        finally:
            os.unlink(config_file)

    def test_remove_config_from_argv(self, merger):
        """Test removing --config and its value from argv."""
        argv = [
            "--config",
            "config.yaml",
            "--port",
            "30000",
            "--tensor-parallel-size",
            "2",
        ]
        result = merger.remove_config_from_argv(argv)

        assert "--config" not in result
        assert "config.yaml" not in result
        assert "--port" in result
        assert "30000" in result

    def test_invalid_choice_raises_error(self, merger):
        """Test that invalid choice values raise ValueError."""
        config_data = {
            "model-path": "test-model",
            "quantization": "invalid_quantization_method",
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            config_file = f.name

        try:
            cli_args = ["--config", config_file]
            with pytest.raises(ValueError, match="Invalid value"):
                merger.parse_config(cli_args)
        finally:
            os.unlink(config_file)


class TestPrepareServerArgs:
    """Tests for prepare_server_args function."""

    def test_basic_config_loading(self):
        """Test loading basic configuration from YAML file."""
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
            argv = ["--config", config_file]
            server_args = prepare_server_args(argv)

            assert server_args.model_path == "microsoft/DialoGPT-medium"
            assert server_args.host == "0.0.0.0"
            assert server_args.port == 30000
            assert server_args.tp_size == 1
            assert server_args.max_running_requests == 256
        finally:
            os.unlink(config_file)

    def test_cli_overrides_config(self):
        """Test that CLI arguments override config file values."""
        config_data = {
            "model-path": "microsoft/DialoGPT-medium",
            "port": 30000,
            "tensor-parallel-size": 1,
            "max-running-requests": 128,
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            config_file = f.name

        try:
            argv = [
                "--config",
                config_file,
                "--port",
                "40000",
                "--tensor-parallel-size",
                "2",
            ]
            server_args = prepare_server_args(argv)

            # Config values that are NOT overridden
            assert server_args.model_path == "microsoft/DialoGPT-medium"
            assert server_args.max_running_requests == 128

            # CLI values that override config
            assert server_args.port == 40000
            assert server_args.tp_size == 2
        finally:
            os.unlink(config_file)

    def test_boolean_config_values(self):
        """Test boolean values from config file."""
        config_data = {
            "model-path": "microsoft/DialoGPT-medium",
            "enable-metrics": True,
            "log-requests": True,
            "show-time-cost": True,
            "skip-server-warmup": True,
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            config_file = f.name

        try:
            argv = ["--config", config_file]
            server_args = prepare_server_args(argv)

            assert server_args.enable_metrics is True
            assert server_args.log_requests is True
            assert server_args.show_time_cost is True
            assert server_args.skip_server_warmup is True
        finally:
            os.unlink(config_file)

    def test_boolean_false_config_values(self):
        """Test that False boolean values are properly applied."""
        config_data = {
            "model-path": "microsoft/DialoGPT-medium",
            "enable-metrics": False,
            "log-requests": False,
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            config_file = f.name

        try:
            argv = ["--config", config_file]
            server_args = prepare_server_args(argv)

            assert server_args.enable_metrics is False
            assert server_args.log_requests is False
        finally:
            os.unlink(config_file)


class TestErrorHandling:
    """Tests for error handling in config parsing."""

    def test_nonexistent_config_file(self):
        """Test error handling for non-existent config file."""
        with pytest.raises(ValueError, match="Config file not found"):
            argv = ["--config", "non-existent-config-file.yaml"]
            prepare_server_args(argv)

    def test_invalid_yaml_syntax(self):
        """Test error handling for invalid YAML syntax."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("invalid: yaml: content: [")
            invalid_yaml_file = f.name

        try:
            with pytest.raises(Exception):
                argv = ["--config", invalid_yaml_file]
                prepare_server_args(argv)
        finally:
            os.unlink(invalid_yaml_file)

    def test_non_yaml_file_extension(self):
        """Test error handling for non-YAML file extension."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write('{"model-path": "test"}')
            non_yaml_file = f.name

        try:
            with pytest.raises(ValueError, match="Config file must be YAML format"):
                argv = ["--config", non_yaml_file]
                prepare_server_args(argv)
        finally:
            os.unlink(non_yaml_file)

    def test_multiple_config_files_error(self, merger):
        """Test error when multiple config files are specified."""
        with pytest.raises(ValueError, match="Multiple config files specified"):
            merger.parse_config(["--config", "file1.yaml", "--config", "file2.yaml"])

    def test_config_flag_without_path(self, merger):
        """Test error when --config flag has no path."""
        with pytest.raises(ValueError, match="No config file specified"):
            merger.parse_config(["--config"])

    def test_non_dict_root_config(self, merger):
        """Test error when config file root is not a dictionary."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("- item1\n- item2\n- item3")
            config_file = f.name

        try:
            with pytest.raises(ValueError, match="dictionary at root level"):
                merger._parse_yaml_config(config_file)
        finally:
            os.unlink(config_file)


class TestEdgeCases:
    """Tests for edge cases in config parsing."""

    def test_empty_config_file(self, merger):
        """Test handling of empty config file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("")
            config_file = f.name

        try:
            config_args = merger._parse_yaml_config(config_file)
            assert config_args == {}
        finally:
            os.unlink(config_file)

    def test_no_config_returns_empty_dict(self, merger):
        """Test that parse_config returns empty dict when no config specified."""
        result = merger.parse_config(["--port", "30000"])
        assert result == {}

    def test_config_with_underscore_keys(self, merger):
        """Test that config keys with underscores are properly handled."""
        config_data = {
            "model_path": "microsoft/DialoGPT-medium",
            "max_running_requests": 128,
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            config_file = f.name

        try:
            cli_args = ["--config", config_file]
            parsed_config = merger.parse_config(cli_args)

            assert "model_path" in parsed_config
            assert parsed_config["model_path"] == "microsoft/DialoGPT-medium"
        finally:
            os.unlink(config_file)

    def test_config_with_list_values(self):
        """Test config file with list values."""
        config_data = {
            "model-path": "microsoft/DialoGPT-medium",
            "encoder-urls": ["url1", "url2"],
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            config_file = f.name

        try:
            argv = ["--config", config_file]
            server_args = prepare_server_args(argv)

            assert server_args.encoder_urls == ["url1", "url2"]
        finally:
            os.unlink(config_file)

    def test_yml_extension(self, merger):
        """Test that .yml extension is also accepted."""
        config_data = {
            "model-path": "microsoft/DialoGPT-medium",
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            yaml.dump(config_data, f)
            config_file = f.name

        try:
            config_args = merger._parse_yaml_config(config_file)
            assert config_args["model-path"] == "microsoft/DialoGPT-medium"
        finally:
            os.unlink(config_file)


class TestTypeValidation:
    """Tests for type validation in config parsing."""

    def test_store_true_rejects_integer(self, merger):
        """Test that store_true actions reject integer values."""
        config_data = {
            "model-path": "test-model",
            "trust-remote-code": 1,  # Should be boolean, not int
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            config_file = f.name

        try:
            cli_args = ["--config", config_file]
            with pytest.raises(ValueError, match="Expected boolean"):
                merger.parse_config(cli_args)
        finally:
            os.unlink(config_file)

    def test_store_true_rejects_string(self, merger):
        """Test that store_true actions reject string values."""
        config_data = {
            "model-path": "test-model",
            "enable-metrics": "yes",  # Should be boolean, not string
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            config_file = f.name

        try:
            cli_args = ["--config", config_file]
            with pytest.raises(ValueError, match="Expected boolean"):
                merger.parse_config(cli_args)
        finally:
            os.unlink(config_file)

    def test_store_true_accepts_boolean_true(self, merger):
        """Test that store_true actions accept True value."""
        config_data = {
            "model-path": "test-model",
            "trust-remote-code": True,
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            config_file = f.name

        try:
            cli_args = ["--config", config_file]
            parsed_config = merger.parse_config(cli_args)
            assert parsed_config["trust_remote_code"] is True
        finally:
            os.unlink(config_file)

    def test_store_true_accepts_boolean_false(self, merger):
        """Test that store_true actions accept False value."""
        config_data = {
            "model-path": "test-model",
            "trust-remote-code": False,
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            config_file = f.name

        try:
            cli_args = ["--config", config_file]
            parsed_config = merger.parse_config(cli_args)
            assert parsed_config["trust_remote_code"] is False
        finally:
            os.unlink(config_file)

    def test_type_conversion_int(self, merger):
        """Test that integer type conversion is applied."""
        config_data = {
            "model-path": "test-model",
            "port": "30000",  # String that should be converted to int
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            config_file = f.name

        try:
            cli_args = ["--config", config_file]
            parsed_config = merger.parse_config(cli_args)
            assert parsed_config["port"] == 30000
            assert isinstance(parsed_config["port"], int)
        finally:
            os.unlink(config_file)

    def test_type_conversion_failure(self, merger):
        """Test that invalid type conversion raises ValueError."""
        config_data = {
            "model-path": "test-model",
            "port": "not_a_number",  # Cannot be converted to int
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            config_file = f.name

        try:
            cli_args = ["--config", config_file]
            with pytest.raises(ValueError, match="Type conversion failed"):
                merger.parse_config(cli_args)
        finally:
            os.unlink(config_file)

    def test_type_conversion_float(self, merger):
        """Test that float type conversion is applied."""
        config_data = {
            "model-path": "test-model",
            "mem-fraction-static": "0.85",  # String that should be converted to float
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            config_file = f.name

        try:
            cli_args = ["--config", config_file]
            parsed_config = merger.parse_config(cli_args)
            assert parsed_config["mem_fraction_static"] == 0.85
            assert isinstance(parsed_config["mem_fraction_static"], float)
        finally:
            os.unlink(config_file)


if __name__ == "__main__":
    pytest.main([__file__])
