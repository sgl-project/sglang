"""
Test script to verify SGLang config file integration.

Tests the ConfigArgumentMerger and prepare_server_args functionality,
ensuring proper merging of YAML config files with CLI arguments.
"""

import argparse
import os
import tempfile
from contextlib import contextmanager

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


@contextmanager
def temp_yaml_config(config_data, suffix=".yaml"):
    """Context manager for creating temporary YAML config files."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=suffix, delete=False) as f:
        yaml.dump(config_data, f)
        config_file = f.name
    try:
        yield config_file
    finally:
        os.unlink(config_file)


class TestConfigArgumentMerger:
    """Tests for ConfigArgumentMerger class."""

    def test_parse_config_basic(self, merger):
        """Test parsing and normalizing config keys."""
        config = {"model-path": "test-model", "max-running-requests": 128}
        with temp_yaml_config(config) as f:
            result = merger.parse_config(["--config", f])
            assert result["model_path"] == "test-model"
            assert result["max_running_requests"] == 128

    def test_parse_config_with_booleans(self, merger):
        """Test boolean values parsing."""
        config = {
            "model-path": "test",
            "trust-remote-code": True,
            "enable-metrics": False,
        }
        with temp_yaml_config(config) as f:
            result = merger.parse_config(["--config", f])
            assert result["trust_remote_code"] is True
            assert result["enable_metrics"] is False

    def test_remove_config_from_argv(self, merger):
        """Test removing --config from argv."""
        argv = ["--config", "config.yaml", "--port", "30000"]
        result = merger.remove_config_from_argv(argv)
        assert result == ["--port", "30000"]

    def test_invalid_choice_raises_error(self, merger):
        """Test invalid choice raises ValueError."""
        config = {"model-path": "test", "quantization": "invalid_method"}
        with temp_yaml_config(config) as f:
            with pytest.raises(ValueError, match="Invalid value"):
                merger.parse_config(["--config", f])


class TestPrepareServerArgs:
    """Tests for prepare_server_args function."""

    def test_basic_config_loading(self):
        """Test loading config into ServerArgs."""
        config = {"model-path": "test-model", "port": 30000, "tensor-parallel-size": 1}
        with temp_yaml_config(config) as f:
            args = prepare_server_args(["--config", f])
            assert args.model_path == "test-model"
            assert args.port == 30000
            assert args.tp_size == 1

    def test_cli_overrides_config(self):
        """Test CLI arguments override config values."""
        config = {"model-path": "test-model", "port": 30000}
        with temp_yaml_config(config) as f:
            args = prepare_server_args(["--config", f, "--port", "40000"])
            assert args.model_path == "test-model"  # from config
            assert args.port == 40000  # CLI override


class TestErrorHandling:
    """Tests for error handling."""

    def test_nonexistent_config_file(self):
        """Test non-existent file error."""
        with pytest.raises(ValueError, match="Config file not found"):
            prepare_server_args(["--config", "non-existent.yaml"])

    def test_non_yaml_extension(self):
        """Test non-YAML extension error."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            f.write(b'{"key": "value"}')
            try:
                with pytest.raises(ValueError, match="YAML format"):
                    prepare_server_args(["--config", f.name])
            finally:
                os.unlink(f.name)

    def test_multiple_config_files(self, merger):
        """Test multiple config files error."""
        with pytest.raises(ValueError, match="Multiple config"):
            merger.parse_config(["--config", "a.yaml", "--config", "b.yaml"])

    def test_config_flag_without_path(self, merger):
        """Test --config without path error."""
        with pytest.raises(ValueError, match="No config file specified"):
            merger.parse_config(["--config"])


class TestTypeValidation:
    """Tests for type validation in config parsing."""

    def test_store_true_rejects_non_bool(self, merger):
        """Test store_true rejects non-boolean values."""
        for bad_value in [1, "yes", "true"]:
            config = {"model-path": "test", "trust-remote-code": bad_value}
            with temp_yaml_config(config) as f:
                with pytest.raises(ValueError, match="Expected boolean"):
                    merger.parse_config(["--config", f])

    def test_store_true_accepts_bool(self, merger):
        """Test store_true accepts True/False."""
        for val in [True, False]:
            config = {"model-path": "test", "trust-remote-code": val}
            with temp_yaml_config(config) as f:
                result = merger.parse_config(["--config", f])
                assert result["trust_remote_code"] is val

    def test_type_conversion(self, merger):
        """Test type conversion for int/float."""
        config = {"model-path": "test", "port": "30000", "mem-fraction-static": "0.85"}
        with temp_yaml_config(config) as f:
            result = merger.parse_config(["--config", f])
            assert result["port"] == 30000 and isinstance(result["port"], int)
            assert result["mem_fraction_static"] == 0.85

    def test_type_conversion_failure(self, merger):
        """Test invalid type conversion raises error."""
        config = {"model-path": "test", "port": "not_a_number"}
        with temp_yaml_config(config) as f:
            with pytest.raises(ValueError, match="Type conversion failed"):
                merger.parse_config(["--config", f])

    def test_null_passthrough(self, merger):
        """Test null values pass through."""
        config = {"model-path": "test", "quantization-param-path": None}
        with temp_yaml_config(config) as f:
            result = merger.parse_config(["--config", f])
            assert result["quantization_param_path"] is None

    def test_dict_skips_json_conversion(self, merger):
        """Test dict values skip json.loads."""
        config = {"model-path": "test", "mm-process-config": {"image": {"resize": 224}}}
        with temp_yaml_config(config) as f:
            result = merger.parse_config(["--config", f])
            assert result["mm_process_config"] == {"image": {"resize": 224}}

    def test_list_values(self):
        """Test list values with nargs."""
        config = {
            "model-path": "test",
            "log-requests": True,
            "log-requests-target": ["a", "b"],
        }
        with temp_yaml_config(config) as f:
            args = prepare_server_args(["--config", f])
            assert args.log_requests_target == ["a", "b"]


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_config(self, merger):
        """Test empty config file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("")
            try:
                assert merger._parse_yaml_config(f.name) == {}
            finally:
                os.unlink(f.name)

    def test_no_config_returns_empty(self, merger):
        """Test no --config returns empty dict."""
        assert merger.parse_config(["--port", "30000"]) == {}

    def test_yml_extension(self, merger):
        """Test .yml extension works."""
        config = {"model-path": "test"}
        with temp_yaml_config(config, suffix=".yml") as f:
            result = merger._parse_yaml_config(f)
            assert result["model-path"] == "test"


if __name__ == "__main__":
    pytest.main([__file__])
