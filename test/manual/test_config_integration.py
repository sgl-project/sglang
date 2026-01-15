"""
Test script to verify SGLang config file integration.

Tests the ConfigArgumentMerger and prepare_server_args functionality,
ensuring proper merging of YAML config files with CLI arguments.
"""

import argparse
import json
import os
import tempfile
from contextlib import contextmanager
from unittest.mock import patch

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


class DummyConfig:
    def __init__(self):
        self.architectures = ["LlamaForCausalLM"]
        self.model_type = "llama"
        self.max_position_embeddings = 4096
        self.num_hidden_layers = 32
        self.num_attention_heads = 32
        self.hidden_size = 4096
        self.vocab_size = 32000
        self.quantization_config = None

    def to_dict(self):
        return {"quantization_config": self.quantization_config}

    def __getitem__(self, key):
        return getattr(self, key)

    def get(self, key, default=None):
        return getattr(self, key, default)


@pytest.fixture(autouse=True)
def mock_model_loading():
    with patch("transformers.AutoConfig.from_pretrained", return_value=DummyConfig()):
        with patch("transformers.AutoTokenizer.from_pretrained"):
            yield


class TestConfigArgumentMerger:
    """Tests for ConfigArgumentMerger class."""

    def test_parse_config(self, merger):
        """Test parsing and normalizing config keys."""
        config = {
            "model-path": "microsoft/DialoGPT-medium",
            "max-running-requests": 128,
            "port": "30001",
            "mem-fraction-static": "0.85",
            "trust-remote-code": True,
            "enable-metrics": False,
            "log-requests-target": ["path1", "path2"],
            "quantization-param-path": "path3",
        }
        with temp_yaml_config(config) as f:
            result = merger.parse_config(["--config", f])
            assert result["model_path"] == "microsoft/DialoGPT-medium"
            assert result["max_running_requests"] == 128
            assert result["port"] == 30001
            assert result["mem_fraction_static"] == 0.85
            assert result["trust_remote_code"] is True
            assert result["enable_metrics"] is False
            assert result["log_requests_target"] == ["path1", "path2"]
            assert result["quantization_param_path"] == "path3"

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
        config = {
            "model-path": "microsoft/DialoGPT-medium",
            "port": 30000,
            "tensor-parallel-size": 1,
        }
        with temp_yaml_config(config) as f:
            args = prepare_server_args(["--config", f])
            assert args.model_path == "microsoft/DialoGPT-medium"
            assert args.port == 30000
            assert args.tp_size == 1

    def test_json_config_loading(self):
        """Test JSON config file loading."""
        config = {"model-path": "microsoft/DialoGPT-medium", "port": 30000}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config, f)
            try:
                args = prepare_server_args(["--config", f.name])
                assert args.model_path == "microsoft/DialoGPT-medium"
                assert args.port == 30000
            finally:
                os.unlink(f.name)

    def test_cli_overrides_config(self):
        """Test CLI arguments override config values."""
        config = {"model-path": "microsoft/DialoGPT-medium", "port": 30000}
        with temp_yaml_config(config) as f:
            args = prepare_server_args(["--config", f, "--port", "40000"])
            assert args.model_path == "microsoft/DialoGPT-medium"  # from config
            assert args.port == 40000  # CLI override


class TestErrorHandling:
    """Tests for error handling."""

    def test_nonexistent_config_file(self):
        """Test non-existent file error."""
        with pytest.raises(ValueError, match="Config file not found"):
            prepare_server_args(["--config", "non-existent.yaml"])

    def test_multiple_config_files(self, merger):
        """Test multiple config files error."""
        with pytest.raises(ValueError, match="Multiple config"):
            merger.parse_config(["--config", "a.yaml", "--config", "b.yaml"])

    def test_config_flag_without_path(self, merger):
        """Test --config without path error."""
        with pytest.raises(ValueError, match="No config file specified"):
            merger.parse_config(["--config"])

    def test_unsupported_extension(self):
        """Test unsupported extension error."""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"key: value")
            try:
                with pytest.raises(ValueError, match="YAML or JSON format"):
                    prepare_server_args(["--config", f.name])
            finally:
                os.unlink(f.name)

    def test_rejected_options_in_config(self, merger):
        """Test that config/help/h cannot be specified in config file."""
        for option in ["config", "help", "h", "hybrid-kvcache-ratio", "lora-path"]:
            config = {"model-path": "test", option: "value"}
            with temp_yaml_config(config) as f:
                with pytest.raises(ValueError, match="cannot be specified in config"):
                    merger.parse_config(["--config", f])


class TestTypeValidation:
    """Tests for type validation in config parsing."""

    def test_store_true_rejects_non_bool(self, merger):
        """Test store_true rejects non-boolean values."""
        for bad_value in [1, "hello"]:
            config = {"model-path": "test", "trust-remote-code": bad_value}
            with temp_yaml_config(config) as f:
                with pytest.raises(ValueError, match="Expected boolean"):
                    merger.parse_config(["--config", f])

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


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_config(self, merger):
        """Test empty config file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("")
            try:
                assert merger._parse_config_file(f.name) == {}
            finally:
                os.unlink(f.name)

    def test_no_config_returns_empty(self, merger):
        """Test no --config returns empty dict."""
        assert merger.parse_config(["--port", "30000"]) == {}

    def test_yml_extension(self, merger):
        """Test .yml extension works."""
        config = {"model-path": "test"}
        with temp_yaml_config(config, suffix=".yml") as f:
            result = merger._parse_config_file(f)
            assert result["model-path"] == "test"

    def test_unknown_option_ignored_with_warning(self, merger):
        """Test unknown config options are ignored with warning."""
        config = {"model-path": "test", "unknown-option-xyz": "value"}
        with temp_yaml_config(config) as f:
            # Should not raise, just log warning
            result = merger.parse_config(["--config", f])
            assert "unknown_option_xyz" not in result
            assert result["model_path"] == "test"

    def test_non_dict_root_raises_error(self, merger):
        """Test non-dict root config raises error."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("- item1\n- item2")
            try:
                with pytest.raises(ValueError, match="dictionary at root level"):
                    merger._parse_config_file(f.name)
            finally:
                os.unlink(f.name)

    def test_nullable_str_with_numeric_value(self, merger):
        """Test nullable_str converts numeric values to string."""
        config = {"model-path": "test", "quantization-param-path": 12345}
        with temp_yaml_config(config) as f:
            result = merger.parse_config(["--config", f])
            # Should be converted to string "12345"
            assert result["quantization_param_path"] == "12345"


if __name__ == "__main__":
    pytest.main([__file__])
