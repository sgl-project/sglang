"""Unit tests for diffusion server args config validation."""

import json
from pathlib import Path
import pytest

from sglang.multimodal_gen.runtime.server_args.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def test_load_config_file_valid_yaml(tmp_path: Path):
    config_file = tmp_path / "valid.yaml"
    config_file.write_text("port: 8000\nhost: 127.0.0.1\n")
    data = ServerArgs.load_config_file(str(config_file))
    assert data == {"port": 8000, "host": "127.0.0.1"}


def test_load_config_file_valid_json(tmp_path: Path):
    config_file = tmp_path / "valid.json"
    config_file.write_text(json.dumps({"port": 8000, "host": "127.0.0.1"}))
    data = ServerArgs.load_config_file(str(config_file))
    assert data == {"port": 8000, "host": "127.0.0.1"}


def test_load_config_file_empty_yaml(tmp_path: Path):
    config_file = tmp_path / "empty.yaml"
    config_file.write_text("")
    data = ServerArgs.load_config_file(str(config_file))
    assert data == {}


def test_load_config_file_comment_only_yaml(tmp_path: Path):
    config_file = tmp_path / "comments.yaml"
    config_file.write_text("# This is a comment only file\n# Another comment\n")
    data = ServerArgs.load_config_file(str(config_file))
    assert data == {}


def test_load_config_file_null_json(tmp_path: Path):
    config_file = tmp_path / "null.json"
    config_file.write_text("null")
    data = ServerArgs.load_config_file(str(config_file))
    assert data == {}


def test_load_config_file_rejects_yaml_list(tmp_path: Path):
    config_file = tmp_path / "list.yaml"
    config_file.write_text("- item1\n- item2\n")
    with pytest.raises(ValueError, match="must contain a mapping/dictionary at the root level"):
        ServerArgs.load_config_file(str(config_file))


def test_load_config_file_rejects_json_list(tmp_path: Path):
    config_file = tmp_path / "list.json"
    config_file.write_text(json.dumps(["item1", "item2"]))
    with pytest.raises(ValueError, match="must contain a mapping/dictionary at the root level"):
        ServerArgs.load_config_file(str(config_file))


def test_load_config_file_rejects_scalar(tmp_path: Path):
    config_file = tmp_path / "scalar.yaml"
    config_file.write_text("just_a_string\n")
    with pytest.raises(ValueError, match="must contain a mapping/dictionary at the root level"):
        ServerArgs.load_config_file(str(config_file))


def test_load_config_file_rejects_unsupported_format(tmp_path: Path):
    config_file = tmp_path / "config.toml"
    config_file.write_text("port = 8000\n")
    with pytest.raises(ValueError, match="Unsupported config file format"):
        ServerArgs.load_config_file(str(config_file))
