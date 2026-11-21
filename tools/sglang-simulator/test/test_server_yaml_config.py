"""Simulator startup preserves native YAML parsing and CLI precedence."""

import argparse
import importlib.util
import sys
from pathlib import Path

import pytest
from sglang_simulator.compat import merge_server_config


@pytest.fixture
def native_parser(monkeypatch):
    path = (
        Path(__file__).resolve().parents[3]
        / "python/sglang/srt/utils/server_args_config_parser.py"
    )
    name = "sglang.srt.utils.server_args_config_parser"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setitem(sys.modules, name, module)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--max-running-requests", type=int, default=1)
    parser.add_argument("--skip-server-warmup", action="store_true")
    return parser


def test_native_yaml_and_explicit_cli_precedence(native_parser, tmp_path):
    path = tmp_path / "server.yaml"
    path.write_text(
        "model-path: /models/test\nmax-running-requests: 8\nskip-server-warmup: true\n",
        encoding="utf-8",
    )
    argv = ["--config", str(path), "--max-running-requests", "3"]
    parsed = native_parser.parse_args(merge_server_config(native_parser, argv))
    assert parsed.model_path == "/models/test"
    assert parsed.max_running_requests == 3
    assert parsed.skip_server_warmup


def test_direct_flags_require_no_yaml_merge():
    argv = ["--model-path", "/models/test"]
    assert merge_server_config(object(), argv) is argv


def test_native_config_errors_propagate(native_parser, tmp_path):
    with pytest.raises(ValueError, match="not found"):
        merge_server_config(native_parser, ["--config", str(tmp_path / "missing.yaml")])
