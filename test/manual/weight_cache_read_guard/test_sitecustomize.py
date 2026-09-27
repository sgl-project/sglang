# SPDX-License-Identifier: Apache-2.0
"""Fresh-interpreter checks for the opt-in weight-read acceptance guard."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file


@pytest.mark.parametrize("blocked", [False, True])
@pytest.mark.parametrize(
    "api", ["tensor", "slice", "load_file", "readonly", "reader", "torch"]
)
def test_guard_catches_actual_tensor_readers(tmp_path, blocked, api):
    checkpoint = tmp_path / "weights.safetensors"
    save_file({"weight": torch.ones(2)}, checkpoint)
    if api == "torch":
        checkpoint = tmp_path / "weights.pt"
        torch.save({"weight": torch.ones(2)}, checkpoint)
    logs = tmp_path / "events"
    logs.mkdir()
    script = """
import sys
from unittest.mock import patch
import torch
from safetensors import safe_open
from safetensors.torch import load_file
from sglang.multimodal_gen.runtime.loader.readonly_safetensors import load_safetensors_readonly
from sglang.multimodal_gen.runtime.loader.weight_readers.safetensors_mmap import SafetensorsMmapReader
path, api = sys.argv[1:]
if api in ('tensor', 'slice'):
    with safe_open(path, framework='pt') as handle:
        assert handle.keys() == ['weight']
        value = handle.get_tensor('weight') if api == 'tensor' else handle.get_slice('weight')[:]
elif api == 'load_file':
    value = load_file(path)['weight']
elif api == 'readonly':
    value = load_safetensors_readonly(path)['weight']
elif api == 'reader':
    with patch('sglang.multimodal_gen.runtime.loader.weight_readers.safetensors_mmap.host_copies_are_redundant', return_value=True):
        value = dict(SafetensorsMmapReader().iter_weights([path], device='cpu', to_cpu=True, show_progress=False))['weight']
else:
    value = torch.load(path)['weight']
assert torch.equal(value, torch.ones(2))
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(checkpoint), api],
        env={
            **os.environ,
            "PYTHONPATH": str(Path(__file__).parent)
            + os.pathsep
            + os.environ.get("PYTHONPATH", ""),
            "WC_TEST_BLOCKED_FILES": json.dumps([str(checkpoint)] if blocked else []),
            "WC_TEST_GUARD_LOG_DIR": str(logs),
        },
        capture_output=True,
        text=True,
        timeout=60,
    )
    events = [
        json.loads(line)
        for path in logs.glob("*.jsonl")
        for line in path.read_text().splitlines()
    ]
    assert any(event["event"] == "installed" for event in events)
    if blocked:
        assert result.returncode != 0, result.stdout
        assert "WC_TEST_CACHED_TENSOR_READ" in result.stderr
        assert any(event["event"] == "blocked_tensor" for event in events)
    else:
        assert result.returncode == 0, result.stderr
        assert any(event["event"] == "uncached_tensor" for event in events)


def test_guard_allows_cached_metadata_only(tmp_path):
    checkpoint = tmp_path / "weights.safetensors"
    save_file({"weight": torch.ones(2)}, checkpoint)
    logs = tmp_path / "events"
    logs.mkdir()
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from sglang.multimodal_gen.runtime.loader.readonly_safetensors import safetensors_keys; assert safetensors_keys(sys.argv[1]) == ['weight']",
            str(checkpoint),
        ],
        env={
            **os.environ,
            "PYTHONPATH": str(Path(__file__).parent)
            + os.pathsep
            + os.environ.get("PYTHONPATH", ""),
            "WC_TEST_BLOCKED_FILES": json.dumps([str(checkpoint)]),
            "WC_TEST_GUARD_LOG_DIR": str(logs),
        },
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    events = [
        json.loads(line)["event"]
        for path in logs.glob("*.jsonl")
        for line in path.read_text().splitlines()
    ]
    assert events == ["installed"]
