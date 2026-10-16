"""Unit tests for try_detect_fp4_experts (DeepSeek V4 routed-expert layout).

Regression tests for https://github.com/sgl-project/sglang/issues/44810:
a training-exported BF16 DeepSeek-V4 checkpoint has no `quantization_config`,
so ModelConfig falls back to the SGLANG_DSV4_FP4_EXPERTS env default (True)
unless the detector explicitly returns False for unquantized dtypes.
"""

import json
import os
import struct

import pytest

from sglang.srt.configs.deepseek_v4 import try_detect_fp4_experts
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

_EXPERT_KEY = "model.layers.0.mlp.experts.0.w1.weight"


def _make_checkpoint(tmp_path, dtype: str) -> str:
    """Build a minimal fake checkpoint dir with one safetensors shard whose
    header declares one routed-expert weight tensor of the given dtype."""
    header = json.dumps({_EXPERT_KEY: {"dtype": dtype, "shape": [2048, 4096]}}).encode()
    shard = os.path.join(str(tmp_path), "model.safetensors")
    with open(shard, "wb") as f:
        f.write(struct.pack("<Q", len(header)))
        f.write(header)
    return str(tmp_path)


@pytest.mark.parametrize(
    "dtype", ["BF16", "F16", "F32"], ids=["bf16", "f16", "f32"]
)
def test_unquantized_expert_dtype_detects_not_fp4(tmp_path, dtype):
    """Unquantized checkpoints must route to the unquantized expert path."""
    assert try_detect_fp4_experts(_make_checkpoint(tmp_path, dtype)) is False


@pytest.mark.parametrize(
    "dtype,expected", [("U8", True), ("I8", True), ("F4", True), ("F8_E4M3", False)]
)
def test_quantized_expert_dtype_detection_unchanged(tmp_path, dtype, expected):
    """Genuine MXFP4-packed checkpoints keep True; FP8-converted keep False."""
    assert try_detect_fp4_experts(_make_checkpoint(tmp_path, dtype)) is expected


def test_unreadable_header_returns_none(tmp_path):
    """None (fall back to the user/env default) when there is nothing to probe."""
    assert try_detect_fp4_experts(os.path.join(str(tmp_path), "does-not-exist")) is None
    assert try_detect_fp4_experts(str(tmp_path)) is None  # no safetensors shard
