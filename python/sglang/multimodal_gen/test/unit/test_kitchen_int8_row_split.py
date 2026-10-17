# SPDX-License-Identifier: Apache-2.0
"""Hardware defaults and explicit overrides for Kitchen INT8 row splitting."""

import pytest
import torch

from sglang.multimodal_gen.runtime.layers.quantization import (
    convrot_int8_comfy_kitchen as kitchen_int8,
)


@pytest.mark.parametrize(
    "capability,device,dtype,rows,width,expected",
    [
        ((12, 0), "cuda:0", torch.bfloat16, 32700, 16128, None),
        ((12, 0), "cuda:1", torch.bfloat16, 8193, 24832, None),
        ((12, 0), "cuda:0", torch.bfloat16, 14850, 28672, 8192),
        ((8, 9), "cuda:0", torch.bfloat16, 32700, 16128, 8192),
        ((9, 0), "cuda:0", torch.bfloat16, 32700, 16128, 8192),
        ((12, 0), "cuda:0", torch.float16, 32700, 16128, 8192),
        ((12, 0), "cuda:0", torch.float32, 32700, 16128, 8192),
        ((12, 0), "cpu", torch.bfloat16, 32700, 16128, 8192),
        ((12, 0), "cuda:0", torch.bfloat16, 8192, 16128, None),
        ((8, 9), "cuda:0", torch.bfloat16, 32700, 5376, None),
    ],
)
def test_row_split_hardware_default(
    monkeypatch, capability, device, dtype, rows, width, expected
):
    monkeypatch.setattr(kitchen_int8, "_ROW_SPLIT_OVERRIDDEN", False)
    monkeypatch.setattr(kitchen_int8, "_MAX_ROWS_PER_CALL", 8192)
    monkeypatch.setattr(kitchen_int8, "_MIN_SPLIT_OUTPUT", 8192)
    kitchen_int8._is_sm120.cache_clear()
    devices = []

    def get_capability(device_id):
        devices.append(device_id)
        return capability

    monkeypatch.setattr(
        kitchen_int8.current_platform, "get_device_capability", get_capability
    )
    try:
        assert (
            kitchen_int8._row_split(rows, width, 5376, torch.device(device), dtype)
            == expected
        )
        if devices:
            assert devices == [torch.device(device).index]
    finally:
        kitchen_int8._is_sm120.cache_clear()


@pytest.mark.parametrize("limit,expected", [(0, None), (8192, 8192), (4096, 4096)])
def test_row_split_explicit_override(monkeypatch, limit, expected):
    monkeypatch.setattr(kitchen_int8, "_ROW_SPLIT_OVERRIDDEN", True)
    monkeypatch.setattr(kitchen_int8, "_MAX_ROWS_PER_CALL", limit)
    monkeypatch.setattr(kitchen_int8, "_MIN_SPLIT_OUTPUT", 8192)
    assert (
        kitchen_int8._row_split(
            32700, 16128, 5376, torch.device("cuda:0"), torch.bfloat16
        )
        == expected
    )


def test_row_split_preserves_large_reduction_policy(monkeypatch):
    monkeypatch.setattr(kitchen_int8, "_ROW_SPLIT_OVERRIDDEN", False)
    monkeypatch.setattr(kitchen_int8, "_MAX_ROWS_PER_CALL", 8192)
    monkeypatch.setattr(kitchen_int8, "_MIN_SPLIT_OUTPUT", 8192)
    assert (
        kitchen_int8._row_split(
            12288, 21504, 28672, torch.device("cuda:0"), torch.bfloat16
        )
        == 8192
    )
