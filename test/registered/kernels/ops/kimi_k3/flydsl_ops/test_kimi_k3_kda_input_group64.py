# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import importlib

import pytest
import torch

from aiter.jit.utils.chip_info import get_gfx
from sglang.kernels.ops.kimi_k3.flydsl.kernels.kimi_k3_kda_input_group64_gfx950 import (
    build_kimi_k3_kda_input_group64_module,
)
from sglang.kernels.ops.kimi_k3.flydsl.kimi_k3_kda_input_group64 import (
    kimi_k3_kda_input_group64,
    quantize_kimi_k3_kda_input_group64,
    supports_kimi_k3_kda_input_group64,
)

group64_module = importlib.import_module("sglang.kernels.ops.kimi_k3.flydsl.kimi_k3_kda_input_group64")


def test_support_predicate_fails_closed_off_gpu() -> None:
    tensor = torch.empty(1)
    assert not supports_kimi_k3_kda_input_group64(tensor, tensor, tensor)


def test_quantizer_rejects_non_cuda_input() -> None:
    with pytest.raises(ValueError, match="contiguous CUDA BF16"):
        quantize_kimi_k3_kda_input_group64(torch.empty(1, dtype=torch.bfloat16))


def test_wrapper_uses_validated_gfx950_schedule_by_default(monkeypatch) -> None:
    hidden = torch.empty((1, 7168), dtype=torch.bfloat16)
    weight = torch.empty((6284, 7168), dtype=torch.float8_e4m3fn)
    scale = torch.empty((6284, 112), dtype=torch.float32)
    output = torch.empty((1, 6288), dtype=torch.bfloat16)
    schedules = []
    launches = []

    def fake_builder(**kwargs):
        schedules.append(kwargs)

        def launch(*args, **kwargs):
            launches.append((args, kwargs))

        return launch

    monkeypatch.setattr(
        group64_module,
        "supports_kimi_k3_kda_input_group64",
        lambda *args: True,
    )
    monkeypatch.setattr(
        group64_module,
        "build_kimi_k3_kda_input_group64_module",
        fake_builder,
    )
    monkeypatch.setattr(group64_module, "ptr_arg", lambda tensor: tensor)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: None)
    group64_module._launcher.cache_clear()

    assert kimi_k3_kda_input_group64(hidden, weight, scale, output) is output
    assert schedules == [
        {
            "num_tokens": 1,
            "rows_per_wave": 2,
            "cu_count": 256,
            "waves_per_eu": 0,
            "weight_cache_modifier": 2,
            "hidden_to_lds": True,
        }
    ]
    assert len(launches) == 1
    group64_module._launcher.cache_clear()


@pytest.mark.parametrize("rows_per_wave", [0, 5])
def test_builder_rejects_unsupported_rows_per_wave(rows_per_wave: int) -> None:
    with pytest.raises(ValueError, match="rows_per_wave"):
        build_kimi_k3_kda_input_group64_module(rows_per_wave=rows_per_wave)


@pytest.mark.parametrize("cu_count", [0, 257])
def test_builder_rejects_unsupported_cu_count(cu_count: int) -> None:
    with pytest.raises(ValueError, match="cu_count"):
        build_kimi_k3_kda_input_group64_module(cu_count=cu_count)


def test_builder_rejects_negative_waves_per_eu() -> None:
    with pytest.raises(ValueError, match="waves_per_eu"):
        build_kimi_k3_kda_input_group64_module(waves_per_eu=-1)


@pytest.mark.parametrize("cache_modifier", [-1, 4])
def test_builder_rejects_unsupported_cache_modifier(
    cache_modifier: int,
) -> None:
    with pytest.raises(ValueError, match="weight_cache_modifier"):
        build_kimi_k3_kda_input_group64_module(weight_cache_modifier=cache_modifier)


@pytest.mark.parametrize("num_tokens", [1, 2])
@torch.inference_mode()
def test_group64_projection_matches_dequantized_reference_on_gfx950(
    num_tokens: int,
) -> None:
    if not torch.cuda.is_available() or get_gfx() != "gfx950":
        pytest.skip("Kimi-K3 group64 projection is gfx950-only")

    hidden_size = 7168
    stored_rows = 6284
    output_rows = 6288
    group_size = 64
    groups_per_row = hidden_size // group_size

    columns = torch.arange(hidden_size, device="cuda", dtype=torch.float32)
    rows = torch.arange(stored_rows, device="cuda", dtype=torch.float32)
    hidden = torch.stack(
        [
            torch.sin(columns * (0.03125 + token * 0.001)).to(torch.bfloat16)
            for token in range(num_tokens)
        ]
    )
    stored_weight = (
        ((columns.to(torch.int32) % 17).float() - 8.0).unsqueeze(0)
        * ((rows.to(torch.int32) % 13).float() + 1.0).unsqueeze(1)
        * 0.002
    ).to(torch.bfloat16)
    weight = torch.zeros(
        output_rows,
        hidden_size,
        device="cuda",
        dtype=torch.bfloat16,
    )
    weight[:stored_rows].copy_(stored_weight)

    packed, scale = quantize_kimi_k3_kda_input_group64(weight)
    actual = kimi_k3_kda_input_group64(hidden, packed, scale)

    hidden_groups = hidden.float().reshape(num_tokens, groups_per_row, group_size)
    dequantized = (
        packed.to(torch.bfloat16)
        .float()
        .reshape(stored_rows, groups_per_row, group_size)
        * scale[..., None]
    )
    expected = torch.einsum("tgc,rgc->tr", hidden_groups, dequantized).to(
        torch.bfloat16
    )
    actual_fp32 = actual[:, :stored_rows].float()
    expected_fp32 = expected.float()
    delta = actual_fp32 - expected_fp32
    relative_rmse = delta.square().mean().sqrt() / expected_fp32.square().mean().sqrt()
    cosine = torch.nn.functional.cosine_similarity(
        actual_fp32,
        expected_fp32,
        dim=1,
    )

    assert torch.isfinite(actual).all()
    assert relative_rmse <= 5e-4
    assert torch.all(cosine >= 0.99999)
    assert torch.count_nonzero(actual[:, stored_rows:]).item() == 0
