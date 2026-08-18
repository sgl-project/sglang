# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

from aiter.jit.utils.chip_info import get_gfx_runtime
from sglang.kernels.ops.kimi_k3.flydsl.kernels.latent_moe_tail_fp8_gfx950 import (
    build_b1_latent_moe_tail_fp8_persistent_module,
)
from sglang.kernels.ops.kimi_k3.flydsl.latent_moe_tail_fp8 import (
    latent_moe_tail_fp8,
    quantize_latent_moe_tail_weight,
    supports_latent_moe_tail_fp8,
)

LATENT_DIM = 3584
HIDDEN_DIM = 7168
EPSILON = 1.0e-6


def _relative_rmse(actual: torch.Tensor, expected: torch.Tensor) -> float:
    error = (actual.float() - expected.float()).square().mean().sqrt()
    reference = expected.float().square().mean().sqrt().clamp_min(1.0e-12)
    return (error / reference).item()


def test_support_predicate_fails_closed_on_cpu():
    bf16 = torch.empty(1, dtype=torch.bfloat16)
    fp8 = torch.empty(1, dtype=torch.float8_e4m3fn)
    fp32 = torch.empty(1, dtype=torch.float32)

    assert not supports_latent_moe_tail_fp8(
        bf16,
        bf16,
        bf16,
        fp8,
        fp32,
        EPSILON,
    )


def test_quantizer_rejects_non_cuda_input():
    with pytest.raises(ValueError, match="contiguous CUDA BF16"):
        quantize_latent_moe_tail_weight(torch.empty(1, dtype=torch.bfloat16))


@pytest.mark.parametrize(
    ("keyword", "value", "message"),
    [
        ("rows_per_wave", 0, "rows_per_wave"),
        ("cu_count", 257, "cu_count"),
        ("waves_per_eu", -1, "waves_per_eu"),
        ("weight_cache_modifier", 4, "weight_cache_modifier"),
    ],
)
def test_builder_rejects_invalid_schedule(keyword: str, value: int, message: str):
    with pytest.raises(ValueError, match=message):
        build_b1_latent_moe_tail_fp8_persistent_module(**{keyword: value})


@pytest.mark.skipif(
    not torch.cuda.is_available() or get_gfx_runtime() != "gfx950",
    reason="Kimi-K3 FP8 latent-tail specialization requires gfx950",
)
@torch.inference_mode()
def test_latent_moe_tail_fp8_matches_dequantized_oracle_and_replays():
    generator = torch.Generator(device="cpu").manual_seed(20260730)
    routed = torch.randn((1, LATENT_DIM), generator=generator).bfloat16().cuda()
    shared = torch.randn((1, HIDDEN_DIM), generator=generator).bfloat16().cuda()
    rms_weight = torch.randn(LATENT_DIM, generator=generator).bfloat16().cuda()
    up_weight = (
        torch.randn((HIDDEN_DIM, LATENT_DIM), generator=generator)
        .mul_(LATENT_DIM**-0.5)
        .bfloat16()
        .cuda()
    )
    packed, scale = quantize_latent_moe_tail_weight(up_weight)

    inverse_rms = torch.rsqrt(
        routed.float().square().mean(dim=-1, keepdim=True) + EPSILON
    )
    normalized = (routed.float() * inverse_rms * rms_weight.float()).bfloat16()
    dequantized = packed.float() * scale[:, None]
    dequantized_oracle = (
        torch.mm(normalized.float(), dequantized.t()).bfloat16().float()
        + shared.float()
    ).bfloat16()
    bf16_oracle = (
        torch.mm(normalized.float(), up_weight.float().t()).bfloat16().float()
        + shared.float()
    ).bfloat16()

    out = torch.empty_like(shared)
    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup_stream):
        latent_moe_tail_fp8(
            routed,
            shared,
            rms_weight,
            packed,
            scale,
            EPSILON,
            out=out,
        )
    torch.cuda.current_stream().wait_stream(warmup_stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = latent_moe_tail_fp8(
            routed,
            shared,
            rms_weight,
            packed,
            scale,
            EPSILON,
            out=out,
        )
    graph.replay()
    torch.cuda.synchronize()

    assert actual is out
    torch.testing.assert_close(actual, dequantized_oracle, rtol=0.01, atol=0.015625)
    assert _relative_rmse(actual, bf16_oracle) < 0.03
    assert (
        torch.nn.functional.cosine_similarity(
            actual.float(),
            bf16_oracle.float(),
        ).item()
        > 0.999
    )

    previous = actual.clone()
    routed.add_(0.5)
    graph.replay()
    torch.cuda.synchronize()
    assert not torch.equal(previous, actual)
