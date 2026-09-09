from __future__ import annotations

import sys
from contextlib import contextmanager
from unittest.mock import patch

import pytest
import torch

from sglang.kernels.jit.utils import (
    get_ci_test_range,
    is_arch_support_pdl,
    is_hip_runtime,
)
from sglang.kernels.ops.attention.dsa import quant_k_cache
from sglang.kernels.ops.attention.dsa.quant_k_cache import (
    _quantize_k_cache_fast_separate_cuda,
    _quantize_k_cache_fast_separate_cuda_module,
    _quantize_k_cache_fast_separate_triton,
    quantize_k_cache_separate,
)
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=45, stage="base-b-kernel-unit", runner_config="1-gpu-large")

DEVICE = "cuda"
FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = 448.0
NOPE_DIM = 512
ROPE_DIM = 64
SCALE_OFFSET = 512
NOPE_PART_BYTES = 528
ROPE_PART_BYTES = 128
GROUP_SIZES = [8, 16, 32, 64, 128, 256, 512]

TOKEN_COUNTS = get_ci_test_range(
    [1, 2, 3, 4, 5, 7, 64, 257, 499, 500, 512, 624, 625, 8192],
    [1, 3, 5, 64, 257, 499, 500, 512, 624, 625, 8192],
)


def _skip_if_unavailable() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if is_hip_runtime():
        pytest.skip("MLA K-cache quantization is currently CUDA-only")
    if not hasattr(torch, "float8_e4m3fn"):
        pytest.skip("torch.float8_e4m3fn is unavailable")


def _run_cuda(
    k_nope: torch.Tensor,
    k_rope: torch.Tensor,
    *,
    group_size: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    nope_part, rope_part = _quantize_k_cache_fast_separate_cuda(
        k_nope, k_rope, group_size
    )
    torch.cuda.synchronize()
    return nope_part, rope_part


def _payload_views(
    nope_part: torch.Tensor, rope_part: torch.Tensor, dim_nope: int = NOPE_DIM
):
    nope_rows = nope_part[:, 0, :].contiguous()
    rope_rows = rope_part[:, 0, :].contiguous()
    quant = nope_rows[:, :dim_nope].view(FP8_DTYPE).float()
    scales = nope_rows[:, dim_nope:].contiguous().view(torch.float32)
    rope = (
        rope_rows.view(-1)
        .view(torch.bfloat16)
        .view(rope_rows.shape[0], rope_rows.shape[1] // 2)
    )
    return quant, scales, rope


def _check_payload(
    nope_part: torch.Tensor,
    rope_part: torch.Tensor,
    k_nope: torch.Tensor,
    k_rope: torch.Tensor,
    group_size: int = 128,
) -> None:
    dim_nope = k_nope.shape[-1]
    num_groups = dim_nope // group_size
    k_nope = k_nope.reshape(k_nope.shape[0], dim_nope)
    k_rope = k_rope.reshape(k_rope.shape[0], k_rope.shape[-1])
    quant, scales, rope = _payload_views(nope_part, rope_part, dim_nope)

    groups = k_nope.float().reshape(-1, num_groups, group_size)
    expected_scales = groups.abs().amax(dim=-1) / FP8_MAX
    torch.testing.assert_close(scales, expected_scales, rtol=1e-6, atol=0.0)
    assert torch.equal(rope, k_rope)

    expected_quant = (
        (groups / expected_scales.unsqueeze(-1))
        .clamp(-FP8_MAX, FP8_MAX)
        .to(FP8_DTYPE)
        .float()
    )
    # FP32 reference and device reciprocal can straddle an FP8 rounding tie.
    torch.testing.assert_close(
        quant.reshape_as(expected_quant), expected_quant, rtol=0.125, atol=2**-9
    )
    reconstructed = quant.reshape_as(groups) * scales.unsqueeze(-1)
    torch.testing.assert_close(
        reconstructed.reshape(-1, dim_nope),
        k_nope.float(),
        rtol=0.15,
        atol=5e-2,
    )


@contextmanager
def _check_cuda_dispatch():
    with (
        patch.object(
            quant_k_cache,
            "_quantize_k_cache_fast_separate_cuda_module",
            wraps=_quantize_k_cache_fast_separate_cuda_module,
        ) as cuda_module,
        patch.object(
            quant_k_cache,
            "_quantize_k_cache_fast_separate_triton",
            wraps=_quantize_k_cache_fast_separate_triton,
        ) as triton_impl,
    ):
        yield
    cuda_module.assert_called_once()
    triton_impl.assert_not_called()


@pytest.mark.parametrize("num_tokens", TOKEN_COUNTS)
def test_cuda_matches_triton(num_tokens: int):
    _skip_if_unavailable()
    torch.manual_seed(0)

    k_nope = torch.randn((num_tokens, NOPE_DIM), dtype=torch.bfloat16, device=DEVICE)
    k_rope = torch.randn((num_tokens, ROPE_DIM), dtype=torch.bfloat16, device=DEVICE)

    nope_part, rope_part = _run_cuda(k_nope, k_rope)
    triton_nope_part, triton_rope_part = _quantize_k_cache_fast_separate_triton(
        k_nope, k_rope
    )

    assert nope_part.shape == (num_tokens, 1, NOPE_PART_BYTES)
    assert rope_part.shape == (num_tokens, 1, ROPE_PART_BYTES)
    assert nope_part.dtype == torch.uint8
    assert rope_part.dtype == torch.uint8
    assert torch.equal(nope_part, triton_nope_part)
    assert torch.equal(rope_part, triton_rope_part)
    _check_payload(nope_part, rope_part, k_nope, k_rope)


def test_group_reduction_across_input_sizes():
    _skip_if_unavailable()
    # Repeat identical groups across small and large inputs to check that
    # automatic scheduling preserves the result, including zero-group bytes.
    num_tokens = 4
    positions = torch.tensor([0, 31, 32, 63, 64, 95, 96, 127] * 2, device=DEVICE)
    group_values = (
        ((torch.arange(128, dtype=torch.float32, device=DEVICE) + 1) / 256)
        .to(torch.bfloat16)
        .repeat(16, 1)
    )
    # Adjacent values differ in sign to expose swapped or duplicated FP8 pairs.
    group_values[:, 1::2].neg_()
    peaks = torch.arange(8, 24, dtype=torch.bfloat16, device=DEVICE)
    peaks[8:].neg_()
    group_values[torch.arange(16, device=DEVICE), positions] = peaks
    k_nope = group_values.view(num_tokens, NOPE_DIM)
    k_rope = (
        torch.arange(num_tokens * ROPE_DIM, dtype=torch.float32, device=DEVICE)
        .to(torch.bfloat16)
        .view(num_tokens, ROPE_DIM)
    )

    for include_zero_groups in (False, True):
        if include_zero_groups:
            # Exercise zero scales in both single-group and persistent CTAs.
            group_values[::3].zero_()
        actual = _run_cuda(k_nope, k_rope)
        repeats = 8192 // num_tokens
        repeated = _run_cuda(k_nope.repeat(repeats, 1), k_rope.repeat(repeats, 1))
        expected = _quantize_k_cache_fast_separate_triton(k_nope, k_rope)
        for actual_part, repeated_part in zip(actual, repeated):
            assert torch.equal(actual_part.repeat(repeats, 1, 1), repeated_part)
        group_maxima = group_values.float().abs().amax(dim=-1)
        nonzero_groups = group_maxima != 0
        actual_quant = actual[0][:, 0, :SCALE_OFFSET].reshape(-1, 128)
        expected_quant = expected[0][:, 0, :SCALE_OFFSET].reshape(-1, 128)
        # Zero groups produce 0 * inf before clipping. CUDA and Triton clamp
        # that NaN differently, so preserve the existing CUDA bytes above.
        # Every nonzero group's quantized bytes, all scales, and all RoPE bytes
        # must still match Triton exactly.
        assert torch.equal(actual_quant[nonzero_groups], expected_quant[nonzero_groups])
        assert torch.equal(
            actual[0][:, :, SCALE_OFFSET:], expected[0][:, :, SCALE_OFFSET:]
        )
        assert torch.equal(actual[1], expected[1])
        _, scales, rope = _payload_views(*actual)
        expected_scales = group_maxima.view(num_tokens, 4) / FP8_MAX
        torch.testing.assert_close(scales, expected_scales, rtol=1e-6, atol=0.0)
        if include_zero_groups:
            assert torch.all(scales.reshape(-1)[~nonzero_groups] == 0).item()
        assert torch.equal(rope, k_rope)


@pytest.mark.parametrize("num_tokens", [7, 8192])
@pytest.mark.parametrize("group_size", [64, 128, 512])
def test_strided_inputs(group_size: int, num_tokens: int):
    _skip_if_unavailable()
    combined = torch.randn(
        (num_tokens, NOPE_DIM + ROPE_DIM),
        dtype=torch.bfloat16,
        device=DEVICE,
    )
    k_nope = combined[:, :NOPE_DIM]
    k_rope = combined[:, NOPE_DIM:]
    assert k_nope.stride(0) == NOPE_DIM + ROPE_DIM
    assert k_rope.stride(0) == NOPE_DIM + ROPE_DIM

    nope_part, rope_part = _run_cuda(k_nope, k_rope, group_size=group_size)
    triton_nope_part, triton_rope_part = _quantize_k_cache_fast_separate_triton(
        k_nope, k_rope, group_size
    )
    assert torch.equal(nope_part, triton_nope_part)
    assert torch.equal(rope_part, triton_rope_part)
    _check_payload(nope_part, rope_part, k_nope, k_rope, group_size)


@pytest.mark.parametrize("group_size", GROUP_SIZES)
@pytest.mark.parametrize("num_tokens", [1, 7, 8192])
def test_group_sizes_use_cuda(group_size: int, num_tokens: int):
    _skip_if_unavailable()
    torch.manual_seed(0)

    num_groups = NOPE_DIM // group_size
    k_nope = torch.randn((num_tokens, NOPE_DIM), dtype=torch.bfloat16, device=DEVICE)
    k_rope = torch.randn((num_tokens, ROPE_DIM), dtype=torch.bfloat16, device=DEVICE)

    with _check_cuda_dispatch():
        actual = _run_cuda(k_nope, k_rope, group_size=group_size)
    expected = _quantize_k_cache_fast_separate_triton(k_nope, k_rope, group_size)
    assert actual[0].shape == (num_tokens, 1, NOPE_DIM + num_groups * 4)
    assert actual[1].shape == (num_tokens, 1, ROPE_PART_BYTES)
    assert torch.equal(actual[0], expected[0])
    assert torch.equal(actual[1], expected[1])
    _check_payload(*actual, k_nope, k_rope, group_size)


@pytest.mark.parametrize("group_size", GROUP_SIZES)
@pytest.mark.parametrize("num_tokens", [5, 8192])
def test_group_and_vector_boundaries(group_size: int, num_tokens: int):
    _skip_if_unavailable()
    # Distinct neighboring maxima detect reductions mixing groups or tokens.
    num_groups = NOPE_DIM // group_size
    k_nope = torch.full(
        (num_tokens, NOPE_DIM), 0.125, dtype=torch.bfloat16, device=DEVICE
    )
    maxima = (
        (torch.arange(num_tokens * num_groups, device=DEVICE) % 127 + 1)
        .to(torch.bfloat16)
        .reshape(num_tokens, num_groups)
    )
    positions = torch.tensor((0, group_size - 1, 7, group_size // 2), device=DEVICE)
    tokens = torch.arange(num_tokens, device=DEVICE)[:, None]
    groups = torch.arange(num_groups, device=DEVICE)[None, :]
    position = positions[(tokens + groups) % len(positions)]
    sign = 1 - 2 * ((tokens + groups) % 2)
    k_nope[tokens, groups * group_size + position] = sign * maxima
    k_rope = (
        torch.arange(num_tokens * ROPE_DIM, dtype=torch.float32, device=DEVICE)
        .to(torch.bfloat16)
        .reshape(num_tokens, ROPE_DIM)
    )

    with _check_cuda_dispatch():
        nope_part, rope_part = _run_cuda(k_nope, k_rope, group_size=group_size)
    _, scales, rope = _payload_views(nope_part, rope_part)
    expected = maxima.float() / FP8_MAX
    torch.testing.assert_close(scales, expected, rtol=1e-6, atol=0.0)
    assert torch.equal(rope, k_rope)
    expected_parts = _quantize_k_cache_fast_separate_triton(k_nope, k_rope, group_size)
    assert torch.equal(nope_part, expected_parts[0])
    assert torch.equal(rope_part, expected_parts[1])
    _check_payload(nope_part, rope_part, k_nope, k_rope, group_size)


@pytest.mark.parametrize("group_size", [8, 64, 128, 512])
def test_empty_input_returns_empty_parts(group_size: int):
    _skip_if_unavailable()
    k_nope = torch.empty((0, NOPE_DIM), dtype=torch.bfloat16, device=DEVICE)
    k_rope = torch.empty((0, ROPE_DIM), dtype=torch.bfloat16, device=DEVICE)

    nope_part, rope_part = _quantize_k_cache_fast_separate_cuda(
        k_nope, k_rope, group_size
    )
    assert nope_part.shape == (0, 1, NOPE_DIM + (NOPE_DIM // group_size) * 4)
    assert rope_part.shape == (0, 1, ROPE_PART_BYTES)
    assert nope_part.dtype == torch.uint8
    assert rope_part.dtype == torch.uint8


@pytest.mark.parametrize("env_value", [None, "0", "1"])
def test_dispatch_respects_env(monkeypatch, env_value):
    env_name = envs.SGLANG_OPT_USE_CUDA_MLA_K_CACHE_QUANT.name
    if env_value is None:
        monkeypatch.delenv(env_name, raising=False)
    else:
        monkeypatch.setenv(env_name, env_value)

    k_nope, k_rope = object(), object()
    expected = (object(), object())
    with (
        patch.object(
            quant_k_cache, "_quantize_k_cache_fast_separate_cuda"
        ) as cuda_impl,
        patch.object(
            quant_k_cache, "_quantize_k_cache_fast_separate_triton"
        ) as triton_impl,
    ):
        selected, unused = (
            (cuda_impl, triton_impl) if env_value == "1" else (triton_impl, cuda_impl)
        )
        selected.return_value = expected
        actual = quant_k_cache._quantize_k_cache_fast_separate(k_nope, k_rope, 64)

    selected.assert_called_once_with(k_nope, k_rope, 64)
    unused.assert_not_called()
    assert actual is expected


@pytest.mark.parametrize(
    "num_tokens,with_head_dim", [(1, False), (10, True), (8192, False)]
)
@pytest.mark.parametrize("tile_size", GROUP_SIZES)
def test_dispatch_uses_cuda(
    num_tokens: int, with_head_dim: bool, tile_size: int, monkeypatch
):
    _skip_if_unavailable()
    monkeypatch.setenv(envs.SGLANG_OPT_USE_CUDA_MLA_K_CACHE_QUANT.name, "1")
    torch.manual_seed(0)
    k_nope = torch.randn((num_tokens, NOPE_DIM), dtype=torch.bfloat16, device=DEVICE)
    k_rope = torch.randn((num_tokens, ROPE_DIM), dtype=torch.bfloat16, device=DEVICE)

    with _check_cuda_dispatch():
        actual = quantize_k_cache_separate(
            k_nope.unsqueeze(1) if with_head_dim else k_nope,
            k_rope.unsqueeze(1) if with_head_dim else k_rope,
            tile_size=tile_size,
        )

    # With CUDA enabled, the public entry must use it for every supported group
    # size, including single-token and multi-iteration workloads.
    num_groups = NOPE_DIM // tile_size
    assert actual[0].shape == (num_tokens, 1, NOPE_DIM + num_groups * 4)
    assert actual[1].shape == (num_tokens, 1, ROPE_PART_BYTES)
    expected = _quantize_k_cache_fast_separate_triton(k_nope, k_rope, tile_size)
    assert torch.equal(actual[0], expected[0])
    assert torch.equal(actual[1], expected[1])
    _check_payload(*actual, k_nope, k_rope, tile_size)


@pytest.mark.parametrize("num_tokens", [1, 512, 8192])
def test_cuda_graph_replay(num_tokens: int):
    _skip_if_unavailable()
    torch.manual_seed(0)
    k_nope = torch.randn((num_tokens, NOPE_DIM), dtype=torch.bfloat16, device=DEVICE)
    k_rope = torch.randn((num_tokens, ROPE_DIM), dtype=torch.bfloat16, device=DEVICE)

    # Complete JIT compilation and initialize the host launch path before capture.
    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup_stream):
        _quantize_k_cache_fast_separate_cuda(k_nope, k_rope)
    torch.cuda.current_stream().wait_stream(warmup_stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = _quantize_k_cache_fast_separate_cuda(k_nope, k_rope)

    for _ in range(2):
        # Replay must read the current inputs, including the partial final CTA.
        k_nope.normal_()
        k_rope.normal_()
        expected = _quantize_k_cache_fast_separate_triton(k_nope, k_rope)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(actual[0], expected[0])
        assert torch.equal(actual[1], expected[1])


@pytest.mark.parametrize(
    "num_tokens,group_size",
    [
        (7, 64),
        (8192, 64),
        (7, 128),
        (499, 128),
        (500, 128),
        (625, 128),
        (8192, 128),
        (7, 256),
        (8192, 256),
        (7, 512),
        (8192, 512),
    ],
)
def test_preallocated_outputs_preserve_guards(num_tokens: int, group_size: int):
    _skip_if_unavailable()
    torch.manual_seed(0)
    guard_rows = 8
    nope_part_bytes = NOPE_DIM + (NOPE_DIM // group_size) * 4
    # Keep extra input rows readable so an incorrect final-iteration mask
    # leaves observable output corruption instead of an unrelated input fault.
    k_nope_storage = torch.randn(
        (num_tokens + 2 * guard_rows, NOPE_DIM),
        dtype=torch.bfloat16,
        device=DEVICE,
    )
    k_rope_storage = torch.randn(
        (num_tokens + 2 * guard_rows, ROPE_DIM),
        dtype=torch.bfloat16,
        device=DEVICE,
    )
    k_nope = k_nope_storage[guard_rows : guard_rows + num_tokens]
    k_rope = k_rope_storage[guard_rows : guard_rows + num_tokens]

    # A 32-byte pad preserves the natural row alignment. For group_size=512,
    # the 516-byte payload leaves alternate rows only 4-byte aligned.
    nope_storage = torch.full(
        (num_tokens + 2 * guard_rows, nope_part_bytes + 32),
        0xA5,
        dtype=torch.uint8,
        device=DEVICE,
    )
    rope_storage = torch.full(
        (num_tokens + 2 * guard_rows, ROPE_PART_BYTES + 32),
        0x5A,
        dtype=torch.uint8,
        device=DEVICE,
    )
    nope_part = nope_storage[guard_rows : guard_rows + num_tokens, :nope_part_bytes]
    rope_part = rope_storage[guard_rows : guard_rows + num_tokens, :ROPE_PART_BYTES]
    module = _quantize_k_cache_fast_separate_cuda_module(
        torch.bfloat16,
        k_nope.shape[1],
        k_rope.shape[1],
        group_size,
        is_arch_support_pdl(),
    )
    module.quantize_k_cache_fast_separate_cuda(nope_part, rope_part, k_nope, k_rope)
    torch.cuda.synchronize()
    expected = _quantize_k_cache_fast_separate_triton(k_nope, k_rope, group_size)
    assert torch.equal(nope_part, expected[0][:, 0])
    assert torch.equal(rope_part, expected[1][:, 0])

    # Guards detect invalid tensor/padding writes, but cannot prove exclusive
    # ownership when two CTAs write the same valid element. Review the CTA
    # interval partition separately; run GPU memory diagnostics as well.
    for storage, width, sentinel in (
        (nope_storage, nope_part_bytes, 0xA5),
        (rope_storage, ROPE_PART_BYTES, 0x5A),
    ):
        assert torch.all(storage[:guard_rows] == sentinel).item()
        assert torch.all(storage[guard_rows + num_tokens :] == sentinel).item()
        assert torch.all(storage[:, width:] == sentinel).item()


def test_input_device_differs_from_current_device():
    _skip_if_unavailable()
    if torch.cuda.device_count() < 2:
        pytest.skip("Requires two CUDA devices")
    current_device = torch.cuda.current_device()
    input_device = 1 if current_device == 0 else 0
    if torch.cuda.get_device_capability(
        current_device
    ) != torch.cuda.get_device_capability(input_device):
        pytest.skip("JIT architecture caching requires matching GPU capabilities")
    k_nope = torch.randn(
        (10, NOPE_DIM), dtype=torch.bfloat16, device=f"cuda:{input_device}"
    )
    k_rope = torch.randn(
        (10, ROPE_DIM), dtype=torch.bfloat16, device=f"cuda:{input_device}"
    )
    actual = _quantize_k_cache_fast_separate_cuda(k_nope, k_rope)
    assert torch.cuda.current_device() == current_device
    assert actual[0].device == k_nope.device
    assert actual[1].device == k_rope.device
    with torch.cuda.device(input_device):
        expected = _quantize_k_cache_fast_separate_triton(k_nope, k_rope)
        torch.cuda.synchronize()
        assert torch.equal(actual[0], expected[0])
        assert torch.equal(actual[1], expected[1])


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
