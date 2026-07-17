from __future__ import annotations

import math
import sys

import pytest
import torch

from sglang.jit_kernel.dsv4 import (
    CompressorDecodePlan,
    CompressorPrefillPlan,
    compress_128_online_decode_fused,
    compress_forward,
    compress_norm_rope_store,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=45, stage="base-b-kernel-unit", runner_config="1-gpu-large")

HEAD_DIM = 512
RATIO = 128
FP8_PAGE_SIZE = 2
FP8_TOKEN_BYTES = 584
FP8_VALUE_BYTES = 576
NORM_EPS = 1e-6


def _make_freq_cis(max_position: int) -> torch.Tensor:
    positions = torch.arange(max_position, dtype=torch.float32, device="cuda")
    inv_freq = 1.0 / (
        10000 ** (torch.arange(0, 64, 2, dtype=torch.float32, device="cuda") / 64)
    )
    angles = positions[:, None] * inv_freq[None, :]
    return torch.polar(torch.ones_like(angles), angles)


def _make_decode_plan(seq_lens: torch.Tensor) -> CompressorDecodePlan:
    batch_size = seq_lens.shape[0]
    slots = torch.arange(batch_size, dtype=torch.int32, device="cuda")
    plan_i32 = torch.empty((batch_size, 4), dtype=torch.int32, device="cuda")
    plan_i32[:, 0] = seq_lens.to(torch.int32)
    plan_i32[:, 1] = slots
    plan_i32[:, 2] = slots
    plan_i32[:, 3] = -1
    return CompressorDecodePlan(RATIO, plan_i32.view(torch.uint8))


def _make_state(batch_size: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    state = torch.empty(
        (batch_size, 1, HEAD_DIM * 3), dtype=torch.float32, device="cuda"
    )
    state[..., :HEAD_DIM].normal_(generator=generator)
    state[..., HEAD_DIM : 2 * HEAD_DIM].uniform_(0.5, 2.0, generator=generator)
    state[..., 2 * HEAD_DIM :].normal_(generator=generator)
    return state


def _make_fp8_cache(num_slots: int) -> torch.Tensor:
    page_bytes = (
        math.ceil(FP8_TOKEN_BYTES * FP8_PAGE_SIZE / FP8_VALUE_BYTES) * FP8_VALUE_BYTES
    )
    num_pages = math.ceil(num_slots / FP8_PAGE_SIZE)
    return torch.zeros((num_pages, page_bytes), dtype=torch.uint8, device="cuda")


def _extract_fp8_token(cache: torch.Tensor, loc: int) -> torch.Tensor:
    page = loc // FP8_PAGE_SIZE
    offset = loc % FP8_PAGE_SIZE
    value_start = offset * FP8_VALUE_BYTES
    scale_start = FP8_PAGE_SIZE * FP8_VALUE_BYTES + offset * 8
    return torch.cat(
        (
            cache[page, value_start : value_start + FP8_VALUE_BYTES],
            cache[page, scale_start : scale_start + 8],
        )
    )


def _run_reference(
    state: torch.Tensor,
    kv_score_input: torch.Tensor,
    ape: torch.Tensor,
    plan: CompressorDecodePlan,
    norm_weight: torch.Tensor,
    freq_cis: torch.Tensor,
    out_loc: torch.Tensor,
    cache: torch.Tensor,
    *,
    page_size: int = FP8_PAGE_SIZE,
    bf16_store: bool = False,
) -> None:
    compressed = compress_forward(
        state,
        kv_score_input,
        ape,
        plan,
        head_dim=HEAD_DIM,
        compress_ratio=RATIO,
        is_online=True,
    )
    compress_norm_rope_store(
        compressed,
        plan,
        norm_weight=norm_weight,
        norm_eps=NORM_EPS,
        freq_cis=freq_cis,
        out_loc=out_loc,
        kvcache=cache,
        page_size=page_size,
        bf16_store=bf16_store,
    )


def _run_fused(
    state: torch.Tensor,
    kv_score_input: torch.Tensor,
    ape: torch.Tensor,
    plan: CompressorDecodePlan,
    norm_weight: torch.Tensor,
    freq_cis: torch.Tensor,
    out_loc: torch.Tensor,
    cache: torch.Tensor,
    *,
    page_size: int = FP8_PAGE_SIZE,
    bf16_store: bool = False,
    dcp_world_size: int = 1,
    dcp_rank: int = 0,
) -> None:
    compress_128_online_decode_fused(
        state,
        kv_score_input,
        ape,
        plan,
        norm_weight=norm_weight,
        norm_eps=NORM_EPS,
        freq_cis=freq_cis,
        out_loc=out_loc,
        kvcache=cache,
        page_size=page_size,
        bf16_store=bf16_store,
        dcp_world_size=dcp_world_size,
        dcp_rank=dcp_rank,
    )


def test_mixed_decode_positions_match_two_kernel_reference() -> None:
    seq_lens = torch.tensor(
        [1, 2, 64, 127, 128, 129, 256], dtype=torch.int64, device="cuda"
    )
    batch_size = seq_lens.shape[0]
    generator = torch.Generator(device="cuda").manual_seed(20260717)
    state = _make_state(batch_size, seed=1)
    kv_score_input = torch.randn(
        (batch_size, HEAD_DIM * 2),
        dtype=torch.float32,
        device="cuda",
        generator=generator,
    )
    ape = torch.randn(
        (RATIO, HEAD_DIM),
        dtype=torch.float32,
        device="cuda",
        generator=generator,
    )
    norm_weight = torch.randn(
        HEAD_DIM, dtype=torch.float32, device="cuda", generator=generator
    )
    freq_cis = _make_freq_cis(384)
    plan = _make_decode_plan(seq_lens)
    out_loc = torch.arange(batch_size, dtype=torch.int64, device="cuda")

    reference_state = state.clone()
    fused_state = state.clone()
    reference_cache = _make_fp8_cache(batch_size)
    fused_cache = reference_cache.clone()
    _run_reference(
        reference_state,
        kv_score_input,
        ape,
        plan,
        norm_weight,
        freq_cis,
        out_loc,
        reference_cache,
    )
    _run_fused(
        fused_state,
        kv_score_input,
        ape,
        plan,
        norm_weight,
        freq_cis,
        out_loc,
        fused_cache,
    )

    torch.testing.assert_close(fused_state, reference_state, rtol=0, atol=0)
    torch.testing.assert_close(fused_cache, reference_cache, rtol=0, atol=0)


def test_online_prefill_partial_state_then_boundary_decode() -> None:
    prefix_len = 127
    generator = torch.Generator(device="cuda").manual_seed(128)
    ape = torch.randn(
        (RATIO, HEAD_DIM),
        dtype=torch.float32,
        device="cuda",
        generator=generator,
    )
    prefix_input = torch.randn(
        (prefix_len, HEAD_DIM * 2),
        dtype=torch.float32,
        device="cuda",
        generator=generator,
    )
    decode_input = torch.randn(
        (1, HEAD_DIM * 2),
        dtype=torch.float32,
        device="cuda",
        generator=generator,
    )
    norm_weight = torch.randn(
        HEAD_DIM, dtype=torch.float32, device="cuda", generator=generator
    )
    freq_cis = _make_freq_cis(256)
    req_pool_indices = torch.zeros(1, dtype=torch.int64, device="cuda")
    req_to_token = torch.zeros((1, 256), dtype=torch.int32, device="cuda")
    prefill_plan = CompressorPrefillPlan.generate_online(
        torch.tensor([prefix_len], dtype=torch.int64, device="cuda"),
        torch.tensor([prefix_len], dtype=torch.int64, device="cuda"),
        req_pool_indices,
        req_to_token,
        prefix_len,
    )
    reference_state = torch.zeros(
        (1, 1, HEAD_DIM * 3), dtype=torch.float32, device="cuda"
    )
    fused_state = reference_state.clone()
    for state in (reference_state, fused_state):
        compress_forward(
            state,
            prefix_input,
            ape,
            prefill_plan,
            head_dim=HEAD_DIM,
            compress_ratio=RATIO,
            is_online=True,
        )

    decode_plan = _make_decode_plan(
        torch.tensor([RATIO], dtype=torch.int64, device="cuda")
    )
    out_loc = torch.zeros(1, dtype=torch.int64, device="cuda")
    reference_cache = _make_fp8_cache(1)
    fused_cache = reference_cache.clone()
    _run_reference(
        reference_state,
        decode_input,
        ape,
        decode_plan,
        norm_weight,
        freq_cis,
        out_loc,
        reference_cache,
    )
    _run_fused(
        fused_state,
        decode_input,
        ape,
        decode_plan,
        norm_weight,
        freq_cis,
        out_loc,
        fused_cache,
    )

    torch.testing.assert_close(fused_state, reference_state, rtol=0, atol=0)
    torch.testing.assert_close(fused_cache, reference_cache, rtol=0, atol=0)


def test_dcp_owner_store_matches_global_cache() -> None:
    batch_size = 4
    generator = torch.Generator(device="cuda").manual_seed(2)
    seq_lens = torch.full((batch_size,), RATIO, dtype=torch.int64, device="cuda")
    plan = _make_decode_plan(seq_lens)
    state = _make_state(batch_size, seed=3)
    kv_score_input = torch.randn(
        (batch_size, HEAD_DIM * 2),
        dtype=torch.float32,
        device="cuda",
        generator=generator,
    )
    ape = torch.randn(
        (RATIO, HEAD_DIM),
        dtype=torch.float32,
        device="cuda",
        generator=generator,
    )
    norm_weight = torch.randn(
        HEAD_DIM, dtype=torch.float32, device="cuda", generator=generator
    )
    freq_cis = _make_freq_cis(256)
    out_loc = torch.arange(batch_size, dtype=torch.int64, device="cuda")
    global_cache = _make_fp8_cache(batch_size)
    _run_reference(
        state.clone(),
        kv_score_input,
        ape,
        plan,
        norm_weight,
        freq_cis,
        out_loc,
        global_cache,
    )

    for rank in range(2):
        local_cache = _make_fp8_cache(batch_size // 2)
        _run_fused(
            state.clone(),
            kv_score_input,
            ape,
            plan,
            norm_weight,
            freq_cis,
            out_loc,
            local_cache,
            dcp_world_size=2,
            dcp_rank=rank,
        )
        for local_loc, global_loc in enumerate(range(rank, batch_size, 2)):
            torch.testing.assert_close(
                _extract_fp8_token(local_cache, local_loc),
                _extract_fp8_token(global_cache, global_loc),
                rtol=0,
                atol=0,
            )


def test_unified_bf16_boundary_store_matches_reference() -> None:
    batch_size = 3
    generator = torch.Generator(device="cuda").manual_seed(4)
    seq_lens = torch.full((batch_size,), RATIO, dtype=torch.int64, device="cuda")
    plan = _make_decode_plan(seq_lens)
    state = _make_state(batch_size, seed=5)
    kv_score_input = torch.randn(
        (batch_size, HEAD_DIM * 2),
        dtype=torch.float32,
        device="cuda",
        generator=generator,
    )
    ape = torch.randn(
        (RATIO, HEAD_DIM),
        dtype=torch.float32,
        device="cuda",
        generator=generator,
    )
    norm_weight = torch.randn(
        HEAD_DIM, dtype=torch.float32, device="cuda", generator=generator
    )
    freq_cis = _make_freq_cis(256)
    out_loc = torch.arange(batch_size, dtype=torch.int64, device="cuda")
    reference_cache = torch.zeros(
        (batch_size, HEAD_DIM * 2), dtype=torch.uint8, device="cuda"
    )
    fused_cache = reference_cache.clone()
    _run_reference(
        state.clone(),
        kv_score_input,
        ape,
        plan,
        norm_weight,
        freq_cis,
        out_loc,
        reference_cache,
        page_size=1,
        bf16_store=True,
    )
    _run_fused(
        state.clone(),
        kv_score_input,
        ape,
        plan,
        norm_weight,
        freq_cis,
        out_loc,
        fused_cache,
        page_size=1,
        bf16_store=True,
    )
    torch.testing.assert_close(fused_cache, reference_cache, rtol=0, atol=0)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
