from __future__ import annotations

import sys
from typing import Tuple, Union

import pytest
import torch
import triton

from sglang.jit_kernel.benchmark.bench_activation import register_cuda_ci
from sglang.jit_kernel.dsv4 import compress_forward
from sglang.jit_kernel.tests.deepseek_v4.common import (
    LegacyContext,
    PagedContext,
    make_legacy_context,
    make_paged_context,
    make_state_pool,
    to_seq_extend,
)

register_cuda_ci(est_time=30, suite="stage-b-kernel-unit-1-gpu-large")
register_cuda_ci(est_time=30, suite="nightly-kernel-1-gpu", nightly=True)

Context = Union[LegacyContext, PagedContext]

# c4 input row layout: | kv_overlap | kv | score_overlap | score |
HEAD_DIM = 512
RATIO = 4
WINDOW = 8  # = 2 * RATIO (overlap + current)
ATOL = 5e-3
RTOL = 5e-3


# -----------------------------------------------------------------------------
# fp64 ground truth (single compress event over a 8-token window).
# -----------------------------------------------------------------------------


def _gt_compress(
    kv_score_input_cpu: torch.Tensor,  # [num_q, head_dim*4]
    ape_cpu: torch.Tensor,  # [8, head_dim]
    P: int,
    head_dim: int,
) -> torch.Tensor:
    """fp64 reference for compress event at ragged position ``P``.

    Tokens at positions [P-7..P-4] contribute their *overlap* halves, tokens
    at [P-3..P] contribute their *fresh* halves. Bias[0..3] for overlap,
    bias[4..7] for fresh. When P < 7, the overlap is masked (kv=0, score=-inf)
    so the softmax sees only the 4 fresh tokens.
    """
    if P < 7:
        kv_ov = torch.zeros(4, head_dim, dtype=torch.float64)
        sc_ov = torch.full((4, head_dim), float("-inf"), dtype=torch.float64)
    else:
        kv_ov = kv_score_input_cpu[P - 7 : P - 3, :head_dim].double()
        sc_ov = kv_score_input_cpu[P - 7 : P - 3, 2 * head_dim : 3 * head_dim].double()
    kv_fr = kv_score_input_cpu[P - 3 : P + 1, head_dim : 2 * head_dim].double()
    sc_fr = kv_score_input_cpu[P - 3 : P + 1, 3 * head_dim :].double()
    kv = torch.cat([kv_ov, kv_fr], dim=0)
    sc = torch.cat([sc_ov, sc_fr], dim=0) + ape_cpu.double()
    return ((kv * sc.softmax(dim=0)).sum(dim=0)).float()


# -----------------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------------


def _run_prefill(
    ctx: Context,
    pool: torch.Tensor,
    kv_score_input: torch.Tensor,
    ape: torch.Tensor,
    seq_lens_cpu: torch.Tensor,
    extend_lens_cpu: torch.Tensor,
) -> torch.Tensor:
    num_q = int(extend_lens_cpu.sum().item())
    plan = ctx.make_prefill_plan(seq_lens_cpu, extend_lens_cpu, num_q)
    return compress_forward(
        pool,
        kv_score_input,
        ape,
        plan,
        head_dim=ctx.head_dim,
        compress_ratio=RATIO,
    )


def _run_decode(
    ctx: Context,
    pool: torch.Tensor,
    kv_score_input: torch.Tensor,
    ape: torch.Tensor,
    seq_lens_gpu: torch.Tensor,
) -> torch.Tensor:
    plan = ctx.make_decode_plan(seq_lens_gpu)
    return compress_forward(
        pool,
        kv_score_input,
        ape,
        plan,
        head_dim=ctx.head_dim,
        compress_ratio=RATIO,
    )


def _make_inputs(
    num_q: int, head_dim: int, seed: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator(device="cpu").manual_seed(seed)
    kv_score_input_cpu = torch.randn(
        num_q, head_dim * 4, generator=g, dtype=torch.float32
    )
    ape_cpu = torch.randn(WINDOW, head_dim, generator=g, dtype=torch.float32)
    return kv_score_input_cpu, ape_cpu


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["legacy", "paged"])
@pytest.mark.parametrize("seq_len", [4, 8, 32, 256, 1024])
def test_prefill_no_context(mode: str, seq_len: int) -> None:
    """Prefill once, no prefix. Every compress event must match fp64 GT."""
    if mode == "legacy":
        ctx: Context = make_legacy_context(
            bs=1, compress_ratio=RATIO, head_dim=HEAD_DIM
        )
    else:
        ctx = make_paged_context(bs=1, compress_ratio=RATIO, head_dim=HEAD_DIM)

    seq_lens_cpu, extend_lens_cpu, num_q = to_seq_extend([(seq_len, seq_len)])
    kv_in_cpu, ape_cpu = _make_inputs(num_q, ctx.head_dim, seed=seq_len)

    pool = make_state_pool(ctx.num_pages, RATIO, ctx.head_dim)
    out = _run_prefill(
        ctx, pool, kv_in_cpu.cuda(), ape_cpu.cuda(), seq_lens_cpu, extend_lens_cpu
    )

    # Compact prefill output: row per compress plan, in CPU-planner order
    # (batch-major, position-ascending).
    for plan_id, P in enumerate(range(RATIO - 1, seq_len, RATIO)):
        gt = _gt_compress(kv_in_cpu, ape_cpu, P=P, head_dim=ctx.head_dim)
        triton.testing.assert_close(out[plan_id].cpu(), gt, atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize("mode", ["legacy", "paged"])
@pytest.mark.parametrize("prefix_len", [4, 256])
def test_prefill_then_decode(mode: str, prefix_len: int) -> None:
    """Prefill once, then decode 4 more tokens through one compress boundary."""
    extend_decode = 4
    seq_len = prefix_len + extend_decode

    if mode == "legacy":
        ctx: Context = make_legacy_context(
            bs=1, compress_ratio=RATIO, head_dim=HEAD_DIM
        )
    else:
        ctx = make_paged_context(bs=1, compress_ratio=RATIO, head_dim=HEAD_DIM)

    kv_full_cpu, ape_cpu = _make_inputs(
        seq_len, ctx.head_dim, seed=seq_len + prefix_len
    )
    pool = make_state_pool(ctx.num_pages, RATIO, ctx.head_dim)

    # Prefill the prefix.
    seq_lens_cpu, extend_lens_cpu, _ = to_seq_extend([(prefix_len, prefix_len)])
    _run_prefill(
        ctx,
        pool,
        kv_full_cpu[:prefix_len].cuda(),
        ape_cpu.cuda(),
        seq_lens_cpu,
        extend_lens_cpu,
    )

    # Decode `extend_decode` tokens one at a time.
    final_out = None
    for k in range(extend_decode):
        cur_seq_len = prefix_len + k + 1
        seq_lens_gpu = torch.tensor([cur_seq_len], dtype=torch.int64, device="cuda")
        kv_step = kv_full_cpu[prefix_len + k : prefix_len + k + 1].cuda()
        out = _run_decode(ctx, pool, kv_step, ape_cpu.cuda(), seq_lens_gpu)
        if cur_seq_len % RATIO == 0:
            final_out = out

    # Check the trailing compress: position P = seq_len - 1 = prefix + 3.
    P = seq_len - 1
    gt = _gt_compress(kv_full_cpu, ape_cpu, P=P, head_dim=ctx.head_dim)
    assert final_out is not None
    triton.testing.assert_close(final_out[0].cpu(), gt, atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize("mode", ["legacy", "paged"])
@pytest.mark.parametrize("prefix_len", [256, 512, 768])
def test_prefill_then_extend(mode: str, prefix_len: int) -> None:
    """Prefill once, then prefill an extend that crosses one compress event.

    The first prefill ends at a swa_page boundary (only relevant for paged),
    so the second prefill's overlap must be read out of the buffer.
    """
    extend_len = 4

    if mode == "legacy":
        ctx: Context = make_legacy_context(
            bs=1, compress_ratio=RATIO, head_dim=HEAD_DIM
        )
    else:
        ctx = make_paged_context(bs=1, compress_ratio=RATIO, head_dim=HEAD_DIM)

    seq_len = prefix_len + extend_len
    kv_full_cpu, ape_cpu = _make_inputs(seq_len, ctx.head_dim, seed=prefix_len)
    pool = make_state_pool(ctx.num_pages, RATIO, ctx.head_dim)

    # First prefill: seq=prefix, ext=prefix.
    seq_lens_cpu, extend_lens_cpu, _ = to_seq_extend([(prefix_len, prefix_len)])
    _run_prefill(
        ctx,
        pool,
        kv_full_cpu[:prefix_len].cuda(),
        ape_cpu.cuda(),
        seq_lens_cpu,
        extend_lens_cpu,
    )

    # Second prefill: seq=prefix+extend, ext=extend, prefix=prefix_len.
    seq_lens_cpu, extend_lens_cpu, num_q = to_seq_extend([(seq_len, extend_len)])
    out = _run_prefill(
        ctx,
        pool,
        kv_full_cpu[prefix_len:].cuda(),
        ape_cpu.cuda(),
        seq_lens_cpu,
        extend_lens_cpu,
    )

    P = seq_len - 1
    gt = _gt_compress(kv_full_cpu, ape_cpu, P=P, head_dim=ctx.head_dim)
    # Single compress event in this extend; compact plan_id 0.
    triton.testing.assert_close(out[0].cpu(), gt, atol=ATOL, rtol=RTOL)


def test_paged_buffer_intermediate() -> None:
    """Paged-only: after a multi-page prefill, verify the trailing 4 tokens of
    every swa_page sit in the correct state-pool slots.

    These slots are what radix-cache resume reads when prefix-matching from a
    swa_page boundary, so they MUST match the original token data.
    """
    ctx = make_paged_context(
        bs=1,
        compress_ratio=RATIO,
        head_dim=HEAD_DIM,
        swa_page_size=256,
        ring_size=8,
        num_swa_pages_per_req=8,
    )
    seq_len = 1024  # 4 swa_pages
    seq_lens_cpu, extend_lens_cpu, num_q = to_seq_extend([(seq_len, seq_len)])
    kv_in_cpu, ape_cpu = _make_inputs(num_q, ctx.head_dim, seed=42)

    pool = make_state_pool(ctx.num_pages, RATIO, ctx.head_dim)
    _run_prefill(
        ctx, pool, kv_in_cpu.cuda(), ape_cpu.cuda(), seq_lens_cpu, extend_lens_cpu
    )

    pool_cpu = pool.cpu()
    # For each swa_page boundary, the trailing `RATIO` tokens must have been
    # written. The state slot for token at position p is
    # `state_loc(0, p) = (p // swa_page_size) * ring_size + p % ring_size`.
    for swa_page_end in range(ctx.swa_page_size, seq_len + 1, ctx.swa_page_size):
        for offset in range(RATIO):
            p = swa_page_end - RATIO + offset
            sl = ctx.state_loc(0, p)
            page_idx = sl // RATIO
            slot_idx = sl % RATIO
            actual = pool_cpu[page_idx, slot_idx]
            # Token-row layout: the c4 prefill write copies the full
            # head_dim*4 row from kv_input verbatim into the state pool.
            expected = kv_in_cpu[p]
            triton.testing.assert_close(
                actual,
                expected,
                atol=ATOL,
                rtol=RTOL,
            )


@pytest.mark.parametrize("mode", ["legacy", "paged"])
def test_prefill_multibatch(mode: str) -> None:
    """Multi-batch prefill, both modes."""
    seq_extend = [(8, 8), (256, 256), (260, 260), (1023, 1023)]
    bs = len(seq_extend)
    if mode == "legacy":
        ctx: Context = make_legacy_context(
            bs=bs, compress_ratio=RATIO, head_dim=HEAD_DIM
        )
    else:
        ctx = make_paged_context(bs=bs, compress_ratio=RATIO, head_dim=HEAD_DIM)

    seq_lens_cpu, extend_lens_cpu, num_q = to_seq_extend(seq_extend)
    kv_in_cpu, ape_cpu = _make_inputs(num_q, ctx.head_dim, seed=99)
    pool = make_state_pool(ctx.num_pages, RATIO, ctx.head_dim)
    out = _run_prefill(
        ctx, pool, kv_in_cpu.cuda(), ape_cpu.cuda(), seq_lens_cpu, extend_lens_cpu
    )

    # Compact: walk batches in order, then positions in order; matches the
    # CPU planner's emit order for plan_c.
    base = 0
    plan_id = 0
    for b, (seq, ext) in enumerate(seq_extend):
        for j in range(ext):
            P = j  # prefix=0 here
            if (P + 1) % RATIO != 0:
                continue
            gt = _gt_compress(
                kv_in_cpu[base : base + ext],
                ape_cpu,
                P=P,
                head_dim=ctx.head_dim,
            )
            triton.testing.assert_close(
                out[plan_id].cpu(),
                gt,
                atol=ATOL,
                rtol=RTOL,
            )
            plan_id += 1
        base += ext


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
