from __future__ import annotations

from typing import Tuple, Union

import pytest
import torch
import triton

from sglang.jit_kernel.dsv4 import compress_forward

from .common import (
    LegacyContext,
    PagedContext,
    make_legacy_context,
    make_paged_context,
    make_state_pool,
    to_seq_extend,
)

Context = Union[LegacyContext, PagedContext]

# c128 input row layout: | kv | score |  each [head_dim]
HEAD_DIM = 512
RATIO = 128
ATOL = 5e-3
RTOL = 5e-3


def _gt_compress(
    kv_score_input_cpu: torch.Tensor,  # [num_q, head_dim*2]
    ape_cpu: torch.Tensor,  # [128, head_dim]
    P: int,
    head_dim: int,
) -> torch.Tensor:
    """fp64 reference for compress event at ragged position ``P`` (P % 128 == 127)."""
    lo = P - (RATIO - 1)
    kv = kv_score_input_cpu[lo : P + 1, :head_dim].double()
    sc = kv_score_input_cpu[lo : P + 1, head_dim:].double()
    return ((kv * (sc + ape_cpu.double()).softmax(dim=0)).sum(dim=0)).float()


def _make_inputs(
    num_q: int, head_dim: int, seed: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator(device="cpu").manual_seed(seed)
    kv_score_input_cpu = torch.randn(
        num_q, head_dim * 2, generator=g, dtype=torch.float32
    )
    ape_cpu = torch.randn(RATIO, head_dim, generator=g, dtype=torch.float32)
    return kv_score_input_cpu, ape_cpu


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


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["legacy", "paged"])
@pytest.mark.parametrize("seq_len", [128, 256, 512])
def test_prefill_no_context(mode: str, seq_len: int) -> None:
    """Single-shot prefill, no prefix. Every compress event must match fp64 GT."""
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

    # Compact prefill output: row per compress plan, in CPU-planner order.
    for plan_id, P in enumerate(range(RATIO - 1, seq_len, RATIO)):
        gt = _gt_compress(kv_in_cpu, ape_cpu, P=P, head_dim=ctx.head_dim)
        triton.testing.assert_close(out[plan_id].cpu(), gt, atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize("mode", ["legacy", "paged"])
@pytest.mark.parametrize("prefix_len", [0, 128, 256])
def test_prefill_then_decode(mode: str, prefix_len: int) -> None:
    """Prefill ``prefix_len`` tokens, then decode through to the next 128 boundary."""
    seq_len = prefix_len + RATIO  # one full compress chunk after prefix

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

    if prefix_len > 0:
        seq_lens_cpu, extend_lens_cpu, _ = to_seq_extend([(prefix_len, prefix_len)])
        _run_prefill(
            ctx,
            pool,
            kv_full_cpu[:prefix_len].cuda(),
            ape_cpu.cuda(),
            seq_lens_cpu,
            extend_lens_cpu,
        )

    final_out = None
    for k in range(RATIO):
        cur_seq_len = prefix_len + k + 1
        seq_lens_gpu = torch.tensor([cur_seq_len], dtype=torch.int64, device="cuda")
        kv_step = kv_full_cpu[prefix_len + k : prefix_len + k + 1].cuda()
        out = _run_decode(ctx, pool, kv_step, ape_cpu.cuda(), seq_lens_gpu)
        if cur_seq_len % RATIO == 0:
            final_out = out

    P = seq_len - 1
    gt = _gt_compress(kv_full_cpu, ape_cpu, P=P, head_dim=ctx.head_dim)
    assert final_out is not None
    triton.testing.assert_close(final_out[0].cpu(), gt, atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize("mode", ["legacy", "paged"])
@pytest.mark.parametrize("prefix_len", [128, 256])
def test_prefill_then_extend(mode: str, prefix_len: int) -> None:
    """Prefill once, then a second prefill that extends across one compress event.

    First prefill ends at a 128-boundary so the second prefill starts fresh.
    """
    extend_len = RATIO
    seq_len = prefix_len + extend_len

    if mode == "legacy":
        ctx: Context = make_legacy_context(
            bs=1, compress_ratio=RATIO, head_dim=HEAD_DIM
        )
    else:
        ctx = make_paged_context(bs=1, compress_ratio=RATIO, head_dim=HEAD_DIM)

    kv_full_cpu, ape_cpu = _make_inputs(seq_len, ctx.head_dim, seed=prefix_len)
    pool = make_state_pool(ctx.num_pages, RATIO, ctx.head_dim)

    seq_lens_cpu, extend_lens_cpu, _ = to_seq_extend([(prefix_len, prefix_len)])
    _run_prefill(
        ctx,
        pool,
        kv_full_cpu[:prefix_len].cuda(),
        ape_cpu.cuda(),
        seq_lens_cpu,
        extend_lens_cpu,
    )

    seq_lens_cpu, extend_lens_cpu, _ = to_seq_extend([(seq_len, extend_len)])
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


@pytest.mark.parametrize("mode", ["legacy", "paged"])
def test_prefill_multibatch(mode: str) -> None:
    """Multi-batch prefill, each batch ending at a different chunk count."""
    seq_extend = [(128, 128), (256, 256), (384, 384)]
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
            P = j  # prefix=0
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
    pytest.main([__file__, "-v"])
