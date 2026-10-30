"""Muse Spark compact A4 verify experts (gfx950): the EP4 64-expert launch matches two EP8 32-expert launches."""

import pytest
import torch

from sglang.srt.utils import is_gfx95_supported, is_hip
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=60, suite="stage-b-test-1-gpu-small-amd-mi35x")

pytestmark = pytest.mark.skipif(
    not (is_hip() and is_gfx95_supported()), reason="Tuned for MI355X (gfx950)."
)


def _inputs(m, local_start, experts):
    torch.manual_seed(m * 1000 + experts)
    dev = "cuda"
    x = (torch.randn(m, 4096, device=dev) * 0.5).to(torch.bfloat16)
    # Random MXFP4 nibbles and E8M0 scales near 2^0 keep the GEMMs finite.
    w13 = torch.randint(0, 256, (experts, 8192, 2048), device=dev, dtype=torch.uint8)
    w2 = torch.randint(0, 256, (experts, 4096, 2048), device=dev, dtype=torch.uint8)
    s13 = torch.randint(118, 124, (experts, 8192, 128), device=dev, dtype=torch.uint8)
    s2 = torch.randint(118, 124, (experts, 4096, 128), device=dev, dtype=torch.uint8)
    # Each token routes to 8 distinct experts; about a third land outside this rank.
    pool = torch.arange(
        local_start - experts // 2, local_start + experts + experts // 2, device=dev
    )
    ids = torch.stack(
        [pool[torch.randperm(pool.numel(), device=dev)[:8]] for _ in range(m)]
    )
    return x, ids.int().contiguous(), w13, w2, s13, s2


@pytest.mark.parametrize("m", [8, 16, 64, 192, 512])
def test_compact_a4_local64_matches_two_local32(m):
    from sglang.kernels.ops.moe.mxfp4_compact_a4_moe import compact_a4_experts

    start = 64
    x, ids, w13, w2, s13, s2 = _inputs(m, start, 64)
    got = compact_a4_experts(x, ids, w13, w2, s13, s2, local_start=start)
    flat = ids.flatten()
    for half in (0, 1):
        lo = start + 32 * half
        sl = slice(32 * half, 32 * half + 32)
        ref = compact_a4_experts(
            x,
            ids,
            w13[sl].contiguous(),
            w2[sl].contiguous(),
            s13[sl].contiguous(),
            s2[sl].contiguous(),
            local_start=lo,
        )
        rows = ((flat >= lo) & (flat < lo + 32)).nonzero().flatten()
        assert rows.numel() > 0
        assert torch.equal(got[rows], ref[rows])


@pytest.mark.parametrize("experts", [32, 64])
@pytest.mark.parametrize("m", [8, 128, 512])
def test_compact_a4_reads_a4w4_preshuffled_weights(experts, m):
    """The a4w4 preshuffled copy (the only one kept when both do not fit) gives the stored-layout result."""
    from aiter import dtypes
    from aiter.ops.shuffle import (
        shuffle_scale_a16w4,
        shuffle_weight,
        shuffle_weight_a16w4,
    )
    from aiter.utility.fp4_utils import e8m0_shuffle

    from sglang.kernels.ops.moe.mxfp4_compact_a4_moe import compact_a4_experts

    start = experts
    x, ids, w13, w2, s13, s2 = _inputs(m, start, experts)
    e = experts
    shuffled = (
        shuffle_weight(w13.view(dtypes.fp4x2), (16, 16)),
        shuffle_weight_a16w4(w2.view(dtypes.fp4x2), 16, False),
        e8m0_shuffle(s13.view(dtypes.fp8_e8m0).view(e * 8192, 128)),
        shuffle_scale_a16w4(s2.view(dtypes.fp8_e8m0).view(e * 4096, 128), e, False),
    )
    views = [
        t.view(torch.uint8).view(e, n, k)
        for t, (n, k) in zip(
            shuffled, ((8192, 2048), (4096, 2048), (8192, 128), (4096, 128))
        )
    ]
    ref = compact_a4_experts(x, ids, w13, w2, s13, s2, local_start=start)
    got = compact_a4_experts(x, ids, *views, local_start=start, shuffled=True)
    flat = ids.flatten()
    rows = ((flat >= start) & (flat < start + e)).nonzero().flatten()
    assert torch.equal(got[rows], ref[rows])


@pytest.mark.parametrize("m", [8, 64, 512])
def test_compact_a4_prequantized_input(m):
    """An input quantized by the caller (the routing epilogue) gives the same expert rows."""
    from sglang.kernels.ops.moe.mxfp4_compact_a4_moe import (
        _prepare_a4,
        compact_a4_experts,
    )

    start = 64
    x, ids, w13, w2, s13, s2 = _inputs(m, start, 64)
    quant = [torch.empty(m, n, device="cuda", dtype=torch.uint8) for n in (2048, 128)]
    _prepare_a4[(m * 4,)](x, *quant, num_warps=4)
    ref = compact_a4_experts(x, ids, w13, w2, s13, s2, local_start=start)
    got = compact_a4_experts(
        None, ids, w13, w2, s13, s2, local_start=start, quantized=tuple(quant)
    )
    flat = ids.flatten()
    rows = ((flat >= start) & (flat < start + 64)).nonzero().flatten()
    assert torch.equal(got[rows], ref[rows])
