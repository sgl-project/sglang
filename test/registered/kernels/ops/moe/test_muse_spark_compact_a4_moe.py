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
