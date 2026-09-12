"""Ratio-2 target-verify compressor including RMSNorm, RoPE and cache writes."""

import torch

from sglang.kernels.jit.benchmark import marker
from sglang.kernels.ops.attention.dsv4.c2 import c2_verify_norm_rope_store
from sglang.srt.mem_cache.deepseek_v4_memory_pool import get_compress_state_ring_size
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=15, stage="base-b-kernel-benchmark", runner_config="4-gpu-b200"
)


@marker.parametrize("batch", [1, 8, 64, 128], [1, 64])
@marker.benchmark("draft_len", [2, 6])
def benchmark(batch: int, draft_len: int):
    if torch.cuda.get_device_capability()[0] < 10:
        marker.skip("FP4 packing requires SM100")
    device, dim, page = "cuda", 512, 128
    n = batch * draft_len
    ring = get_compress_state_ring_size(2, True, draft_len)
    inputs = torch.randn(n, 2 * dim, device=device)
    state = torch.randn((batch + 1) * ring, 2 * dim, device=device)
    norm = torch.randn(dim, device=device, dtype=torch.bfloat16)
    req = torch.arange(batch, device=device).repeat_interleave(draft_len)
    pos = (
        torch.arange(batch, device=device)[:, None] % 2
        + 8
        + torch.arange(draft_len, device=device)
    ).flatten()
    loc = torch.arange(n, device=device) * 2 + 3
    angles = torch.randn(32, 32, device=device)
    freqs = torch.view_as_real(torch.polar(torch.ones_like(angles), angles)).flatten(-2)
    cache = torch.zeros(
        n // page + 2, -(-584 * page // 576) * 576, device=device, dtype=torch.uint8
    )
    out = torch.empty(n, dim, device=device, dtype=torch.bfloat16)
    return marker.do_bench(
        c2_verify_norm_rope_store,
        input_args=(inputs, state, norm, pos, req, loc, 1e-6, freqs, cache),
        input_kwargs=dict(page_size=page, ring_size=ring, draft_len=draft_len, out=out),
        memory_output=(out, state, cache),
    )


if __name__ == "__main__":
    benchmark.run()
