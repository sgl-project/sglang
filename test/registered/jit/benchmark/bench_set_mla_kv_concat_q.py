"""Fused MLA KV-scatter + absorbed-q concat vs the two-kernel sequence.

Both workloads are launch-bound at decode batch sizes, so the interesting
number is latency (one launch saved), not bandwidth.
"""

import torch

from sglang.kernels.jit.benchmark import marker
from sglang.kernels.ops.attention.concat_mla import concat_mla_absorb_q
from sglang.kernels.ops.attention.set_mla_kv_concat_q import set_mla_kv_concat_q
from sglang.kernels.ops.kvcache.set_mla_kv_buffer import set_mla_kv_buffer
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=6, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)

NOPE_DIM = 512
ROPE_DIM = 64
TOTAL_DIM = NOPE_DIM + ROPE_DIM
PAGES = 8192


def _two_kernel(pool, loc, k_nope, k_rope, q_nope, q_rope):
    set_mla_kv_buffer(pool, loc, k_nope, k_rope)
    return concat_mla_absorb_q(q_nope, q_rope)


FN_MAP = {
    "fused": set_mla_kv_concat_q,
    "two_kernel": _two_kernel,
}


@marker.parametrize("bs", [1, 8, 64, 256, 1024], [1, 64])
@marker.parametrize("heads", [8, 16], [8])
@marker.benchmark("impl", ["fused", "two_kernel"])
def benchmark(bs: int, heads: int, impl: str):
    if torch.cuda.get_device_capability()[0] < 9:
        marker.skip("requires SM90+ (TMA bulk store)")

    gen = torch.Generator(device="cuda").manual_seed(bs)

    def rand(*shape):
        return (
            torch.randn(*shape, generator=gen, device="cuda", dtype=torch.float32)
            .mul(0.1)
            .to(torch.bfloat16)
        )

    pool = rand(PAGES, TOTAL_DIM)
    latent = rand(bs, TOTAL_DIM)
    q_all = rand(bs, heads, TOTAL_DIM)
    loc = torch.randperm(PAGES, generator=gen, device="cuda")[:bs]
    k_nope, k_rope = latent[:, :NOPE_DIM], latent[:, NOPE_DIM:]
    q_nope, q_rope = q_all[..., :NOPE_DIM], q_all[..., NOPE_DIM:]

    return marker.do_bench(
        FN_MAP[impl],
        input_args=(pool, loc, k_nope, k_rope, q_nope, q_rope),
        # Reads: loc + latent halves + q halves. The pool is written (scatter),
        # clone it too so replayed iterations don't accumulate L2-hot rows.
        graph_clone_args=(0, 1, 2, 3, 4, 5),
        # Bytes actually moved: latent row + q rows read, q_out returned via
        # the "out" default. The scattered pool rows (~same as latent) are
        # not expressible here; latency is the number that matters.
        memory_args=(latent, q_all, loc),
    )


if __name__ == "__main__":
    benchmark.run()
