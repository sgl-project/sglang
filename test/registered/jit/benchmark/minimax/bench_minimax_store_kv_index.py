"""Benchmark: fused MiniMax-M3 KV + index cache store (1 launch) vs the separate
per-buffer index_put_ stores (main K, main V, index K, optional index V)."""

import torch

from sglang.jit_kernel.benchmark import marker
from sglang.jit_kernel.minimax_store_kv_index import store_kv_index
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(
    est_time=6, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)
register_amd_ci(est_time=6, suite="jit-kernel-unit-test-amd")

HEAD_DIM = 128
NUM_KV_HEADS = 1
HAS_V = False
N = 1 << 20
DTYPE = torch.bfloat16


def _separate(k, v, kc, vc, ik, ikc, loc, **_):
    kc[loc] = k
    vc[loc] = v
    ikc[loc] = ik


def _fused(k, v, kc, vc, ik, ikc, loc, *, num_kv_heads, head_bytes):
    store_kv_index(
        k,
        v,
        kc,
        vc,
        ik,
        ikc,
        None,
        None,
        loc,
        num_kv_heads=num_kv_heads,
        head_bytes=head_bytes,
    )


FN_MAP = {"fused": _fused, "separate": _separate}


@marker.parametrize("T", [16, 64, 256, 1024, 4096, 16384], [256, 4096])
@marker.benchmark("impl", ["fused", "separate"])
def benchmark(T: int, impl: str):
    k = torch.randn(T, NUM_KV_HEADS * HEAD_DIM, dtype=DTYPE, device="cuda")
    v = torch.randn_like(k)
    ik = torch.randn(T, HEAD_DIM, dtype=DTYPE, device="cuda")
    kc = torch.zeros(N, NUM_KV_HEADS * HEAD_DIM, dtype=DTYPE, device="cuda")
    vc = torch.zeros_like(kc)
    ikc = torch.zeros(N, HEAD_DIM, dtype=DTYPE, device="cuda")
    loc = torch.randperm(N, device="cuda")[:T]
    extra_kwargs = dict(num_kv_heads=NUM_KV_HEADS, head_bytes=HEAD_DIM * DTYPE.itemsize)
    return marker.do_bench(
        FN_MAP[impl],
        input_args=(k, v, kc, vc, ik, ikc, loc),
        input_kwargs=extra_kwargs if impl == "fused" else {},
        # Read inputs cloned per iter; caches are write targets (kept hot).
        graph_clone_args=(0, 1, 4, 6),
        memory_args=(k, v, ik, loc),
        memory_output=(k, v, ik),
    )


if __name__ == "__main__":
    benchmark.run()
