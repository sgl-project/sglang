import torch

from sglang.jit_kernel.benchmark import marker
from sglang.kernels.ops.kvcache.fused_fp8_qkv_kv_cache import fused_fp8_qkv_kv_cache
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=6, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)

FP8 = torch.float8_e4m3fn
D = 128


def fused_qkv(q, k, v, k_cache, v_cache, cache_loc, k_scale, v_scale):
    return fused_fp8_qkv_kv_cache(
        q, k, v, k_cache, v_cache, cache_loc, k_scale, v_scale
    )


def fused_kv_only(q, k, v, k_cache, v_cache, cache_loc, k_scale, v_scale):
    fused_fp8_qkv_kv_cache(None, k, v, k_cache, v_cache, cache_loc, k_scale, v_scale)
    return q.to(FP8)


FN_MAP = {"fused_qkv": fused_qkv, "fused_kv_only": fused_kv_only}


@marker.parametrize("num_tokens", [8, 128, 2048, 4096, 8192, 16384], [8, 2048])
@marker.parametrize("hq,hkv", [(64, 2), (16, 1), (8, 1)])
@marker.benchmark("impl", ["fused_qkv", "fused_kv_only"])
def benchmark(num_tokens: int, hq: int, hkv: int, impl: str):
    qd, kvd = hq * D, hkv * D
    qkv = torch.randn(num_tokens, qd + 2 * kvd, dtype=torch.bfloat16, device="cuda")
    q = qkv[:, :qd]
    k = qkv[:, qd : qd + kvd].view(num_tokens, hkv, D)
    v = qkv[:, qd + kvd :].view(num_tokens, hkv, D)
    slots = num_tokens + 16
    k_cache = torch.zeros(slots, hkv, D, dtype=FP8, device="cuda")
    v_cache = torch.zeros(slots, hkv, D, dtype=FP8, device="cuda")
    cache_loc = torch.arange(num_tokens, dtype=torch.int64, device="cuda")
    k_scale = torch.tensor(0.5, dtype=torch.float32, device="cuda")
    v_scale = torch.tensor(0.7, dtype=torch.float32, device="cuda")
    return marker.do_bench(
        FN_MAP[impl],
        input_args=(q, k, v, k_cache, v_cache, cache_loc, k_scale, v_scale),
        graph_clone_args=(0,),
        memory_output=(k_cache, v_cache),
    )


if __name__ == "__main__":
    benchmark.run()
