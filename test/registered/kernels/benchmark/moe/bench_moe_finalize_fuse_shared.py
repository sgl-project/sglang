import torch

from sglang.kernels.jit.benchmark import marker
from sglang.kernels.ops.moe.moe_finalize_fuse_shared import moe_finalize_fuse_shared
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=10, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)

# name -> (hidden size, top-k, shared row): the trtllm deferred-finalize models
CONFIGS = {
    "kimi_k3": (3584, 16, False),
    "deepseek_v4_flash": (4096, 6, True),
    "deepseek_v4_pro": (7168, 6, True),
    "deepseek_v3": (7168, 8, True),
    "glm5": (6144, 8, True),
    "qwen3_5_moe": (4096, 10, True),
}


@marker.parametrize("config", list(CONFIGS), ["kimi_k3", "deepseek_v4_flash"])
@marker.parametrize("num_tokens", [1, 8, 32, 64, 128, 192, 256, 384], [1, 128])
@marker.benchmark("provider", ["jit"])
def benchmark(num_tokens: int, config: str, provider: str):
    hidden, top_k, has_shared = CONFIGS[config]
    rows = num_tokens * top_k + 256
    gemm2_out = torch.randn(rows, hidden, device="cuda", dtype=torch.bfloat16)
    idx = torch.randperm(rows, device="cuda")[: num_tokens * top_k].to(torch.int32)
    weights = torch.rand(num_tokens, top_k, device="cuda")
    shared = (
        torch.randn(num_tokens, hidden, device="cuda", dtype=torch.bfloat16)
        if has_shared
        else None
    )
    out = torch.empty(num_tokens, hidden, device="cuda", dtype=torch.bfloat16)
    return marker.do_bench(
        moe_finalize_fuse_shared,
        input_args=(gemm2_out, idx, weights, shared, top_k),
        input_kwargs={"out": out},
    )


if __name__ == "__main__":
    benchmark.run()
