import torch

from sglang.kernels.jit.benchmark import marker
from sglang.kernels.ops.attention.qk_norm_rope import qk_norm_rope
from sglang.kernels.ops.attention.rope import apply_rope_inplace
from sglang.kernels.ops.layernorm.norm import fused_inplace_qknorm
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=6, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)


def prepare(qkv, qw, kw, cache, positions, fused):
    q, k, _ = qkv.split([32 * 128, 8 * 128, 8 * 128], dim=-1)
    if fused:
        return qk_norm_rope(q, k, qw, kw, cache, positions, 128, 1e-6)
    q = q.view(q.shape[0], 32, 128)
    k = k.view(k.shape[0], 8, 128)
    fused_inplace_qknorm(q, k, qw, kw, 1e-6)
    apply_rope_inplace(q, k, cache, positions, is_neox=True)
    return q.contiguous()


@marker.parametrize("tokens", [512, 4096, 8192], [512, 4096])
@marker.parametrize("dtype", [torch.bfloat16, torch.float16])
@marker.benchmark("fused", [False, True])
def benchmark(tokens, dtype, fused):
    qkv = torch.randn(tokens, 48 * 128, device="cuda", dtype=dtype)
    qw = torch.ones(128, device="cuda", dtype=dtype)
    kw = torch.ones_like(qw)
    cache = torch.randn(32768, 128, device="cuda", dtype=torch.float32)
    positions = torch.arange(tokens, device="cuda")
    return marker.do_bench(
        prepare,
        input_args=(qkv, qw, kw, cache, positions, fused),
        graph_clone_args=(0, 1, 2, 3, 4),
    )


if __name__ == "__main__":
    benchmark.run()
