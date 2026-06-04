import torch

from sglang.jit_kernel.benchmark import marker
from sglang.jit_kernel.benchmark.utils import create_random
from sglang.jit_kernel.kimi_k2_moe_fused_gate import (
    kimi_k2_moe_fused_gate as jit_kimi_k2_moe_fused_gate,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=6, suite="base-b-kernel-benchmark-1-gpu-large")

NUM_EXPERTS = 384
TOPK = 8

# fp32 is the legacy host-upcast input; bf16 is the new native router-logit input.
# Both should hit the same kernel speed (bf16 is widened in-register), which is the
# point of this axis: the dtype change must not regress the kernel itself.
DTYPE_MAP = {"fp32": torch.float32, "bf16": torch.bfloat16}


@torch.compile()
def torch_impl(input: torch.Tensor, bias: torch.Tensor):
    scores = torch.sigmoid(input.to(torch.float32))
    biased = scores + bias.to(torch.float32).unsqueeze(0)
    _, idx = torch.topk(biased, k=TOPK, dim=-1, sorted=False)
    weights = scores.gather(1, idx)
    s = weights.sum(dim=-1, keepdim=True)
    weights = weights / s
    return weights.to(torch.float32), idx.to(torch.int32)


def jit_impl(input: torch.Tensor, bias: torch.Tensor):
    return jit_kimi_k2_moe_fused_gate(input, bias, topk=TOPK, renormalize=True)


FN_MAP = {"jit": jit_impl, "torch": torch_impl}


@marker.parametrize("num_rows", [16, 64, 128, 512, 1024, 4096, 16384], [128, 4096])
@marker.parametrize("dtype", ["fp32", "bf16"])
@marker.benchmark("impl", ["jit", "torch"])
def benchmark(num_rows: int, dtype: str, impl: str):
    torch_dtype = DTYPE_MAP[dtype]
    input = create_random(num_rows, NUM_EXPERTS, dtype=torch_dtype)
    bias = create_random(NUM_EXPERTS, dtype=torch_dtype)
    return marker.do_bench(
        FN_MAP[impl],
        input_args=(input, bias),
        graph_clone_args=(0, 1),
    )


if __name__ == "__main__":
    benchmark.run()
