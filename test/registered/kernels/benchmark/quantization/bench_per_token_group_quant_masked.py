import math

import torch

from sglang.kernels.jit.benchmark import marker
from sglang.kernels.jit.benchmark.utils import create_empty, create_random
from sglang.kernels.ops.quantization.fp8_kernel import (
    create_per_token_group_quant_fp8_output_scale,
    fp8_dtype,
    fp8_max,
    fp8_min,
)

# per_token_group_quant_8bit_v2 is DEPRECATED (no production call sites); the
# kernel is kept only as the perf baseline for this benchmark.
from sglang.kernels.ops.quantization.per_token_group_quant import (
    per_token_group_quant,
)
from sglang.kernels.ops.quantization.per_token_group_quant_8bit_v2 import (
    per_token_group_quant_8bit_v2,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=25, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)

# name -> (moe_intermediate_size, topk, num_experts, group_size)
MODELS = {
    "deepseek_v4": (3072, 6, 384, 32),  # DeepSeek-V4 Pro
    "deepseek_v3": (2048, 8, 256, 128),  # DeepSeek-V3/R1
    "qwen3_235b": (1536, 8, 128, 128),  # Qwen3-235B-A22B
}


def _jit_v2(G, x, x_q, x_s, masked_m, expected_m, fuse):
    per_token_group_quant_8bit_v2(
        x,
        x_q,
        x_s,
        G,
        1e-10,
        float(fp8_min),
        float(fp8_max),
        scale_ue8m0=True,
        fuse_silu_and_mul=fuse,
        masked_m=masked_m,
    )


def _current(G, x, x_q, x_s, masked_m, expected_m, fuse):
    per_token_group_quant(
        x,
        x_q,
        x_s,
        G,
        scale_ue8m0=True,
        fuse_silu_and_mul=fuse,
        masked_m=masked_m,
        expected_m=expected_m,
    )


FN = {"jit_v2": _jit_v2, "current": _current}


@marker.parametrize("model", list(MODELS), ci_vals=["deepseek_v3"])
@marker.parametrize("num_gpus", [4, 8], ci_vals=[4])
@marker.parametrize("fuse_silu", [True, False], ci_vals=[False])
@marker.parametrize("balanced", [True, False], ci_vals=[True])
@marker.parametrize("num_tokens", [2**n for n in range(8)], ci_vals=[1, 128])
@marker.benchmark("impl", ["jit_v2", "current"], unit="us")
def benchmark(
    model: str,
    fuse_silu: bool,
    num_gpus: int,
    num_tokens: int,
    balanced: bool,
    impl: str,
) -> marker.BenchResult:
    torch.cuda.random.manual_seed(42)
    max_tokens = 128  # TODO: test other size
    hidden_size, topk, num_experts, group_size = MODELS[model]
    if num_experts % num_gpus != 0 or topk * num_gpus > num_experts:
        marker.skip("Incompatible model configuration")
    if impl == "jit_v2" and (hidden_size // group_size) % 16 != 0:
        marker.skip("v2 masked requires num_groups % 16 == 0")
    if num_tokens > max_tokens:
        marker.skip("num_tokens exceeds max_tokens")

    num_local_experts = num_experts // num_gpus
    padded_tokens = max_tokens * num_gpus
    expected_m = math.ceil(max_tokens * topk / num_local_experts)
    in_hidden = hidden_size * (2 if fuse_silu else 1)
    x = create_random(num_local_experts, padded_tokens, in_hidden)
    x_q = create_empty(num_local_experts, padded_tokens, hidden_size, dtype=fp8_dtype)
    x_s = create_per_token_group_quant_fp8_output_scale(
        x_shape=(num_local_experts, padded_tokens, hidden_size),
        device="cuda",
        group_size=group_size,
        column_major_scales=True,
        scale_tma_aligned=True,
        scale_ue8m0=True,
    )
    if balanced:  # simulation
        topk_ids = torch.randint(0, num_local_experts, (num_tokens * topk,))
        masked_m = torch.bincount(topk_ids, minlength=num_local_experts)
        masked_m = masked_m.cuda().int()
    else:  # only the last few experts receive all tokens
        masked_m = create_empty(num_local_experts, dtype=torch.int32)
        masked_m[:-topk].zero_()
        masked_m[-topk:].fill_(num_tokens)
    return marker.do_bench(
        FN[impl],
        input_args=(group_size, x, x_q, x_s, masked_m, expected_m, fuse_silu),
        graph_clone_args=(0,),
        memory_args=(x[:topk, :num_tokens], masked_m),
        memory_output=(x_q[:topk, :num_tokens], x_s[:topk, :num_tokens]),
    )


if __name__ == "__main__":
    benchmark.run()
