"""Cold-L2 latency of the DeepGEMM down-activation preparation (no GEMMs).

intermediate_size is the local post-SwiGLU MoE width, not model hidden_size.
GLM-5.3-Flash has moe_intermediate_size=2048: TP1/2/4/8 give 2048/1024/512/256.
Qwen3-235B-A22B and Qwen3-Coder-480B-A35B have MoE widths 1536 and 2560:
https://huggingface.co/Qwen/Qwen3-235B-A22B/blob/main/config.json
https://huggingface.co/Qwen/Qwen3-Coder-480B-A35B-Instruct/blob/main/config.json
These are dimension-derived cases using GLM's clamped activation, not full
execution/accuracy claims for those models. Wider synthetic cases exercise
the other rows-per-CTA branches through the supported width limit.
"""

import torch

from sglang.kernels.jit.benchmark import marker
from sglang.kernels.jit.benchmark.utils import create_random
from sglang.kernels.ops.activation.silu_and_mul_clamp import (
    silu_and_mul_clamp,
    silu_and_mul_contig_post_quant,
)
from sglang.kernels.ops.moe.ep_moe_kernels import tma_align_input_scale
from sglang.kernels.ops.quantization.fp8_kernel import (
    create_per_token_group_quant_fp8_output_scale,
    sglang_per_token_group_quant_fp8,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=60, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)


def separate(x, activated, q, scale):
    silu_and_mul_clamp(x, activated, 10.0)
    sglang_per_token_group_quant_fp8(activated, 128, output_q=q, output_s=scale)
    return q, tma_align_input_scale(scale)


def fused(x, q, scale):
    hidden = q.shape[1]
    rows_per_cta = 4 if hidden <= 2048 else 2 if hidden <= 4096 else 1
    silu_and_mul_contig_post_quant(
        x,
        q,
        scale,
        128,
        swiglu_limit=10.0,
        round_to_bf16=True,
        rows_per_cta=rows_per_cta,
    )
    return q, scale


def measure(rows, intermediate_size, provider):
    x = create_random(rows, intermediate_size * 2)
    q = torch.empty(
        (rows, intermediate_size), device=x.device, dtype=torch.float8_e4m3fn
    )
    aligned = provider == "fused"
    scale = create_per_token_group_quant_fp8_output_scale(
        q.shape, x.device, 128, aligned, aligned, False
    )
    if aligned:
        fn, args = fused, (x, q, scale)
    else:
        activated = torch.empty(
            (rows, intermediate_size), device=x.device, dtype=x.dtype
        )
        fn, args = separate, (x, activated, q, scale)
    # All read and written input/output tensors rotate. At the smallest case
    # the buffers are large enough to avoid the marker's 100-rotation cap.
    return marker.do_bench(
        fn,
        input_args=args,
        graph_clone_args="all",
        warmup_iters=5,
        replay_iters=1000,
        disable_log_bandwidth=True,
    )


# Cover actual GLM TP widths and all three rows-per-CTA branches. Keep large
# token counts at the original widths to bound the rotating buffer footprint.
_SHAPES = [
    (rows, intermediate_size)
    for rows in (8192, 32768, 65536)
    for intermediate_size in (256, 512, 1024, 1536, 2048, 2560, 3072, 4096, 8192)
] + [
    (rows, intermediate_size)
    for rows in (262144, 294912)
    for intermediate_size in (512, 2048)
]
_CI_SHAPES = [(8192, width) for width in (512, 2048, 2560, 4096, 8192)]


@marker.parametrize("rows,intermediate_size", _SHAPES, _CI_SHAPES)
@marker.benchmark("provider", ["separate", "fused"])
def benchmark(rows, intermediate_size, provider):
    return measure(rows, intermediate_size, provider)


if __name__ == "__main__":
    benchmark.run()
