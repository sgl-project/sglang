import torch
from sgl_kernel import sgl_per_token_group_quant_8bit

from sglang.jit_kernel.benchmark import marker
from sglang.jit_kernel.benchmark.utils import create_random
from sglang.jit_kernel.per_token_group_quant_8bit_v2 import (
    per_token_group_quant_8bit_v2,
)
from sglang.srt.layers.quantization.fp8_kernel import (
    create_per_token_group_quant_fp8_output_scale,
    fp8_dtype,
    fp8_max,
    fp8_min,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=6, suite="base-b-kernel-benchmark-1-gpu-large")

G = 128
HIDDEN = 4096


def _aot_v2(x, x_q, x_s):
    # Low-level AOT op writing into the same preallocated x_q/x_s as the JIT
    # path, so this is a kernel-vs-kernel comparison (no wrapper / no realloc).
    sgl_per_token_group_quant_8bit(
        x,
        x_q,
        x_s,
        G,
        1e-10,
        float(fp8_min),
        float(fp8_max),
        False,  # scale_ue8m0
        False,  # fuse_silu_and_mul
        None,  # masked_m
        enable_v2=True,
    )


def _jit_v2(x, x_q, x_s):
    per_token_group_quant_8bit_v2(x, x_q, x_s, G, 1e-10, float(fp8_min), float(fp8_max))


FN = {"aot_v2": _aot_v2, "jit_v2": _jit_v2}


@marker.parametrize("num_tokens", [1, 8, 64, 512, 4096], ci_vals=[1, 512])
@marker.benchmark("impl", ["aot_v2", "jit_v2"])
def benchmark(num_tokens: int, impl: str):
    x = create_random(num_tokens, HIDDEN)
    x_q = torch.empty(num_tokens, HIDDEN, device="cuda", dtype=fp8_dtype)
    x_s = create_per_token_group_quant_fp8_output_scale(
        x_shape=(num_tokens, HIDDEN),
        device="cuda",
        group_size=G,
        column_major_scales=True,
        scale_tma_aligned=True,
        scale_ue8m0=False,
    )
    return marker.do_bench(FN[impl], input_args=(x, x_q, x_s), graph_clone_args=(0,))


if __name__ == "__main__":
    benchmark.run()
