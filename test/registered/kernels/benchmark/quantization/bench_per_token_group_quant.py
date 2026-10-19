from sglang.kernels.jit.benchmark import marker
from sglang.kernels.jit.benchmark.utils import create_empty, create_random
from sglang.kernels.ops.activation.activation import silu_and_mul
from sglang.kernels.ops.quantization.fp8_kernel import (
    create_per_token_group_quant_fp8_output_scale,
    fp8_dtype,
    fp8_max,
    fp8_min,
)

# per_token_group_quant_8bit_v2 is the perf baseline; production only routes
# fp32 UE8M0 scales to it.
from sglang.kernels.ops.quantization.per_token_group_quant import (
    per_token_group_quant,
)
from sglang.kernels.ops.quantization.per_token_group_quant_8bit_v2 import (
    per_token_group_quant_8bit_v2,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=35, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)

HIDDEN = 2048
LAYOUTS = {
    "row_major_fp32": (False, False),
    "col_major_fp32": (True, False),
    "col_major_ue8m0": (True, True),
}


def _jit_v2(G, x, x_q, x_s, scale_ue8m0):
    per_token_group_quant_8bit_v2(
        x,
        x_q,
        x_s,
        G,
        1e-10,
        float(fp8_min),
        float(fp8_max),
        scale_ue8m0=scale_ue8m0,
    )


def _current(G, x, x_q, x_s, scale_ue8m0):
    per_token_group_quant(x, x_q, x_s, G, scale_ue8m0=scale_ue8m0)


FN = {"jit_v2": _jit_v2, "current": _current}


@marker.parametrize("group_size", [32, 64, 128], ci_vals=[128])
@marker.parametrize("layout", list(LAYOUTS), ci_vals=["col_major_ue8m0"])
@marker.parametrize("num_tokens", [2**n for n in range(0, 14)], ci_vals=[1, 32, 2048])
@marker.benchmark("impl", ["jit_v2", "current"])
def benchmark(group_size: int, layout: str, num_tokens: int, impl: str):
    column_major, scale_ue8m0 = LAYOUTS[layout]
    x = create_random(num_tokens, HIDDEN)
    x_q = create_empty(num_tokens, HIDDEN, dtype=fp8_dtype)
    x_s = create_per_token_group_quant_fp8_output_scale(
        x_shape=(num_tokens, HIDDEN),
        device="cuda",
        group_size=group_size,
        column_major_scales=column_major,
        scale_tma_aligned=column_major,
        scale_ue8m0=scale_ue8m0,
    )
    return marker.do_bench(
        FN[impl],
        input_args=(group_size, x, x_q, x_s, scale_ue8m0),
        memory_args=(x,),
        memory_output=(x_q, x_s),
    )


def _dense_activation_quant(x, fused):
    return per_token_group_quant(
        x if fused else silu_and_mul(x),
        fuse_silu_and_mul=fused,
        silu_and_mul_mode="dense" if fused else "legacy",
        scale_ue8m0=True,
        column_major_scales=True,
    )


@marker.parametrize("num_tokens", [1, 32, 128], ci_vals=[1, 32])
@marker.benchmark("fused", [False, True])
def benchmark_dense(num_tokens: int, fused: bool):
    x = create_random(num_tokens, 8192)
    return marker.do_bench(_dense_activation_quant, input_args=(x, fused))


if __name__ == "__main__":
    benchmark.run()
    benchmark_dense.run()
