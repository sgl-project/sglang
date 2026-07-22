from sglang.jit_kernel.benchmark import marker
from sglang.jit_kernel.benchmark.utils import create_empty, create_random

# per_token_group_quant_8bit_v2 is DEPRECATED (no production call sites); the
# kernel is kept only as the perf baseline for this benchmark.
from sglang.jit_kernel.per_token_group_quant import per_token_group_quant
from sglang.jit_kernel.per_token_group_quant_8bit_v2 import (
    per_token_group_quant_8bit_v2,
)
from sglang.kernels.ops.quantization.fp8_kernel import (
    create_per_token_group_quant_fp8_output_scale,
    fp8_dtype,
    fp8_max,
    fp8_min,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=25, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
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
        graph_clone_args=(1,),
        memory_args=(x,),
        memory_output=(x_q, x_s),
    )


if __name__ == "__main__":
    benchmark.run()
