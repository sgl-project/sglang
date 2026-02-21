import torch
import triton
import triton.testing
from sgl_kernel.elementwise import downcast_fp8 as aot_downcast_fp8

from sglang.jit_kernel.benchmark.utils import is_in_ci
from sglang.jit_kernel.cast import downcast_fp8 as jit_downcast_fp8

IS_CI = is_in_ci()

DEVICE = "cuda"
DTYPE = torch.bfloat16

if IS_CI:
    CONFIGS = [(4, 8, 128, 16)]
else:
    CONFIGS = [(sl, h, d, sl * 2) for sl in [4, 16, 64] for h in [8, 32] for d in [128]]

LINE_VALS = ["jit", "aot"]
LINE_NAMES = ["SGL JIT Kernel", "SGL AOT Kernel"]
STYLES = [("orange", "-"), ("blue", "--")]


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["input_sl", "head", "dim", "out_sl"],
        x_vals=CONFIGS,
        line_arg="provider",
        line_vals=LINE_VALS,
        line_names=LINE_NAMES,
        styles=STYLES,
        ylabel="us",
        plot_name="downcast-fp8-performance",
        args={},
    )
)
def benchmark(input_sl, head, dim, out_sl, provider):
    k = torch.randn(input_sl, head, dim, dtype=DTYPE, device=DEVICE)
    v = torch.randn(input_sl, head, dim, dtype=DTYPE, device=DEVICE)
    k_out = torch.zeros(out_sl, head, dim, dtype=torch.uint8, device=DEVICE)
    v_out = torch.zeros(out_sl, head, dim, dtype=torch.uint8, device=DEVICE)
    k_scale = torch.tensor([1.0], dtype=torch.float32, device=DEVICE)
    v_scale = torch.tensor([1.0], dtype=torch.float32, device=DEVICE)
    loc = torch.arange(input_sl, dtype=torch.int64, device=DEVICE)

    FN_MAP = {
        "jit": jit_downcast_fp8,
        "aot": aot_downcast_fp8,
    }
    fn = lambda: FN_MAP[provider](k, v, k_out, v_out, k_scale, v_scale, loc)
    quantiles = [0.5, 0.2, 0.8]
    ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(fn, quantiles=quantiles)  # type: ignore
    return 1000 * ms, 1000 * max_ms, 1000 * min_ms


if __name__ == "__main__":
    benchmark.run(print_data=True)
