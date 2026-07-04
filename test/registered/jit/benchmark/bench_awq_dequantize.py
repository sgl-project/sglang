import torch

from sglang.jit_kernel.awq_dequantize import awq_dequantize as jit_awq_dequantize
from sglang.jit_kernel.benchmark import marker
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=5, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)

try:
    from sgl_kernel import awq_dequantize as aot_awq_dequantize

    AOT_AVAILABLE = True
except ImportError:
    AOT_AVAILABLE = False


LINE_VALS = ["jit", "aot"] if AOT_AVAILABLE else ["jit"]


@marker.parametrize("qweight_row", [128, 256, 512, 1024, 3584], [128])
@marker.parametrize("qweight_col", [16, 32, 64, 128, 448], [16])
@marker.benchmark("provider", LINE_VALS)
def benchmark(qweight_row: int, qweight_col: int, provider: str):
    if provider == "aot" and not AOT_AVAILABLE:
        marker.skip("sgl_kernel AOT not available")

    device = torch.device("cuda")
    qweight = torch.randint(
        0,
        torch.iinfo(torch.int32).max,
        (qweight_row, qweight_col),
        dtype=torch.int32,
        device=device,
    )
    group_size = qweight_row
    scales_row = qweight_row // group_size
    scales_col = qweight_col * 8
    scales = torch.rand(scales_row, scales_col, dtype=torch.float16, device=device)
    qzeros = torch.randint(
        0,
        torch.iinfo(torch.int32).max,
        (scales_row, qweight_col),
        dtype=torch.int32,
        device=device,
    )

    if provider == "jit":
        fn = jit_awq_dequantize
    elif provider == "aot":
        fn = aot_awq_dequantize
    else:
        raise ValueError(f"Unknown provider: {provider}")

    return marker.do_bench(
        fn,
        input_args=(qweight, scales, qzeros),
        graph_clone_args=(0, 1, 2),
    )


if __name__ == "__main__":
    benchmark.run()
