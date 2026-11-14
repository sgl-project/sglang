import torch

from sglang.kernels.jit.benchmark import marker
from sglang.kernels.ops.quantization.groupwise_int8_dequant import _dequant_into
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=15, stage="base-b-kernel-benchmark", runner_config="1-gpu-small"
)


def _torch_dequant(weight, scales, output):
    rows, width = weight.shape
    output.copy_(
        (weight.reshape(rows, -1, 128).float() * scales[..., None]).reshape(rows, width)
    )


@marker.parametrize("shape", [(2048, 2048), (8192, 2048), (2048, 8192), (12288, 2048)])
@marker.benchmark("impl", ["torch", "fused"])
def benchmark(shape, impl):
    rows, width = shape
    weight = torch.randint(-128, 128, shape, dtype=torch.int8, device="cuda")
    scales = torch.rand((rows, width // 128), device="cuda")
    output = torch.empty(shape, dtype=torch.bfloat16, device="cuda")
    fn = _torch_dequant if impl == "torch" else _dequant_into
    # Rotate outputs as well as both inputs to avoid hot-L2 output reuse.
    return marker.do_bench(
        fn,
        input_args=(weight, scales, output),
        memory_args=(weight, scales),
        memory_output=(output,),
    )


if __name__ == "__main__":
    benchmark.run()
