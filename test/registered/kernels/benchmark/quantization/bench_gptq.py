import torch
from sgl_kernel import gptq_gemm as aot_gptq_gemm

from sglang.kernels.jit.benchmark import marker
from sglang.kernels.ops.quantization import gptq_gemm
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=40, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)

FN_MAP = {"jit": gptq_gemm, "aot": aot_gptq_gemm}


def pack_rows(values: torch.Tensor, bit: int) -> torch.Tensor:
    pack = 32 // bit
    out = torch.zeros(
        (values.shape[0] // pack, values.shape[1]),
        dtype=torch.int32,
        device=values.device,
    )
    for index in range(pack):
        out |= values[index::pack].to(torch.int32) << (index * bit)
    return out


def pack_cols(values: torch.Tensor, bit: int) -> torch.Tensor:
    pack = 32 // bit
    out = torch.zeros(
        (values.shape[0], values.shape[1] // pack),
        dtype=torch.int32,
        device=values.device,
    )
    for index in range(pack):
        out |= values[:, index::pack].to(torch.int32) << (index * bit)
    return out


def make_args(m: int, n: int, k: int, group_size: int = 128):
    bit = 4
    groups = k // group_size
    weight = torch.randn(k, n, dtype=torch.float16, device="cuda")
    grouped = weight.reshape(groups, group_size, n)
    maximum = grouped.amax(dim=1, keepdim=True)
    minimum = grouped.amin(dim=1, keepdim=True)
    scales = ((maximum - minimum) / 15).clamp(min=1e-6)
    zeros = (-minimum / scales).round()
    quant = ((grouped / scales + zeros).round().clamp(0, 15)).to(torch.uint8)
    q_weight = pack_rows(quant.reshape(k, n), bit)
    q_zeros = pack_cols((zeros.to(torch.uint8) - 1).reshape(groups, n), bit)
    g_idx = torch.arange(k, dtype=torch.int32, device="cuda") // group_size
    a = torch.randn(m, k, dtype=torch.float16, device="cuda")
    return a, q_weight, q_zeros, scales.squeeze(1), g_idx, False, bit


@marker.parametrize("m", [1, 8, 16, 128], [1, 128])
@marker.parametrize("n,k", [(2048, 2048), (4096, 4096)], [(4096, 4096)])
@marker.benchmark("impl", ["jit", "aot"])
def benchmark(m: int, n: int, k: int, impl: str):
    args = make_args(m, n, k)
    return marker.do_bench(
        FN_MAP[impl], input_args=args, graph_clone_args=(0,), disable_log_bandwidth=True
    )


if __name__ == "__main__":
    benchmark.run()
