import torch

from sglang.kernels.jit.benchmark import marker
from sglang.kernels.jit.benchmark.utils import create_random
from sglang.kernels.ops.quantization.mxfp8_group_quant import mxfp8_group_quant
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=8, stage="base-b-kernel-benchmark", runner_config="1-gpu-small"
)


@torch.compile()
def _torch_mxfp8_group_quant(x: torch.Tensor):
    m, k = x.shape
    blocks = x.float().view(m, k // 32, 32)
    dequant = blocks.abs().amax(dim=-1, keepdim=True) / 448.0
    scale_bits = (dequant.view(torch.int32) + 0x007FFFFF) & 0x7F800000
    d = scale_bits.view(torch.float32)
    quant_scale = torch.where(d == 0, torch.zeros_like(d), 1.0 / d)
    q = (blocks * quant_scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    return q.view(m, k), (scale_bits >> 23).to(torch.uint8).view(m, k // 32)


FN_MAP = {
    "jit": mxfp8_group_quant,
    "torch": _torch_mxfp8_group_quant,
}


# Shapes are flattened MoE expert weights ([E * N, K]) and dense weights.
@marker.parametrize(
    "m,k",
    [(4096, 7168), (7168, 2048), (32 * 5760, 2880), (16 * 4096, 7168)],
    [(4096, 7168), (32 * 5760, 2880)],
)
@marker.parametrize("dtype", [torch.bfloat16, torch.float32], [torch.bfloat16])
@marker.benchmark("impl", ["jit", "torch"])
def benchmark(m: int, k: int, dtype: torch.dtype, impl: str):
    x = create_random(m, k, dtype=dtype)
    return marker.do_bench(FN_MAP[impl], input_args=(x,), graph_clone_args=(0,))


if __name__ == "__main__":
    benchmark.run()
