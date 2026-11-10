import torch

from sglang.kernels.jit.benchmark import marker
from sglang.kernels.ops.quantization.mxfp4_dequant import mxfp4_dequant
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=6, stage="base-b-kernel-benchmark", runner_config="1-gpu-small"
)

_E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
_E2M1_LUT = torch.tensor(_E2M1 + [-v for v in _E2M1], dtype=torch.float32)


@torch.compile()
def _torch_mxfp4_dequant(packed: torch.Tensor, scale: torch.Tensor, lut: torch.Tensor):
    rows, packed_k = packed.shape
    k = packed_k * 2
    p = packed.long()
    values = torch.stack((lut[p & 0xF], lut[p >> 4]), dim=-1).view(rows, k // 32, 32)
    block_scale = (scale.to(torch.int32) << 23).view(torch.float32).unsqueeze(-1)
    return (values * block_scale).view(rows, k).to(torch.bfloat16)


# [E * N, K] rows of MoE expert weights (gpt-oss-20b / -120b w13 and w2).
@marker.parametrize(
    "rows,k",
    [(32 * 5760, 2880), (32 * 2880, 2880), (8 * 5760, 2880)],
    [(32 * 5760, 2880)],
)
@marker.benchmark("impl", ["jit", "torch"])
def benchmark(rows: int, k: int, impl: str):
    packed = torch.randint(0, 256, (rows, k // 2), dtype=torch.uint8, device="cuda")
    scale = torch.randint(110, 135, (rows, k // 32), dtype=torch.uint8, device="cuda")
    if impl == "jit":
        fn, args = mxfp4_dequant, (packed, scale)
    else:
        fn, args = _torch_mxfp4_dequant, (packed, scale, _E2M1_LUT.cuda())
    return marker.do_bench(fn, input_args=args, graph_clone_args=(0, 1))


if __name__ == "__main__":
    benchmark.run()
