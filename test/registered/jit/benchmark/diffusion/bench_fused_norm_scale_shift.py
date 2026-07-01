# Benchmarks SGLang fused layernorm/rmsnorm scale shift kernels
# 1. fused_norm_scale_shift
# 2. fused_scale_residual_norm_scale_shift
import torch

from sglang.jit_kernel.benchmark import marker
from sglang.multimodal_gen.runtime.layers.layernorm import (
    LayerNormScaleShift,
    RMSNormScaleShift,
    ScaleResidualLayerNormScaleShift,
    ScaleResidualRMSNormScaleShift,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=17,
    stage="base-b-kernel-benchmark",
    runner_config="1-gpu-large",
    disabled="Temporarily skipped to unblock flashinfer upgrade. Ref: https://github.com/sgl-project/sglang/actions/runs/23735552939/job/69139238979?pr=21422",
)

B_RANGE = [1]
S_RANGE = [128, 1024, 4096]
D_RANGE = [1024, 3072, 4096]
NORM_TYPE_RANGE = ["layer", "rms"]
AFFINE_RANGE = [True, False]
DTYPE = torch.bfloat16
DEVICE = "cuda"
EPS = 1e-5


def preprocess_layer(layer, affine: bool, D: int, DTYPE: torch.dtype):
    if affine:
        weight = torch.randn(D, dtype=DTYPE, device=DEVICE)
        bias = torch.randn(D, dtype=DTYPE, device=DEVICE)
        with torch.no_grad():
            layer.norm.weight.copy_(weight)
            if hasattr(layer.norm, "bias"):
                layer.norm.bias.copy_(bias)
    layer.requires_grad_(False)
    return layer.to(DEVICE)


# ============================================================================
# Benchmark 1: fused_norm_scale_shift
# ============================================================================
@marker.parametrize("B", B_RANGE, [1])
@marker.parametrize("S", S_RANGE, [128])
@marker.parametrize("D", D_RANGE, [1024])
@marker.parametrize("norm_type", NORM_TYPE_RANGE, ["layer"])
@marker.parametrize("affine", AFFINE_RANGE, [True])
@marker.benchmark("provider", ["native", "cuda"])
def bench_fused_norm_scale_shift(
    B: int, S: int, D: int, norm_type: str, affine: bool, provider: str
):
    x = torch.randn(B, S, D, dtype=DTYPE, device=DEVICE)
    scale = torch.randn(B, S, D, dtype=DTYPE, device=DEVICE)
    shift = torch.randn(B, S, D, dtype=DTYPE, device=DEVICE)
    if norm_type == "layer":
        layer = LayerNormScaleShift(D, EPS, affine, dtype=DTYPE)
    else:
        layer = RMSNormScaleShift(D, EPS, affine, dtype=DTYPE)
    layer = preprocess_layer(layer, affine, D, DTYPE)
    if provider == "native":
        fn = layer.forward_native
    else:
        fn = layer.forward_cuda

    # Rotate the read tensors per iteration (do_bench clones input_args); a
    # zero-arg closure would keep them L2-hot and report wrongly fast numbers.
    return marker.do_bench(fn, input_args=(x, shift, scale))


# ============================================================================
# Benchmark 2: fused_scale_residual_norm_scale_shift
# ============================================================================
@marker.parametrize("B", B_RANGE, [1])
@marker.parametrize("S", S_RANGE, [128])
@marker.parametrize("D", D_RANGE, [1024])
@marker.parametrize("norm_type", NORM_TYPE_RANGE, ["layer"])
@marker.parametrize("affine", AFFINE_RANGE, [True])
@marker.benchmark("provider", ["native", "cuda"])
def bench_fused_scale_residual_norm_scale_shift(
    B: int, S: int, D: int, norm_type: str, affine: bool, provider: str
):
    residual = torch.randn(B, S, D, dtype=DTYPE, device=DEVICE)
    x = torch.randn(B, S, D, dtype=DTYPE, device=DEVICE)
    scale = torch.randn(B, S, D, dtype=DTYPE, device=DEVICE)
    shift = torch.randn(B, S, D, dtype=DTYPE, device=DEVICE)
    gate = torch.randn(B, 1, D, dtype=DTYPE, device=DEVICE)
    if norm_type == "layer":
        layer = ScaleResidualLayerNormScaleShift(D, EPS, affine, dtype=DTYPE).to(DEVICE)
    else:
        layer = ScaleResidualRMSNormScaleShift(D, EPS, affine, dtype=DTYPE).to(DEVICE)
    layer = preprocess_layer(layer, affine, D, DTYPE)
    if provider == "native":
        fn = layer.forward_native
    else:
        fn = layer.forward_cuda

    # Rotate the read tensors per iteration (do_bench clones input_args); a
    # zero-arg closure would keep them L2-hot and report wrongly fast numbers.
    return marker.do_bench(fn, input_args=(residual, x, gate, shift, scale))


SEP = "=" * 80


if __name__ == "__main__":
    print(f"\n{SEP}")
    print("Benchmark: fused_norm_scale_shift")
    print(f"{SEP}\n")
    bench_fused_norm_scale_shift.run()

    print(f"\n{SEP}")
    print("Benchmark: fused_scale_residual_norm_scale_shift")
    print(f"{SEP}\n")
    bench_fused_scale_residual_norm_scale_shift.run()
