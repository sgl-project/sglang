import torch

from sglang.jit_kernel.benchmark import marker
from sglang.jit_kernel.diffusion.triton.norm import norm_infer
from sglang.jit_kernel.diffusion.triton.scale_shift import (
    fuse_layernorm_scale_shift_gate_select01_kernel,
    fuse_residual_layernorm_scale_shift_gate_select01_kernel,
)
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(
    est_time=13, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)
register_amd_ci(est_time=13, stage="jit-kernel-benchmark", runner_config="amd")

DTYPE = torch.bfloat16
DEVICE = "cuda"
EPS = 1e-6
SEP = "=" * 80
B_RANGE = [1, 2]
S_RANGE = [128, 512, 2048]
D_RANGE = [1024, 1536, 3072]


def _make_common_inputs(batch_size: int, seq_len: int, hidden_size: int):
    x = torch.randn(batch_size, seq_len, hidden_size, dtype=DTYPE, device=DEVICE)
    weight = torch.randn(hidden_size, dtype=DTYPE, device=DEVICE)
    bias = torch.randn(hidden_size, dtype=DTYPE, device=DEVICE)
    index = torch.randint(0, 2, (batch_size, seq_len), dtype=torch.int32, device=DEVICE)
    scale0 = torch.randn(batch_size, hidden_size, dtype=DTYPE, device=DEVICE)
    shift0 = torch.randn(batch_size, hidden_size, dtype=DTYPE, device=DEVICE)
    gate0 = torch.randn(batch_size, hidden_size, dtype=DTYPE, device=DEVICE)
    scale1 = torch.randn(batch_size, hidden_size, dtype=DTYPE, device=DEVICE)
    shift1 = torch.randn(batch_size, hidden_size, dtype=DTYPE, device=DEVICE)
    gate1 = torch.randn(batch_size, hidden_size, dtype=DTYPE, device=DEVICE)
    return x, weight, bias, index, scale0, shift0, gate0, scale1, shift1, gate1


def _apply_select01_modulation(
    x: torch.Tensor,
    scale0: torch.Tensor,
    shift0: torch.Tensor,
    gate0: torch.Tensor,
    scale1: torch.Tensor,
    shift1: torch.Tensor,
    gate1: torch.Tensor,
    index: torch.Tensor,
):
    idx = index.bool().unsqueeze(-1)
    scale = torch.where(idx, scale1.unsqueeze(1), scale0.unsqueeze(1))
    shift = torch.where(idx, shift1.unsqueeze(1), shift0.unsqueeze(1))
    gate = torch.where(idx, gate1.unsqueeze(1), gate0.unsqueeze(1))
    return x * (1 + scale) + shift, gate


@marker.parametrize("B", B_RANGE, [1])
@marker.parametrize("S", S_RANGE, [128])
@marker.parametrize("D", D_RANGE, [3072])
@marker.benchmark("provider", ["split", "fused"])
def bench_layernorm_scale_shift_gate_select01(B: int, S: int, D: int, provider: str):
    x, weight, bias, index, scale0, shift0, gate0, scale1, shift1, gate1 = (
        _make_common_inputs(B, S, D)
    )

    if provider == "split":

        def fn(x):
            normalized = norm_infer(
                x.view(-1, x.shape[-1]),
                weight,
                bias,
                eps=EPS,
                is_rms_norm=False,
            ).view_as(x)
            return _apply_select01_modulation(
                normalized, scale0, shift0, gate0, scale1, shift1, gate1, index
            )

    else:

        def fn(x):
            return fuse_layernorm_scale_shift_gate_select01_kernel(
                x,
                weight=weight,
                bias=bias,
                scale0=scale0,
                shift0=shift0,
                gate0=gate0,
                scale1=scale1,
                shift1=shift1,
                gate1=gate1,
                index=index,
                eps=EPS,
            )

    # Rotate the dominant read tensor x per iteration (do_bench clones
    # input_args); a zero-arg closure would keep it L2-hot and report wrongly
    # fast numbers.
    return marker.do_bench(fn, input_args=(x,))


@marker.parametrize("B", B_RANGE, [1])
@marker.parametrize("S", S_RANGE, [128])
@marker.parametrize("D", D_RANGE, [3072])
@marker.benchmark("provider", ["split", "fused"])
def bench_residual_layernorm_scale_shift_gate_select01(
    B: int, S: int, D: int, provider: str
):
    x, weight, bias, index, scale0, shift0, gate0, scale1, shift1, gate1 = (
        _make_common_inputs(B, S, D)
    )
    residual = torch.randn_like(x)
    residual_gate = torch.randn_like(x)

    if provider == "split":

        def fn(x, residual, residual_gate):
            residual_out = residual + residual_gate * x
            normalized = norm_infer(
                residual_out.view(-1, residual_out.shape[-1]),
                weight,
                bias,
                eps=EPS,
                is_rms_norm=False,
            ).view_as(residual_out)
            return _apply_select01_modulation(
                normalized, scale0, shift0, gate0, scale1, shift1, gate1, index
            )

    else:

        def fn(x, residual, residual_gate):
            return fuse_residual_layernorm_scale_shift_gate_select01_kernel(
                x,
                residual=residual,
                residual_gate=residual_gate,
                weight=weight,
                bias=bias,
                scale0=scale0,
                shift0=shift0,
                gate0=gate0,
                scale1=scale1,
                shift1=shift1,
                gate1=gate1,
                index=index,
                eps=EPS,
            )

    # Rotate the read tensors per iteration (do_bench clones input_args); a
    # zero-arg closure would keep them L2-hot and report wrongly fast numbers.
    return marker.do_bench(fn, input_args=(x, residual, residual_gate))


if __name__ == "__main__":
    print(f"\n{SEP}")
    print("Benchmark: qwen_image layernorm + scale_shift_gate_select01")
    print(f"{SEP}\n")
    bench_layernorm_scale_shift_gate_select01.run()

    print(f"\n{SEP}")
    print("Benchmark: qwen_image residual + layernorm + scale_shift_gate_select01")
    print(f"{SEP}\n")
    bench_residual_layernorm_scale_shift_gate_select01.run()
