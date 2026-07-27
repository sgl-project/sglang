import itertools
import sys

import pytest
import torch
import triton

from sglang.kernels.jit.utils import get_ci_test_range
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=37, stage="base-b-kernel-unit", runner_config="1-gpu-large")
# Nightly is not redundant here: it sets SGLANG_JIT_KERNEL_RUN_FULL_TESTS=1 to expand get_ci_test_range sweeps.
register_cuda_ci(est_time=148, suite="nightly-kernel-1-gpu", nightly=True)


def sglang_aot_qknorm(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
) -> None:
    from sgl_kernel import rmsnorm

    head_dim = q.shape[-1]
    q = q.view(-1, head_dim)
    k = k.view(-1, head_dim)
    rmsnorm(q, q_weight, out=q)
    rmsnorm(k, k_weight, out=k)


def sglang_jit_qknorm(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
) -> None:
    from sglang.kernels.ops.layernorm.norm import fused_inplace_qknorm

    fused_inplace_qknorm(q, k, q_weight, k_weight)


def flashinfer_qknorm(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
) -> None:
    from flashinfer.norm import rmsnorm

    rmsnorm(q, q_weight, out=q)
    rmsnorm(k, k_weight, out=k)


@torch.compile()
def torch_impl_qknorm(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    eps: float = 1e-6,
) -> None:
    q_mean = q.float().pow(2).mean(dim=-1, keepdim=True)
    k_mean = k.float().pow(2).mean(dim=-1, keepdim=True)
    q_norm = (q_mean + eps).rsqrt()
    k_norm = (k_mean + eps).rsqrt()
    q.copy_(q.float() * q_norm * q_weight.float())
    k.copy_(k.float() * k_norm * k_weight.float())


def torch_impl_hf_qknorm(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference the HF cast-before-weight RMSNorm ordering."""
    q_norm = q.float() * (q.float().pow(2).mean(dim=-1, keepdim=True) + eps).rsqrt()
    k_norm = k.float() * (k.float().pow(2).mean(dim=-1, keepdim=True) + eps).rsqrt()
    return q_weight * q_norm.to(q.dtype), k_weight * k_norm.to(k.dtype)


def torch_impl_standard_qknorm(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference the standard multiply-before-output-cast RMSNorm ordering."""
    q_norm = (q.float().pow(2).mean(dim=-1, keepdim=True) + eps).rsqrt()
    k_norm = (k.float().pow(2).mean(dim=-1, keepdim=True) + eps).rsqrt()
    return (
        (q.float() * q_norm * q_weight.float()).to(q.dtype),
        (k.float() * k_norm * k_weight.float()).to(k.dtype),
    )


BS_LIST = [2**n for n in range(0, 14)]
BS_LIST += [x + 1 + i for i, x in enumerate(BS_LIST)]
BS_LIST = get_ci_test_range(BS_LIST, [1, 9, 256, 4109])
N_K_LIST = get_ci_test_range([2, 4], [2, 4])
N_Q_LIST = get_ci_test_range([8, 16], [8, 16])
HEAD_DIM_LIST = get_ci_test_range([64, 128, 256, 512, 1024], [64, 256, 1024])
DEVICE = "cuda"
DTYPE = torch.bfloat16

# NOTE(dark): sgl_kernel use flashinfer template, which is bitwise identical to flashinfer impl.
# However, sgl-jit-kernel, flashinfer, torch_impl, may have small numerical differences.
# so we allow a small rel/abs tolerance in correctness test.


@pytest.mark.parametrize(
    "batch_size,n_k,n_q,head_dim",
    list(itertools.product(BS_LIST, N_K_LIST, N_Q_LIST, HEAD_DIM_LIST)),
)
def test_qknorm(batch_size: int, n_k: int, n_q: int, head_dim: int) -> None:
    q = torch.randn(batch_size, n_q, head_dim, device=DEVICE, dtype=DTYPE)
    k = torch.randn(batch_size, n_k, head_dim, device=DEVICE, dtype=DTYPE)
    q_weight = torch.randn(head_dim, device=DEVICE, dtype=DTYPE)
    k_weight = torch.randn(head_dim, device=DEVICE, dtype=DTYPE)
    q_k_aot = (q.clone(), k.clone())
    q_k_jit = (q.clone(), k.clone())
    sglang_aot_qknorm(q_k_aot[0], q_k_aot[1], q_weight, k_weight)
    sglang_jit_qknorm(q_k_jit[0], q_k_jit[1], q_weight, k_weight)
    triton.testing.assert_close(q_k_aot[0], q_k_jit[0], atol=1e-2, rtol=1e-2)
    triton.testing.assert_close(q_k_aot[1], q_k_jit[1], atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_qknorm_hf_cast_order(dtype: torch.dtype) -> None:
    """The optional path must round normalized values before weight multiply."""
    from sglang.jit_kernel.norm import fused_inplace_qknorm

    torch.manual_seed(0)
    q = torch.randn(64, 28, 128, device=DEVICE, dtype=dtype)
    k = torch.randn(64, 4, 128, device=DEVICE, dtype=dtype)
    q_weight = torch.randn(128, device=DEVICE, dtype=dtype)
    k_weight = torch.randn(128, device=DEVICE, dtype=dtype)
    expected_q, expected_k = torch_impl_hf_qknorm(q, k, q_weight, k_weight)
    standard_q, standard_k = torch_impl_standard_qknorm(q, k, q_weight, k_weight)

    actual_q = q.clone()
    actual_k = k.clone()
    fused_inplace_qknorm(
        actual_q,
        actual_k,
        q_weight,
        k_weight,
        cast_x_before_out_mul=True,
    )

    torch.testing.assert_close(actual_q, expected_q, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(actual_k, expected_k, atol=1e-2, rtol=1e-2)
    hf_error = max(
        (actual_q.float() - expected_q.float()).abs().max().item(),
        (actual_k.float() - expected_k.float()).abs().max().item(),
    )
    standard_error = max(
        (actual_q.float() - standard_q.float()).abs().max().item(),
        (actual_k.float() - standard_k.float()).abs().max().item(),
    )
    assert (expected_q != standard_q).any() or (expected_k != standard_k).any()
    assert hf_error < standard_error, (
        "cast-before-weight kernel is not closer to the HF reference: "
        f"hf={hf_error}, standard={standard_error}"
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
