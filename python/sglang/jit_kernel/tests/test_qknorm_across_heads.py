import itertools

import pytest
import torch
import triton


def sglang_jit_qknorm_across_heads(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
) -> None:
    from sglang.jit_kernel.norm import fused_inplace_qknorm_across_heads

    fused_inplace_qknorm_across_heads(q, k, q_weight, k_weight)


def sglang_aot_qknorm_across_heads(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
) -> None:
    from sgl_kernel import rmsnorm

    rmsnorm(q, q_weight, out=q)
    rmsnorm(k, k_weight, out=k)


@torch.compile()
def torch_impl_qknorm_across_heads(
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


BS_LIST = [2**n for n in range(0, 14)]
BS_LIST += [x + 1 + i for i, x in enumerate(BS_LIST)]
HIDDEN_DIM_LIST = [512, 1024, 2048, 4096]
DEVICE = "cuda"
DTYPE = torch.bfloat16


@pytest.mark.parametrize(
    "batch_size,hidden_dim",
    list(itertools.product(BS_LIST, HIDDEN_DIM_LIST)),
)
def test_qknorm_across_heads(batch_size: int, hidden_dim: int) -> None:
    q = torch.randn(batch_size, hidden_dim, device=DEVICE, dtype=DTYPE)
    k = torch.randn(batch_size, hidden_dim, device=DEVICE, dtype=DTYPE)
    q_weight = torch.randn(hidden_dim, device=DEVICE, dtype=DTYPE)
    k_weight = torch.randn(hidden_dim, device=DEVICE, dtype=DTYPE)

    q_k_jit = (q.clone(), k.clone())
    q_k_aot = (q.clone(), k.clone())

    sglang_jit_qknorm_across_heads(q_k_jit[0], q_k_jit[1], q_weight, k_weight)
    sglang_aot_qknorm_across_heads(q_k_aot[0], q_k_aot[1], q_weight, k_weight)

    triton.testing.assert_close(q_k_jit[0], q_k_aot[0], atol=1e-2, rtol=1e-2)
    triton.testing.assert_close(q_k_jit[1], q_k_aot[1], atol=1e-2, rtol=1e-2)


if __name__ == "__main__":
    pytest.main([__file__])
