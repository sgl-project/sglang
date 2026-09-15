"""FP32 router logits with a fixed per-token reduction over the hidden dimension.

This optional small-router path keeps the reduction independent of how tokens
are partitioned into batches. It does not make the rest of a model deterministic.
"""

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["M", "N"])
def _router_kernel(
    X,
    W,
    Y,
    M,
    N,
    K: tl.constexpr,
    SX0,
    SX1,
    SW0,
    SW1,
    SY0,
    EXPERTS: tl.constexpr,
    PADDED_K: tl.constexpr,
):
    row = tl.program_id(0)
    experts = tl.program_id(1) * EXPERTS + tl.arange(0, EXPERTS)
    hidden = tl.arange(0, PADDED_K)
    x = tl.load(
        X + row * SX0 + hidden * SX1,
        mask=(row < M) & (hidden < K),
        other=0,
    ).to(tl.float32)
    w = tl.load(
        W + experts[:, None] * SW0 + hidden[None, :] * SW1,
        mask=(experts[:, None] < N) & (hidden[None, :] < K),
        other=0,
    ).to(tl.float32)
    # Each row uses the same reduction tree, including at microbatch boundaries.
    logits = tl.sum(w * x[None, :], axis=1)
    tl.store(Y + row * SY0 + experts, logits, mask=(row < M) & (experts < N))


def batch_invariant_router_gemm(
    hidden_states: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    """Return FP32 [tokens, experts] logits from BF16 [tokens, hidden] inputs.

    Weights may be BF16 or FP32. The fixed reduction supports hidden dimensions
    up to 8192, including non-power-of-two tails, and strided input matrices.
    Empty token/expert dimensions require no kernel launch. No global precision
    settings, communication resources, or synchronization policies are changed.
    """
    x, w = hidden_states, weight
    if x.layout != torch.strided or w.layout != torch.strided:
        raise ValueError("Batch-invariant router requires strided tensors")
    if x.ndim != 2 or w.ndim != 2 or x.shape[1] != w.shape[1]:
        raise ValueError("Expected router input [M,K] and weight [N,K]")
    if x.device.type != "cuda" or x.device != w.device:
        raise ValueError("Batch-invariant router inputs must share a CUDA device")
    if x.dtype != torch.bfloat16 or w.dtype not in (torch.bfloat16, torch.float32):
        raise ValueError("Batch-invariant router needs BF16 input and BF16/FP32 weight")
    m, k = x.shape
    n = w.shape[0]
    if k > 8192:
        raise ValueError("Batch-invariant router supports hidden dimensions up to 8192")
    output = torch.empty((m, n), dtype=torch.float32, device=x.device)
    if not m or not n:
        return output
    if not k:
        return output.zero_()
    with torch.cuda.device(x.device):
        _router_kernel[(m, triton.cdiv(n, 4))](
            x,
            w,
            output,
            m,
            n,
            k,
            x.stride(0),
            x.stride(1),
            w.stride(0),
            w.stride(1),
            output.stride(0),
            EXPERTS=4,
            PADDED_K=triton.next_power_of_2(k),
            num_warps=4,
            num_stages=1,
            enable_fp_fusion=False,
        )
    return output
