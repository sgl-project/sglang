"""HC mix up projection fused with its gate for decode-size row counts (hc = 4).

``out[m, j] = bf16(sum_g sigmoid(bf16(logit[m, j*4 + g])) * x[m, g*hs + j] / 4)`` with
``logit = t @ w_up.T`` accumulated in FP32 over K in ascending order. The logits never reach
memory; each is rounded to bf16 before the sigmoid, as the torch.compile chain's separate up
GEMM output is, and the products are summed left to right from g = 0, as that chain's compiled
mean unrolls. Given the same ``t`` the result matches that chain bit for bit on VR200.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _gate_term(acc, x_ptr, offs_m, offs_j, mask_m, stride_x, g: tl.constexpr, HS: tl.constexpr):
    logit = acc.to(x_ptr.dtype.element_ty).to(tl.float32)
    xg = tl.load(
        x_ptr + offs_m[:, None] * stride_x + (g * HS + offs_j)[None, :],
        mask=mask_m[:, None],
        other=0.0,
    ).to(tl.float32)
    return tl.sigmoid(logit) * xg


@triton.jit
def _hc_up_gate_kernel(
    t_ptr,
    w_ptr,
    x_ptr,
    out_ptr,
    M,
    stride_t,
    stride_w,
    stride_x,
    stride_out,
    HS: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_J: tl.constexpr,
    BLOCK_K: tl.constexpr,
    USE_PDL: tl.constexpr,
):
    # w is hc_mix_splitk.permute_pad_up_weight's layout: row j*4 + g, so one tile of 4*BLOCK_J rows.
    pid_j = tl.program_id(0)
    pid_m = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_j = pid_j * BLOCK_J + tl.arange(0, BLOCK_J)
    offs_n = pid_j * (4 * BLOCK_J) + tl.arange(0, 4 * BLOCK_J)
    offs_k = tl.arange(0, BLOCK_K)
    mask_m = offs_m < M
    if USE_PDL:
        # The weight does not depend on the previous kernel: pull this CTA's rows into L2
        # (one prefetch per 128-byte line) before waiting for t and x.
        lines = tl.minimum(tl.arange(0, 8), (K * 2 - 1) // 128) * 64
        tl.inline_asm_elementwise(
            "prefetch.global.L2 [$1];\n mov.u32 $0, 0;",
            "=r,l",
            [w_ptr + offs_n[:, None] * stride_w + lines[None, :]],
            dtype=tl.int32,
            is_pure=False,
            pack=1,
        )
        tl.extra.cuda.gdc_wait()
    acc = tl.zeros((BLOCK_M, 4 * BLOCK_J), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        a = tl.load(
            t_ptr + offs_m[:, None] * stride_t + (k0 + offs_k)[None, :],
            mask=mask_m[:, None],
            other=0.0,
        )
        b = tl.load(w_ptr + offs_n[:, None] * stride_w + (k0 + offs_k)[None, :])
        acc = tl.dot(a, tl.trans(b), acc)
    # Column jl*4 + g -> [BM, BJ, 2 (g_hi), 2 (g_lo)] with g = 2*g_hi + g_lo.
    acc_lo0, acc_lo1 = tl.split(tl.reshape(acc, (BLOCK_M, BLOCK_J, 2, 2)))
    acc_g0, acc_g2 = tl.split(acc_lo0)
    acc_g1, acc_g3 = tl.split(acc_lo1)
    total = _gate_term(acc_g0, x_ptr, offs_m, offs_j, mask_m, stride_x, 0, HS)
    total = total + _gate_term(acc_g1, x_ptr, offs_m, offs_j, mask_m, stride_x, 1, HS)
    total = total + _gate_term(acc_g2, x_ptr, offs_m, offs_j, mask_m, stride_x, 2, HS)
    total = total + _gate_term(acc_g3, x_ptr, offs_m, offs_j, mask_m, stride_x, 3, HS)
    tl.store(
        out_ptr + offs_m[:, None] * stride_out + offs_j[None, :],
        (total / 4).to(out_ptr.dtype.element_ty),
        mask=mask_m[:, None],
    )


def hc_up_gate(
    t: torch.Tensor,
    w_up_permuted: torch.Tensor,
    x: torch.Tensor,
    hs: int,
    k: int,
    *,
    config: tuple[int, int, int, int, int],
    pdl: bool,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """``t`` [rows, >= k] (silu'd low-rank activations), ``w_up_permuted`` [4 * hs, >= k] with rows j*4 + g,
    ``x`` [rows, 4 * hs]; the first ``k`` columns of ``t`` and the weight are the reduction.
    ``config`` = (BLOCK_M, BLOCK_J, BLOCK_K, num_warps, num_stages); hs % BLOCK_J == 0, k % BLOCK_K == 0."""
    block_m, block_j, block_k, num_warps, num_stages = config
    rows = x.shape[0]
    if (
        hs % block_j
        or k % block_k
        or t.stride(1) != 1
        or x.stride(1) != 1
        or w_up_permuted.stride(1) != 1
    ):
        raise ValueError(f"hc_up_gate: unsupported layout or config {config} for hs={hs}, k={k}")
    if out is None:
        out = torch.empty((rows, hs), dtype=x.dtype, device=x.device)
    if rows == 0:
        return out
    _hc_up_gate_kernel[(hs // block_j, triton.cdiv(rows, block_m))](
        t,
        w_up_permuted,
        x,
        out,
        rows,
        t.stride(0),
        w_up_permuted.stride(0),
        x.stride(0),
        out.stride(0),
        HS=hs,
        K=k,
        BLOCK_M=block_m,
        BLOCK_J=block_j,
        BLOCK_K=block_k,
        USE_PDL=pdl,
        num_warps=num_warps,
        num_stages=num_stages,
        **({"launch_pdl": True} if pdl else {}),
    )
    return out
