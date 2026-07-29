# SPDX-License-Identifier: Apache-2.0
# Fused Triton prologue for the KDA Blackwell pipeline.
#
# In ONE pass per (chunk, head) it computes the per-chunk cumsum g_cu and the five
# pre-scaled key/query tensors the cutedsl kernels consume, replacing ~30 separate
# PyTorch elementwise ops + copies:
#
#   g_cu      = cumsum_within_chunk(g)            [T, Hv, K]  (fp32, for kernel_h decay)
#   g_last[d] = g_cu at the chunk's last token    (= total sum over the chunk)
#   kL  = k * exp(g_cu - g_last)    (kkt KKT-left)
#   kR  = k * exp(g_last - g_cu)    (kkt KKT-right == kernel_h kg == kernel_o Aqk-K)
#   kgw = k * exp(g_cu)             (kkt W operand)
#   qg  = scale * q * exp(g_cu)             (kernel_o Q@H)
#   qg2 = scale * q * exp(g_cu - g_last)    (kernel_o Aqk-Q)
import torch
import triton
import triton.language as tl


@triton.jit
def _kda_prologue_kernel(
    q_ptr,
    k_ptr,
    g_ptr,
    kL_ptr,
    kR_ptr,
    kgw_ptr,
    qg_ptr,
    qg2_ptr,
    gcu_ptr,
    cu_seqlens_ptr,
    chunk_indices_ptr,
    scale,
    Hv: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
):
    chunk = tl.program_id(0)
    head = tl.program_id(1)

    seq_id = tl.load(chunk_indices_ptr + chunk * 2 + 0)
    chunk_id = tl.load(chunk_indices_ptr + chunk * 2 + 1)
    bos = tl.load(cu_seqlens_ptr + seq_id)
    eos = tl.load(cu_seqlens_ptr + seq_id + 1)
    off_t = bos + chunk_id * BT

    row = off_t + tl.arange(0, BT)
    col = tl.arange(0, K)
    mask_row = row < eos
    offs = row[:, None] * (Hv * K) + head * K + col[None, :]
    mask = mask_row[:, None]

    g = tl.load(g_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    q = tl.load(q_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    k = tl.load(k_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    g_cu = tl.cumsum(g, axis=0)  # [BT, K]
    g_last = tl.sum(g, axis=0)  # [K]  (OOB rows contributed 0)
    gml = g_cu - g_last[None, :]  # g_cu - g_last  (>= 0, since g_cu>=g_last)
    e_gcu = tl.exp(g_cu)  # <= 1
    e_gml = tl.exp(gml)  # >= 1 (kL side; huge entries get masked)
    e_lmg = tl.exp(-gml)  # <= 1 (bounded: kR / kg)

    tl.store(gcu_ptr + offs, g_cu, mask=mask)
    tl.store(kL_ptr + offs, (k * e_gml).to(kL_ptr.dtype.element_ty), mask=mask)
    tl.store(kR_ptr + offs, (k * e_lmg).to(kR_ptr.dtype.element_ty), mask=mask)
    tl.store(kgw_ptr + offs, (k * e_gcu).to(kgw_ptr.dtype.element_ty), mask=mask)
    tl.store(qg_ptr + offs, (scale * q * e_gcu).to(qg_ptr.dtype.element_ty), mask=mask)
    tl.store(
        qg2_ptr + offs, (scale * q * e_gml).to(qg2_ptr.dtype.element_ty), mask=mask
    )


def kda_prologue(q, k, g_act, scale, cu_seqlens, chunk_indices, num_chunks):
    """q/k/g_act: [T, Hv, K]. Returns (kL, kR, kgw, qg, qg2) bf16 + g_cu fp32."""
    T, Hv, K = q.shape
    kL = torch.empty_like(q, dtype=torch.bfloat16)
    kR = torch.empty_like(q, dtype=torch.bfloat16)
    kgw = torch.empty_like(q, dtype=torch.bfloat16)
    qg = torch.empty_like(q, dtype=torch.bfloat16)
    qg2 = torch.empty_like(q, dtype=torch.bfloat16)
    g_cu = torch.empty_like(q, dtype=torch.float32)
    grid = (num_chunks, Hv)
    _kda_prologue_kernel[grid](
        q,
        k,
        g_act,
        kL,
        kR,
        kgw,
        qg,
        qg2,
        g_cu,
        cu_seqlens,
        chunk_indices,
        scale,
        Hv=Hv,
        K=K,
        BT=64,
        num_warps=8,
    )
    return kL, kR, kgw, qg, qg2, g_cu
