from __future__ import annotations
import math
import torch
import triton
import triton.language as tl

@triton.jit
def _qk_matmul_kernel(Q_ptr, C_ptr, O_ptr,
                      M, N, K,
                      stride_qm, stride_qk,
                      stride_cn, stride_ck,
                      stride_om, stride_on,
                      BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                      SCALE):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    q_ptrs = Q_ptr + (offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk)
    c_ptrs = C_ptr + (offs_n[:, None] * stride_cn + offs_k[None, :] * stride_ck)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        q = tl.load(q_ptrs + k * stride_qk, mask=(offs_m[:, None] < M) & (offs_k[None, :] + k < K), other=0.0)
        c = tl.load(c_ptrs + k * stride_ck, mask=(offs_n[:, None] < N) & (offs_k[None, :] + k < K), other=0.0)
        acc += tl.dot(q, tl.trans(c))
    acc *= SCALE
    tl.store(O_ptr + (offs_m[:, None] * stride_om + offs_n[None, :] * stride_on), acc,
             mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

def _matmul_q_c(q_S_D, c_Sc_D, scale):
    Sq, D = q_S_D.shape
    Sc, _ = c_Sc_D.shape
    BLOCK_M, BLOCK_N, BLOCK_K = 64, 128, 64
    out = q_S_D.new_empty((Sq, Sc), dtype=torch.float32)
    grid = (triton.cdiv(Sq, BLOCK_M), triton.cdiv(Sc, BLOCK_N))
    _qk_matmul_kernel[grid](
        q_S_D, c_Sc_D, out,
        Sq, Sc, D,
        q_S_D.stride(0), q_S_D.stride(1),
        c_Sc_D.stride(0), c_Sc_D.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        SCALE=scale,
    )
    return out

@torch.no_grad()
def stage1_scores_bshd(q_BSHD: torch.Tensor,
                       c1_BSHD: torch.Tensor,
                       c2_BSHD: torch.Tensor,
                       hg: int,
                       valid_sc1_len: torch.Tensor | None = None) -> torch.Tensor:
    """Compute approx scores = mean_h softmax(q_h @ c1^T / sqrt(D)), using c2 for LSE.
    Args:
      q_BSHD:  [B, Sq, Hq, D]
      c1_BSHD: [B, Sc1, Hk, D]
      c2_BSHD: [B, Sc2, Hk, D]
      hg:      heads per group (= Hq/Hk)
      valid_sc1_len: optional [B,Hk] valid length for c1 blocks
    Returns: scores [B, Sq, Hk, Sc1] (float32)
    """
    B, Sq, Hq, D = q_BSHD.shape
    _, Sc1, Hk, _ = c1_BSHD.shape
    _, Sc2, Hk2, _ = c2_BSHD.shape
    assert Hk == Hk2 and Hq % hg == 0 and (Hq // hg) == Hk
    scale = 1.0 / math.sqrt(D)
    out = q_BSHD.new_zeros((B, Sq, Hk, Sc1), dtype=torch.float32)

    for b in range(B):
        for hk in range(Hk):
            c1 = c1_BSHD[b, :, hk, :].contiguous()   # [Sc1,D]
            c2 = c2_BSHD[b, :, hk, :].contiguous()   # [Sc2,D]
            acc = None
            for g in range(hg):
                hq = hk * hg + g
                q = q_BSHD[b, :, hq, :].contiguous()  # [Sq,D]
                logits2 = (q @ c2.transpose(0,1)) * scale
                lse2 = torch.logsumexp(logits2.to(torch.float32), dim=-1)  # [Sq]
                logits1 = _matmul_q_c(q, c1, scale)  # [Sq,Sc1] float32
                probs   = torch.exp(logits1 - lse2.unsqueeze(-1))
                if valid_sc1_len is not None:
                    L = int(valid_sc1_len[b, hk].item())
                    if L < Sc1:
                        probs[:, L:] = 0.0
                acc = probs if acc is None else (acc + probs)
            out[b, :, hk, :] = acc / hg
    return out