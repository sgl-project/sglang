from __future__ import annotations
import math
import torch
import triton
import triton.language as tl

@triton.jit
def _attn_fwd_kernel(Q, K, V, O,
                     B, H, M, N, D,
                     stride_qb, stride_qh, stride_qm, stride_qd,
                     stride_kb, stride_kh, stride_kn, stride_kd,
                     stride_vb, stride_vh, stride_vn, stride_vd,
                     stride_ob, stride_oh, stride_om, stride_od,
                     BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
                     SCALE):
    bh = tl.program_id(0)
    m0 = tl.program_id(1) * BLOCK_M
    offs_m = m0 + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    b = bh // H
    h = bh % H
    Q_ptrs = Q + b*stride_qb + h*stride_qh + offs_m[:, None]*stride_qm + offs_d[None, :]*stride_qd
    K_ptrs = K + b*stride_kb + h*stride_kh + offs_n[:, None]*stride_kn + offs_d[None, :]*stride_kd
    V_ptrs = V + b*stride_vb + h*stride_vh + offs_n[:, None]*stride_vn + offs_d[None, :]*stride_vd
    O_ptrs = O + b*stride_ob + h*stride_oh + offs_m[:, None]*stride_om + offs_d[None, :]*stride_od
    m_i = tl.full([BLOCK_M], -float('inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    for n0 in range(0, N, BLOCK_N):
        k = tl.load(K_ptrs + n0*stride_kn, mask=(offs_n[None, :] + n0 < N) & (offs_d[:, None] < D), other=0.0)
        v = tl.load(V_ptrs + n0*stride_vn, mask=(offs_n[None, :] + n0 < N) & (offs_d[:, None] < D), other=0.0)
        q = tl.load(Q_ptrs, mask=(offs_m[:, None] < M) & (offs_d[None, :] < D), other=0.0)
        qk = tl.dot(q, tl.trans(k)) * SCALE
        m_ij = tl.maximum(m_i[:, None], tl.max(qk, axis=1))
        p = tl.exp(qk - m_ij)
        l_ij = l_i[:, None] * tl.exp(m_i[:, None] - m_ij) + tl.sum(p, axis=1)
        acc = acc * (l_i / l_ij)[:, None] * tl.exp(m_i - m_ij)[:, None] + tl.dot(p, v)
        l_i = l_ij; m_i = m_ij
    o = acc / l_i[:, None]
    tl.store(O_ptrs, o, mask=(offs_m[:, None] < M) & (offs_d[None, :] < D))

def _fused_attn(Q_bhmd, K_bhnd, V_bhnd):
    B, H, M, D = Q_bhmd.shape
    _, _, N, _ = K_bhnd.shape
    O = torch.empty_like(Q_bhmd, dtype=torch.float32)
    BLOCK_M, BLOCK_N, BLOCK_D = 64, 128, triton.next_power_of_2(D)
    grid = (B*H, (M + BLOCK_M - 1)//BLOCK_M)
    scale = 1.0 / math.sqrt(D)
    _attn_fwd_kernel[grid](
        Q_bhmd, K_bhnd, V_bhnd, O,
        B, H, M, N, D,
        Q_bhmd.stride(0), Q_bhmd.stride(1), Q_bhmd.stride(2), Q_bhmd.stride(3),
        K_bhnd.stride(0), K_bhnd.stride(1), K_bhnd.stride(2), K_bhnd.stride(3),
        V_bhnd.stride(0), V_bhnd.stride(1), V_bhnd.stride(2), V_bhnd.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_DMODEL=BLOCK_D,
        SCALE=scale,
    )
    return O

@torch.no_grad()
def sparse_attn_stage2(q_BSHD: torch.Tensor,
                       k_BSHD: torch.Tensor,
                       v_BSHD: torch.Tensor,
                       topk_idx_BSHK: torch.Tensor,
                       block_size: int = 64,
                       sw_span: int | None = None,
                       sink_len: int | None = None) -> torch.Tensor:
    B, Sq, Hq, D = q_BSHD.shape
    Sk = k_BSHD.shape[1]
    Hk = k_BSHD.shape[2]
    TopK = topk_idx_BSHK.shape[-1]
    assert Hq % Hk == 0
    hg = Hq // Hk
    out = q_BSHD.new_empty((B, Sq, Hq, D), dtype=torch.float32)

    n_blocks = (Sk + block_size - 1) // block_size
    sink_blocks = 0
    if sink_len is not None and sink_len > 0:
        sink_blocks = min(n_blocks, (sink_len + block_size - 1) // block_size)
    sw_begin_block = n_blocks
    if sw_span is not None and sw_span > 0:
        sw_begin_tok = max(0, Sk - sw_span)
        sw_begin_block = min(n_blocks - 1, max(0, sw_begin_tok // block_size))

    for b in range(B):
        for hk in range(Hk):
            q_hg = q_BSHD[b, :, hk*hg:(hk+1)*hg, :]
            for sq in range(Sq):
                blocks = topk_idx_BSHK[b, sq, hk].tolist() if TopK > 0 else []
                if sink_blocks > 0:
                    blocks.extend(list(range(0, sink_blocks)))
                if sw_begin_block < n_blocks:
                    blocks.extend(list(range(sw_begin_block, n_blocks)))
                if not blocks:
                    out[b, sq, hk*hg:(hk+1)*hg, :] = 0
                    continue
                uniq = torch.tensor(sorted(set(blocks)), device=q_BSHD.device, dtype=torch.long)
                idx_list = []
                for t in uniq.tolist():
                    s = t * block_size; e = min(s + block_size, Sk)
                    if s < e:
                        idx_list.append(torch.arange(s, e, device=q_BSHD.device))
                tok_idx = torch.cat(idx_list, dim=0) if len(idx_list) else torch.empty(0, device=q_BSHD.device, dtype=torch.long)
                if tok_idx.numel() == 0:
                    out[b, sq, hk*hg:(hk+1)*hg, :] = 0
                    continue
                Ksel = k_BSHD[b, tok_idx, hk, :].contiguous().unsqueeze(0).unsqueeze(0)
                Vsel = v_BSHD[b, tok_idx, hk, :].contiguous().unsqueeze(0).unsqueeze(0)
                Q = q_hg[sq].unsqueeze(0).permute(1,0,2).unsqueeze(0)
                O = _fused_attn(Q, Ksel.expand(1, hg, -1, -1), Vsel.expand(1, hg, -1, -1))
                out[b, sq, hk*hg:(hk+1)*hg, :] = O[0, :, 0, :]
    return out.to(dtype=q_BSHD.dtype)