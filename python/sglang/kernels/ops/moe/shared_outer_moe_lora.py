"""Decode-sized MoE-LoRA delta for experts with shared outer LoRA factors.

With ``experts_shared_outer_loras`` the gate/up projections share one LoRA A
across all experts and the down projection shares one LoRA B. The generic
per-expert path (``fused_moe_lora``) expands the shared factor to every expert
and recomputes the same rank-r shrink for each routed (token, expert) block with
64-row tiles, which at decode (a few hundred token-expert pairs) is almost
entirely padding. Here the shared shrink runs once per token, and the per-expert
factor is applied per (token, expert) pair:

    gate/up:  s_m   = bf16(A_gu @ x_m)                  (once per token, rank 2r)
              out[m,k,:N]       += bf16(B_gate[e_mk] @ s_m[:r])
              out[m,k,Np:Np+N]  += bf16(B_up[e_mk]   @ s_m[r:])
    down:     t_mk  = bf16(A_down[e_mk] @ inter_mk[:N])   (per pair, rank r)
              out[m,k,:]        += bf16(w_mk * (B_down @ t_mk))

The rounding points (bf16 shrink output, fp32 expand accumulation, bf16 delta,
bf16 add) are the stock path's. Tokens whose adapter id is -1 or whose adapter
has no MoE weights are skipped; negative expert ids are skipped.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _so_token_shrink_kernel(
    x_ptr,
    a_ptr,
    s_ptr,
    map_ptr,
    en_ptr,
    H,
    stride_xm,
    stride_al,
    stride_ar,
    stride_ah,
    stride_sm,
    stride_ss,
    R2: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """fp32 shrink partial s_part[m, split, :] over one BLOCK_H slice of H.
    Split-H keeps every SM busy at decode; the consumer sums the partials in a
    fixed order (no atomics), so the result is run-to-run deterministic."""
    m = tl.program_id(0)
    sp = tl.program_id(1)
    offs_r = tl.arange(0, R2)
    lora = tl.load(map_ptr + m)
    if lora < 0:
        return
    if tl.load(en_ptr + lora) == 0:
        return
    offs_h = sp * BLOCK_H + tl.arange(0, BLOCK_H)
    mh = offs_h < H
    xv = tl.load(x_ptr + m * stride_xm + offs_h, mask=mh, other=0.0).to(tl.float32)
    av = tl.load(
        a_ptr
        + lora * stride_al
        + offs_r[:, None] * stride_ar
        + offs_h[None, :] * stride_ah,
        mask=mh[None, :],
        other=0.0,
    ).to(tl.float32)
    tl.store(
        s_ptr + m * stride_sm + sp * stride_ss + offs_r,
        tl.sum(av * xv[None, :], axis=1),
    )


@triton.jit
def _so_gate_up_expand_kernel(
    out_ptr,
    s_ptr,
    b_ptr,
    topk_ids_ptr,
    map_ptr,
    en_ptr,
    N,
    NPAD,
    stride_om,
    stride_ok,
    stride_sm,
    stride_ss,
    stride_bl,
    stride_be,
    stride_bn,
    stride_br,
    TOPK: tl.constexpr,
    R: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NSPLIT: tl.constexpr,
):
    p = tl.program_id(0)
    nb = tl.program_id(1)
    m = p // TOPK
    k = p % TOPK
    lora = tl.load(map_ptr + m)
    if lora < 0:
        return
    if tl.load(en_ptr + lora) == 0:
        return
    e = tl.load(topk_ids_ptr + p)
    if e < 0:
        return
    offs_n = nb * BLOCK_N + tl.arange(0, BLOCK_N)
    mn = offs_n < N
    offs_r = tl.arange(0, R)
    sg = tl.zeros([R], dtype=tl.float32)
    su = tl.zeros([R], dtype=tl.float32)
    for sp in range(0, NSPLIT):
        sg += tl.load(s_ptr + m * stride_sm + sp * stride_ss + offs_r)
        su += tl.load(s_ptr + m * stride_sm + sp * stride_ss + R + offs_r)
    sg = sg.to(out_ptr.dtype.element_ty).to(tl.float32)
    su = su.to(out_ptr.dtype.element_ty).to(tl.float32)
    b_base = b_ptr + lora * stride_bl + e.to(tl.int64) * stride_be
    b_gate = tl.load(
        b_base + offs_n[:, None] * stride_bn + offs_r[None, :] * stride_br,
        mask=mn[:, None],
        other=0.0,
    )
    b_up = tl.load(
        b_base + (N + offs_n)[:, None] * stride_bn + offs_r[None, :] * stride_br,
        mask=mn[:, None],
        other=0.0,
    )
    dg = (
        tl.sum(b_gate.to(tl.float32) * sg[None, :], axis=1)
        .to(out_ptr.dtype.element_ty)
        .to(tl.float32)
    )
    du = (
        tl.sum(b_up.to(tl.float32) * su[None, :], axis=1)
        .to(out_ptr.dtype.element_ty)
        .to(tl.float32)
    )
    o = out_ptr + m * stride_om + k * stride_ok
    og = tl.load(o + offs_n, mask=mn, other=0.0).to(tl.float32)
    tl.store(o + offs_n, (og + dg).to(out_ptr.dtype.element_ty), mask=mn)
    ou = tl.load(o + NPAD + offs_n, mask=mn, other=0.0).to(tl.float32)
    tl.store(o + NPAD + offs_n, (ou + du).to(out_ptr.dtype.element_ty), mask=mn)


@triton.jit
def _so_pair_shrink_kernel(
    in_ptr,
    a_ptr,
    t_ptr,
    topk_ids_ptr,
    map_ptr,
    en_ptr,
    N,
    stride_ip,
    stride_al,
    stride_ae,
    stride_ar,
    stride_an,
    stride_tp,
    TOPK: tl.constexpr,
    R: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    p = tl.program_id(0)
    m = p // TOPK
    offs_r = tl.arange(0, R)
    lora = tl.load(map_ptr + m)
    if lora < 0:
        return
    if tl.load(en_ptr + lora) == 0:
        return
    e = tl.load(topk_ids_ptr + p)
    if e < 0:
        return
    a_base = a_ptr + lora * stride_al + e.to(tl.int64) * stride_ae
    acc = tl.zeros([R], dtype=tl.float32)
    for n0 in range(0, N, BLOCK_N):
        offs_n = n0 + tl.arange(0, BLOCK_N)
        mn = offs_n < N
        xv = tl.load(in_ptr + p * stride_ip + offs_n, mask=mn, other=0.0).to(tl.float32)
        av = tl.load(
            a_base + offs_r[:, None] * stride_ar + offs_n[None, :] * stride_an,
            mask=mn[None, :],
            other=0.0,
        ).to(tl.float32)
        acc += tl.sum(av * xv[None, :], axis=1)
    tl.store(t_ptr + p * stride_tp + offs_r, acc.to(t_ptr.dtype.element_ty))


@triton.jit
def _so_down_expand_kernel(
    out_ptr,
    t_ptr,
    b_ptr,
    w_ptr,
    topk_ids_ptr,
    map_ptr,
    en_ptr,
    H,
    stride_op,
    stride_tp,
    stride_bl,
    stride_bh,
    stride_br,
    TOPK: tl.constexpr,
    R: tl.constexpr,
    BLOCK_H: tl.constexpr,
    MUL_W: tl.constexpr,
):
    p = tl.program_id(0)
    hb = tl.program_id(1)
    m = p // TOPK
    lora = tl.load(map_ptr + m)
    if lora < 0:
        return
    if tl.load(en_ptr + lora) == 0:
        return
    e = tl.load(topk_ids_ptr + p)
    if e < 0:
        return
    offs_r = tl.arange(0, R)
    offs_h = hb * BLOCK_H + tl.arange(0, BLOCK_H)
    mh = offs_h < H
    t = tl.load(t_ptr + p * stride_tp + offs_r).to(tl.float32)
    b = tl.load(
        b_ptr
        + lora * stride_bl
        + offs_h[:, None] * stride_bh
        + offs_r[None, :] * stride_br,
        mask=mh[:, None],
        other=0.0,
    )
    d = tl.sum(b.to(tl.float32) * t[None, :], axis=1)
    if MUL_W:
        d = d * tl.load(w_ptr + p).to(tl.float32)
    d = d.to(out_ptr.dtype.element_ty).to(tl.float32)
    o = out_ptr + p * stride_op + offs_h
    ov = tl.load(o, mask=mh, other=0.0).to(tl.float32)
    tl.store(o, (ov + d).to(out_ptr.dtype.element_ty), mask=mh)


def shared_outer_gate_up_lora_add(
    out: torch.Tensor,  # [M, top_k, 2 * N_pad], in place
    x: torch.Tensor,  # [M, H]
    gate_up_a: torch.Tensor,  # [L, 1, 2r, H] (shared over experts)
    gate_up_b: torch.Tensor,  # [L, E, 2N, r]
    topk_ids: torch.Tensor,  # [M, top_k]
    token_lora_mapping: torch.Tensor,  # [>= M] int, -1 = no adapter
    adapter_enabled: torch.Tensor,  # [L] int
) -> None:
    M, top_k, two_npad = out.shape
    r2 = gate_up_a.shape[2]
    r = r2 // 2
    n = gate_up_b.shape[2] // 2
    npad = two_npad // 2
    H = x.shape[1]
    assert (
        gate_up_a.shape[1] == 1
        and r == triton.next_power_of_2(r)
        and gate_up_b.shape[3] == r
    )
    assert out.stride(2) == 1 and x.stride(1) == 1
    block_h = 256
    nsplit = triton.cdiv(H, block_h)
    # fp32 partials; rows of skipped tokens are never read
    s = torch.empty((M, nsplit, r2), dtype=torch.float32, device=out.device)
    _so_token_shrink_kernel[(M, nsplit)](
        x,
        gate_up_a,
        s,
        token_lora_mapping,
        adapter_enabled,
        H,
        x.stride(0),
        gate_up_a.stride(0),
        gate_up_a.stride(2),
        gate_up_a.stride(3),
        s.stride(0),
        s.stride(1),
        R2=r2,
        BLOCK_H=block_h,
        num_warps=4,
    )
    tids = topk_ids.reshape(-1)
    block_n = 64
    _so_gate_up_expand_kernel[(M * top_k, triton.cdiv(n, block_n))](
        out,
        s,
        gate_up_b,
        tids,
        token_lora_mapping,
        adapter_enabled,
        n,
        npad,
        out.stride(0),
        out.stride(1),
        s.stride(0),
        s.stride(1),
        gate_up_b.stride(0),
        gate_up_b.stride(1),
        gate_up_b.stride(2),
        gate_up_b.stride(3),
        TOPK=top_k,
        R=r,
        BLOCK_N=block_n,
        NSPLIT=nsplit,
        num_warps=2,
    )


def shared_outer_down_lora_add(
    out: torch.Tensor,  # [M, top_k, H], in place
    inter: torch.Tensor,  # [M * top_k, >= N] (Marlin activation, possibly padded)
    down_a: torch.Tensor,  # [L, E, r, N]
    down_b: torch.Tensor,  # [L, 1, H, r] (shared over experts)
    topk_ids: torch.Tensor,  # [M, top_k]
    topk_weights: torch.Tensor,  # [M, top_k]
    token_lora_mapping: torch.Tensor,
    adapter_enabled: torch.Tensor,
    mul_routed_weight: bool = True,
) -> None:
    M, top_k, H = out.shape
    r = down_a.shape[2]
    n = down_a.shape[3]
    assert (
        down_b.shape[1] == 1 and down_b.shape[3] == r and r == triton.next_power_of_2(r)
    )
    assert out.stride(2) == 1 and inter.stride(1) == 1 and topk_weights.is_contiguous()
    P = M * top_k
    t = torch.empty(
        (P, r), dtype=out.dtype, device=out.device
    )  # rows of skipped pairs are never read
    tids = topk_ids.reshape(-1)
    _so_pair_shrink_kernel[(P,)](
        inter,
        down_a,
        t,
        tids,
        token_lora_mapping,
        adapter_enabled,
        n,
        inter.stride(0),
        down_a.stride(0),
        down_a.stride(1),
        down_a.stride(2),
        down_a.stride(3),
        t.stride(0),
        TOPK=top_k,
        R=r,
        BLOCK_N=128,
        num_warps=2,
    )
    out2 = out.view(P, H)
    block_h = 256
    _so_down_expand_kernel[(P, triton.cdiv(H, block_h))](
        out2,
        t,
        down_b,
        topk_weights.reshape(-1),
        tids,
        token_lora_mapping,
        adapter_enabled,
        H,
        out2.stride(0),
        t.stride(0),
        down_b.stride(0),
        down_b.stride(2),
        down_b.stride(3),
        TOPK=top_k,
        R=r,
        BLOCK_H=block_h,
        MUL_W=mul_routed_weight,
        num_warps=4,
    )
