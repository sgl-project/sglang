# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
# Vendored from the NVIDIA KDA_prefill package (benchmark/ Blackwell path)
# for the Kimi-K3 chunked prefill forward. Local deltas: fla.* imports
# re-pointed to sglang's vendored fla subset, flat sibling imports made
# package-relative, RCP_LN2 inlined. INTERNAL COLLABORATION ONLY.
# ruff: noqa  -- vendored kernel library, minimal local deltas

"""
cuTile Kernel 4: Fused KDA forward kernel.

Equal-length: (B, num_chunks, BT, H, d) tensors, Grid=(B*H,)

Varlen v1 (legacy): per-seq tensor views as lists + runtime list[seq_id].
        Partial chunks require clone+zero in launcher.

Varlen v2 (zero-copy): Array.slice() on flat packed tensors.
        Each block slices its sequence range from the flat tensor — zero copy.
        Partial chunks: ct.load with ZERO padding auto-zeros OOB rows.
        ct.store silently drops OOB writes. No clone, no alloc, no lists.
        Grid=(N_seqs*H,), single launch, zero data movement.
"""

import math

import cuda.tile as ct
import torch

ConstInt = ct.Constant[int]


@ct.kernel
def fused_kda_kernel_v3(
    S_in,
    S_out,
    A_qk,
    A_kk,
    K_scaled,
    V,
    Kg,
    Q_scaled,
    Beta,
    gk_last_exp,
    O,
    d: ConstInt,
    chunk_size: ConstInt,
    H: ConstInt,
):
    """Equal-length kernel. Grid: (B * H,).

    S_in:  (B, H, d, d) fp32, read-only initial state in (K, V) layout.
    S_out: (B, H, d, d) fp32, write-only final state in (K, V) layout.
    Internal state s is (K, V) — used directly as right MMA operand,
    eliminating per-iteration transpose.
    """
    num_chunks = A_qk.shape[1]
    bid = ct.bid(0)
    b = bid // H
    h = bid % H
    zp = ct.PaddingMode.ZERO

    s = ct.astype(
        ct.reshape(ct.load(S_in, (b, h, 0, 0), (1, 1, d, d), padding_mode=zp), (d, d)),
        ct.float32,
    )
    causal = ct.where(
        ct.arange(chunk_size, dtype=ct.int32)[:, None]
        >= ct.arange(chunk_size, dtype=ct.int32)[None, :],
        1.0,
        0.0,
    )

    for c in range(num_chunks):
        aq = ct.reshape(
            ct.load(
                A_qk,
                (b, c, 0, h, 0),
                (1, 1, chunk_size, 1, chunk_size),
                padding_mode=zp,
            ),
            (chunk_size, chunk_size),
        )
        ak = ct.reshape(
            ct.load(
                A_kk,
                (b, c, 0, h, 0),
                (1, 1, chunk_size, 1, chunk_size),
                padding_mode=zp,
            ),
            (chunk_size, chunk_size),
        )
        ks = ct.reshape(
            ct.load(
                K_scaled, (b, c, 0, h, 0), (1, 1, chunk_size, 1, d), padding_mode=zp
            ),
            (chunk_size, d),
        )
        vr = ct.reshape(
            ct.load(V, (b, c, 0, h, 0), (1, 1, chunk_size, 1, d), padding_mode=zp),
            (chunk_size, d),
        )
        kg = ct.reshape(
            ct.load(Kg, (b, c, 0, h, 0), (1, 1, chunk_size, 1, d), padding_mode=zp),
            (chunk_size, d),
        )
        qs = ct.reshape(
            ct.load(
                Q_scaled, (b, c, 0, h, 0), (1, 1, chunk_size, 1, d), padding_mode=zp
            ),
            (chunk_size, d),
        )
        be = ct.reshape(
            ct.load(Beta, (b, c, 0, h), (1, 1, chunk_size, 1), padding_mode=zp),
            (1, chunk_size),
        )
        gk = ct.reshape(
            ct.load(gk_last_exp, (b, c, h, 0), (1, 1, 1, d), padding_mode=zp), (1, d)
        )

        ab = ct.astype(ak * be, K_scaled.dtype)
        aq_c = ct.astype(ct.where(causal > 0.0, aq, 0.0), K_scaled.dtype)
        sb = ct.astype(s, K_scaled.dtype)
        w = ct.mma(ab, ks, ct.full((chunk_size, d), 0.0, dtype=ct.float32))
        u = ct.mma(ab, vr, ct.full((chunk_size, d), 0.0, dtype=ct.float32))
        ws = ct.mma(
            ct.astype(w, K_scaled.dtype),
            sb,
            ct.full((chunk_size, d), 0.0, dtype=ct.float32),
        )
        nv = ct.astype(u - ws, K_scaled.dtype)
        oi = ct.mma(qs, sb, ct.full((chunk_size, d), 0.0, dtype=ct.float32))
        oa = ct.mma(aq_c, nv, ct.full((chunk_size, d), 0.0, dtype=ct.float32))
        ct.store(
            O,
            (b, c, 0, h, 0),
            ct.reshape(ct.astype(oi + oa, O.dtype), (1, 1, chunk_size, 1, d)),
        )
        s = s * ct.transpose(gk) + ct.mma(
            ct.transpose(kg), nv, ct.full((d, d), 0.0, dtype=ct.float32)
        )

    ct.store(S_out, (b, h, 0, 0), ct.astype(ct.reshape(s, (1, 1, d, d)), S_out.dtype))


@ct.kernel
def fused_kda_kernel_varlen(
    S,  # (N_seqs, H, d, d)
    A_qk_list,
    A_kk_list,
    K_list,
    V_list,
    Kg_list,
    Q_list,
    Beta_list,
    gk_list,
    O_list,
    d: ConstInt,
    chunk_size: ConstInt,
    H: ConstInt,
):
    """
    Varlen kernel — runtime list[seq_id], partial chunks handled in-kernel.
    Grid: (N_seqs * H,). Zero copy, zero code bloat, single launch.
    Partial chunk rows are masked inside the kernel — no launcher clone needed.
    """
    bid = ct.bid(0)
    seq = bid // H
    h = bid % H
    zp = ct.PaddingMode.ZERO

    s = ct.astype(
        ct.reshape(ct.load(S, (seq, h, 0, 0), (1, 1, d, d), padding_mode=zp), (d, d)),
        ct.float32,
    )
    causal = ct.where(
        ct.arange(chunk_size, dtype=ct.int32)[:, None]
        >= ct.arange(chunk_size, dtype=ct.int32)[None, :],
        1.0,
        0.0,
    )

    # Runtime list index — single pointer lookup, no code bloat
    Aq = A_qk_list[seq]
    Ak = A_kk_list[seq]
    K = K_list[seq]
    V = V_list[seq]
    Kg_t = Kg_list[seq]
    Q = Q_list[seq]
    Be = Beta_list[seq]
    Gk = gk_list[seq]
    Og = O_list[seq]

    num_chunks = Aq.shape[0]

    for c in range(num_chunks):
        aq = ct.reshape(
            ct.load(Aq, (c, 0, h, 0), (1, chunk_size, 1, chunk_size), padding_mode=zp),
            (chunk_size, chunk_size),
        )
        ak = ct.reshape(
            ct.load(Ak, (c, 0, h, 0), (1, chunk_size, 1, chunk_size), padding_mode=zp),
            (chunk_size, chunk_size),
        )
        ks = ct.reshape(
            ct.load(K, (c, 0, h, 0), (1, chunk_size, 1, d), padding_mode=zp),
            (chunk_size, d),
        )
        vr = ct.reshape(
            ct.load(V, (c, 0, h, 0), (1, chunk_size, 1, d), padding_mode=zp),
            (chunk_size, d),
        )
        kg = ct.reshape(
            ct.load(Kg_t, (c, 0, h, 0), (1, chunk_size, 1, d), padding_mode=zp),
            (chunk_size, d),
        )
        qs = ct.reshape(
            ct.load(Q, (c, 0, h, 0), (1, chunk_size, 1, d), padding_mode=zp),
            (chunk_size, d),
        )
        be = ct.reshape(
            ct.load(Be, (c, 0, h), (1, chunk_size, 1), padding_mode=zp), (1, chunk_size)
        )
        gk_c = ct.reshape(ct.load(Gk, (c, h, 0), (1, 1, d), padding_mode=zp), (1, d))

        # Row mask: for partial last chunk, zero invalid rows of K1 outputs & output.
        # For full chunks: valid_rows = chunk_size → mask all 1s → no-op multiply.
        ab = ct.astype(ak * be, K.dtype)
        aq_c = ct.astype(ct.where(causal > 0.0, aq, 0.0), K.dtype)
        sb = ct.astype(s, K.dtype)
        st = ct.transpose(sb)
        w = ct.mma(ab, ks, ct.full((chunk_size, d), 0.0, dtype=ct.float32))
        u = ct.mma(ab, vr, ct.full((chunk_size, d), 0.0, dtype=ct.float32))
        ws = ct.mma(
            ct.astype(w, K.dtype), st, ct.full((chunk_size, d), 0.0, dtype=ct.float32)
        )
        nv = ct.astype(u - ws, K.dtype)
        oi = ct.mma(qs, st, ct.full((chunk_size, d), 0.0, dtype=ct.float32))
        oa = ct.mma(aq_c, nv, ct.full((chunk_size, d), 0.0, dtype=ct.float32))
        o_out = ct.astype(oi + oa, Og.dtype)
        ct.store(Og, (c, 0, h, 0), ct.reshape(o_out, (1, chunk_size, 1, d)))
        s = s * gk_c + ct.mma(
            ct.transpose(nv), kg, ct.full((d, d), 0.0, dtype=ct.float32)
        )

    ct.store(S, (seq, h, 0, 0), ct.astype(ct.reshape(s, (1, 1, d, d)), S.dtype))


@ct.kernel
def fused_kda_kernel_varlen_v2(
    S_in,  # (N_seqs, H, d, d) — initial state, read-only, fp32, (K,V) layout
    S_out,  # (N_seqs, H, d, d) — final state, write-only, fp32, (K,V) layout
    A_qk_flat,  # (1, T, H, BT) — packed flat tensor, bf16
    A_kk_flat,  # (1, T, H, BT) — packed flat tensor, bf16
    K_flat,  # (1, T, H, d)  — packed flat tensor, bf16
    V_flat,  # (1, T, H, d)  — packed flat tensor, bf16
    Kg_flat,  # (1, T, H, d)  — packed flat tensor, bf16
    Q_flat,  # (1, T, H, d)  — packed flat tensor, bf16
    Beta_flat,  # (1, T, H)     — packed flat tensor, bf16
    gk_flat,  # (1, total_chunks, H, d) — per-chunk gate, fp32
    O_flat,  # (1, T, H, d)  — output, bf16
    cu_seqlens,  # (N_seqs + 1,) — int32, cumulative sequence lengths
    chunk_offsets,  # (N_seqs + 1,) — int32, cumulative chunk counts
    d: ConstInt,
    chunk_size: ConstInt,
    H: ConstInt,
):
    """
    Varlen K4 v2 — zero copy via Array.slice() + OOB zero-padding.

    S_in/S_out in (K, V) layout — no launcher transpose needed.
    Grid: (N_seqs * H,). Single launch, zero data movement.
    """
    bid = ct.bid(0)
    seq = bid // H
    h = bid % H
    zp = ct.PaddingMode.ZERO

    bos = ct.astype(ct.load(cu_seqlens, index=seq, shape=()), ct.int32)
    eos = ct.astype(ct.load(cu_seqlens, index=seq + 1, shape=()), ct.int32)
    co = ct.astype(ct.load(chunk_offsets, index=seq, shape=()), ct.int32)

    seq_len = eos - bos
    num_chunks = ct.cdiv(seq_len, chunk_size)

    Aq = A_qk_flat.slice(axis=1, start=bos, stop=eos)
    Ak = A_kk_flat.slice(axis=1, start=bos, stop=eos)
    K = K_flat.slice(axis=1, start=bos, stop=eos)
    Vt = V_flat.slice(axis=1, start=bos, stop=eos)
    Kg = Kg_flat.slice(axis=1, start=bos, stop=eos)
    Q = Q_flat.slice(axis=1, start=bos, stop=eos)
    Be = Beta_flat.slice(axis=1, start=bos, stop=eos)
    Og = O_flat.slice(axis=1, start=bos, stop=eos)

    s = ct.astype(
        ct.reshape(
            ct.load(S_in, (seq, h, 0, 0), (1, 1, d, d), padding_mode=zp), (d, d)
        ),
        ct.float32,
    )

    causal = ct.where(
        ct.arange(chunk_size, dtype=ct.int32)[:, None]
        >= ct.arange(chunk_size, dtype=ct.int32)[None, :],
        1.0,
        0.0,
    )

    for c in range(num_chunks):
        aq = ct.reshape(
            ct.load(Aq, (0, c, h, 0), (1, chunk_size, 1, chunk_size), padding_mode=zp),
            (chunk_size, chunk_size),
        )
        ak = ct.reshape(
            ct.load(Ak, (0, c, h, 0), (1, chunk_size, 1, chunk_size), padding_mode=zp),
            (chunk_size, chunk_size),
        )
        ks = ct.reshape(
            ct.load(K, (0, c, h, 0), (1, chunk_size, 1, d), padding_mode=zp),
            (chunk_size, d),
        )
        vr = ct.reshape(
            ct.load(Vt, (0, c, h, 0), (1, chunk_size, 1, d), padding_mode=zp),
            (chunk_size, d),
        )
        kg = ct.reshape(
            ct.load(Kg, (0, c, h, 0), (1, chunk_size, 1, d), padding_mode=zp),
            (chunk_size, d),
        )
        qs = ct.reshape(
            ct.load(Q, (0, c, h, 0), (1, chunk_size, 1, d), padding_mode=zp),
            (chunk_size, d),
        )
        be = ct.reshape(
            ct.load(Be, (0, c, h), (1, chunk_size, 1), padding_mode=zp), (1, chunk_size)
        )
        gk_c = ct.reshape(
            ct.load(gk_flat, (0, co + c, h, 0), (1, 1, 1, d), padding_mode=zp), (1, d)
        )

        ab = ct.astype(ak * be, K_flat.dtype)
        aq_c = ct.astype(ct.where(causal > 0.0, aq, 0.0), K_flat.dtype)
        sb = ct.astype(s, K_flat.dtype)
        w = ct.mma(ab, ks, ct.full((chunk_size, d), 0.0, dtype=ct.float32))
        u = ct.mma(ab, vr, ct.full((chunk_size, d), 0.0, dtype=ct.float32))
        ws = ct.mma(
            ct.astype(w, K_flat.dtype),
            sb,
            ct.full((chunk_size, d), 0.0, dtype=ct.float32),
        )
        nv = ct.astype(u - ws, K_flat.dtype)
        oi = ct.mma(qs, sb, ct.full((chunk_size, d), 0.0, dtype=ct.float32))
        oa = ct.mma(aq_c, nv, ct.full((chunk_size, d), 0.0, dtype=ct.float32))
        o_out = ct.astype(oi + oa, O_flat.dtype)
        ct.store(Og, (0, c, h, 0), ct.reshape(o_out, (1, chunk_size, 1, d)))
        s = s * ct.transpose(gk_c) + ct.mma(
            ct.transpose(kg), nv, ct.full((d, d), 0.0, dtype=ct.float32)
        )

    ct.store(S_out, (seq, h, 0, 0), ct.astype(ct.reshape(s, (1, 1, d, d)), S_out.dtype))


def launch_kda_kernel(
    S_in, S_out, A_qk, A_kk, K_scaled, V, Kg, Q_scaled, Beta, gk_last_exp, chunk_size
):
    """Launch equal-length K4.

    S_in:  (B, H, K, V) fp32, read-only initial state (K, V) layout.
    S_out: (B, H, K, V) fp32, write-only final state (K, V) layout.
    No clone or transpose needed — kernel reads/writes (K, V) directly.
    """
    B, nc, cs, H, d = K_scaled.shape
    O = torch.empty(B, nc, cs, H, d, dtype=K_scaled.dtype, device=K_scaled.device)
    with torch.cuda.device(K_scaled.device):
        ct.launch(
            torch.cuda.current_stream(),
            (B * H,),
            fused_kda_kernel_v3,
            (
                S_in,
                S_out,
                A_qk,
                A_kk,
                K_scaled,
                V,
                Kg,
                Q_scaled,
                Beta,
                gk_last_exp,
                O,
                d,
                chunk_size,
                H,
            ),
        )
    return O


def launch_kda_kernel_varlen(
    S,
    A_qk_flat,
    A_kk_flat,
    K_scaled_flat,
    V_flat,
    Kg_flat,
    Q_scaled_flat,
    Beta_flat,
    gk_last_exp,
    O_flat,
    cu_seqlens,
    chunk_offsets,
    chunk_size=64,
):
    """Launch varlen K4 — zero overhead launcher.

    Pure views (no clone, no alloc). Partial chunks handled IN-KERNEL via row masking.
    """
    N = len(cu_seqlens) - 1
    device = S.device
    H = K_scaled_flat.shape[2]
    d = K_scaled_flat.shape[3]
    V_dim = V_flat.shape[3]
    BT = chunk_size

    cu_cpu = cu_seqlens.cpu().tolist()
    co_cpu = chunk_offsets.cpu().tolist()

    # Build per-seq views (zero copy for aligned, clone+zero for partial chunks)
    aqk_l, akk_l, k_l, v_l, kg_l, q_l, be_l, gk_l, o_l = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for i in range(N):
        bos, eos = cu_cpu[i], cu_cpu[i + 1]
        seq_len = eos - bos
        n_c = math.ceil(seq_len / BT)
        co = co_cpu[i]
        sl = n_c * BT
        partial = seq_len % BT

        # A_qk/A_kk: zero-init safe. V/Beta: A_kk=0 → no contribution. gk: per-chunk.
        aqk_l.append(A_qk_flat[0, bos : bos + sl].reshape(n_c, BT, H, BT))
        akk_l.append(A_kk_flat[0, bos : bos + sl].reshape(n_c, BT, H, BT))
        v_l.append(V_flat[0, bos : bos + sl].reshape(n_c, BT, H, V_dim))
        be_l.append(Beta_flat[0, bos : bos + sl].reshape(n_c, BT, H))
        gk_l.append(gk_last_exp[0, co : co + n_c])

        if partial == 0:
            k_l.append(K_scaled_flat[0, bos : bos + sl].reshape(n_c, BT, H, d))
            kg_l.append(Kg_flat[0, bos : bos + sl].reshape(n_c, BT, H, d))
            q_l.append(Q_scaled_flat[0, bos : bos + sl].reshape(n_c, BT, H, d))
            o_l.append(O_flat[0, bos : bos + sl].reshape(n_c, BT, H, V_dim))
        else:
            # Clone+zero K1 outputs for partial last chunk
            kv = K_scaled_flat[0, bos : bos + sl].clone().reshape(n_c, BT, H, d)
            kgv = Kg_flat[0, bos : bos + sl].clone().reshape(n_c, BT, H, d)
            qv = Q_scaled_flat[0, bos : bos + sl].clone().reshape(n_c, BT, H, d)
            kv[-1, partial:] = 0
            kgv[-1, partial:] = 0
            qv[-1, partial:] = 0
            k_l.append(kv)
            kg_l.append(kgv)
            q_l.append(qv)
            o_l.append(
                torch.zeros(n_c, BT, H, V_dim, dtype=K_scaled_flat.dtype, device=device)
            )

    with torch.cuda.device(device):
        ct.launch(
            torch.cuda.current_stream(),
            (N * H,),
            fused_kda_kernel_varlen,
            (S, aqk_l, akk_l, k_l, v_l, kg_l, q_l, be_l, gk_l, o_l, d, chunk_size, H),
        )

    # Gather output for non-aligned sequences
    for i in range(N):
        bos, eos = cu_cpu[i], cu_cpu[i + 1]
        seq_len = eos - bos
        if seq_len % BT != 0:
            n_c = math.ceil(seq_len / BT)
            O_flat[0, bos:eos] = o_l[i].reshape(n_c * BT, H, V_dim)[:seq_len]


def launch_kda_kernel_varlen_v2(
    S_in,
    S_out,
    A_qk_flat,
    A_kk_flat,
    K_scaled_flat,
    V_flat,
    Kg_flat,
    Q_scaled_flat,
    Beta_flat,
    gk_last_exp,
    O_flat,
    cu_seqlens,
    chunk_offsets,
    chunk_size=64,
):
    """Launch varlen K4 v2 — true zero-copy, zero-overhead launcher.

    S_in:  (N_seqs, H, K, V) fp32, read-only initial state (K, V) layout.
    S_out: (N_seqs, H, K, V) fp32, write-only final state (K, V) layout.
    No clone or transpose needed.
    """
    N = cu_seqlens.shape[0] - 1
    device = S_in.device
    H = K_scaled_flat.shape[2]
    d = K_scaled_flat.shape[3]

    with torch.cuda.device(device):
        ct.launch(
            torch.cuda.current_stream(),
            (N * H,),
            fused_kda_kernel_varlen_v2,
            (
                S_in,
                S_out,
                A_qk_flat,
                A_kk_flat,
                K_scaled_flat,
                V_flat,
                Kg_flat,
                Q_scaled_flat,
                Beta_flat,
                gk_last_exp,
                O_flat,
                cu_seqlens,
                chunk_offsets,
                d,
                chunk_size,
                H,
            ),
        )
