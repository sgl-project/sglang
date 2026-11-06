"""Grouped-head split-KV verify for the gfx950 monokernel's DFLASH MLA draft.

Specializes ``verify_mla.py`` for sliding windows and optional BF16 prefix dots
over FP8 KV. Each program handles a group of query heads and the whole draft
block; the caller must enforce topk == 1.
"""

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.attention.verify_splitkv import _AMD_LAUNCH_KWARGS

MAX_N_SPLITS = 32  # Grid split dim upper bound
TARGET_PROGRAMS = 512  # Target total stage-1 programs
# Narrow V heads (the K3 dspark GQA draft, Dv 64) keep cheap partials, so they split
# further: at bs1 / 131K the 2 head blocks x 32 splits leave most MI350X CUs idle.
NARROW_V_HEAD_DIM = 128
NARROW_MAX_N_SPLITS = 128
NARROW_TARGET_PROGRAMS = 1024
# Splits the combine reads per step; the per-split online loop pays one dependent
# round trip per split.
NARROW_SPLIT_BLOCK = 16
# Absorbed MLA (Dv 512, the DFlash2 MLA draft) splits further too: at bs1 / 64K the
# 3 head blocks x 32 splits walk 2K keys each and leave most MI350X CUs idle.
MLA_MAX_N_SPLITS = 128
MLA_SPLIT_BLOCK = 16
# draft rows per combine program on the MLA path; the other paths take all L_EXT
MLA_COMBINE_ROWS = 2
# The shared launch's waves_per_eu 4 caps stage 1 at 128 VGPRs, where both draft tiles
# (MLA and narrow) spill on gfx950; 2 compiles them spill-free.
DRAFT_STAGE1_WAVES_PER_EU = 2

DEFAULT_BLOCK_H = (
    4  # BLOCK_H must be a power of 2 (tl.arange); heads beyond H_Q are masked.
)
DEFAULT_BLOCK_N = 64
DEFAULT_NUM_WARPS = 8
_BLOCK_CONFIG = {
    # head_dim: (BLOCK_H, BLOCK_N, num_warps)
    256: (4, 64, 8),  # Qwen3.5 TP2 / TP4 / TP8
    # 4 warps beat 8 on MI350X; 2 warps, 128-key or 32-key tiles miscompute (32 keys
    # fails test_verify_shared_kv). Two heads a tile: twice the head blocks, so a
    # 1K-key window (the MLA5 draft) fills more of the CUs.
    576: (2, 64, 4),  # K3 MLA (kv_lora_rank 512 + qk_rope 64)
    64: (4, 256, 4),  # K3 GQA (dspark draft attention)
}


def block_config(head_dim):
    """Return (BLOCK_H, BLOCK_N, num_warps), with a power-of-two head tile."""
    return _BLOCK_CONFIG.get(
        head_dim, (DEFAULT_BLOCK_H, DEFAULT_BLOCK_N, DEFAULT_NUM_WARPS)
    )


@triton.jit
def _active_splits(seqlen, num_splits, BLOCK_N: tl.constexpr):
    """Floor partitioning keeps every active split non-empty."""
    return tl.maximum(1, tl.minimum(num_splits, seqlen // BLOCK_N))


@triton.jit
def _verify_mla_prefix_stage1(
    Q,
    K_Buffer,
    V_Buffer,
    sm_scale,
    k_scale,
    v_scale,
    qo_indptr,
    kv_indptr,
    kv_indices,
    Att_Out,  # [BS, H_Q, MAX_N_SPLITS, L_EXT, Dv] bf16
    Att_Lse,  # [BS, H_Q, MAX_N_SPLITS, L_EXT]     fp32
    num_splits,  # launched split count
    stride_qbs,
    stride_qh,
    stride_buf_kbs,
    stride_buf_vbs,
    stride_buf_kh,
    stride_buf_vh,
    stride_ob,
    stride_oh,
    stride_os,
    stride_ol,
    stride_lb,
    stride_lh,
    stride_ls,
    H_Q: tl.constexpr,
    L_EXT: tl.constexpr,
    BLOCK_H: tl.constexpr,
    KV_GROUP_NUM: tl.constexpr,
    HAS_KV_HEADS: tl.constexpr,
    NOPE_DIM: tl.constexpr,
    PE_DIM: tl.constexpr,
    V_HEAD_DIM: tl.constexpr,
    BLOCK_DNOPE: tl.constexpr,
    BLOCK_DPE: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BF16_DOT: tl.constexpr = False,
    WINDOW: tl.constexpr = 0,
):
    cur_batch = tl.program_id(0)
    head_block = tl.program_id(1)
    split_kv_id = tl.program_id(2)

    # row tile size for each workgroup
    R: tl.constexpr = BLOCK_H * L_EXT

    cur_batch_kv_start_idx = tl.load(kv_indptr + cur_batch)
    cur_batch_seq_len = tl.load(kv_indptr + cur_batch + 1) - cur_batch_kv_start_idx
    if WINDOW > 0:
        # Keys before (length - WINDOW) are outside every row's window: start the
        # rail there, so the per-row mask below drops at most L_EXT - 1 keys.
        skip = tl.maximum(cur_batch_seq_len - WINDOW, 0)
        cur_batch_kv_start_idx += skip
        cur_batch_seq_len -= skip
    active = _active_splits(cur_batch_seq_len, num_splits, BLOCK_N)

    # skip idle workgroups
    if split_kv_id < active:
        head_start = head_block * BLOCK_H
        offs_h = head_start + tl.arange(0, BLOCK_H)
        # The caller guarantees KV_GROUP_NUM % BLOCK_H == 0, so every query head
        # in this block maps to the same KV head and the K/V tiles stay 2D loads.
        # MLA has a single latent head; keeping the offset out of that path
        # entirely preserves the original address arithmetic in the inner loop.
        if HAS_KV_HEADS:
            kv_head_off_k = (head_start // KV_GROUP_NUM) * stride_buf_kh
            kv_head_off_v = (head_start // KV_GROUP_NUM) * stride_buf_vh
        else:
            kv_head_off_k = 0
            kv_head_off_v = 0
        offs_l = tl.arange(0, L_EXT)
        offs_dn = tl.arange(0, BLOCK_DNOPE)
        offs_dp = tl.arange(0, BLOCK_DPE)
        offs_dv = tl.arange(0, BLOCK_DV)

        cur_q_start = tl.load(qo_indptr + cur_batch)
        l_ext = tl.load(qo_indptr + cur_batch + 1) - cur_q_start
        row_mask = tl.reshape((offs_h[:, None] < H_Q) & (offs_l[None, :] < l_ext), (R,))

        # the last kv split completes the remaining
        kv_len_per_split = cur_batch_seq_len // active
        split_start = kv_len_per_split * split_kv_id
        split_end = tl.where(
            split_kv_id == active - 1, cur_batch_seq_len, split_start + kv_len_per_split
        )

        # For absorbed MLA, NOPE_DIM is the latent width and PE_DIM is the
        # appended RoPE width. For ordinary shared-KV attention (Qwen3.5),
        # NOPE_DIM is the complete already-rotated Q/K head and PE_DIM is zero.
        q_row = tl.reshape(
            (cur_q_start + offs_l)[None, :] * stride_qbs + offs_h[:, None] * stride_qh,
            (R,),
        )
        # q, the probabilities and the dots run in the cache dtype (FP8 dots over
        # an FP8 cache), or with BF16_DOT in q's dtype, upcasting K/V instead.
        dot_dtype = Q.dtype.element_ty if BF16_DOT else K_Buffer.dtype.element_ty
        q_nope = tl.load(
            Q + q_row[:, None] + offs_dn[None, :],
            mask=row_mask[:, None] & (offs_dn[None, :] < NOPE_DIM),
            other=0.0,
        ).to(dot_dtype)
        if PE_DIM > 0:
            q_pe = tl.load(
                Q + q_row[:, None] + (NOPE_DIM + offs_dp)[None, :],
                mask=row_mask[:, None] & (offs_dp[None, :] < PE_DIM),
                other=0.0,
            ).to(dot_dtype)

        e_max = tl.zeros([R], dtype=tl.float32) - float("inf")
        e_sum = tl.zeros([R], dtype=tl.float32)
        acc = tl.zeros([R, BLOCK_DV], dtype=tl.float32)

        # calculate attention scores for each kv split
        for start_n in tl.range(split_start, split_end, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            n_mask = offs_n < split_end
            kv_loc = tl.load(
                kv_indices + cur_batch_kv_start_idx + offs_n, mask=n_mask, other=0
            )
            base = kv_loc[None, :] * stride_buf_kbs + kv_head_off_k
            k_nope = tl.load(
                K_Buffer + base + offs_dn[:, None],
                mask=(offs_dn[:, None] < NOPE_DIM) & n_mask[None, :],
                other=0.0,
            )
            qk = tl.dot(q_nope, k_nope.to(dot_dtype))
            if PE_DIM > 0:
                k_pe = tl.load(
                    K_Buffer + base + (NOPE_DIM + offs_dp)[:, None],
                    mask=(offs_dp[:, None] < PE_DIM) & n_mask[None, :],
                    other=0.0,
                )
                qk += tl.dot(q_pe, k_pe.to(dot_dtype))
            qk *= sm_scale * k_scale
            qk = tl.where(n_mask[None, :], qk, float("-inf"))
            if WINDOW > 0:
                # extend_attention_fwd's prefix window: draft row l sits at
                # (rail length + l) and keeps rail key n iff q <= n + WINDOW.
                # On the shortened rail it masks at most L_EXT - 1 keys at its
                # start, inside the first tile of split 0, so no row loses a
                # whole tile.
                q_pos = cur_batch_seq_len + tl.reshape(
                    offs_h[:, None] * 0 + offs_l[None, :], (R,)
                )
                qk = tl.where(
                    q_pos[:, None] <= offs_n[None, :] + WINDOW, qk, float("-inf")
                )

            # MLA exposes its latent V through V_Buffer; ordinary shared-KV
            # attention has an independent V cache. Both use this same load.
            v = tl.load(
                V_Buffer
                + kv_loc[:, None] * stride_buf_vbs
                + kv_head_off_v
                + offs_dv[None, :],
                mask=n_mask[:, None] & (offs_dv[None, :] < V_HEAD_DIM),
                other=0.0,
            )
            if BF16_DOT:
                v = v.to(dot_dtype)
            n_e_max = tl.maximum(tl.max(qk, 1), e_max)
            re_scale = tl.exp(e_max - n_e_max)
            p = tl.exp(qk - n_e_max[:, None])
            acc *= re_scale[:, None]
            acc += tl.dot(p.to(v.dtype), v)
            e_sum = e_sum * re_scale + tl.sum(p, 1)
            e_max = n_e_max

        # fp8 dequant of prefix V: scale the accumulated (pre-normalised) output.
        acc *= v_scale
        o_row = tl.reshape(
            cur_batch * stride_ob
            + offs_h[:, None] * stride_oh
            + split_kv_id * stride_os
            + offs_l[None, :] * stride_ol,
            (R,),
        )
        tl.store(
            Att_Out + o_row[:, None] + offs_dv[None, :],
            (acc / e_sum[:, None]).to(Att_Out.dtype.element_ty),
            mask=row_mask[:, None] & (offs_dv[None, :] < V_HEAD_DIM),
        )
        lse_row = tl.reshape(
            cur_batch * stride_lb
            + offs_h[:, None] * stride_lh
            + split_kv_id * stride_ls
            + offs_l[None, :],
            (R,),
        )
        tl.store(Att_Lse + lse_row, e_max + tl.log(e_sum), mask=row_mask)


@triton.jit
def _verify_mla_combine_stage2(
    Att_Out,
    Att_Lse,
    Q,
    K_Extend,
    V_Extend,
    O_Out,
    sm_scale,
    qo_indptr,
    kv_indptr,
    num_splits,
    stride_ob,
    stride_oh,
    stride_os,
    stride_ol,
    stride_lb,
    stride_lh,
    stride_ls,
    stride_qbs,
    stride_qh,
    stride_kebs,
    stride_vebs,
    stride_keh,
    stride_veh,
    stride_oobs,
    stride_ooh,
    L_EXT: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    V_HEAD_DIM: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_N: tl.constexpr,
    KV_GROUP_NUM: tl.constexpr,
    HAS_KV_HEADS: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    SPLIT_BLOCK: tl.constexpr = 1,
    ROW_BLOCK: tl.constexpr = 0,
    WINDOW: tl.constexpr = 0,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    # this program's query rows (ROW_BLOCK; 0: all L_EXT) against every draft key
    ROWS: tl.constexpr = L_EXT if ROW_BLOCK == 0 else ROW_BLOCK
    row0 = tl.program_id(2) * ROWS
    if HAS_KV_HEADS:
        kv_head_off_ke = (cur_head // KV_GROUP_NUM) * stride_keh
        kv_head_off_ve = (cur_head // KV_GROUP_NUM) * stride_veh
    else:
        kv_head_off_ke = 0
        kv_head_off_ve = 0

    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_dv = tl.arange(0, BLOCK_DV)
    offs_l = row0 + tl.arange(0, ROWS)
    offs_k = tl.arange(0, L_EXT)

    cur_q_start = tl.load(qo_indptr + cur_batch)
    l_ext = tl.load(qo_indptr + cur_batch + 1) - cur_q_start
    mask_l = offs_l < l_ext
    mask_k = offs_k < l_ext

    # ---- (a) combine prefix splits (online logsumexp over active splits) ---
    seqlen = tl.load(kv_indptr + cur_batch + 1) - tl.load(kv_indptr + cur_batch)
    if WINDOW > 0:
        seqlen = tl.minimum(seqlen, WINDOW)  # stage 1's shortened rail
    # Stage 1 still runs split 0 of an empty prefix and stores its 0/0 partial;
    # skip it here so only the draft-draft part reaches the output.
    active = tl.where(seqlen > 0, _active_splits(seqlen, num_splits, BLOCK_N), 0)

    m = tl.zeros([ROWS], dtype=tl.float32) - float("inf")
    l_acc = tl.zeros([ROWS], dtype=tl.float32)
    acc = tl.zeros([ROWS, BLOCK_DV], dtype=tl.float32)
    if SPLIT_BLOCK > 1:
        offs_s = tl.arange(0, SPLIT_BLOCK)
        lse_base = Att_Lse + cur_batch * stride_lb + cur_head * stride_lh
        o_base = Att_Out + cur_batch * stride_ob + cur_head * stride_oh
        for s0 in range(0, active, SPLIT_BLOCK):
            s_mask = (s0 + offs_s) < active
            lse_blk = tl.load(
                lse_base + (s0 + offs_s)[:, None] * stride_ls + offs_l[None, :],
                mask=s_mask[:, None] & mask_l[None, :],
                other=float("-inf"),
            )
            m = tl.maximum(m, tl.max(lse_blk, 0))
        m_ref = tl.where(m == float("-inf"), 0.0, m)
        for s0 in range(0, active, SPLIT_BLOCK):
            s_mask = (s0 + offs_s) < active
            lse_blk = tl.load(
                lse_base + (s0 + offs_s)[:, None] * stride_ls + offs_l[None, :],
                mask=s_mask[:, None] & mask_l[None, :],
                other=float("-inf"),
            )
            w = tl.exp(lse_blk - m_ref[None, :])
            o_blk = tl.load(
                o_base
                + (s0 + offs_s)[:, None, None] * stride_os
                + offs_l[None, :, None] * stride_ol
                + offs_dv[None, None, :],
                mask=s_mask[:, None, None]
                & mask_l[None, :, None]
                & (offs_dv[None, None, :] < V_HEAD_DIM),
                other=0.0,
            ).to(tl.float32)
            acc += tl.sum(o_blk * w[:, :, None], 0)
            l_acc += tl.sum(w, 0)
        m = m_ref
    else:
        for s in range(active):
            lse_s = tl.load(
                Att_Lse
                + cur_batch * stride_lb
                + cur_head * stride_lh
                + s * stride_ls
                + offs_l,
                mask=mask_l,
                other=float("-inf"),
            )
            o_s = tl.load(
                Att_Out
                + cur_batch * stride_ob
                + cur_head * stride_oh
                + s * stride_os
                + offs_l[:, None] * stride_ol
                + offs_dv[None, :],
                mask=mask_l[:, None] & (offs_dv[None, :] < V_HEAD_DIM),
                other=0.0,
            ).to(tl.float32)
            new_m = tl.maximum(m, lse_s)
            alpha = tl.exp(m - new_m)
            beta = tl.exp(lse_s - new_m)
            acc = acc * alpha[:, None] + o_s * beta[:, None]
            l_acc = l_acc * alpha + beta
            m = new_m
    # An empty prefix (no active split) leaves acc = l_acc = 0: keep o_prefix at 0
    # instead of 0/0, so its zero weight in the merge below drops it cleanly.
    o_prefix = acc / tl.where(l_acc > 0, l_acc, 1.0)[:, None]
    lse_prefix = m + tl.log(l_acc)

    # ---- (b) draft-draft causal attention (L_EXT x L_EXT) -----------------
    # load draft queries [L_EXT, D], draft K/V [L_EXT, D]/[L_EXT, Dv]
    offs_q = (
        (cur_q_start + offs_l)[:, None] * stride_qbs
        + cur_head * stride_qh
        + offs_d[None, :]
    )
    q = tl.load(
        Q + offs_q, mask=mask_l[:, None] & (offs_d[None, :] < HEAD_DIM), other=0.0
    ).to(tl.float32)
    offs_ke = (
        (cur_q_start + offs_k)[:, None] * stride_kebs + kv_head_off_ke + offs_d[None, :]
    )
    ke = tl.load(
        K_Extend + offs_ke,
        mask=mask_k[:, None] & (offs_d[None, :] < HEAD_DIM),
        other=0.0,
    ).to(tl.float32)
    offs_ve = (
        (cur_q_start + offs_k)[:, None] * stride_vebs
        + kv_head_off_ve
        + offs_dv[None, :]
    )
    ve = tl.load(
        V_Extend + offs_ve,
        mask=mask_k[:, None] & (offs_dv[None, :] < V_HEAD_DIM),
        other=0.0,
    ).to(tl.float32)

    # scores[i,j] = q_i . k_j  (i query, j key)  -> [L_EXT, L_EXT]
    qk = tl.sum(q[:, None, :] * ke[None, :, :], 2) * sm_scale
    # causal: query i sees key j iff j <= i
    # non-causal: query i sees all keys
    both_valid = mask_k[None, :] & mask_l[:, None]
    if IS_CAUSAL:
        vis = (offs_k[None, :] <= offs_l[:, None]) & both_valid
    else:
        vis = both_valid
    qk = tl.where(vis, qk, float("-inf"))
    m_d = tl.max(qk, 1)
    pd = tl.exp(qk - m_d[:, None])
    denom_d = tl.sum(pd, 1)
    o_draft = tl.sum(pd[:, :, None] * ve[None, :, :], 1) / denom_d[:, None]
    lse_draft = m_d + tl.log(denom_d)

    # ---- (c) final LSE merge (prefix vs draft) ----------------------------
    mm = tl.maximum(lse_prefix, lse_draft)
    wp = tl.exp(lse_prefix - mm)
    wd = tl.exp(lse_draft - mm)
    o = (o_prefix * wp[:, None] + o_draft * wd[:, None]) / (wp + wd)[:, None]

    offs_oo = (
        (cur_q_start + offs_l)[:, None] * stride_oobs
        + cur_head * stride_ooh
        + offs_dv[None, :]
    )
    tl.store(
        O_Out + offs_oo,
        o.to(O_Out.dtype.element_ty),
        mask=mask_l[:, None] & (offs_dv[None, :] < V_HEAD_DIM),
    )


class VerifyMLA:
    def __init__(
        self,
        max_bs,
        h_q,
        head_dim,
        v_head_dim,
        l_ext,
        device="cuda",
        block_h=DEFAULT_BLOCK_H,
        block_n=DEFAULT_BLOCK_N,
        num_warps=DEFAULT_NUM_WARPS,
        kv_group_num=None,
    ):
        self.h_q = h_q
        # MLA is the h_kv == 1 case (kv_group_num == h_q); a GQA draft passes a
        # smaller group. block_h must divide it so a head block maps to one KV
        # head -- can_handle enforces that before this is constructed.
        self.kv_group_num = h_q if kv_group_num is None else kv_group_num
        # h_kv == 1 (MLA / MQA) needs no KV-head offset at all; making that a
        # constexpr keeps the inner-loop addressing identical to the original.
        self.has_kv_heads = self.kv_group_num < h_q
        self.head_dim = head_dim
        self.v_head_dim = v_head_dim
        self.nope_dim = v_head_dim
        self.pe_dim = head_dim - v_head_dim
        self.l_ext = l_ext
        # tl.dot requires BLOCK_H * L_EXT to cover at least 16 rows.
        min_l_pad = triton.next_power_of_2(triton.cdiv(16, block_h))
        self.l_pad = max(min_l_pad, triton.next_power_of_2(l_ext))
        self.device = device
        self.block_h = block_h
        self.block_n = block_n
        self.num_warps = num_warps
        self.n_head_blocks = triton.cdiv(h_q, block_h)
        narrow = v_head_dim <= NARROW_V_HEAD_DIM
        mla = not narrow and head_dim > v_head_dim
        if narrow:
            self.max_splits, self.split_block = NARROW_MAX_N_SPLITS, NARROW_SPLIT_BLOCK
        elif mla:
            self.max_splits, self.split_block = MLA_MAX_N_SPLITS, MLA_SPLIT_BLOCK
        else:
            self.max_splits, self.split_block = MAX_N_SPLITS, 1
        self.target_programs = NARROW_TARGET_PROGRAMS if narrow else TARGET_PROGRAMS
        self.row_block = min(MLA_COMBINE_ROWS, self.l_pad) if mla else 0
        self.stage1_launch = (
            dict(_AMD_LAUNCH_KWARGS, waves_per_eu=DRAFT_STAGE1_WAVES_PER_EU)
            if (mla or narrow) and _AMD_LAUNCH_KWARGS
            else _AMD_LAUNCH_KWARGS
        )
        self._alloc(max_bs)

    def _alloc(self, max_bs):
        self.max_bs = max_bs
        # (request, split) pairs a launch can need: bs x _num_splits(bs) peaks well
        # below max_bs x max_splits, so the partials are flat and viewed per call.
        pairs = max(bs * self._num_splits(bs) for bs in range(1, max_bs + 1))
        # bf16 partials (halves scratch traffic vs fp32); lse stays fp32.
        self.att_out_flat = torch.empty(
            pairs * self.h_q * self.l_pad * self.v_head_dim,
            dtype=torch.bfloat16,
            device=self.device,
        )
        self.att_lse_flat = torch.empty(
            pairs * self.h_q * self.l_pad, dtype=torch.float32, device=self.device
        )

    def _partials(self, bs, num_splits):
        """Views of the flat partials for one launch.

        [bs, H_Q, num_splits, L_PAD, Dv] bf16 and [bs, H_Q, num_splits, L_PAD] fp32."""
        n = bs * self.h_q * num_splits * self.l_pad
        att_out = self.att_out_flat[: n * self.v_head_dim].view(
            bs, self.h_q, num_splits, self.l_pad, self.v_head_dim
        )
        att_lse = self.att_lse_flat[:n].view(bs, self.h_q, num_splits, self.l_pad)
        return att_out, att_lse

    def grow_buffers(self, max_bs):
        if max_bs > self.max_bs:
            self._alloc(max_bs)

    def _num_splits(self, bs):
        budget = self.target_programs // max(1, bs * self.n_head_blocks)
        return max(1, min(self.max_splits, budget))

    def _run_prefix_kernel(
        self,
        bs,
        num_splits,
        att_out,
        att_lse,
        q_extend,
        k_buffer,
        v_buffer,
        qo_indptr,
        kv_indptr,
        kv_indices,
        sm_scale,
        k_scale,
        v_scale,
        bf16_dot=False,
        window=0,
    ):
        grid = (bs, self.n_head_blocks, num_splits)
        _verify_mla_prefix_stage1[grid](
            q_extend,
            k_buffer,
            v_buffer,
            sm_scale,
            k_scale,
            v_scale,
            qo_indptr,
            kv_indptr,
            kv_indices,
            att_out,
            att_lse,
            num_splits,
            q_extend.stride(0),
            q_extend.stride(1),
            k_buffer.stride(0),
            v_buffer.stride(0),
            k_buffer.stride(1),
            v_buffer.stride(1),
            att_out.stride(0),
            att_out.stride(1),
            att_out.stride(2),
            att_out.stride(3),
            att_lse.stride(0),
            att_lse.stride(1),
            att_lse.stride(2),
            H_Q=self.h_q,
            L_EXT=self.l_pad,
            BLOCK_H=self.block_h,
            KV_GROUP_NUM=self.kv_group_num,
            HAS_KV_HEADS=self.has_kv_heads,
            NOPE_DIM=self.nope_dim,
            PE_DIM=self.pe_dim,
            V_HEAD_DIM=self.v_head_dim,
            BLOCK_DNOPE=triton.next_power_of_2(self.nope_dim),
            BLOCK_DPE=max(1, triton.next_power_of_2(self.pe_dim)),
            BLOCK_DV=triton.next_power_of_2(self.v_head_dim),
            BLOCK_N=self.block_n,
            BF16_DOT=bf16_dot,
            WINDOW=window,
            num_warps=self.num_warps,
            # 2 stages spill on the MLA tile and miscompute
            num_stages=1,
            **self.stage1_launch,
        )

    def _run_combine_kernel(
        self,
        bs,
        num_splits,
        att_out,
        att_lse,
        q_extend,
        k_extend,
        v_extend,
        o_out,
        qo_indptr,
        kv_indptr,
        sm_scale,
        is_causal=True,
        window=0,
    ):
        grid = (bs, self.h_q, self.l_pad // self.row_block if self.row_block else 1)
        _verify_mla_combine_stage2[grid](
            att_out,
            att_lse,
            q_extend,
            k_extend,
            v_extend,
            o_out,
            sm_scale,
            qo_indptr,
            kv_indptr,
            num_splits,
            att_out.stride(0),
            att_out.stride(1),
            att_out.stride(2),
            att_out.stride(3),
            att_lse.stride(0),
            att_lse.stride(1),
            att_lse.stride(2),
            q_extend.stride(0),
            q_extend.stride(1),
            k_extend.stride(0),
            v_extend.stride(0),
            k_extend.stride(1),
            v_extend.stride(1),
            o_out.stride(0),
            o_out.stride(1),
            L_EXT=self.l_pad,
            HEAD_DIM=self.head_dim,
            V_HEAD_DIM=self.v_head_dim,
            BLOCK_DMODEL=triton.next_power_of_2(self.head_dim),
            BLOCK_DV=triton.next_power_of_2(self.v_head_dim),
            BLOCK_N=self.block_n,
            KV_GROUP_NUM=self.kv_group_num,
            HAS_KV_HEADS=self.has_kv_heads,
            IS_CAUSAL=is_causal,
            SPLIT_BLOCK=self.split_block,
            ROW_BLOCK=self.row_block,
            WINDOW=window,
            num_warps=4,
            num_stages=1,
        )

    def __call__(
        self,
        q_extend,
        k_extend,
        v_extend,
        k_buffer,
        v_buffer,
        qo_indptr,
        kv_indptr,
        kv_indices,
        sm_scale,
        o_out=None,
        k_scale=1.0,
        v_scale=1.0,
        is_causal=True,
        bf16_dot=False,
        window=0,
    ):
        if o_out is None:
            o_out = torch.empty(
                (q_extend.shape[0], self.h_q, self.v_head_dim),
                dtype=q_extend.dtype,
                device=q_extend.device,
            )
        bs = qo_indptr.shape[0] - 1
        # One split count for both stages (they must agree on the active count).
        num_splits = self._num_splits(bs)
        att_out, att_lse = self._partials(bs, num_splits)
        self._run_prefix_kernel(
            bs,
            num_splits,
            att_out,
            att_lse,
            q_extend,
            k_buffer,
            v_buffer,
            qo_indptr,
            kv_indptr,
            kv_indices,
            sm_scale,
            k_scale,
            v_scale,
            bf16_dot=bf16_dot,
            window=window,
        )
        self._run_combine_kernel(
            bs,
            num_splits,
            att_out,
            att_lse,
            q_extend,
            k_extend,
            v_extend,
            o_out,
            qo_indptr,
            kv_indptr,
            sm_scale,
            is_causal=is_causal,
            window=window,
        )
        return o_out


_VMLA_CACHE = {}


def _get_vmla(max_bs, h_q, head_dim, v_head_dim, l_ext, device, kv_group_num=None):
    key = (h_q, head_dim, v_head_dim, l_ext, str(device), kv_group_num)
    vk = _VMLA_CACHE.get(key)
    if vk is None:
        block_h, block_n, num_warps = block_config(head_dim)
        if (
            kv_group_num is not None
            and kv_group_num < h_q  # i.e. h_kv > 1, so the offset is live
            and kv_group_num % block_h != 0
        ):
            # Shrink the head block so it stays inside one KV head. can_handle
            # only admits power-of-two groups in that case, so this stays valid
            # for tl.arange(0, BLOCK_H).
            block_h = kv_group_num
        vk = VerifyMLA(
            max_bs,
            h_q,
            head_dim,
            v_head_dim,
            l_ext,
            device=device,
            block_h=block_h,
            block_n=block_n,
            num_warps=num_warps,
            kv_group_num=kv_group_num,
        )
        _VMLA_CACHE[key] = vk
    else:
        vk.grow_buffers(max_bs)
    return vk


def can_handle(
    q_extend,
    k_extend,
    v_extend,
    k_buffer,
    v_buffer,
    qo_indptr,
    kv_indptr,
    kv_indices,
    custom_mask,
    is_causal,
    mask_indptr,
    max_len_extend,
    sliding_window_size=-1,
    sinks=None,
    logit_cap=0.0,
    xai_temperature_len=-1,
):
    """Check static eligibility; unsupported inputs fall back to extend_attention_fwd.

    The caller must enforce topk == 1: only then does a causal mask match the
    draft tree. Reading custom_mask values would synchronize during graph capture.
    """
    if sinks is not None:
        return False
    # A sliding window is masked on the prefix (WINDOW in stage 1); inside the
    # draft block it never binds as long as it spans the block.
    if sliding_window_size is not None and sliding_window_size > 0:
        try:
            if int(sliding_window_size) < int(max_len_extend):
                return False
        except (TypeError, ValueError):
            return False
    if logit_cap and logit_cap > 0:
        return False
    if xai_temperature_len is not None and xai_temperature_len > 0:
        return False
    if not is_causal and custom_mask is not None:
        return False
    # q layout must be [tokens, H_Q, D]; head dims handled by power-of-2 pad.
    if q_extend.dim() != 3 or k_extend.dim() != 3 or v_extend.dim() != 3:
        return False
    # GQA group must divide evenly.
    h_q = q_extend.shape[1]
    h_kv = k_extend.shape[1]
    if h_kv == 0 or h_q % h_kv != 0:
        return False
    kv_group_num = h_q // h_kv
    if h_kv > 1 and (kv_group_num & (kv_group_num - 1)) != 0:
        return False
    # head dims must match buffers.
    if k_buffer.shape[1] != h_kv or v_buffer.shape[1] != h_kv:
        return False
    if q_extend.shape[2] != k_extend.shape[2]:
        return False
    if q_extend.shape[2] != k_buffer.shape[2]:
        return False
    if v_extend.shape[2] != v_buffer.shape[2]:
        return False
    # NOTE: must NOT read any tensor *values* here (no .item()/.cpu()): the
    # target-verify step runs inside a captured CUDA/HIP graph, where a
    # device->host sync raises hipErrorStreamCaptureUnsupported. We therefore
    # gate purely on static shapes/dtypes/python scalars.
    bs = qo_indptr.shape[0] - 1
    if bs < 1:
        return False
    # max_len_extend must be a known positive python int (it is the static
    # server_args.speculative_num_draft_tokens for the verify path). For
    # topk=1 the per-seq extend len is constant == num_draft_tokens ==
    # max_len_extend by construction of qo_indptr (arange with that step), so
    # the L_EXT row-tile mask is exactly right and the tree custom_mask equals
    # causal -- no value inspection required.
    try:
        mle = int(max_len_extend)
    except (TypeError, ValueError):
        return False
    if mle < 1:
        return False
    # The packed extend tensor must hold exactly bs * max_len_extend rows
    # (constant extend len). This is a pure shape check (no sync) and rejects
    # any ragged/variable-extend batch -> falls back to the baseline.
    if q_extend.shape[0] != bs * mle:
        return False
    return True


def verify_shared_kv_fwd(
    q_extend,
    k_extend,
    v_extend,
    o_extend,
    k_buffer,
    v_buffer,
    qo_indptr,
    kv_indptr,
    kv_indices,
    custom_mask,
    is_causal,
    mask_indptr,
    max_len_extend,
    k_scale,
    v_scale,
    sm_scale=None,
    logit_cap=0.0,
    skip_prefix_custom_mask=True,
    sliding_window_size=-1,
    sinks=None,
    window_kv_offsets=None,
    xai_temperature_len=-1,
    max_bs=None,
    bf16_dot=False,
):
    """Write topk == 1 verify output; return False when the caller must fall back.

    ``bf16_dot`` upcasts FP8 cached K/V for prefix dots in q's dtype. Draft-draft
    attention remains FP32; a BF16 cache needs no upcast.
    """
    if not can_handle(
        q_extend,
        k_extend,
        v_extend,
        k_buffer,
        v_buffer,
        qo_indptr,
        kv_indptr,
        kv_indices,
        custom_mask,
        is_causal,
        mask_indptr,
        max_len_extend,
        sliding_window_size=sliding_window_size,
        sinks=sinks,
        logit_cap=logit_cap,
        xai_temperature_len=xai_temperature_len,
    ):
        return False
    if q_extend.shape[2] < v_extend.shape[2]:
        return False
    if kv_indices.numel() == 0:
        return False

    bs = qo_indptr.shape[0] - 1
    h_q = q_extend.shape[1]
    h_kv = k_extend.shape[1]
    head_dim = q_extend.shape[2]
    v_head_dim = v_extend.shape[2]
    l_ext = int(max_len_extend)
    kv_group_num = h_q // h_kv

    if sm_scale is None:
        sm_scale = 1.0 / (head_dim**0.5)
    try:
        k_scale = float(k_scale)
    except (TypeError, ValueError):
        k_scale = 1.0
    try:
        v_scale = float(v_scale)
    except (TypeError, ValueError):
        v_scale = 1.0

    if max_bs is None or max_bs < bs:
        max_bs = bs
    vk = _get_vmla(
        max_bs, h_q, head_dim, v_head_dim, l_ext, q_extend.device, kv_group_num
    )
    vk(
        q_extend,
        k_extend.contiguous(),
        v_extend.contiguous(),
        k_buffer,
        v_buffer,
        qo_indptr,
        kv_indptr,
        kv_indices,
        sm_scale,
        o_out=o_extend,
        k_scale=k_scale,
        v_scale=v_scale,
        is_causal=is_causal,
        bf16_dot=bf16_dot,
        window=(
            int(sliding_window_size)
            if sliding_window_size is not None and sliding_window_size > 0
            else 0
        ),
    )
    return True
