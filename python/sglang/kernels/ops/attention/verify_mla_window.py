"""Split-KV verify of the MiniMax-M3.1 DSpark draft's windowed, non-causal block.

``verify_mla``'s stage 1 masks the prefix by the window; every draft key sits inside
it, so this combine merges the prefix splits and the draft keys in one pass, several
draft rows per program.
"""

import triton
import triton.language as tl

from sglang.kernels.ops.attention.verify_mla import (
    MAX_N_SPLITS,
    VerifyMLA,
    _active_splits,
)

_BLOCK_CONFIG = {
    # M3.1 DSpark draft, 16 q heads per KV head; measured on gfx950 at bs 8 / 32
    128: (8, 32, 4),
}


# draft rows per program of the combine, measured on gfx950 for the M3.1 draft
# (16 heads, 7 rows, window 4095) at 1-64 requests
def _combine_rows(bs: int) -> int:
    return 2 if bs <= 16 else 8


@triton.jit
def _verify_window_combine_stage2(
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
    ROWS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    V_HEAD_DIM: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_SPLITS: tl.constexpr,
    KV_GROUP_NUM: tl.constexpr,
    HAS_KV_HEADS: tl.constexpr,
):
    """_verify_mla_combine_stage2 for a windowed, non-causal draft block, ROWS rows per program.

    The splits merge in one pass instead of a loop over them.
    """
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    offs_r = tl.program_id(2) * ROWS + tl.arange(0, ROWS)
    cur_q_start = tl.load(qo_indptr + cur_batch)
    l_ext = tl.load(qo_indptr + cur_batch + 1) - cur_q_start
    if tl.program_id(2) * ROWS < l_ext:
        if HAS_KV_HEADS:
            kv_head_off_ke = (cur_head // KV_GROUP_NUM) * stride_keh
            kv_head_off_ve = (cur_head // KV_GROUP_NUM) * stride_veh
        else:
            kv_head_off_ke = 0
            kv_head_off_ve = 0
        offs_d = tl.arange(0, BLOCK_DMODEL)
        offs_dv = tl.arange(0, BLOCK_DV)
        offs_l = tl.arange(0, L_EXT)
        mask_r = offs_r < l_ext
        mask_l = offs_l < l_ext
        mask_dv = offs_dv < V_HEAD_DIM

        # ---- (a) the prefix splits (a windowed split can be empty, lse -inf) ----
        seqlen = tl.load(kv_indptr + cur_batch + 1) - tl.load(kv_indptr + cur_batch)
        offs_s = tl.arange(0, BLOCK_SPLITS)
        mask_s = offs_s < _active_splits(seqlen, num_splits, BLOCK_N)
        lse = tl.load(
            Att_Lse
            + cur_batch * stride_lb
            + cur_head * stride_lh
            + offs_s[:, None] * stride_ls
            + offs_r[None, :],
            mask=mask_s[:, None] & mask_r[None, :],
            other=float("-inf"),
        )
        m = tl.max(lse, axis=0)
        w = tl.exp(lse - tl.where(m == float("-inf"), 0.0, m)[None, :])
        o_s = tl.load(
            Att_Out
            + cur_batch * stride_ob
            + cur_head * stride_oh
            + offs_s[:, None, None] * stride_os
            + offs_r[None, :, None] * stride_ol
            + offs_dv[None, None, :],
            mask=mask_s[:, None, None] & mask_r[None, :, None] & mask_dv[None, None, :],
            other=0.0,
        ).to(tl.float32)
        l_acc = tl.sum(w, axis=0)
        o_prefix = tl.sum(o_s * w[:, :, None], axis=0) / l_acc[:, None]
        lse_prefix = m + tl.log(l_acc)

        # ---- (b) these rows against the draft keys ----------------------------
        q = tl.load(
            Q
            + (cur_q_start + offs_r)[:, None] * stride_qbs
            + cur_head * stride_qh
            + offs_d[None, :],
            mask=mask_r[:, None] & (offs_d[None, :] < HEAD_DIM),
            other=0.0,
        ).to(tl.float32)
        ke = tl.load(
            K_Extend
            + (cur_q_start + offs_l)[:, None] * stride_kebs
            + kv_head_off_ke
            + offs_d[None, :],
            mask=mask_l[:, None] & (offs_d[None, :] < HEAD_DIM),
            other=0.0,
        ).to(tl.float32)
        ve = tl.load(
            V_Extend
            + (cur_q_start + offs_l)[:, None] * stride_vebs
            + kv_head_off_ve
            + offs_dv[None, :],
            mask=mask_l[:, None] & mask_dv[None, :],
            other=0.0,
        ).to(tl.float32)
        qk = tl.sum(q[:, None, :] * ke[None, :, :], 2) * sm_scale
        vis = mask_l[None, :] & mask_r[:, None]
        qk = tl.where(vis, qk, float("-inf"))
        m_d = tl.max(qk, 1)
        pd = tl.exp(qk - m_d[:, None])
        denom_d = tl.sum(pd, 1)
        o_draft = tl.sum(pd[:, :, None] * ve[None, :, :], 1) / denom_d[:, None]
        lse_draft = m_d + tl.log(denom_d)

        # ---- (c) prefix vs draft ----------------------------------------------
        mm = tl.maximum(lse_prefix, lse_draft)
        wp = tl.exp(lse_prefix - mm)
        wd = tl.exp(lse_draft - mm)
        o = (o_prefix * wp[:, None] + o_draft * wd[:, None]) / (wp + wd)[:, None]
        tl.store(
            O_Out
            + (cur_q_start + offs_r)[:, None] * stride_oobs
            + cur_head * stride_ooh
            + offs_dv[None, :],
            o.to(O_Out.dtype.element_ty),
            mask=mask_r[:, None] & mask_dv[None, :],
        )


class WindowedVerifyMLA(VerifyMLA):
    """``VerifyMLA`` with the one-pass windowed combine; ``can_handle`` admits only a
    non-causal block."""

    # measured on gfx950 for the M3.1 draft window
    prefix_num_stages = 2

    @staticmethod
    def block_config(head_dim):
        return _BLOCK_CONFIG.get(head_dim) or VerifyMLA.block_config(head_dim)

    def _run_combine_kernel(
        self,
        bs,
        num_splits,
        q_extend,
        k_extend,
        v_extend,
        o_out,
        qo_indptr,
        kv_indptr,
        sm_scale,
        is_causal=True,
    ):
        rows = _combine_rows(bs)
        _verify_window_combine_stage2[(bs, self.h_q, triton.cdiv(self.l_pad, rows))](
            self.att_out,
            self.att_lse,
            q_extend,
            k_extend,
            v_extend,
            o_out,
            sm_scale,
            qo_indptr,
            kv_indptr,
            num_splits,
            self.att_out.stride(0),
            self.att_out.stride(1),
            self.att_out.stride(2),
            self.att_out.stride(3),
            self.att_lse.stride(0),
            self.att_lse.stride(1),
            self.att_lse.stride(2),
            q_extend.stride(0),
            q_extend.stride(1),
            k_extend.stride(0),
            v_extend.stride(0),
            k_extend.stride(1),
            v_extend.stride(1),
            o_out.stride(0),
            o_out.stride(1),
            L_EXT=self.l_pad,
            ROWS=rows,
            HEAD_DIM=self.head_dim,
            V_HEAD_DIM=self.v_head_dim,
            BLOCK_DMODEL=triton.next_power_of_2(self.head_dim),
            BLOCK_DV=triton.next_power_of_2(self.v_head_dim),
            BLOCK_N=self.block_n,
            BLOCK_SPLITS=MAX_N_SPLITS,
            KV_GROUP_NUM=self.kv_group_num,
            HAS_KV_HEADS=self.has_kv_heads,
            num_warps=4,
            num_stages=1,
        )
