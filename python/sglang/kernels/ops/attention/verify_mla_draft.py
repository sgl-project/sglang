"""Split-KV verify of the gfx950 monokernel's DFLASH MLA draft.

It launches ``verify_mla``'s kernel bodies with constexpr options the other launches
leave off: BF16 prefix dots over an FP8 cache, a sliding window trimmed from the
front of each rail, empty prefixes, and a combine over blocks of splits and row tiles.
"""

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.attention.verify_mla import (
    DEFAULT_BLOCK_H,
    DEFAULT_BLOCK_N,
    DEFAULT_NUM_WARPS,
    TARGET_PROGRAMS,
    VerifyMLA,
    _verify_mla_combine_stage2_body,
    _verify_mla_prefix_stage1_body,
)
from sglang.kernels.ops.attention.verify_splitkv import _AMD_LAUNCH_KWARGS

# The MLA draft (Dv 512) splits further: at bs1 / 64K the 3 head blocks x 32 splits
# walk 2K keys each and leave most MI350X CUs idle.
MLA_MAX_N_SPLITS = 128
# splits the combine reads per step; the per-split online loop pays one dependent
# round trip per split
MLA_SPLIT_BLOCK = 16
# draft rows per combine program; VerifyMLA's combine takes all L_EXT rows
MLA_COMBINE_ROWS = 2
# The shared launch's waves_per_eu 4 caps stage 1 at 128 VGPRs, where the MLA draft
# tile spills on gfx950; 2 compiles it spill-free.
DRAFT_STAGE1_WAVES_PER_EU = 2

_DRAFT_BLOCK_CONFIG = {
    # head_dim: (BLOCK_H, BLOCK_N, num_warps)
    # 2 warps or 32/128-key tiles miscompute; 2 heads a tile fill more CUs on a 1K window
    576: (2, 64, 4),  # K3 MLA (kv_lora_rank 512 + qk_rope 64)
}


# The draft's launched kernels: the bodies' parameters plus the draft's constexprs.
# fmt: off
@triton.jit
def _verify_mla_draft_prefix_stage1(
    Q, K_Buffer, V_Buffer, sm_scale, k_scale, v_scale, qo_indptr, kv_indptr, kv_indices,
    Att_Out, Att_Lse, num_splits, stride_qbs, stride_qh, stride_buf_kbs, stride_buf_vbs,
    stride_buf_kh, stride_buf_vh, stride_ob, stride_oh, stride_os, stride_ol, stride_lb,
    stride_lh, stride_ls,
    H_Q: tl.constexpr, L_EXT: tl.constexpr, BLOCK_H: tl.constexpr,
    KV_GROUP_NUM: tl.constexpr, HAS_KV_HEADS: tl.constexpr, NOPE_DIM: tl.constexpr,
    PE_DIM: tl.constexpr, V_HEAD_DIM: tl.constexpr, BLOCK_DNOPE: tl.constexpr,
    BLOCK_DPE: tl.constexpr, BLOCK_DV: tl.constexpr, BLOCK_N: tl.constexpr,
    BF16_DOT: tl.constexpr = False, TRIM_WINDOW: tl.constexpr = 0,
):
    _verify_mla_prefix_stage1_body(
        Q, K_Buffer, V_Buffer, sm_scale, k_scale, v_scale, qo_indptr, kv_indptr, kv_indices,
        Att_Out, Att_Lse, num_splits, stride_qbs, stride_qh, stride_buf_kbs, stride_buf_vbs,
        stride_buf_kh, stride_buf_vh, stride_ob, stride_oh, stride_os, stride_ol, stride_lb,
        stride_lh, stride_ls, H_Q, L_EXT, BLOCK_H, KV_GROUP_NUM, HAS_KV_HEADS, NOPE_DIM,
        PE_DIM, V_HEAD_DIM, BLOCK_DNOPE, BLOCK_DPE, BLOCK_DV, BLOCK_N, -1, BF16_DOT,
        TRIM_WINDOW,
    )


@triton.jit
def _verify_mla_draft_combine_stage2(
    Att_Out, Att_Lse, Q, K_Extend, V_Extend, O_Out, sm_scale, qo_indptr, kv_indptr,
    num_splits, stride_ob, stride_oh, stride_os, stride_ol, stride_lb, stride_lh,
    stride_ls, stride_qbs, stride_qh, stride_kebs, stride_vebs, stride_keh, stride_veh,
    stride_oobs, stride_ooh,
    L_EXT: tl.constexpr, HEAD_DIM: tl.constexpr, V_HEAD_DIM: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr, BLOCK_DV: tl.constexpr, BLOCK_N: tl.constexpr,
    KV_GROUP_NUM: tl.constexpr, HAS_KV_HEADS: tl.constexpr, IS_CAUSAL: tl.constexpr,
    SPLIT_BLOCK: tl.constexpr = 1, ROW_BLOCK: tl.constexpr = 0,
    TRIM_WINDOW: tl.constexpr = 0,
):
    _verify_mla_combine_stage2_body(
        Att_Out, Att_Lse, Q, K_Extend, V_Extend, O_Out, sm_scale, qo_indptr, kv_indptr,
        num_splits, stride_ob, stride_oh, stride_os, stride_ol, stride_lb, stride_lh,
        stride_ls, stride_qbs, stride_qh, stride_kebs, stride_vebs, stride_keh, stride_veh,
        stride_oobs, stride_ooh, L_EXT, HEAD_DIM, V_HEAD_DIM, BLOCK_DMODEL, BLOCK_DV,
        BLOCK_N, KV_GROUP_NUM, HAS_KV_HEADS, IS_CAUSAL, SPLIT_BLOCK, ROW_BLOCK, TRIM_WINDOW,
        True,
    )
# fmt: on


class DraftVerifyMLA(VerifyMLA):
    """``VerifyMLA`` for the gfx950 monokernel's DFLASH MLA draft.

    It serves empty prefixes and a window trimmed from each rail, splits further,
    combines blocks of splits over row tiles, and compiles stage 1 spill-free.
    """

    # the base constructor's _alloc sizes the partials by max_splits
    max_splits = MLA_MAX_N_SPLITS
    split_block = MLA_SPLIT_BLOCK
    stage1_launch = (
        dict(_AMD_LAUNCH_KWARGS, waves_per_eu=DRAFT_STAGE1_WAVES_PER_EU)
        if _AMD_LAUNCH_KWARGS
        else _AMD_LAUNCH_KWARGS
    )

    @staticmethod
    def block_config(head_dim):
        """Return (BLOCK_H, BLOCK_N, num_warps), with a power-of-two head tile."""
        return _DRAFT_BLOCK_CONFIG.get(
            head_dim, (DEFAULT_BLOCK_H, DEFAULT_BLOCK_N, DEFAULT_NUM_WARPS)
        )

    def __init__(self, max_bs, h_q, head_dim, v_head_dim, l_ext, **kwargs):
        super().__init__(max_bs, h_q, head_dim, v_head_dim, l_ext, **kwargs)
        self.row_block = min(MLA_COMBINE_ROWS, self.l_pad)

    def _alloc(self, max_bs):
        self.max_bs = max_bs
        # (request, split) pairs a launch can need: bs x _num_splits(bs) peaks well
        # below max_bs x max_splits, so the partials are flat and viewed per call.
        pairs = max(bs * self._num_splits(bs) for bs in range(1, max_bs + 1))
        self.att_out = torch.empty(
            pairs * self.h_q * self.l_pad * self.v_head_dim,
            dtype=torch.bfloat16,
            device=self.device,
        )
        self.att_lse = torch.empty(
            pairs * self.h_q * self.l_pad, dtype=torch.float32, device=self.device
        )

    def _partials(self, bs, num_splits):
        """Views of the flat partials for one launch.

        [bs, H_Q, num_splits, L_PAD, Dv] bf16 and [bs, H_Q, num_splits, L_PAD] fp32."""
        n = bs * self.h_q * num_splits * self.l_pad
        att_out = self.att_out[: n * self.v_head_dim].view(
            bs, self.h_q, num_splits, self.l_pad, self.v_head_dim
        )
        att_lse = self.att_lse[:n].view(bs, self.h_q, num_splits, self.l_pad)
        return att_out, att_lse

    def _num_splits(self, bs):
        budget = TARGET_PROGRAMS // max(1, bs * self.n_head_blocks)
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
        _verify_mla_draft_prefix_stage1[grid](
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
            TRIM_WINDOW=window,
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
        grid = (bs, self.h_q, self.l_pad // self.row_block)
        _verify_mla_draft_combine_stage2[grid](
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
            TRIM_WINDOW=window,
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
