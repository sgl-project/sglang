"""
Grouped-head split-KV attention for speculative *verify* (topk==1).
Following the pattern of ``python/sglang/kernels/ops/attention/verify_splitkv.py``.

Grid is ``(bs, n_head_blocks, split)``; each program handles ``BLOCK_H`` query
heads x ALL ``L_EXT`` draft queries. It supports absorbed MLA and ordinary
MHA/GQA when exactly one TP-local KV head is shared by all local query heads.

Correctness matches ``extend_attention_fwd`` for the topk==1 causal verify case.

score(h,i,t) = q_nope[i,h] · c_KV[t]  +  q_pe[i,h] · k_pe[t]      # 512 dot + 64 dot
out(h,i)     = Σ_t softmax_t · c_KV[t]                            # V = c_KV
"""

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.attention.verify_splitkv import _AMD_LAUNCH_KWARGS

MAX_N_SPLITS = 32  # Grid split dim upper bound
TARGET_PROGRAMS = 512  # Target total stage-1 programs

DEFAULT_BLOCK_H = (
    4  # BLOCK_H must be a power of 2 (tl.arange); heads beyond H_Q are masked.
)
DEFAULT_BLOCK_N = 64
DEFAULT_NUM_WARPS = 8
_BLOCK_CONFIG = {
    # head_dim: (BLOCK_H, BLOCK_N, num_warps)
    256: (4, 64, 8),  # Qwen3.5 TP2 / TP4 / TP8
    576: (4, 64, 8),  # K3 MLA (kv_lora_rank 512 + qk_rope 64)
}


def block_config(head_dim):
    """
    Return (BLOCK_H, BLOCK_N, num_warps) for a head_dim; default for untuned
    dims. BLOCK_H must be a power of 2 (heads beyond H_Q are masked).
    """
    return _BLOCK_CONFIG.get(
        head_dim, (DEFAULT_BLOCK_H, DEFAULT_BLOCK_N, DEFAULT_NUM_WARPS)
    )


@triton.jit
def _active_splits(seqlen, num_splits, BLOCK_N: tl.constexpr):
    """
    The launched split count, capped by ``seqlen // BLOCK_N``.
    Floor partitioning keeps every active split non-empty.
    """
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
    NOPE_DIM: tl.constexpr,
    PE_DIM: tl.constexpr,
    V_HEAD_DIM: tl.constexpr,
    BLOCK_DNOPE: tl.constexpr,
    BLOCK_DPE: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    head_block = tl.program_id(1)
    split_kv_id = tl.program_id(2)

    # row tile size for each workgroup
    R: tl.constexpr = BLOCK_H * L_EXT

    cur_batch_kv_start_idx = tl.load(kv_indptr + cur_batch)
    cur_batch_seq_len = tl.load(kv_indptr + cur_batch + 1) - cur_batch_kv_start_idx
    active = _active_splits(cur_batch_seq_len, num_splits, BLOCK_N)

    # skip idle workgroups
    if split_kv_id < active:
        head_start = head_block * BLOCK_H
        offs_h = head_start + tl.arange(0, BLOCK_H)
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
        q_nope = tl.load(
            Q + q_row[:, None] + offs_dn[None, :],
            mask=row_mask[:, None] & (offs_dn[None, :] < NOPE_DIM),
            other=0.0,
        ).to(K_Buffer.dtype.element_ty)
        if PE_DIM > 0:
            q_pe = tl.load(
                Q + q_row[:, None] + (NOPE_DIM + offs_dp)[None, :],
                mask=row_mask[:, None] & (offs_dp[None, :] < PE_DIM),
                other=0.0,
            ).to(K_Buffer.dtype.element_ty)

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
            base = kv_loc[None, :] * stride_buf_kbs
            k_nope = tl.load(
                K_Buffer + base + offs_dn[:, None],
                mask=(offs_dn[:, None] < NOPE_DIM) & n_mask[None, :],
                other=0.0,
            )
            qk = tl.dot(q_nope, k_nope)
            if PE_DIM > 0:
                k_pe = tl.load(
                    K_Buffer + base + (NOPE_DIM + offs_dp)[:, None],
                    mask=(offs_dp[:, None] < PE_DIM) & n_mask[None, :],
                    other=0.0,
                )
                qk += tl.dot(q_pe, k_pe)
            qk *= sm_scale * k_scale
            qk = tl.where(n_mask[None, :], qk, float("-inf"))

            # MLA exposes its latent V through V_Buffer; ordinary shared-KV
            # attention has an independent V cache. Both use this same load.
            v = tl.load(
                V_Buffer + kv_loc[:, None] * stride_buf_vbs + offs_dv[None, :],
                mask=n_mask[:, None] & (offs_dv[None, :] < V_HEAD_DIM),
                other=0.0,
            )
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
    stride_oobs,
    stride_ooh,
    L_EXT: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    V_HEAD_DIM: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)

    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_dv = tl.arange(0, BLOCK_DV)
    offs_l = tl.arange(0, L_EXT)

    cur_q_start = tl.load(qo_indptr + cur_batch)
    l_ext = tl.load(qo_indptr + cur_batch + 1) - cur_q_start
    mask_l = offs_l < l_ext

    # ---- (a) combine prefix splits (online logsumexp over active splits) ---
    seqlen = tl.load(kv_indptr + cur_batch + 1) - tl.load(kv_indptr + cur_batch)
    active = _active_splits(seqlen, num_splits, BLOCK_N)

    m = tl.zeros([L_EXT], dtype=tl.float32) - float("inf")
    l_acc = tl.zeros([L_EXT], dtype=tl.float32)
    acc = tl.zeros([L_EXT, BLOCK_DV], dtype=tl.float32)
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
    o_prefix = acc / l_acc[:, None]
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
    offs_ke = (cur_q_start + offs_l)[:, None] * stride_kebs + offs_d[None, :]
    ke = tl.load(
        K_Extend + offs_ke,
        mask=mask_l[:, None] & (offs_d[None, :] < HEAD_DIM),
        other=0.0,
    ).to(tl.float32)
    offs_ve = (cur_q_start + offs_l)[:, None] * stride_vebs + offs_dv[None, :]
    ve = tl.load(
        V_Extend + offs_ve,
        mask=mask_l[:, None] & (offs_dv[None, :] < V_HEAD_DIM),
        other=0.0,
    ).to(tl.float32)

    # scores[i,j] = q_i . k_j  (i query, j key)  -> [L_EXT, L_EXT]
    qk = tl.sum(q[:, None, :] * ke[None, :, :], 2) * sm_scale
    # causal among drafts: query i sees key j iff j <= i, and both valid
    causal = (offs_l[None, :] <= offs_l[:, None]) & mask_l[None, :] & mask_l[:, None]
    qk = tl.where(causal, qk, float("-inf"))
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
    ):
        self.h_q = h_q
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
        self._alloc(max_bs)

    def _alloc(self, max_bs):
        self.max_bs = max_bs
        # bf16 partials (halves scratch traffic vs fp32); lse stays fp32.
        self.att_out = torch.empty(
            (max_bs, self.h_q, MAX_N_SPLITS, self.l_pad, self.v_head_dim),
            dtype=torch.bfloat16,
            device=self.device,
        )
        self.att_lse = torch.empty(
            (max_bs, self.h_q, MAX_N_SPLITS, self.l_pad),
            dtype=torch.float32,
            device=self.device,
        )

    def grow_buffers(self, max_bs):
        if max_bs > self.max_bs:
            self._alloc(max_bs)

    def _num_splits(self, bs):
        budget = TARGET_PROGRAMS // max(1, bs * self.n_head_blocks)
        return max(1, min(MAX_N_SPLITS, budget))

    def _run_prefix_kernel(
        self,
        bs,
        num_splits,
        q_extend,
        k_buffer,
        v_buffer,
        qo_indptr,
        kv_indptr,
        kv_indices,
        sm_scale,
        k_scale,
        v_scale,
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
            self.att_out,
            self.att_lse,
            num_splits,
            q_extend.stride(0),
            q_extend.stride(1),
            k_buffer.stride(0),
            v_buffer.stride(0),
            self.att_out.stride(0),
            self.att_out.stride(1),
            self.att_out.stride(2),
            self.att_out.stride(3),
            self.att_lse.stride(0),
            self.att_lse.stride(1),
            self.att_lse.stride(2),
            H_Q=self.h_q,
            L_EXT=self.l_pad,
            BLOCK_H=self.block_h,
            NOPE_DIM=self.nope_dim,
            PE_DIM=self.pe_dim,
            V_HEAD_DIM=self.v_head_dim,
            BLOCK_DNOPE=triton.next_power_of_2(self.nope_dim),
            BLOCK_DPE=max(1, triton.next_power_of_2(self.pe_dim)),
            BLOCK_DV=triton.next_power_of_2(self.v_head_dim),
            BLOCK_N=self.block_n,
            num_warps=self.num_warps,
            num_stages=1,
            **_AMD_LAUNCH_KWARGS,
        )

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
    ):
        grid = (bs, self.h_q)
        _verify_mla_combine_stage2[grid](
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
            o_out.stride(0),
            o_out.stride(1),
            L_EXT=self.l_pad,
            HEAD_DIM=self.head_dim,
            V_HEAD_DIM=self.v_head_dim,
            BLOCK_DMODEL=triton.next_power_of_2(self.head_dim),
            BLOCK_DV=triton.next_power_of_2(self.v_head_dim),
            BLOCK_N=self.block_n,
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
        self._run_prefix_kernel(
            bs,
            num_splits,
            q_extend,
            k_buffer,
            v_buffer,
            qo_indptr,
            kv_indptr,
            kv_indices,
            sm_scale,
            k_scale,
            v_scale,
        )
        self._run_combine_kernel(
            bs,
            num_splits,
            q_extend,
            k_extend,
            v_extend,
            o_out,
            qo_indptr,
            kv_indptr,
            sm_scale,
        )
        return o_out


_VMLA_CACHE = {}


def _get_vmla(max_bs, h_q, head_dim, v_head_dim, l_ext, device):
    key = (h_q, head_dim, v_head_dim, l_ext, str(device))
    vk = _VMLA_CACHE.get(key)
    if vk is None:
        block_h, block_n, num_warps = block_config(head_dim)
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
    """Return True iff the grouped-head split-KV verify path can serve this
    exact problem with the same result as extend_attention_fwd. Conservative:
    anything not explicitly handled -> False -> caller falls back to the
    baseline.

    IMPORTANT: ``custom_mask`` is intentionally NOT inspected (its values can't
    be read inside a captured HIP graph without a host sync). The kernel always
    computes pure-causal attention, which equals the tree mask ONLY at
    speculative topk == 1. The caller therefore MUST gate enablement on topk == 1
    (TritonAttnBackend does: ``use_verify_shared_kv = ... and self.topk == 1``).
    At topk > 1 the tree is not causal and this path must stay disabled."""
    # No exotic features.
    if sinks is not None:
        return False
    if sliding_window_size is not None and sliding_window_size > 0:
        return False
    if logit_cap and logit_cap > 0:
        return False
    if xai_temperature_len is not None and xai_temperature_len > 0:
        return False
    if not is_causal:
        return False
    # q layout must be [tokens, H_Q, D]; head dims handled by power-of-2 pad.
    if q_extend.dim() != 3 or k_extend.dim() != 3 or v_extend.dim() != 3:
        return False
    # GQA group must divide evenly.
    h_q = q_extend.shape[1]
    h_kv = k_extend.shape[1]
    if h_kv == 0 or h_q % h_kv != 0:
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
):
    """
    Grouped-head drop-in for extend_attention_fwd on a topk==1 target-verify
    shape. Returns True if it ran (o_extend written), False if unsupported
    (caller falls back). Requires exactly one TP-local KV head.
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
    if k_extend.shape[1] != 1:
        return False
    if q_extend.shape[2] < v_extend.shape[2]:
        return False
    if kv_indices.numel() == 0:
        return False

    bs = qo_indptr.shape[0] - 1
    h_q = q_extend.shape[1]
    head_dim = q_extend.shape[2]
    v_head_dim = v_extend.shape[2]
    l_ext = int(max_len_extend)

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
    vk = _get_vmla(max_bs, h_q, head_dim, v_head_dim, l_ext, q_extend.device)
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
    )
    return True
