"""Split-KV (flash-decode) attention for EAGLE speculative *verify*.

Only valid when speculative ``topk == 1`` (the EAGLE tree reduces to a pure
causal chain); the caller gates on that. ``topk > 1`` trees fall back to
``extend_attention_fwd``.

On the Triton backend, EAGLE target-verify runs through the prefill
``extend_attention_fwd``, which loops the (long) prefix KV serially per
(sequence, head). With only a few draft-token queries, that leaves the GPU
memory system far under-utilized at long context. This kernel instead splits
the prefix KV across parallel programs (flash-decode style) and combines the
partials with a log-sum-exp merge, then handles the small causal draft-draft
block -- recovering memory bandwidth on the verify path.

Two Triton kernels:
  * ``_verify_prefix_stage1``: split-KV over the shared prefix. Applies the fp8
    dequant multipliers ``k_scale`` (on the QK score) and ``v_scale`` (on the
    prefix output), matching ``extend_attention_fwd``'s ``_fwd_kernel``
    (qk *= sm_scale * k_scale; acc += dot(p, v) * v_scale on the prefix loop;
    NO scaling on the draft-draft loop, whose K/V are the fresh bf16 draft
    tensors, not the fp8 pool). fp8 K/V buffers are handled by casting q to the
    buffer dtype before the dot (mirrors ``q.to(k.dtype)`` in the baseline).
  * ``_verify_combine_stage2``: combines the prefix splits (LSE merge) with the
    small causal draft-draft block and writes the output.

``verify_splitkv_fwd(...)`` takes the SAME positional args as
``extend_attention_fwd``; it runs the split-KV path when it can serve the case
bit-equivalently and returns True, otherwise returns False (doing nothing) so
the caller falls back to ``extend_attention_fwd``. Supported case: causal
(topk=1) verify with a constant per-sequence extend length, no sinks /
sliding-window / logit-cap / xai-temperature. Correctness is never violated.
"""

import torch
import triton
import triton.language as tl

from sglang.srt.utils import is_hip

_MIN_BLOCK_KV = 32

# AMD/CDNA-only Triton launch hints (waves_per_eu, matrix_instr_nonkdim); NVIDIA's
# Triton rejects these kwargs, so only pass them on ROCm. In production this kernel
# is dispatched only on AMD (see TritonAttnBackend); keeping it NV-safe lets the
# numerics test run on the CUDA CI lane.
_IS_HIP = is_hip()
_AMD_LAUNCH_KWARGS = {"waves_per_eu": 4, "matrix_instr_nonkdim": 16} if _IS_HIP else {}

# Block-size config keyed on head_dim. The (BLOCK_N, num_warps) tile that best
# hides latency depends on head_dim: at head_dim=256 (Qwen3 family) a narrower
# BLOCK_N with more warps wins, since the 256-wide QK/PV tiles are register
# heavy. head_dim=256 is the value validated on MI350X; other head dims use a
# conservative default. Block size affects PERFORMANCE only, never correctness
# (any valid block size produces the same result).
DEFAULT_N_SPLITS = 8
DEFAULT_BLOCK_N = 32
DEFAULT_NUM_WARPS = 4
_BLOCK_CONFIG = {
    # head_dim: (BLOCK_N, num_warps)
    256: (32, 4),
}


def block_config(head_dim):
    """Return (BLOCK_N, num_warps) for a head_dim; default for untuned dims."""
    return _BLOCK_CONFIG.get(head_dim, (DEFAULT_BLOCK_N, DEFAULT_NUM_WARPS))


# ---------------------------------------------------------------------------
# Adaptive N_SPLITS.
# ---------------------------------------------------------------------------
# The prefix split-KV stage launches a (bs, h_q, N_SPLITS) grid; each (b,h,s)
# program handles kv_len_per_split = cdiv(cdiv(seqlen, N_SPLITS), MIN)*MIN keys.
# A fixed N_SPLITS=16 over-splits short/mid contexts (each split does too little
# work -> launch + reduction overhead dominates) and under-splits very long ones
# (too few parallel waves to saturate the device, raising tail latency on the
# slow split). Mirror the decode kernel's intent (decode_attention.py
# get_num_kv_splits): pick the split count per-dispatch from the representative
# sequence length, growing gradually with seqlen and capped at MAX.
#
# CRITICAL: this must be computed from STATIC shapes only (no .item()/.cpu()
# sync), because the verify/draft-extend step runs inside a captured HIP graph
# where a device->host copy raises hipErrorStreamCaptureUnsupported. We use the
# average prefix length = kv_indices.shape[0] / bs, which is a pure python int
# from tensor shapes -- no device read. N_SPLITS is then a power of two so the
# stage2 reduction tile (tl.arange(0, N_SPLITS)) stays cheap.
#
# Split-count bounds (internal constants). MAX=16 is the MI350X cap: 32
# oversubscribes the device and regresses, per tuning.
ADAPTIVE_SPLITS = True
MAX_N_SPLITS = 16
MIN_N_SPLITS = 4


def choose_n_splits(avg_seqlen):
    """Pick N_SPLITS (power of two, in [MIN_N_SPLITS, MAX_N_SPLITS]) from the
    average prefix length. Tuned by the real-shape sweep (head_dim=256, BS*H_Q
    =128 base programs on ~132 CUs):

        ctx  <  4k -> 4   (short: extra splits add launch/reduction overhead)
        4k <= ctx < 8k -> 8   (sweet spot: best across 1k-16k in the sweep)
        ctx >= 8k      -> 16  (long: a few more splits help latency-bound tail)

    Never 32 (4096 grid blocks oversubscribes the device and regresses, per the
    sweep). Computed from a static shape (avg prefix = kv_indices.shape[0]/bs),
    so it is HIP-graph-capture safe (no device->host sync)."""
    if not ADAPTIVE_SPLITS:
        return DEFAULT_N_SPLITS
    s = int(avg_seqlen)
    if s < 4096:
        n = 4
    elif s < 8192:
        n = 8
    else:
        n = 16
    if n < MIN_N_SPLITS:
        n = MIN_N_SPLITS
    if n > MAX_N_SPLITS:
        n = MAX_N_SPLITS
    return n


@triton.jit
def _verify_prefix_stage1(
    Q,  # [extend_tokens, H_Q, D]
    K_Buffer,  # [pool_tokens, H_KV, D]
    V_Buffer,  # [pool_tokens, H_KV, Dv]
    sm_scale,
    k_scale,  # fp8 dequant multiplier for prefix K (1.0 if bf16)
    v_scale,  # fp8 dequant multiplier for prefix V (1.0 if bf16)
    qo_indptr,  # [BS+1] int32  -> rows of Q (draft queries)
    kv_indptr,  # [BS+1] int32  -> rows of kv_indices (prefix)
    kv_indices,  # [sum prefix] int64
    Att_Out,  # [BS, H_Q, N_SPLITS, L_EXT, Dv]  fp32
    Att_Lse,  # [BS, H_Q, N_SPLITS, L_EXT]      fp32
    stride_qbs,
    stride_qh,
    stride_buf_kbs,
    stride_buf_kh,
    stride_buf_vbs,
    stride_buf_vh,
    stride_ob,
    stride_oh,
    stride_os,
    stride_ol,
    stride_lb,
    stride_lh,
    stride_ls,
    kv_group_num: tl.constexpr,
    N_SPLITS: tl.constexpr,
    L_EXT: tl.constexpr,  # padded power-of-2 row tile (>= real l_ext)
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_N: tl.constexpr,
    MIN_BLOCK_KV: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    split_kv_id = tl.program_id(2)

    cur_kv_head = cur_head // kv_group_num

    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_dv = tl.arange(0, BLOCK_DV)
    offs_l = tl.arange(0, L_EXT)

    # real number of draft query tokens for this seq
    cur_q_start = tl.load(qo_indptr + cur_batch)
    l_ext = tl.load(qo_indptr + cur_batch + 1) - cur_q_start
    mask_l = offs_l < l_ext

    cur_batch_kv_start_idx = tl.load(kv_indptr + cur_batch)
    cur_batch_seq_len = tl.load(kv_indptr + cur_batch + 1) - cur_batch_kv_start_idx

    # split sizing identical to the decode kernel
    kv_len_per_split = (
        tl.cdiv(tl.cdiv(cur_batch_seq_len, N_SPLITS), MIN_BLOCK_KV) * MIN_BLOCK_KV
    )
    split_kv_start = kv_len_per_split * split_kv_id
    split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

    e_max = tl.zeros([L_EXT], dtype=tl.float32) - float("inf")
    e_sum = tl.zeros([L_EXT], dtype=tl.float32)
    acc = tl.zeros([L_EXT, BLOCK_DV], dtype=tl.float32)

    if split_kv_end > split_kv_start:
        # q tile: [L_EXT, D]
        offs_q = (
            (cur_q_start + offs_l)[:, None] * stride_qbs
            + cur_head * stride_qh
            + offs_d[None, :]
        )
        q = tl.load(Q + offs_q, mask=mask_l[:, None], other=0.0)
        q_k = q.to(K_Buffer.dtype.element_ty)

        base_offs_k = cur_kv_head * stride_buf_kh + offs_d[:, None]
        base_offs_v = cur_kv_head * stride_buf_vh + offs_dv[None, :]

        for start_n in tl.range(split_kv_start, split_kv_end, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            n_mask = offs_n < split_kv_end
            kv_loc = tl.load(
                kv_indices + cur_batch_kv_start_idx + offs_n,
                mask=n_mask,
                other=0,
            )
            # K block: [D, BLOCK_N]
            offs_buf_k = kv_loc[None, :] * stride_buf_kbs + base_offs_k
            k = tl.load(K_Buffer + offs_buf_k, mask=n_mask[None, :], other=0.0)
            qk = tl.dot(q_k, k)  # [L_EXT, BLOCK_N]
            qk *= sm_scale * k_scale  # fp8 dequant of prefix K (k_scale==1 if bf16)
            # NO causal mask: full prefix is visible to all draft tokens.
            qk = tl.where(n_mask[None, :], qk, float("-inf"))

            # V block: [BLOCK_N, Dv]
            offs_buf_v = kv_loc[:, None] * stride_buf_vbs + base_offs_v
            v = tl.load(V_Buffer + offs_buf_v, mask=n_mask[:, None], other=0.0)

            n_e_max = tl.maximum(tl.max(qk, 1), e_max)
            re_scale = tl.exp(e_max - n_e_max)
            p = tl.exp(qk - n_e_max[:, None])
            acc *= re_scale[:, None]
            acc += tl.dot(p.to(v.dtype), v)
            e_sum = e_sum * re_scale + tl.sum(p, 1)
            e_max = n_e_max

        # fp8 dequant of prefix V: scale the accumulated (pre-normalised) output.
        acc *= v_scale

        offs_o = (
            cur_batch * stride_ob
            + cur_head * stride_oh
            + split_kv_id * stride_os
            + offs_l[:, None] * stride_ol
            + offs_dv[None, :]
        )
        tl.store(Att_Out + offs_o, acc / e_sum[:, None], mask=mask_l[:, None])

        offs_lse = (
            cur_batch * stride_lb
            + cur_head * stride_lh
            + split_kv_id * stride_ls
            + offs_l
        )
        tl.store(Att_Lse + offs_lse, e_max + tl.log(e_sum), mask=mask_l)
    else:
        # split did not run: write a sentinel lse so stage2 can ignore it.
        offs_lse = (
            cur_batch * stride_lb
            + cur_head * stride_lh
            + split_kv_id * stride_ls
            + offs_l
        )
        tl.store(
            Att_Lse + offs_lse,
            tl.zeros([L_EXT], tl.float32) - float("inf"),
            mask=mask_l,
        )


@triton.jit
def _verify_combine_stage2(
    Att_Out,  # [BS, H_Q, N_SPLITS, L_EXT, Dv]  fp32
    Att_Lse,  # [BS, H_Q, N_SPLITS, L_EXT]      fp32
    Q,  # [extend_tokens, H_Q, D]   (draft queries)
    K_Extend,  # [extend_tokens, H_KV, D]
    V_Extend,  # [extend_tokens, H_KV, Dv]
    O_Out,  # [extend_tokens, H_Q, Dv]  (final, written)
    sm_scale,
    qo_indptr,  # [BS+1] int32
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
    stride_keh,
    stride_vebs,
    stride_veh,
    stride_oobs,
    stride_ooh,
    kv_group_num: tl.constexpr,
    N_SPLITS: tl.constexpr,
    L_EXT: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DV: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    cur_kv_head = cur_head // kv_group_num

    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_dv = tl.arange(0, BLOCK_DV)
    offs_l = tl.arange(0, L_EXT)
    offs_s = tl.arange(0, N_SPLITS)

    cur_q_start = tl.load(qo_indptr + cur_batch)
    l_ext = tl.load(qo_indptr + cur_batch + 1) - cur_q_start
    mask_l = offs_l < l_ext

    # ---- (a) combine prefix splits (logsumexp) ----------------------------
    # lse: [N_SPLITS, L_EXT]
    offs_lse = (
        cur_batch * stride_lb
        + cur_head * stride_lh
        + offs_s[:, None] * stride_ls
        + offs_l[None, :]
    )
    lse = tl.load(offs_lse + Att_Lse)  # [N_SPLITS, L_EXT]
    m_p = tl.max(lse, 0)  # [L_EXT]
    w = tl.exp(lse - m_p[None, :])  # [N_SPLITS, L_EXT]; -inf->0
    denom_p = tl.sum(w, 0)  # [L_EXT]

    # weighted-sum of partial outputs: o_prefix[L_EXT, Dv]
    # Att_Out[b,h,s,l,dv]
    offs_ao = (
        cur_batch * stride_ob
        + cur_head * stride_oh
        + offs_s[:, None, None] * stride_os
        + offs_l[None, :, None] * stride_ol
        + offs_dv[None, None, :]
    )
    ao = tl.load(offs_ao + Att_Out)  # [N_SPLITS, L_EXT, Dv]
    o_prefix = tl.sum(ao * w[:, :, None], 0)  # [L_EXT, Dv]
    o_prefix = o_prefix / denom_p[:, None]
    lse_prefix = m_p + tl.log(denom_p)  # [L_EXT]

    # ---- (b) draft-draft causal attention (L_EXT x L_EXT) -----------------
    # load draft queries [L_EXT, D], draft K/V [L_EXT, D]/[L_EXT, Dv]
    offs_q = (
        (cur_q_start + offs_l)[:, None] * stride_qbs
        + cur_head * stride_qh
        + offs_d[None, :]
    )
    q = tl.load(Q + offs_q, mask=mask_l[:, None], other=0.0).to(tl.float32)

    offs_ke = (
        (cur_q_start + offs_l)[:, None] * stride_kebs
        + cur_kv_head * stride_keh
        + offs_d[None, :]
    )
    ke = tl.load(K_Extend + offs_ke, mask=mask_l[:, None], other=0.0).to(tl.float32)
    offs_ve = (
        (cur_q_start + offs_l)[:, None] * stride_vebs
        + cur_kv_head * stride_veh
        + offs_dv[None, :]
    )
    ve = tl.load(V_Extend + offs_ve, mask=mask_l[:, None], other=0.0).to(tl.float32)

    # scores[i,j] = q_i . k_j  (i query, j key)  -> [L_EXT, L_EXT]
    qk = tl.sum(q[:, None, :] * ke[None, :, :], 2) * sm_scale
    # causal among drafts: query i sees key j iff j <= i, and both valid
    causal = (offs_l[None, :] <= offs_l[:, None]) & mask_l[None, :] & mask_l[:, None]
    qk = tl.where(causal, qk, float("-inf"))
    m_d = tl.max(qk, 1)  # [L_EXT]
    pd = tl.exp(qk - m_d[:, None])  # [L_EXT, L_EXT]
    denom_d = tl.sum(pd, 1)  # [L_EXT]
    o_draft = tl.sum(pd[:, :, None] * ve[None, :, :], 1)  # [L_EXT, Dv]
    o_draft = o_draft / denom_d[:, None]
    lse_draft = m_d + tl.log(denom_d)  # [L_EXT]

    # ---- (c) final LSE merge (prefix vs draft) ----------------------------
    m = tl.maximum(lse_prefix, lse_draft)
    wp = tl.exp(lse_prefix - m)
    wd = tl.exp(lse_draft - m)
    o = (o_prefix * wp[:, None] + o_draft * wd[:, None]) / (wp + wd)[:, None]

    offs_oo = (
        (cur_q_start + offs_l)[:, None] * stride_oobs
        + cur_head * stride_ooh
        + offs_dv[None, :]
    )
    tl.store(O_Out + offs_oo, o.to(O_Out.dtype.element_ty), mask=mask_l[:, None])


class VerifySplitKV:
    """Pre-allocates scratch buffers for a problem shape and runs the split-KV
    verify attention end to end (two Triton launches: prefix split-KV + fused
    combine/draft/merge). Buffers are sized by ``max_bs`` (constant for the
    server lifetime) and reused for every batch size <= max_bs, so their
    addresses stay fixed (CUDA/HIP-graph safe) and GPU memory does not grow per
    batch size. The kernel grid uses the actual per-call bs (<= max_bs)."""

    def __init__(
        self,
        max_bs,
        h_q,
        h_kv,
        head_dim,
        v_head_dim,
        l_ext,
        device="cuda",
        n_splits=DEFAULT_N_SPLITS,
        block_n=DEFAULT_BLOCK_N,
        num_warps=DEFAULT_NUM_WARPS,
    ):
        self.h_q = h_q
        self.h_kv = h_kv
        self.group = h_q // h_kv
        self.head_dim = head_dim
        self.v_head_dim = v_head_dim
        self.l_ext = l_ext  # real draft tokens per seq (fixed == 4)
        self.l_pad = triton.next_power_of_2(l_ext)
        self.device = device
        self.n_splits = n_splits
        self.block_n = block_n
        self.num_warps = num_warps
        self._alloc(max_bs)

    def _alloc(self, max_bs):
        # prefix split partials (fp32), sized for the maximum batch size.
        self.max_bs = max_bs
        self.att_out = torch.empty(
            (max_bs, self.h_q, self.n_splits, self.l_pad, self.v_head_dim),
            dtype=torch.float32,
            device=self.device,
        )
        self.att_lse = torch.empty(
            (max_bs, self.h_q, self.n_splits, self.l_pad),
            dtype=torch.float32,
            device=self.device,
        )

    def grow_buffers(self, max_bs):
        if max_bs > self.max_bs:
            self._alloc(max_bs)

    def _run_prefix_kernel(
        self,
        bs,
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
        grid = (bs, self.h_q, self.n_splits)
        _verify_prefix_stage1[grid](
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
            q_extend.stride(0),
            q_extend.stride(1),
            k_buffer.stride(0),
            k_buffer.stride(1),
            v_buffer.stride(0),
            v_buffer.stride(1),
            self.att_out.stride(0),
            self.att_out.stride(1),
            self.att_out.stride(2),
            self.att_out.stride(3),
            self.att_lse.stride(0),
            self.att_lse.stride(1),
            self.att_lse.stride(2),
            kv_group_num=self.group,
            N_SPLITS=self.n_splits,
            L_EXT=self.l_pad,
            BLOCK_DMODEL=triton.next_power_of_2(self.head_dim),
            BLOCK_DV=triton.next_power_of_2(self.v_head_dim),
            BLOCK_N=self.block_n,
            MIN_BLOCK_KV=_MIN_BLOCK_KV,
            num_warps=self.num_warps,
            num_stages=1,
            **_AMD_LAUNCH_KWARGS,
        )

    def _run_combine_kernel(
        self, bs, q_extend, k_extend, v_extend, o_out, qo_indptr, sm_scale
    ):
        grid = (bs, self.h_q)
        _verify_combine_stage2[grid](
            self.att_out,
            self.att_lse,
            q_extend,
            k_extend,
            v_extend,
            o_out,
            sm_scale,
            qo_indptr,
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
            k_extend.stride(1),
            v_extend.stride(0),
            v_extend.stride(1),
            o_out.stride(0),
            o_out.stride(1),
            kv_group_num=self.group,
            N_SPLITS=self.n_splits,
            L_EXT=self.l_pad,
            BLOCK_DMODEL=triton.next_power_of_2(self.head_dim),
            BLOCK_DV=triton.next_power_of_2(self.v_head_dim),
            num_warps=1,
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
        # actual batch size for this call (<= max_bs); the grid uses it while the
        # scratch buffers stay max_bs-sized (only the first bs slices are touched).
        bs = qo_indptr.shape[0] - 1
        # 1. prefix split-KV
        self._run_prefix_kernel(
            bs,
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
        # 2+3+4. fused combine + draft-draft + merge
        self._run_combine_kernel(
            bs,
            q_extend,
            k_extend,
            v_extend,
            o_out,
            qo_indptr,
            sm_scale,
        )
        return o_out


# ---------------------------------------------------------------------------
# Live-server dispatch entry.
# ---------------------------------------------------------------------------
# Cache one VerifySplitKV instance per (h_q, h_kv, head_dim, v_head_dim, l_ext,
# device, n_splits) shape -- NOT keyed on the dynamic batch size. Buffers are
# sized by the stable max_bs (grown only if a larger one is ever requested), so
# a single instance serves every batch size: addresses stay fixed (graph-safe)
# and GPU memory does not grow per batch size.
_VK_CACHE = {}


def _get_vk(
    max_bs, h_q, h_kv, head_dim, v_head_dim, l_ext, device, n_splits=DEFAULT_N_SPLITS
):
    key = (h_q, h_kv, head_dim, v_head_dim, l_ext, str(device), n_splits)
    vk = _VK_CACHE.get(key)
    if vk is None:
        block_n, num_warps = block_config(head_dim)
        vk = VerifySplitKV(
            max_bs,
            h_q,
            h_kv,
            head_dim,
            v_head_dim,
            l_ext,
            device=device,
            n_splits=n_splits,
            block_n=block_n,
            num_warps=num_warps,
        )
        _VK_CACHE[key] = vk
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
    """Return True iff the split-KV verify path can serve this exact problem
    with the same result as extend_attention_fwd. Conservative: anything not
    explicitly handled -> False -> caller falls back to the baseline.

    IMPORTANT: ``custom_mask`` is intentionally NOT inspected (its values can't
    be read inside a captured HIP graph without a host sync). The kernel always
    computes pure-causal attention, which equals the tree mask ONLY at
    speculative topk == 1. The caller therefore MUST gate enablement on topk == 1
    (TritonAttnBackend does: ``use_verify_splitkv = ... and self.topk == 1``).
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


def verify_splitkv_fwd(
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
    """Drop-in for extend_attention_fwd on the EAGLE target-verify (topk=1)
    shape. Returns True if it ran (o_extend written), False if the case is
    unsupported and the caller must fall back to extend_attention_fwd.

    ``max_bs`` (optional) is the stable maximum batch size used to size the
    cached scratch buffers; the backend passes its req_to_token_pool size. If
    omitted it defaults to this call's bs.

    Arg order mirrors extend_attention_fwd exactly so the call site is a
    one-line swap.
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

    bs = qo_indptr.shape[0] - 1
    h_q = q_extend.shape[1]
    h_kv = k_extend.shape[1]
    head_dim = q_extend.shape[2]
    v_head_dim = v_extend.shape[2]
    l_ext = int(max_len_extend)

    if sm_scale is None:
        sm_scale = 1.0 / (head_dim**0.5)
    # k_scale/v_scale may be float or 0-d tensor; coerce to python float.
    try:
        k_scale = float(k_scale)
    except (TypeError, ValueError):
        k_scale = 1.0
    try:
        v_scale = float(v_scale)
    except (TypeError, ValueError):
        v_scale = 1.0

    # Adaptive split count from the average prefix length. This is a
    # pure-shape derivation (kv_indices.shape[0] / bs) -- no device->host sync,
    # so it is safe inside a captured HIP graph. The whole batch shares one
    # N_SPLITS (the grid dim must be a launch constexpr); the per-split kernel
    # logic still clamps each split's [start,end) to that seq's real length, so
    # mixed-length batches stay correct -- shorter seqs simply write fewer
    # active splits (the rest emit the -inf lse sentinel, ignored in stage2).
    avg_seqlen = kv_indices.shape[0] / max(1, bs)
    n_splits = choose_n_splits(avg_seqlen)

    # Size scratch by the stable max_bs (backend passes req_to_token_pool size);
    # fall back to this call's bs if not provided / smaller.
    if max_bs is None or max_bs < bs:
        max_bs = bs
    vk = _get_vk(
        max_bs,
        h_q,
        h_kv,
        head_dim,
        v_head_dim,
        l_ext,
        q_extend.device,
        n_splits=n_splits,
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
    )

    return True
