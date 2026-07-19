from typing import TypedDict

import torch
import triton
import triton.language as tl

from sglang.srt.utils import is_cuda

PAD_SLOT_ID = -1


class SconvDecodeMetadata(TypedDict):
    cache_mask: torch.Tensor
    safe_idx: torch.Tensor
    cu: torch.Tensor
    si: torch.Tensor


class SconvExtendMetadata(TypedDict):
    cache_mask: torch.Tensor
    safe_idx: torch.Tensor
    cu: torch.Tensor
    si: torch.Tensor


CHUNK_SIZE = 64

# ---------------------------------------------------------------------------
# Causal conv1d forward with cache-loaded prefix (Triton).
#
# Replaces the former Helion kernel: Helion lowers through torch.fx/dynamo, so the
# dynamic extend batch B becomes an *unbacked* symint and an `if is_decode` mask
# branch tripped `GuardOnDataDependentSymNode: Eq(u1, 1)`.  Triton has no symbolic
# -shape guard machinery: `IS_DECODE: tl.constexpr` resolves the mask branch at JIT
# compile time, so the decode specialization compiles with ZERO mask load/multiply
# and the extend specialization keeps the mask multiply — both guard-free.
# ---------------------------------------------------------------------------


def _conv_prefix_autotune_configs() -> list[triton.Config]:
    """Block-size configs for the prefix conv. D=256, W small => memory-bound; favor
    coalesced D loads (D is the contiguous axis; the token axis is strided).

    The x window is loaded ONCE per tile (BLOCK_T + W - 1 unique rows) and the W taps
    are static-offset slices within it, so larger BLOCK_T both amortizes the per-tile
    metadata loads and maximizes window reuse — hence the wide BLOCK_T sweep for the
    large-T extend regime, while small BLOCK_T configs cover decode (T=B is moderate,
    so more channel blocks fill the GPU)."""
    configs = []
    for block_t in (1, 2, 4, 8, 16, 32, 64, 128):
        for block_d in (128, 256):
            # num_warps scaled to the tile so small tiles don't over-subscribe and
            # large tiles get enough parallelism; a couple of num_stages each.
            tile = block_t * block_d
            if tile <= 256:
                warps_opts = (2, 4)
            elif tile <= 4096:
                warps_opts = (4, 8)
            else:
                warps_opts = (8,)
            for num_warps in warps_opts:
                for num_stages in (2, 3, 4):
                    configs.append(
                        triton.Config(
                            {"BLOCK_T": block_t, "BLOCK_D": block_d},
                            num_warps=num_warps,
                            num_stages=num_stages,
                        )
                    )
    return configs


@triton.autotune(configs=_conv_prefix_autotune_configs(), key=["D", "W", "t_bucket"])
@triton.jit
def _causal_conv1d_fwd_with_prefix_kernel(
    x,  # [T, D]
    sconv_cache,  # [max_slots, W-1, D]
    safe_idx,  # [num_seqs] int64 — cache slot per sequence
    cache_mask,  # [num_seqs, 1, 1] RAW metadata (bool 0/1); read only when not IS_DECODE
    weight,  # [D, W]
    cu_seqlens,  # [num_seqs + 1] int64 — packed sequence start offsets
    seq_idx,  # [T] int32 — which sequence each packed token belongs to
    y,  # [T, D] — contiguous output
    t_bucket,  # AUTOTUNE-KEY ONLY (coarse token-count regime); never read in the body.
    stride_x_t,
    stride_x_d,
    stride_cache_slot,
    stride_cache_w,
    stride_cache_d,
    stride_cm,  # cache_mask dim-0 (per-sequence) stride
    stride_weight_d,
    stride_weight_w,
    stride_y_t,
    stride_y_d,
    T,
    D,
    USE_SILU: tl.constexpr,
    USE_RESIDUAL: tl.constexpr,
    IS_DECODE: tl.constexpr,
    W: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Depthwise causal conv1d over a packed [T, D] token stream, with the W-1 prefix
    taps gathered directly from sconv_cache (no intermediate prefix tensor).

    For packed token t in sequence s (bos = cu_seqlens[s], slot = safe_idx[s]) and
    tap iw in 0..W-1:  shifted = t - (W-1) + iw
        shifted >= bos (and < T)         -> tap = x[shifted, d]
        shifted <  bos, pp = shifted-bos+(W-1) in [0, W-1)
                                          -> tap = sconv_cache[slot, pp, d]
                                             (* cache_mask[s] when not IS_DECODE)
        else                              -> tap = 0
    out[t, d] = act(sum_iw tap*weight[d, iw]) [+ x[t, d] if residual], fp32 accumulate.

    MEMORY-BOUND: this conv is dominated by HBM x traffic.  The W taps for a token tile
    read overlapping x rows (positions t0-(W-1) .. t0+BLOCK_T-1).  We warm the whole
    window with one streaming load and pin it in L2 across the W taps via
    `eviction_policy="evict_last"` on the per-tap x loads, so the W overlapping reads
    hit L2 instead of HBM.  The current-token x (tap iw=W-1) is loaded once into
    `x_cur` and reused for both that tap and the residual add (no duplicate load).
    Only the boundary prefix taps touch sconv_cache.

    IS_DECODE (constexpr): when True, the prefix-mask multiply is omitted entirely
    (the decode mask is all-ones, so the multiply was a bit-exact no-op).
    """
    t_off = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
    d_off = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
    t_mask = t_off < T
    d_mask = d_off < D
    td_mask = t_mask[:, None] & d_mask[None, :]

    # Per-token sequence id, sequence start, and cache slot.
    si = tl.load(seq_idx + t_off, mask=t_mask, other=0).to(tl.int64)
    bos = tl.load(cu_seqlens + si, mask=t_mask, other=0).to(tl.int64)
    slot = tl.load(safe_idx + si, mask=t_mask, other=0).to(tl.int64)

    # Per-token prefix mask (extend only). Loaded from the RAW [num_seqs,1,1] bool
    # metadata (indexed on dim0 by si) and cast IN-KERNEL to x's dtype — bool 0/1 ->
    # bf16 0.0/1.0, bit-identical to the old host-side `cache_mask.to(x.dtype)` cast,
    # but with one fewer kernel launch.  Under IS_DECODE this load is constexpr-pruned.
    if not IS_DECODE:
        m_val = tl.load(cache_mask + si * stride_cm, mask=t_mask, other=0).to(
            x.dtype.element_ty
        )

    x_row_base = x + d_off[None, :] * stride_x_d  # [1, BLOCK_D] partial
    weight_base = weight + d_off * stride_weight_d  # [BLOCK_D]
    acc = tl.zeros([BLOCK_T, BLOCK_D], dtype=tl.float32)

    # Current-token x (tap iw = W-1, always in-sequence for valid t): load ONCE,
    # reused for the last tap and the residual add.
    x_cur = tl.load(
        x_row_base + t_off[:, None] * stride_x_t,
        mask=td_mask,
        other=0,
    )

    for iw in tl.static_range(W):
        shifted = t_off - (W - 1) + iw  # [BLOCK_T]

        if iw == W - 1:
            # Current token: in_x is always true for valid t (shifted == t_off).
            x_val = x_cur
        else:
            # Earlier in-sequence x rows.  These overlap heavily across the tile and
            # across taps; evict_last keeps them resident in L2 so the W-1 history
            # taps coalesce onto cached lines instead of re-streaming from HBM.
            in_x = (shifted >= bos) & (shifted < T)
            x_val = tl.load(
                x_row_base + shifted[:, None] * stride_x_t,
                mask=in_x[:, None] & d_mask[None, :],
                other=0,
                eviction_policy="evict_last",
            )

        # prefix tap (fused gather): positions before bos, pp in [0, W-1).
        # (The last tap, iw=W-1, never hits the prefix: shifted==t_off>=bos.)
        if iw == W - 1:
            tap = x_val.to(tl.float32)
        else:
            prefix_pos = shifted - bos + (W - 1)
            in_prefix = (shifted < bos) & (prefix_pos >= 0)
            p_val = tl.load(
                sconv_cache
                + slot[:, None] * stride_cache_slot
                + prefix_pos[:, None] * stride_cache_w
                + d_off[None, :] * stride_cache_d,
                mask=in_prefix[:, None] & d_mask[None, :],
                other=0,
            )
            if not IS_DECODE:
                # bf16 mul by the 0/1 mask == pre-kernel `sconv_cache[safe_idx]*cache_mask`.
                p_val = p_val * m_val[:, None]
            # bf16 add (in_x / in_prefix mutually exclusive => one operand is 0), then
            # cast to fp32 — bit-identical to the v3 Helion `(x_val + p_val).to(f32)`.
            tap = (x_val + p_val).to(tl.float32)

        w_val = tl.load(weight_base + iw * stride_weight_w, mask=d_mask, other=0).to(
            tl.float32
        )
        acc += tap * w_val[None, :]

    if USE_SILU:
        acc = acc * tl.sigmoid(acc)

    if USE_RESIDUAL:
        acc += x_cur.to(tl.float32)  # reuse the once-loaded current-token x

    tl.store(
        y + t_off[:, None] * stride_y_t + d_off[None, :] * stride_y_d,
        acc.to(y.dtype.element_ty),
        mask=td_mask,
    )


# todo(horace): Shift this to be precomputed data
def _seq_idx_from_cu_seqlens(cu_seqlens: torch.Tensor, T: int) -> torch.Tensor:
    """Compute seq_idx from cu_seqlens: for each position, which sequence it belongs to."""
    t = torch.arange(T, dtype=torch.int64, device=cu_seqlens.device)
    # Clamp to [0, num_seqs-1] to prevent OOB when cu_seqlens doesn't span all T
    # tokens (e.g. during CUDA graph capture warmup with dummy zero-length sequences).
    num_seqs = cu_seqlens.shape[0] - 1
    return (
        (torch.searchsorted(cu_seqlens, t, side="right") - 1)
        .clamp(max=num_seqs - 1)
        .to(torch.int32)
    )


@triton.jit
def _fused_decode_metadata_kernel(
    cache_indices_ptr,  # [B] int
    query_start_loc_ptr,  # [B+1] int32 out
    has_initial_state_ptr,  # [B] bool out
    cache_mask_ptr,  # [B] bool out (callers view it [B,1,1])
    safe_idx_ptr,  # [B] int64 out
    cu_ptr,  # [B+1] int64 out
    si_ptr,  # [B] int32 out
    B,
    BLOCK: tl.constexpr,
):
    """All decode sconv metadata in one launch (see fused_decode_sconv_metadata).

    Decode invariants baked in: every token is its own length-1 sequence
    (query_start_loc = cu = arange, si = arange) and always has initial state
    (has_initial_state = ones), so cache_mask reduces to cache_indices != PAD.
    """
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask_b1 = offs < B + 1
    mask_b = offs < B
    tl.store(query_start_loc_ptr + offs, offs.to(tl.int32), mask=mask_b1)
    tl.store(cu_ptr + offs, offs.to(tl.int64), mask=mask_b1)
    tl.store(si_ptr + offs, offs.to(tl.int32), mask=mask_b)
    ones = tl.full([BLOCK], 1, tl.int1)
    tl.store(has_initial_state_ptr + offs, ones, mask=mask_b)
    ci = tl.load(cache_indices_ptr + offs, mask=mask_b, other=-1)
    tl.store(cache_mask_ptr + offs, ci != -1, mask=mask_b)  # PAD_SLOT_ID = -1
    tl.store(safe_idx_ptr + offs, tl.maximum(ci, 0).to(tl.int64), mask=mask_b)


def fused_decode_sconv_metadata(
    B: int, cache_indices: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, SconvDecodeMetadata]:
    """Single-launch replacement for the decode metadata prep: the two arange calls,
    ones, `!= PAD`, `&`, `clamp` and `.long()` that
    ``precompute_helion_decode_metadata`` (+ its callers) issued as ~7 tiny
    elementwise kernels. Returns
    ``(query_start_loc, has_initial_state, SconvDecodeMetadata)`` with tensors
    bit-identical to the unfused path.
    """
    assert cache_indices.shape[0] == B and cache_indices.stride(0) == 1
    device = cache_indices.device
    query_start_loc = torch.empty(B + 1, dtype=torch.int32, device=device)
    has_initial_state = torch.empty(B, dtype=torch.bool, device=device)
    cache_mask = torch.empty((B, 1, 1), dtype=torch.bool, device=device)
    safe_idx = torch.empty(B, dtype=torch.int64, device=device)
    cu = torch.empty(B + 1, dtype=torch.int64, device=device)
    si = torch.empty(B, dtype=torch.int32, device=device)
    BLOCK = 1024
    _fused_decode_metadata_kernel[(triton.cdiv(B + 1, BLOCK),)](
        cache_indices,
        query_start_loc,
        has_initial_state,
        cache_mask,
        safe_idx,
        cu,
        si,
        B,
        BLOCK=BLOCK,
    )
    return (
        query_start_loc,
        has_initial_state,
        SconvDecodeMetadata(cache_mask=cache_mask, safe_idx=safe_idx, cu=cu, si=si),
    )


def precompute_helion_decode_metadata(
    B: int,
    W: int,
    cache_indices: torch.Tensor,
    has_initial_state: torch.Tensor,
) -> SconvDecodeMetadata:
    """Precompute metadata for the helion decode path. Call once, reuse across layers."""
    device = cache_indices.device
    valid = cache_indices != PAD_SLOT_ID
    cache_mask = (has_initial_state & valid)[:, None, None]  # [B, 1, 1]
    safe_idx = cache_indices.clamp(min=0).long()
    # Each sequence has exactly 1 token in the packed [1, B, D] layout
    cu = torch.arange(B + 1, dtype=torch.int64, device=device)
    si = torch.arange(B, dtype=torch.int32, device=device)
    return SconvDecodeMetadata(
        cache_mask=cache_mask,
        safe_idx=safe_idx,
        cu=cu,
        si=si,
    )


# has_initial_state variants for the fused extend metadata kernel.
HIS_ZEROS = 0  # boundary-KV draft extend: force conv to run fresh
HIS_PREFIX = 1  # extend_prefix_lens > 0
HIS_SEQ_MINUS_EXT = 2  # (seq_lens[:B] - extend_seq_lens) > 0 (draft_extend_v2 capture)
HIS_ONES = 3  # target_verify: always has initial state

# The single-tile local cumsum bounds the fused path; larger batches fall back
# to the unfused op sequence.
_FUSED_EXTEND_MAX_B = 1023


@triton.jit
def _fused_extend_metadata_kernel(
    cache_indices_ptr,  # [B] int
    extend_seq_lens_ptr,  # [B]; unused when IS_VERIFY
    his_src_ptr,  # [>=B]: prefix_lens (HIS_PREFIX) / seq_lens (HIS_SEQ_MINUS_EXT)
    query_start_loc_ptr,  # [B+1] int32 out
    has_initial_state_ptr,  # [B] bool out
    cache_mask_ptr,  # [B] bool out (callers view it [B,1,1])
    safe_idx_ptr,  # [B] int64 out
    cu_ptr,  # [B+1] int64 out
    si_ptr,  # [T] int32 out
    B,
    T,
    draft_token_num,  # verify only
    IS_VERIFY: tl.constexpr,
    HIS_MODE: tl.constexpr,
    BLOCK_B: tl.constexpr,  # pow2 >= B+1
    BLOCK_T: tl.constexpr,
):
    """All extend sconv metadata in one launch (see fused_extend_sconv_metadata).

    Grid is (1 + cdiv(T, BLOCK_T),): program 0 writes the [B]-sized outputs,
    programs 1.. fill their si tile. There is no cross-program dependency:
    every program rebuilds cu from extend_seq_lens with a local single-tile
    cumsum (B is small), so no barrier or second launch is needed.
    """
    pid = tl.program_id(0)
    offs_b = tl.arange(0, BLOCK_B)
    mask_b = offs_b < B
    mask_b1 = offs_b < B + 1

    if IS_VERIFY:
        # Uniform draft_token_num tokens per request: cu is a strided arange.
        cu_local = offs_b.to(tl.int64) * draft_token_num
    else:
        # cu_local[i] = sum(extend_seq_lens[:i]); inclusive cumsum of the
        # one-right-shifted lens gives the exclusive prefix sum with cu[0]=0.
        lens = tl.load(
            extend_seq_lens_ptr + offs_b - 1,
            mask=mask_b1 & (offs_b > 0),
            other=0,
        ).to(tl.int64)
        cu_local = tl.cumsum(lens, axis=0)

    if pid == 0:
        tl.store(query_start_loc_ptr + offs_b, cu_local.to(tl.int32), mask=mask_b1)
        tl.store(cu_ptr + offs_b, cu_local, mask=mask_b1)
        if HIS_MODE == 0:  # HIS_ZEROS
            his = offs_b < 0
        elif HIS_MODE == 1:  # HIS_PREFIX
            his = tl.load(his_src_ptr + offs_b, mask=mask_b, other=0).to(tl.int64) > 0
        elif HIS_MODE == 2:  # HIS_SEQ_MINUS_EXT
            seq = tl.load(his_src_ptr + offs_b, mask=mask_b, other=0).to(tl.int64)
            ext = tl.load(extend_seq_lens_ptr + offs_b, mask=mask_b, other=0).to(
                tl.int64
            )
            his = (seq - ext) > 0
        else:  # HIS_ONES
            his = offs_b >= 0
        tl.store(has_initial_state_ptr + offs_b, his, mask=mask_b)
        ci = tl.load(cache_indices_ptr + offs_b, mask=mask_b, other=-1)
        tl.store(cache_mask_ptr + offs_b, his & (ci != -1), mask=mask_b)  # PAD = -1
        tl.store(safe_idx_ptr + offs_b, tl.maximum(ci, 0).to(tl.int64), mask=mask_b)
    else:
        offs_t = (pid - 1) * BLOCK_T + tl.arange(0, BLOCK_T)
        mask_t = offs_t < T
        if IS_VERIFY:
            si = tl.minimum(offs_t // draft_token_num, B - 1)
        else:
            # si[t] = #{s in 1..B : cu[s] <= t}, clamped to B-1 -- identical to
            # searchsorted(cu, t, right) - 1 then clamp (cu[0] = 0 <= t always),
            # including the last-index tie-break for zero-length sequences and
            # the clamp when cu does not span T (dummy capture sequences).
            bounds = tl.where(mask_b1 & (offs_b > 0), cu_local, 9223372036854775807)
            cnt = tl.sum(
                (offs_t[:, None].to(tl.int64) >= bounds[None, :]).to(tl.int32), axis=1
            )
            si = tl.minimum(cnt, B - 1)
        tl.store(si_ptr + offs_t, si.to(tl.int32), mask=mask_t)


def fused_extend_sconv_metadata(
    *,
    B: int,
    T: int,
    cache_indices: torch.Tensor,
    his_mode: int,
    extend_seq_lens: torch.Tensor | None = None,
    his_src: torch.Tensor | None = None,
    draft_token_num: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, SconvExtendMetadata] | None:
    """Single-launch replacement for the extend metadata prep: the
    zeros + cumsum(+scan-init) + slice-copy + compare chain of
    ``_prepare_extend_common_metadata`` plus the != PAD, &, clamp, long, to,
    arange, searchsorted, clamp, int32 chain of
    ``precompute_helion_extend_metadata`` (~10-14 tiny kernels, re-issued per
    owning sconv instance -- and per de-tied draft step under draft_extend_v2).
    Returns ``(query_start_loc, has_initial_state, SconvExtendMetadata)`` with
    tensors bit-identical to the unfused path, or None when the shape falls
    outside the fused kernel's single-tile bound (caller runs unfused).

    ``his_mode`` selects the has_initial_state source: HIS_ZEROS (boundary-KV
    draft extend), HIS_PREFIX (``his_src`` = extend_prefix_lens), HIS_SEQ_MINUS_EXT
    (``his_src`` = seq_lens), HIS_ONES (target_verify; ``draft_token_num`` set,
    ``extend_seq_lens`` unused).
    """
    if B > _FUSED_EXTEND_MAX_B or not cache_indices.is_cuda:
        return None
    assert cache_indices.shape[0] >= B and cache_indices.stride(0) == 1
    is_verify = his_mode == HIS_ONES
    if is_verify:
        assert draft_token_num is not None
    else:
        assert extend_seq_lens is not None and extend_seq_lens.stride(0) == 1
    device = cache_indices.device
    query_start_loc = torch.empty(B + 1, dtype=torch.int32, device=device)
    has_initial_state = torch.empty(B, dtype=torch.bool, device=device)
    cache_mask = torch.empty((B, 1, 1), dtype=torch.bool, device=device)
    safe_idx = torch.empty(B, dtype=torch.int64, device=device)
    cu = torch.empty(B + 1, dtype=torch.int64, device=device)
    si = torch.empty(T, dtype=torch.int32, device=device)
    BLOCK_T = 256
    dummy = cache_indices  # never dereferenced thanks to masks/constexpr
    _fused_extend_metadata_kernel[(1 + triton.cdiv(T, BLOCK_T),)](
        cache_indices,
        extend_seq_lens if extend_seq_lens is not None else dummy,
        his_src if his_src is not None else dummy,
        query_start_loc,
        has_initial_state,
        cache_mask,
        safe_idx,
        cu,
        si,
        B,
        T,
        draft_token_num if draft_token_num is not None else 1,
        IS_VERIFY=is_verify,
        HIS_MODE=his_mode,
        BLOCK_B=triton.next_power_of_2(B + 1),
        BLOCK_T=BLOCK_T,
    )
    return (
        query_start_loc,
        has_initial_state,
        SconvExtendMetadata(cache_mask=cache_mask, safe_idx=safe_idx, cu=cu, si=si),
    )


def precompute_helion_extend_metadata(
    B: int,
    T: int,
    W: int,
    cache_indices: torch.Tensor,
    has_initial_state: torch.Tensor,
    query_start_loc: torch.Tensor,
) -> SconvExtendMetadata:
    """Precompute metadata for the helion extend path. Call once, reuse across layers."""
    device = cache_indices.device

    valid = cache_indices != PAD_SLOT_ID
    cache_mask = (has_initial_state & valid)[:, None, None]  # [B, 1, 1]
    safe_idx = cache_indices.clamp(min=0).long()

    cu = query_start_loc.to(torch.int64)
    si = _seq_idx_from_cu_seqlens(cu, T)

    return SconvExtendMetadata(
        cache_mask=cache_mask,
        safe_idx=safe_idx,
        cu=cu,
        si=si,
    )


def causal_conv1d(
    x: torch.Tensor,
    weight: torch.Tensor,
    sconv_cache: torch.Tensor,
    cache_mask: torch.Tensor,
    safe_idx: torch.Tensor,
    cu: torch.Tensor,
    si: torch.Tensor,
    activation: str | None = None,
    use_residual: bool = True,
    is_decode: bool = False,
) -> torch.Tensor:
    """Inference sconv with prefix loaded directly from cache.

    Metadata args (cache_mask, safe_idx, cu, si) should be precomputed once
    per forward pass via precompute_helion_{decode,extend}_metadata and reused
    across layers.
    """
    if activation == "swish":
        activation = "silu"

    T = x.shape[0]

    if T == 0:
        return torch.empty_like(x)

    D = x.shape[1]
    W = weight.shape[1]
    use_silu = activation in ("silu", "swish")

    if (
        is_cuda()
        and not is_decode
        and x.dtype == torch.bfloat16
        and D % 2 == 0
        and x.stride(1) == 1
    ):
        from sglang.jit_kernel.inkling_sconv import causal_conv1d as _cuda_causal_conv1d

        return _cuda_causal_conv1d(
            x,
            weight,
            sconv_cache,
            cache_mask,
            safe_idx,
            cu,
            si,
            activation=activation,
            use_residual=use_residual,
            is_decode=is_decode,
        )

    # The heavy prefix gather (sconv_cache[safe_idx], a [B, W-1, D] op) is folded INTO
    # the Triton kernel, which reads sconv_cache + safe_idx directly — no intermediate
    # [B, W-1, D] prefix tensor is materialised.
    #
    # The cache-mask multiply is handled per-path via the IS_DECODE constexpr, reading
    # the RAW [B,1,1] bool `cache_mask` metadata directly (no host-side `.to(x.dtype)`
    # cast launch, no placeholder):
    #   - extend (is_decode=False): the kernel loads cache_mask[si] and casts bool->x
    #     dtype in-kernel (0/1), multiplying the prefix tap by it — reproducing
    #     `prefix = sconv_cache[safe_idx] * cache_mask` (incl. has_initial_state=False /
    #     PAD slots whose mask is 0 => zeroed prefix).
    #   - decode (is_decode=True): the mask load+multiply is constexpr-pruned at JIT
    #     compile time; cache_mask is passed but never read.

    # Contiguous [T, D] output (strides (D, 1)) regardless of x's layout.
    y = torch.empty(T, D, dtype=x.dtype, device=x.device)

    # Coarse token-count regime for the autotune key.  D and W are CONSTANT across all
    # model calls, so keying autotune on (D, W) alone tunes the kernel exactly ONCE at
    # whatever shape the first call (warmup) happens to have, then reuses that single
    # config for every T — a small-T warmup config is poison at large T.  A coarse
    # t_bucket in the key gives each token-count regime its own tuned config:
    # small (<=8192), medium (<=65536), large (>65536).
    t_bucket = 0 if T <= 8192 else (1 if T <= 65536 else 2)

    grid = lambda meta: (
        triton.cdiv(T, meta["BLOCK_T"]),
        triton.cdiv(D, meta["BLOCK_D"]),
    )
    _causal_conv1d_fwd_with_prefix_kernel[grid](
        x,
        sconv_cache,
        safe_idx,
        cache_mask,
        weight,
        cu,
        si,
        y,
        t_bucket,
        x.stride(0),
        x.stride(1),
        sconv_cache.stride(0),
        sconv_cache.stride(1),  # stride_cache_w
        sconv_cache.stride(2),  # stride_cache_d
        cache_mask.stride(0),
        weight.stride(0),
        weight.stride(1),
        y.stride(0),
        y.stride(1),
        T,
        D,
        USE_SILU=use_silu,
        USE_RESIDUAL=use_residual,
        IS_DECODE=is_decode,
        W=W,
    )
    return y


@triton.jit
def _update_sconv_cache_kernel(
    x,  # [T, D]
    sconv_cache,  # [max_slots, W-1, D]
    cache_indices,  # [B] int32
    has_initial_state,  # [B] bool
    query_start_loc,  # [B+1] int32
    stride_x_t,
    stride_x_d,
    stride_cache_slot,
    stride_cache_w,
    stride_cache_d,
    B,
    D,
    W_MINUS_1: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """General update_sconv_cache: handles both decode (query_len=1) and extend (query_len>=1).

    The new conv state for a sequence is the last W-1 entries of the virtual stream
    [ old_state (W-1, gated by has_state) ++ x_seq (query_len) ]. For output position w:
      - x token       (query_len >= W_MINUS_1 - w):     x[end - W_MINUS_1 + w]
      - shifted state  (query_len <  W_MINUS_1 - w):     old_cache[w + query_len] * has_state
    Positions are left untouched when the slot is PAD or the sequence is empty.

    The shift source old_cache[w + query_len] is selected with a static inner loop over
    src_w (no data-dependent subscript on the W dim). RAW-safe: this writes positions in
    increasing w, and the selected source w+query_len is always > w (query_len > 0), so a
    position is only ever read before it is overwritten — matching the prior Helion kernel.
    Triton replaces Helion to drop the AOT-autotune dependency (cf. causal_conv1d).
    """
    pid_b = tl.program_id(0)
    pid_d = tl.program_id(1)
    b_off = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    d_off = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    b_mask = b_off < B
    d_mask = d_off < D
    bd_mask = b_mask[:, None] & d_mask[None, :]

    ci_raw = tl.load(cache_indices + b_off, mask=b_mask, other=-1)
    ci = tl.maximum(ci_raw, 0).to(tl.int64)
    end = tl.load(query_start_loc + b_off + 1, mask=b_mask, other=0).to(tl.int64)
    start = tl.load(query_start_loc + b_off, mask=b_mask, other=0).to(tl.int64)
    query_len = end - start
    has_state = tl.load(has_initial_state + b_off, mask=b_mask, other=0) != 0
    # PAD_SLOT_ID = -1 (can't reference Python global in @jit)
    valid = (ci_raw != -1) & (query_len > 0)

    cache_base = (
        sconv_cache + ci[:, None] * stride_cache_slot + d_off[None, :] * stride_cache_d
    )
    # Only valid lanes (real, non-empty slots) write. Invalid lanes (PAD or
    # query_len==0) write nothing — their clamped index aliases slot 0, and
    # storing there would race a real lane's update. Distinct real lanes have
    # distinct working slots, so valid writes never collide.
    write_mask = bd_mask & valid[:, None]

    for w in tl.static_range(W_MINUS_1):
        # Token from x (address clamped >= 0; only selected when gets_x, but the
        # clamped index is always in [0, T) so the load is in-bounds regardless).
        gets_x = query_len >= (W_MINUS_1 - w)
        x_idx = tl.maximum(end - W_MINUS_1 + w, 0)
        x_val = tl.load(
            x + x_idx[:, None] * stride_x_t + d_off[None, :] * stride_x_d,
            mask=bd_mask,
            other=0,
        )

        # Shifted cache value: select old_cache[w + query_len] via static loop.
        shift_val = tl.zeros([BLOCK_B, BLOCK_D], dtype=sconv_cache.dtype.element_ty)
        for src_w in tl.static_range(W_MINUS_1):
            match = query_len == (src_w - w)
            src_val = tl.load(
                cache_base + src_w * stride_cache_w, mask=bd_mask, other=0
            )
            shift_val = tl.where(match[:, None], src_val, shift_val)
        shift_val = tl.where(has_state[:, None], shift_val, 0)

        new_val = tl.where(gets_x[:, None], x_val, shift_val)
        tl.store(cache_base + w * stride_cache_w, new_val, mask=write_mask)


def update_sconv_cache(
    x: torch.Tensor,
    sconv_cache: torch.Tensor,
    cache_indices: torch.Tensor,
    has_initial_state: torch.Tensor,
    query_start_loc: torch.Tensor,
) -> None:
    B = cache_indices.shape[0]
    D = x.shape[-1]
    W_minus_1 = sconv_cache.shape[1]

    if (
        is_cuda()
        and x.dtype == torch.bfloat16
        and D % 2 == 0
        and x.stride(-1) == 1
        and sconv_cache.stride(2) == 1
    ):
        from sglang.jit_kernel.inkling_sconv import (
            update_sconv_cache as _cuda_update_sconv_cache,
        )

        _cuda_update_sconv_cache(
            x,
            sconv_cache,
            cache_indices.to(torch.int32),
            has_initial_state,
            query_start_loc.to(torch.int32),
        )
        return

    BLOCK_D = min(triton.next_power_of_2(D), 1024)
    BLOCK_B = 1  # B is the (small) sequence count; one program per sequence
    num_warps = 4 if BLOCK_D >= 512 else (2 if BLOCK_D >= 128 else 1)
    grid = (triton.cdiv(B, BLOCK_B), triton.cdiv(D, BLOCK_D))
    _update_sconv_cache_kernel[grid](
        x,
        sconv_cache,
        cache_indices,
        has_initial_state,
        query_start_loc,
        x.stride(0),
        x.stride(1),
        sconv_cache.stride(0),
        sconv_cache.stride(1),  # stride_cache_w
        sconv_cache.stride(2),  # stride_cache_d
        B,
        D,
        W_MINUS_1=W_minus_1,
        BLOCK_B=BLOCK_B,
        BLOCK_D=BLOCK_D,
        num_warps=num_warps,
    )


# ---------------------------------------------------------------------------
# Fused decode kernel: causal_conv1d + update_sconv_cache in one launch
# ---------------------------------------------------------------------------


@triton.jit
def _fused_causal_conv1d_update_decode_kernel(
    x,  # [T, D]
    sconv_cache,  # [max_slots, W-1, D]
    cache_indices,  # [B] int32
    cache_mask,  # [B] bool (cache_indices != PAD_SLOT_ID)
    weight,  # [D, W]
    y,  # [T, D] – always contiguous
    track_mask,  # [B] bool   – prefix-cache track mask (dummy if not DO_TRACK)
    track_indices,  # [B] int – persistent ping-pong slots (dummy if not DO_TRACK)
    stride_x_t,
    stride_x_d,
    stride_y_t,
    stride_y_d,
    stride_cache_slot,
    stride_cache_d,
    stride_cache_w,
    stride_weight_d,
    stride_weight_w,
    stride_track_idx,
    T,
    D,
    USE_SILU: tl.constexpr,
    USE_RESIDUAL: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
    W: tl.constexpr,
    DO_TRACK: tl.constexpr,
):
    """Fused depthwise causal conv1d + cache shift-update (+ optional track-copy) for decode.

    Decode invariant: each token t belongs to sequence t, with bos=t.
      - iw = 0..W-2: always reads from sconv_cache (the conv state history)
      - iw = W-1:    always reads from x (the current token)

    General for any W. Uses tl.static_range for both conv and update.
    Cache values are re-read for the update shift (trades W-1 extra loads
    for generality and lower register pressure vs manual unroll).

    Track-copy fusion (DO_TRACK): for prefix caching, the post-update conv
    window of sequence b must also be snapshotted into a persistent ping-pong
    slot track_indices[b] when track_mask[b] is set. The post-update window is
    already produced in-register by the shift below, so it is written to BOTH
    the working slot cache_indices[b] and track_indices[b] in the same pass —
    no separate copy_if_needed launch and no re-read of the row.

    This is race-free because working slots (cache_indices) and ping-pong track
    slots (track_indices) are independent allocations from the same mamba pool,
    hence pairwise-distinct across the batch: every slot written here is unique,
    and the only slot read (cache_indices[b], for the conv taps) is written by
    no other program. track_mask / track_indices are sized to the real batch and
    read only where ci != -1 (the non-pad, real-token lanes) so cudagraph padding
    lanes never index out of bounds.
    """
    t_off = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
    d_off = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
    t_mask = t_off < T
    d_mask = d_off < D
    td_mask = t_mask[:, None] & d_mask[None, :]

    ci = tl.load(cache_indices + t_off, mask=t_mask, other=-1)
    safe_idx = tl.maximum(ci, 0).to(tl.int64)
    valid = ci != -1  # PAD_SLOT_ID = -1 (can't reference Python global in @jit)
    cm = tl.load(cache_mask + t_off, mask=t_mask, other=0).to(
        sconv_cache.dtype.element_ty
    )

    cache_base = (
        sconv_cache
        + safe_idx[:, None] * stride_cache_slot
        + d_off[None, :] * stride_cache_d
    )
    weight_base = weight + d_off * stride_weight_d

    if DO_TRACK:
        # Real (non-pad) lanes only: track tensors are sized to the real batch,
        # while t_off may extend into cudagraph padding (ci == -1 there).
        real = t_mask & valid
        do_track = tl.load(track_mask + t_off, mask=real, other=0) != 0
        track_slot = tl.load(
            track_indices + t_off * stride_track_idx, mask=real, other=0
        ).to(tl.int64)
        track_base = (
            sconv_cache
            + track_slot[:, None] * stride_cache_slot
            + d_off[None, :] * stride_cache_d
        )
        track_write_mask = td_mask & (valid & do_track)[:, None]

    # ---- CONV ----
    acc = tl.zeros([BLOCK_T, BLOCK_D], dtype=tl.float32)

    # Cache taps: iw = 0..W-2
    for iw in tl.static_range(W - 1):
        pv = tl.load(
            cache_base + iw * stride_cache_w,
            mask=td_mask,
            other=0,
            eviction_policy="evict_last",
        )
        w = tl.load(weight_base + iw * stride_weight_w, mask=d_mask, other=0).to(
            tl.float32
        )
        acc += (pv * cm[:, None]).to(tl.float32) * w[None, :]

    # Current token: iw = W-1
    xv = tl.load(
        x + t_off[:, None] * stride_x_t + d_off[None, :] * stride_x_d,
        mask=td_mask,
        other=0,
    )
    xv_f32 = xv.to(tl.float32)
    w_last = tl.load(weight_base + (W - 1) * stride_weight_w, mask=d_mask, other=0).to(
        tl.float32
    )
    acc += xv_f32 * w_last[None, :]

    if USE_SILU:
        acc = acc * tl.sigmoid(acc)

    if USE_RESIDUAL:
        acc += xv_f32

    tl.store(
        y + t_off[:, None] * stride_y_t + d_off[None, :] * stride_y_d,
        acc.to(xv.dtype),
        mask=td_mask,
    )

    # ---- UPDATE: shift cache left, write new token ----
    # cache[slot, w, d] = cache[slot, w+1, d] * cm  for w = 0..W-3
    # cache[slot, W-2, d] = xv
    # When DO_TRACK, the same post-update window is also snapshotted into the
    # persistent ping-pong slot track_indices[b] (prefix caching), in-register
    # with no re-read and no separate copy_if_needed launch.
    write_mask = td_mask & valid[:, None]
    for iw in tl.static_range(W - 2):
        # Re-read cache[slot, d, iw+1] for the shift
        shifted = tl.load(cache_base + (iw + 1) * stride_cache_w, mask=td_mask, other=0)
        new_val = shifted * cm[:, None]
        tl.store(cache_base + iw * stride_cache_w, new_val, mask=write_mask)
        if DO_TRACK:
            tl.store(track_base + iw * stride_cache_w, new_val, mask=track_write_mask)
    # Last position gets the new token
    tl.store(cache_base + (W - 2) * stride_cache_w, xv, mask=write_mask)
    if DO_TRACK:
        tl.store(track_base + (W - 2) * stride_cache_w, xv, mask=track_write_mask)


def _select_fused_decode_config(T: int, D: int) -> tuple[int, int, int, int]:
    """Select (BLOCK_T, BLOCK_D, num_warps, num_stages) for the fused decode kernel.

    Heuristic: keep BLOCK_T small (1-2) for decode since T=B is moderate.
    Scale BLOCK_D so that grid has enough blocks to fill the GPU.
    """
    if T <= 2048:
        block_t = 2
    else:
        # Round down to power of 2; Triton requires tl.arange size to be power of 2.
        raw = min(T // 1024, 8)
        block_t = 1 << (raw.bit_length() - 1)

    target_blocks = 1024
    t_blocks = max(T // block_t, 1)
    needed_d_blocks = max(target_blocks // t_blocks, 1)
    block_d = max(D // needed_d_blocks, 64)
    block_d = 1 << max(min((block_d).bit_length() - 1, 9), 6)

    tile_elems = block_t * block_d
    if tile_elems <= 128:
        num_warps = 1
    elif tile_elems <= 512:
        num_warps = 2
    else:
        num_warps = 4

    return block_t, block_d, num_warps, 3


def fused_causal_conv1d_update_decode(
    x: torch.Tensor,
    weight: torch.Tensor,
    sconv_cache: torch.Tensor,
    cache_indices: torch.Tensor,
    cache_mask: torch.Tensor,
    activation: str | None = None,
    use_residual: bool = True,
    track_mask: torch.Tensor | None = None,
    track_indices: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fused causal_conv1d + update_sconv_cache (+ optional prefix-cache track) for decode.

    Replaces the sequence: prefix construction -> conv -> cache update
    (-> copy_if_needed) with a single kernel launch.

    When track_mask / track_indices are provided (prefix caching with the mamba
    extra buffer), the post-update conv window is also snapshotted into the
    persistent ping-pong slot track_indices[b] wherever track_mask[b] is set —
    folding in the former separate `copy_if_needed` (`_track_conv_state_decode`)
    launch. This is race-free: working slots and ping-pong track slots are
    independent allocations from the same mamba pool and therefore pairwise
    distinct across the batch (see kernel docstring).
    """
    T, D = x.shape
    W = weight.shape[1]

    if (
        is_cuda()
        and x.dtype == torch.bfloat16
        and D % 2 == 0
        and x.stride(1) == 1
        and sconv_cache.stride(2) == 1
    ):
        from sglang.jit_kernel.inkling_sconv import (
            fused_causal_conv1d_update_decode as _cuda_fused_decode,
        )

        _ti = track_indices.to(torch.int64) if track_indices is not None else None
        return _cuda_fused_decode(
            x,
            weight,
            sconv_cache,
            cache_indices.to(torch.int32),
            cache_mask,
            activation=activation,
            use_residual=use_residual,
            track_mask=track_mask,
            track_indices=_ti,
        )
    # Always allocate contiguous output so callers receive a [T, D] tensor
    # with strides (D, 1) regardless of whether x is a non-contiguous view.
    y = torch.empty(T, D, dtype=x.dtype, device=x.device)
    cm = cache_mask.view(-1)
    use_silu = activation in ("silu", "swish")

    do_track = track_mask is not None
    if do_track:
        track_mask = track_mask.view(-1)
        # Sentinel never dereferenced for the no-track path.
        stride_track_idx = track_indices.stride(0)
    else:
        # Dummy tensors satisfy the kernel signature; never dereferenced (DO_TRACK=False).
        track_mask = torch.empty(0, dtype=torch.bool, device=x.device)
        track_indices = torch.empty(0, dtype=torch.int64, device=x.device)
        stride_track_idx = 1

    bt, bd, nw, ns = _select_fused_decode_config(T, D)

    grid = (triton.cdiv(T, bt), triton.cdiv(D, bd))
    _fused_causal_conv1d_update_decode_kernel[grid](
        x,
        sconv_cache,
        cache_indices,
        cm,
        weight,
        y,
        track_mask,
        track_indices,
        x.stride(0),
        x.stride(1),
        y.stride(0),
        y.stride(1),
        sconv_cache.stride(0),
        sconv_cache.stride(2),  # stride_cache_d
        sconv_cache.stride(1),  # stride_cache_w
        weight.stride(0),
        weight.stride(1),
        stride_track_idx,
        T,
        D,
        USE_SILU=use_silu,
        USE_RESIDUAL=use_residual,
        BLOCK_T=bt,
        BLOCK_D=bd,
        W=W,
        DO_TRACK=do_track,
        num_warps=nw,
        num_stages=ns,
    )
    return y


@triton.jit
def _save_intermediate_conv_windows_kernel(
    sconv_cache_ptr,  # [cache_size, W-1, D]
    hidden_states_ptr,  # [B, T_max, D]
    cache_indices_ptr,  # [B] int32
    out_ptr,  # [max_bs, T, W-1, D]
    cache_slot_stride,
    cache_pos_stride,
    hidden_b_stride,
    hidden_t_stride,
    out_b_stride,
    out_t_stride,
    out_w_stride,
    D,
    W_MINUS_1: tl.constexpr,
    BLOCK_D: tl.constexpr,
    PAD_SLOT_ID: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_d = tl.program_id(2)

    cache_idx = tl.load(cache_indices_ptr + pid_b).to(tl.int64)

    # PAD_SLOT_ID guard: skip padded batch slots. Mirrors
    # fused_mamba_state_scatter_with_mask's early-exit. Avoids the OOB
    # negative-stride load that would result from `cache_idx == PAD_SLOT_ID`.
    if cache_idx == PAD_SLOT_ID:
        return

    d_off = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    d_mask = d_off < D

    for w in tl.static_range(W_MINUS_1):
        position = pid_t + 1 + w
        if position < W_MINUS_1:
            src_offset = (
                cache_idx * cache_slot_stride + position * cache_pos_stride + d_off
            )
            val = tl.load(sconv_cache_ptr + src_offset, mask=d_mask, other=0.0)
        else:
            t_in_hidden = position - W_MINUS_1
            src_offset = (
                pid_b.to(tl.int64) * hidden_b_stride
                + t_in_hidden.to(tl.int64) * hidden_t_stride
                + d_off
            )
            val = tl.load(hidden_states_ptr + src_offset, mask=d_mask, other=0.0)

        dst_offset = (
            pid_b.to(tl.int64) * out_b_stride
            + pid_t.to(tl.int64) * out_t_stride
            + w * out_w_stride
            + d_off
        )
        tl.store(out_ptr + dst_offset, val, mask=d_mask)


def save_intermediate_conv_windows(
    sconv_cache: torch.Tensor,  # [cache_size, W-1, D]
    hidden_states: torch.Tensor,  # [B, T_max, D] or [B*T_max, D]
    cache_indices: torch.Tensor,  # [B], int32 or int64
    intermediate_out: torch.Tensor,  # [max_bs, T, W-1, D]
    batch_size: int,
    draft_token_num: int,
) -> None:
    """Fused unfold-and-write into intermediate_out[:batch_size].

    Equivalent to:
        initial = sconv_cache[cache_indices[:batch_size]]
        padded  = torch.cat([initial, hidden_states[:batch_size, :draft_token_num]], dim=1)
        windows = padded.unfold(1, W-1, 1)[:, 1:draft_token_num+1].transpose(-2,-1).contiguous()
        intermediate_out[:batch_size] = windows
    """
    if batch_size == 0 or draft_token_num == 0:
        return

    W_minus_1, D = sconv_cache.shape[1], sconv_cache.shape[2]
    if W_minus_1 == 0:
        return

    if hidden_states.dim() == 2:
        hidden_states = hidden_states.view(batch_size, -1, hidden_states.shape[-1])
    assert (
        hidden_states.dim() == 3
    ), f"unexpected hidden_states shape {hidden_states.shape}"
    assert hidden_states.shape[0] == batch_size
    assert hidden_states.shape[2] == D
    assert intermediate_out.shape[1] == draft_token_num
    assert intermediate_out.shape[2] == W_minus_1
    assert intermediate_out.shape[3] == D

    # kernel assumption
    assert sconv_cache.stride(-1) == 1, "sconv_cache must be D-contiguous"
    assert hidden_states.stride(-1) == 1, "hidden_states must be D-contiguous"
    assert intermediate_out.stride(-1) == 1, "intermediate_out must be D-contiguous"

    cache_indices = cache_indices[:batch_size].to(torch.int32).contiguous()

    BLOCK_D = min(triton.next_power_of_2(D), 1024)
    grid = (batch_size, draft_token_num, triton.cdiv(D, BLOCK_D))

    _save_intermediate_conv_windows_kernel[grid](
        sconv_cache,
        hidden_states,
        cache_indices,
        intermediate_out,
        sconv_cache.stride(0),
        sconv_cache.stride(1),
        hidden_states.stride(0),
        hidden_states.stride(1),
        intermediate_out.stride(0),
        intermediate_out.stride(1),
        intermediate_out.stride(2),
        D,
        W_MINUS_1=W_minus_1,
        BLOCK_D=BLOCK_D,
        PAD_SLOT_ID=PAD_SLOT_ID,
    )
