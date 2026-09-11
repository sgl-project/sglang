"""Triton sparse-MLA forward for the DSA FP8 and BF16 paths.

Two strategies, auto-selected by sequence length:
  1. Single-pass: grid=(seq,), best when seq is large enough to fill CUs.
  2. Split-K: grid=(seq, head_blocks, kv_splits) + reduce, best for short
     sequences (MTP verify/draft with seq=1-6) where single-pass starves the GPU.

Both use the split-dim pattern: D_V processed in NUM_GROUPS chunks of 128
for native CDNA MFMA tile alignment.
"""

import contextlib
import functools

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.quantization.fp8_kernel import is_fp8_fnuz

_ASYNC_COPY_OFF_ARCHES = frozenset({"gfx950"})
_IS_FNUZ = is_fp8_fnuz()
_FP8_MAX = 240.0 if _IS_FNUZ else 448.0
_LOG2E = 1.4426950408889634
_G = tl.constexpr(128)
_PREFERRED_BLOCK_K = 64
_MIN_BLOCK_K = 16
_INDEX_ELEMENT_SIZE = 4

_SUPPORTED_INPUT_DTYPES = (
    torch.bfloat16,
    torch.float8_e4m3fn,
    torch.float8_e4m3fnuz,
)


def _validate_input_dtypes(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    kv: torch.Tensor,
) -> bool:
    """Validate sparse-MLA dtypes and return whether the KV cache is FP8.

    MI300X produces BF16 queries with an FP8 FNUZ KV cache. The kernels cast
    query fragments to the KV element type while loading them, keeping both dot
    operands matched without a separate query-conversion kernel.
    """
    q_dtype = q_nope.dtype
    kv_dtype = kv.dtype
    if q_rope.dtype != q_dtype:
        raise ValueError(
            "Triton sparse MLA requires q_nope and q_rope to have the same "
            f"dtype; got {q_dtype} and {q_rope.dtype}."
        )
    if (
        q_dtype not in _SUPPORTED_INPUT_DTYPES
        or kv_dtype not in _SUPPORTED_INPUT_DTYPES
    ):
        raise ValueError(
            "Triton sparse MLA supports bfloat16 and float8_e4m3 query/KV "
            f"inputs; got query dtype {q_dtype} and KV dtype {kv_dtype}."
        )
    if q_dtype != torch.bfloat16 and q_dtype != kv_dtype:
        raise ValueError(
            "Triton sparse MLA requires FP8 queries to match the KV dtype; "
            f"got query dtype {q_dtype} and KV dtype {kv_dtype}."
        )
    return kv_dtype != torch.bfloat16


@functools.lru_cache(maxsize=None)
def _async_copy_is_harmful_on_device(device: int) -> bool:
    properties = torch.cuda.get_device_properties(device)
    arch = getattr(properties, "gcnArchName", "") or ""
    return arch.split(":", 1)[0] in _ASYNC_COPY_OFF_ARCHES


def _async_copy_is_harmful() -> bool:
    try:
        from triton import knobs
    except ImportError:
        return False

    amd_knobs = getattr(knobs, "amd", None)
    if not hasattr(amd_knobs, "use_async_copy"):
        return False

    # An explicit TRITON_HIP_USE_ASYNC_COPY setting takes precedence.
    if amd_knobs.use_async_copy is not None:
        return False
    if not torch.cuda.is_available():
        return False

    return _async_copy_is_harmful_on_device(torch.cuda.current_device())


@contextlib.contextmanager
def _no_async_copy():
    """Compile enclosed DSA launches without direct-to-LDS async copies.

    Triton compiles lazily and per specialization, so the scope must cover every
    launch. The knob is part of Triton's disk cache key; kernels compiled
    elsewhere retain the backend default.
    """
    if not _async_copy_is_harmful():
        yield
        return

    from triton import knobs

    with knobs.amd.scope():
        knobs.amd.use_async_copy = False
        yield


# ---------------------------------------------------------------------------
# Helper functions for split-K heuristic
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def _cu_count() -> int:
    return torch.cuda.get_device_properties(
        torch.cuda.current_device()
    ).multi_processor_count


@functools.lru_cache(maxsize=None)
def _max_shared_memory(device: int) -> int | None:
    """Return Triton's per-workgroup shared-memory limit for ``device``."""
    try:
        properties = triton.runtime.driver.active.utils.get_device_properties(device)
        return int(properties["max_shared_mem"])
    except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
        return None


def _block_k_for_shared_memory(
    kv_dim: int,
    element_size: int,
    max_shared_memory: int | None,
) -> int:
    """Choose the largest supported KV tile that fits in shared memory.

    Triton's AMD async-copy path stages one KV row plus its int32 page index
    for every BLOCK_K entry. A 64x576 BF16 tile therefore needs 73,984 bytes,
    which exceeds the 64 KiB workgroup limit on gfx942. FP8 needs only 37,120
    bytes, while devices with a larger limit can retain the preferred tile.
    """
    block_k = _PREFERRED_BLOCK_K
    if max_shared_memory is None:
        # Prefer a conservative BF16 tile when the backend cannot report its
        # limit. One-byte FP8 inputs fit the common 64 KiB minimum at D=576.
        return block_k if element_size == 1 else block_k // 2

    bytes_per_entry = kv_dim * element_size + _INDEX_ELEMENT_SIZE
    while block_k > _MIN_BLOCK_K and block_k * bytes_per_entry > max_shared_memory:
        block_k //= 2
    return block_k


def _sparse_mla_block_k(kv: torch.Tensor) -> int:
    device = kv.device.index
    if device is None:
        device = torch.cuda.current_device()
    return _block_k_for_shared_memory(
        kv.shape[-1], kv.element_size(), _max_shared_memory(device)
    )


def _kv_splits_heuristic(
    T: int,
    H: int,
    block_h: int,
    num_cu: int | None = None,
    target_wg_per_cu: float = 2.0,
    max_kv_splits: int = 64,
) -> int:
    if num_cu is None:
        num_cu = _cu_count()
    target_wg = max(1, int(target_wg_per_cu * num_cu))
    head_blocks = max(1, (H + block_h - 1) // block_h)
    base_ctas = max(1, T * head_blocks)
    if base_ctas >= target_wg:
        return 1
    splits_to_fill = max(1, target_wg // base_ctas)
    return _prev_pow2(min(splits_to_fill, max_kv_splits))


def _row_strides(x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
    """Return (tensor, token_stride, head_stride) for a [N, H, D] q tensor.

    The kernels address q by explicit row strides, so a packed [N, H, D] layout is
    not required -- only a unit-stride last dim. Callers that pass an already-
    concatenated q (dsa_backend, GLM-5.2 path) hand us two strided views of one
    [N, H, D_V + D_TAIL] buffer; copying those would cost two extra device kernels
    per layer per forward for nothing. The fallback keeps the kernels' `+ g` / `+ dt`
    addressing valid for exotic layouts -- no caller hits it today.
    """
    if x.stride(-1) != 1:
        x = x.contiguous()
    return x, x.stride(0), x.stride(1)


def _prune_configs(configs, named_args, **kwargs):
    """Drop wasteful configs and retain the established FP8 search space."""
    topk = named_args["topk"]
    max_block_n = _sparse_mla_block_k(named_args["kv_ptr"])
    candidates = configs
    if kwargs["USE_FP8_DOT"]:
        candidates = [
            c
            for c in candidates
            if c.kwargs["BLOCK_N"] in (32, 64) and c.num_stages in (1, 2)
        ]
    keep = [
        c
        for c in candidates
        if c.kwargs["BLOCK_N"] <= topk and c.kwargs["BLOCK_N"] <= max_block_n
    ]
    return keep or [candidates[0]]


_LONG_PREFILL_SEQ_THRESHOLD = 32768


# ---------------------------------------------------------------------------
# Single-pass split-dim kernel (autotuned, for long sequences)
# grid=(seq,), processes D_V in NUM_GROUPS chunks of 128
# ---------------------------------------------------------------------------

_SPLIT_DIM_CONFIGS = [
    triton.Config({"BLOCK_N": bn}, num_warps=w, num_stages=ns)
    for bn in (16, 32, 64)
    for w in (1, 2, 4)
    for ns in (1, 2, 3)
]


@triton.autotune(
    configs=_SPLIT_DIM_CONFIGS,
    key=["topk", "H", "USE_FP8_DOT", "SEQ_BUCKET"],
    prune_configs_by={"early_config_prune": _prune_configs},
)
@triton.jit
def _sparse_mla_fwd_split_dim_kernel(
    q_nope_ptr,  # [seq, H, D_V]   fp8/bf16
    q_rope_ptr,  # [seq, H, D_TAIL] fp8/bf16
    kv_ptr,  # [num_pages, 1, DIM] fp8/bf16
    idx_ptr,  # [seq, topk]      int32
    o_ptr,  # [seq, H, D_V]    bf16
    qk_scale,  # sm_scale * LOG2E (prescaled for exp2)
    fp8_max,
    topk,
    H: tl.constexpr,
    DIM: tl.constexpr,
    D_V: tl.constexpr,
    D_TAIL: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    STRIDE_QN_T: tl.constexpr,
    STRIDE_QN_H: tl.constexpr,
    STRIDE_QR_T: tl.constexpr,
    STRIDE_QR_H: tl.constexpr,
    USE_FP8_DOT: tl.constexpr,
    SEQ_BUCKET: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    s_i = tl.program_id(0)

    h = tl.arange(0, H)
    dt = tl.arange(0, D_TAIL)
    g = tl.arange(0, _G)

    input_type = kv_ptr.dtype.element_ty if USE_FP8_DOT else tl.bfloat16
    q_row = q_nope_ptr + s_i * STRIDE_QN_T + h[:, None] * STRIDE_QN_H
    q0 = tl.load(q_row + g[None, :]).to(input_type)
    if NUM_GROUPS >= 2:
        q1 = tl.load(q_row + (_G + g)[None, :]).to(input_type)
    if NUM_GROUPS >= 3:
        q2 = tl.load(q_row + (2 * _G + g)[None, :]).to(input_type)
    if NUM_GROUPS >= 4:
        q3 = tl.load(q_row + (3 * _G + g)[None, :]).to(input_type)
    q_tail = tl.load(
        q_rope_ptr + s_i * STRIDE_QR_T + h[:, None] * STRIDE_QR_H + dt[None, :]
    ).to(input_type)

    neg_large = -3.4028234663852886e38
    m_i = tl.full([H], neg_large, tl.float32)
    l_i = tl.zeros([H], tl.float32)
    acc0 = tl.zeros([H, _G], tl.float32)
    if NUM_GROUPS >= 2:
        acc1 = tl.zeros([H, _G], tl.float32)
    if NUM_GROUPS >= 3:
        acc2 = tl.zeros([H, _G], tl.float32)
    if NUM_GROUPS >= 4:
        acc3 = tl.zeros([H, _G], tl.float32)

    if USE_FP8_DOT:
        p_dot_scale = 1.0 / fp8_max
    else:
        p_dot_scale = 1.0
    n = tl.arange(0, BLOCK_N)
    for k0 in range(0, topk, BLOCK_N):
        kmask = (k0 + n) < topk
        idx = tl.load(idx_ptr + s_i * topk + k0 + n, mask=kmask, other=-1)
        valid = (idx >= 0) & kmask
        page = tl.where(valid, idx, 0).to(tl.int64)
        kbase = kv_ptr + page[:, None] * DIM

        kv0 = tl.load(kbase + g[None, :], mask=valid[:, None], other=0.0).to(input_type)
        if NUM_GROUPS >= 2:
            kv1 = tl.load(kbase + (_G + g)[None, :], mask=valid[:, None], other=0.0).to(
                input_type
            )
        if NUM_GROUPS >= 3:
            kv2 = tl.load(
                kbase + (2 * _G + g)[None, :], mask=valid[:, None], other=0.0
            ).to(input_type)
        if NUM_GROUPS >= 4:
            kv3 = tl.load(
                kbase + (3 * _G + g)[None, :], mask=valid[:, None], other=0.0
            ).to(input_type)
        kv_tail = tl.load(
            kbase + (D_V + dt)[None, :], mask=valid[:, None], other=0.0
        ).to(input_type)

        qk = tl.dot(q0, tl.trans(kv0))
        if NUM_GROUPS >= 2:
            qk += tl.dot(q1, tl.trans(kv1))
        if NUM_GROUPS >= 3:
            qk += tl.dot(q2, tl.trans(kv2))
        if NUM_GROUPS >= 4:
            qk += tl.dot(q3, tl.trans(kv3))
        qk += tl.dot(q_tail, tl.trans(kv_tail))
        qk = qk * qk_scale
        qk = tl.where(valid[None, :], qk, neg_large)

        m_block = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, m_block)
        alpha = tl.exp2(m_i - m_new)
        p = tl.exp2(qk - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)

        if USE_FP8_DOT:
            p_dot = (p * fp8_max).to(input_type)
        else:
            p_dot = p.to(input_type)
        acc0 = acc0 * alpha[:, None] + tl.dot(p_dot, kv0).to(tl.float32) * p_dot_scale
        if NUM_GROUPS >= 2:
            acc1 = (
                acc1 * alpha[:, None] + tl.dot(p_dot, kv1).to(tl.float32) * p_dot_scale
            )
        if NUM_GROUPS >= 3:
            acc2 = (
                acc2 * alpha[:, None] + tl.dot(p_dot, kv2).to(tl.float32) * p_dot_scale
            )
        if NUM_GROUPS >= 4:
            acc3 = (
                acc3 * alpha[:, None] + tl.dot(p_dot, kv3).to(tl.float32) * p_dot_scale
            )
        m_i = m_new

    l_safe = tl.where(l_i == 0.0, 1.0, l_i)
    inv_l = 1.0 / l_safe
    acc0 = acc0 * inv_l[:, None]
    if NUM_GROUPS >= 2:
        acc1 = acc1 * inv_l[:, None]
    if NUM_GROUPS >= 3:
        acc2 = acc2 * inv_l[:, None]
    if NUM_GROUPS >= 4:
        acc3 = acc3 * inv_l[:, None]

    o_base = o_ptr + s_i * H * D_V
    tl.store(o_base + h[:, None] * D_V + g[None, :], acc0.to(o_ptr.dtype.element_ty))
    if NUM_GROUPS >= 2:
        tl.store(
            o_base + h[:, None] * D_V + (_G + g)[None, :],
            acc1.to(o_ptr.dtype.element_ty),
        )
    if NUM_GROUPS >= 3:
        tl.store(
            o_base + h[:, None] * D_V + (2 * _G + g)[None, :],
            acc2.to(o_ptr.dtype.element_ty),
        )
    if NUM_GROUPS >= 4:
        tl.store(
            o_base + h[:, None] * D_V + (3 * _G + g)[None, :],
            acc3.to(o_ptr.dtype.element_ty),
        )


@_no_async_copy()
def _triton_sparse_mla_fwd_single(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    d_v: int = 512,
) -> torch.Tensor:
    """Single-pass prefill: grid=(seq,), loops over all topk per CTA."""
    is_fp8 = _validate_input_dtypes(q_nope, q_rope, kv)
    use_fp8_dot = is_fp8
    seq, H, d_v_in = q_nope.shape
    assert d_v_in == d_v
    assert d_v % 128 == 0, f"Triton sparse MLA requires d_v divisible by 128, got {d_v}"
    num_groups = d_v // 128
    assert num_groups <= 4, (
        f"Triton sparse MLA supports d_v up to 512 (4 groups), got d_v={d_v}"
    )
    # The single-pass kernel indexes heads with an unmasked `tl.arange(0, H)`,
    # which Triton requires to be a power of two. H < 16 is padded up to 16
    # below; anything larger must already be a power of two.
    assert H <= 16 or (H & (H - 1)) == 0, (
        f"Triton sparse MLA prefill requires a power-of-two head count, got H={H}. "
        "Use a tp_size that divides the model's head count to a power of two, or "
        "pick another DSA prefill backend (--dsa-prefill-backend tilelang)."
    )
    d_tail = q_rope.shape[-1]
    dim = kv.shape[-1]
    topk = indices.shape[-1]
    q_nope, stride_qn_t, stride_qn_h = _row_strides(q_nope)
    q_rope, stride_qr_t, stride_qr_h = _row_strides(q_rope)
    idx_flat = indices.squeeze(1).contiguous() if indices.dim() == 3 else indices
    out = torch.empty(seq, H, d_v, device=q_nope.device, dtype=torch.bfloat16)
    qk_scale = float(sm_scale) * _LOG2E
    # Keep FP8 in one cache bucket while allowing the two target BF16 prefill
    # regimes to retain different autotuned configurations.
    seq_bucket = int(not use_fp8_dot and seq >= _LONG_PREFILL_SEQ_THRESHOLD)
    if H < 16:
        # Pad H to 16 so the FP8 dot path maps to native MFMA tiles on CDNA4.
        # Without padding, M=H<16 FP8 dots fall back to a scalar path.
        H_pad = 16
        q_nope_pad = torch.zeros(
            seq, H_pad, d_v, device=q_nope.device, dtype=q_nope.dtype
        )
        q_rope_pad = torch.zeros(
            seq, H_pad, d_tail, device=q_rope.device, dtype=q_rope.dtype
        )
        q_nope_pad[:, :H, :] = q_nope
        q_rope_pad[:, :H, :] = q_rope
        # Freshly allocated and packed; re-read the strides for the padded shape.
        q_nope_pad, stride_qn_t, stride_qn_h = _row_strides(q_nope_pad)
        q_rope_pad, stride_qr_t, stride_qr_h = _row_strides(q_rope_pad)
        out_pad = torch.empty(
            seq, H_pad, d_v, device=q_nope.device, dtype=torch.bfloat16
        )
        _sparse_mla_fwd_split_dim_kernel[(seq,)](
            q_nope_pad,
            q_rope_pad,
            kv,
            idx_flat,
            out_pad,
            qk_scale,
            _FP8_MAX,
            topk,
            H=H_pad,
            DIM=dim,
            D_V=d_v,
            D_TAIL=d_tail,
            NUM_GROUPS=num_groups,
            STRIDE_QN_T=stride_qn_t,
            STRIDE_QN_H=stride_qn_h,
            STRIDE_QR_T=stride_qr_t,
            STRIDE_QR_H=stride_qr_h,
            USE_FP8_DOT=use_fp8_dot,
            SEQ_BUCKET=seq_bucket,
        )
        out = out_pad[:, :H, :].contiguous()
    else:
        _sparse_mla_fwd_split_dim_kernel[(seq,)](
            q_nope,
            q_rope,
            kv,
            idx_flat,
            out,
            qk_scale,
            _FP8_MAX,
            topk,
            H=H,
            DIM=dim,
            D_V=d_v,
            D_TAIL=d_tail,
            NUM_GROUPS=num_groups,
            STRIDE_QN_T=stride_qn_t,
            STRIDE_QN_H=stride_qn_h,
            STRIDE_QR_T=stride_qr_t,
            STRIDE_QR_H=stride_qr_h,
            USE_FP8_DOT=use_fp8_dot,
            SEQ_BUCKET=seq_bucket,
        )
    return out.unsqueeze(0)


def _prev_pow2(n: int) -> int:
    if n < 1:
        return 1
    return 1 << (n.bit_length() - 1)


def _next_pow2(n: int) -> int:
    if n < 1:
        return 1
    return 1 << (n - 1).bit_length()


# ---------------------------------------------------------------------------
# Split-K kernels (for short sequences: MTP verify/draft, decode)
# grid=(seq, head_blocks, kv_splits) + reduce
# ---------------------------------------------------------------------------


@triton.jit
def _sparse_mla_fused_kernel(
    q_nope_ptr,
    q_rope_ptr,
    kv_ptr,
    idx_ptr,
    out_ptr,
    qk_scale,
    fp8_max,
    topk: tl.constexpr,
    H: tl.constexpr,
    KV_DIM: tl.constexpr,
    D_V: tl.constexpr,
    D_TAIL: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    STRIDE_QN_T: tl.constexpr,
    STRIDE_QN_H: tl.constexpr,
    STRIDE_QR_T: tl.constexpr,
    STRIDE_QR_H: tl.constexpr,
    USE_FP8_DOT: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Single-pass with head-block splitting. grid=(seq, head_blocks)."""
    t = tl.program_id(0)
    pid_h = tl.program_id(1)

    h_offs = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    h_mask = h_offs < H
    dt = tl.arange(0, D_TAIL)
    g = tl.arange(0, _G)

    input_type = kv_ptr.dtype.element_ty if USE_FP8_DOT else tl.bfloat16
    if USE_FP8_DOT:
        p_dot_scale = 1.0 / fp8_max
    else:
        p_dot_scale = 1.0

    qn_row = q_nope_ptr + t * STRIDE_QN_T + h_offs[:, None] * STRIDE_QN_H
    q0 = tl.load(qn_row + g[None, :], mask=h_mask[:, None], other=0.0).to(input_type)
    if NUM_GROUPS >= 2:
        q1 = tl.load(
            qn_row + (_G + g)[None, :],
            mask=h_mask[:, None],
            other=0.0,
        ).to(input_type)
    if NUM_GROUPS >= 3:
        q2 = tl.load(
            qn_row + (2 * _G + g)[None, :],
            mask=h_mask[:, None],
            other=0.0,
        ).to(input_type)
    if NUM_GROUPS >= 4:
        q3 = tl.load(
            qn_row + (3 * _G + g)[None, :],
            mask=h_mask[:, None],
            other=0.0,
        ).to(input_type)
    q_tail = tl.load(
        q_rope_ptr + t * STRIDE_QR_T + h_offs[:, None] * STRIDE_QR_H + dt[None, :],
        mask=h_mask[:, None],
        other=0.0,
    ).to(input_type)

    neg_large = -3.4028234663852886e38
    m_i = tl.full((BLOCK_H,), neg_large, dtype=tl.float32)
    l_i = tl.zeros((BLOCK_H,), dtype=tl.float32)
    acc0 = tl.zeros((BLOCK_H, _G), dtype=tl.float32)
    if NUM_GROUPS >= 2:
        acc1 = tl.zeros((BLOCK_H, _G), dtype=tl.float32)
    if NUM_GROUPS >= 3:
        acc2 = tl.zeros((BLOCK_H, _G), dtype=tl.float32)
    if NUM_GROUPS >= 4:
        acc3 = tl.zeros((BLOCK_H, _G), dtype=tl.float32)

    k_offs = tl.arange(0, BLOCK_K)
    num_tiles = tl.cdiv(topk, BLOCK_K)

    for j in tl.range(0, num_tiles, num_stages=3):
        k_start = j * BLOCK_K
        k_pos = k_start + k_offs
        valid = k_pos < topk

        slot = tl.load(idx_ptr + t * topk + k_pos, mask=valid, other=0)
        valid = valid & (slot >= 0)
        page = tl.where(valid, slot, 0).to(tl.int64)

        kv_base = kv_ptr + page[:, None] * KV_DIM
        kv0 = tl.load(kv_base + g[None, :], mask=valid[:, None], other=0.0).to(
            input_type
        )
        if NUM_GROUPS >= 2:
            kv1 = tl.load(
                kv_base + (_G + g)[None, :], mask=valid[:, None], other=0.0
            ).to(input_type)
        if NUM_GROUPS >= 3:
            kv2 = tl.load(
                kv_base + (2 * _G + g)[None, :], mask=valid[:, None], other=0.0
            ).to(input_type)
        if NUM_GROUPS >= 4:
            kv3 = tl.load(
                kv_base + (3 * _G + g)[None, :], mask=valid[:, None], other=0.0
            ).to(input_type)
        kv_tail = tl.load(
            kv_base + (D_V + dt)[None, :], mask=valid[:, None], other=0.0
        ).to(input_type)

        scores = tl.dot(q0, tl.trans(kv0))
        if NUM_GROUPS >= 2:
            scores += tl.dot(q1, tl.trans(kv1))
        if NUM_GROUPS >= 3:
            scores += tl.dot(q2, tl.trans(kv2))
        if NUM_GROUPS >= 4:
            scores += tl.dot(q3, tl.trans(kv3))
        scores += tl.dot(q_tail, tl.trans(kv_tail))
        scores = scores * qk_scale
        scores = tl.where(valid[None, :], scores, neg_large)

        m_block = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, m_block)
        alpha = tl.exp2(m_i - m_new)
        p = tl.exp2(scores - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)

        if USE_FP8_DOT:
            p_dot = (p * fp8_max).to(input_type)
        else:
            p_dot = p.to(input_type)
        acc0 = acc0 * alpha[:, None] + tl.dot(p_dot, kv0).to(tl.float32) * p_dot_scale
        if NUM_GROUPS >= 2:
            acc1 = (
                acc1 * alpha[:, None] + tl.dot(p_dot, kv1).to(tl.float32) * p_dot_scale
            )
        if NUM_GROUPS >= 3:
            acc2 = (
                acc2 * alpha[:, None] + tl.dot(p_dot, kv2).to(tl.float32) * p_dot_scale
            )
        if NUM_GROUPS >= 4:
            acc3 = (
                acc3 * alpha[:, None] + tl.dot(p_dot, kv3).to(tl.float32) * p_dot_scale
            )
        m_i = m_new

    denom = tl.maximum(l_i, 1.0e-30)
    inv_denom = 1.0 / denom
    acc0 = tl.where(l_i[:, None] > 0.0, acc0 * inv_denom[:, None], 0.0)
    if NUM_GROUPS >= 2:
        acc1 = tl.where(l_i[:, None] > 0.0, acc1 * inv_denom[:, None], 0.0)
    if NUM_GROUPS >= 3:
        acc2 = tl.where(l_i[:, None] > 0.0, acc2 * inv_denom[:, None], 0.0)
    if NUM_GROUPS >= 4:
        acc3 = tl.where(l_i[:, None] > 0.0, acc3 * inv_denom[:, None], 0.0)

    o_base = out_ptr + t * H * D_V
    tl.store(
        o_base + h_offs[:, None] * D_V + g[None, :],
        acc0.to(tl.bfloat16),
        mask=h_mask[:, None],
    )
    if NUM_GROUPS >= 2:
        tl.store(
            o_base + h_offs[:, None] * D_V + (_G + g)[None, :],
            acc1.to(tl.bfloat16),
            mask=h_mask[:, None],
        )
    if NUM_GROUPS >= 3:
        tl.store(
            o_base + h_offs[:, None] * D_V + (2 * _G + g)[None, :],
            acc2.to(tl.bfloat16),
            mask=h_mask[:, None],
        )
    if NUM_GROUPS >= 4:
        tl.store(
            o_base + h_offs[:, None] * D_V + (3 * _G + g)[None, :],
            acc3.to(tl.bfloat16),
            mask=h_mask[:, None],
        )


@triton.jit
def _sparse_mla_split_k_kernel(
    q_nope_ptr,
    q_rope_ptr,
    kv_ptr,
    idx_ptr,
    lse_partial_ptr,
    acc_partial_ptr,
    qk_scale,
    fp8_max,
    topk: tl.constexpr,
    H: tl.constexpr,
    KV_DIM: tl.constexpr,
    D_V: tl.constexpr,
    D_TAIL: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    STRIDE_QN_T: tl.constexpr,
    STRIDE_QN_H: tl.constexpr,
    STRIDE_QR_T: tl.constexpr,
    STRIDE_QR_H: tl.constexpr,
    USE_FP8_DOT: tl.constexpr,
    KV_SPLITS: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Split-K partial kernel. grid=(seq, head_blocks, kv_splits)."""
    t = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_k = tl.program_id(2)

    h_offs = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    h_mask = h_offs < H
    dt = tl.arange(0, D_TAIL)
    g = tl.arange(0, _G)

    input_type = kv_ptr.dtype.element_ty if USE_FP8_DOT else tl.bfloat16
    if USE_FP8_DOT:
        p_dot_scale = 1.0 / fp8_max
    else:
        p_dot_scale = 1.0

    qn_row = q_nope_ptr + t * STRIDE_QN_T + h_offs[:, None] * STRIDE_QN_H
    q0 = tl.load(qn_row + g[None, :], mask=h_mask[:, None], other=0.0).to(input_type)
    if NUM_GROUPS >= 2:
        q1 = tl.load(
            qn_row + (_G + g)[None, :],
            mask=h_mask[:, None],
            other=0.0,
        ).to(input_type)
    if NUM_GROUPS >= 3:
        q2 = tl.load(
            qn_row + (2 * _G + g)[None, :],
            mask=h_mask[:, None],
            other=0.0,
        ).to(input_type)
    if NUM_GROUPS >= 4:
        q3 = tl.load(
            qn_row + (3 * _G + g)[None, :],
            mask=h_mask[:, None],
            other=0.0,
        ).to(input_type)
    q_tail = tl.load(
        q_rope_ptr + t * STRIDE_QR_T + h_offs[:, None] * STRIDE_QR_H + dt[None, :],
        mask=h_mask[:, None],
        other=0.0,
    ).to(input_type)

    tiles_per_segment = tl.cdiv(topk, KV_SPLITS * BLOCK_K)
    if pid_k * tiles_per_segment * BLOCK_K >= topk:
        return
    num_tiles = tl.cdiv(topk, BLOCK_K)
    tile_start = pid_k * tiles_per_segment
    tile_end = tl.minimum((pid_k + 1) * tiles_per_segment, num_tiles)

    neg_large = -3.4028234663852886e38
    m_i = tl.full((BLOCK_H,), neg_large, dtype=tl.float32)
    l_i = tl.zeros((BLOCK_H,), dtype=tl.float32)
    acc0 = tl.zeros((BLOCK_H, _G), dtype=tl.float32)
    if NUM_GROUPS >= 2:
        acc1 = tl.zeros((BLOCK_H, _G), dtype=tl.float32)
    if NUM_GROUPS >= 3:
        acc2 = tl.zeros((BLOCK_H, _G), dtype=tl.float32)
    if NUM_GROUPS >= 4:
        acc3 = tl.zeros((BLOCK_H, _G), dtype=tl.float32)

    k_offs = tl.arange(0, BLOCK_K)
    for j in tl.range(tile_start, tile_end, num_stages=3):
        k_start = j * BLOCK_K
        k_pos = k_start + k_offs
        valid = k_pos < topk

        slot = tl.load(idx_ptr + t * topk + k_pos, mask=valid, other=0)
        valid = valid & (slot >= 0)
        page = tl.where(valid, slot, 0).to(tl.int64)

        kv_base = kv_ptr + page[:, None] * KV_DIM
        kv0 = tl.load(kv_base + g[None, :], mask=valid[:, None], other=0.0).to(
            input_type
        )
        if NUM_GROUPS >= 2:
            kv1 = tl.load(
                kv_base + (_G + g)[None, :], mask=valid[:, None], other=0.0
            ).to(input_type)
        if NUM_GROUPS >= 3:
            kv2 = tl.load(
                kv_base + (2 * _G + g)[None, :], mask=valid[:, None], other=0.0
            ).to(input_type)
        if NUM_GROUPS >= 4:
            kv3 = tl.load(
                kv_base + (3 * _G + g)[None, :], mask=valid[:, None], other=0.0
            ).to(input_type)
        kv_tail = tl.load(
            kv_base + (D_V + dt)[None, :], mask=valid[:, None], other=0.0
        ).to(input_type)

        scores = tl.dot(q0, tl.trans(kv0))
        if NUM_GROUPS >= 2:
            scores += tl.dot(q1, tl.trans(kv1))
        if NUM_GROUPS >= 3:
            scores += tl.dot(q2, tl.trans(kv2))
        if NUM_GROUPS >= 4:
            scores += tl.dot(q3, tl.trans(kv3))
        scores += tl.dot(q_tail, tl.trans(kv_tail))
        scores = scores * qk_scale
        scores = tl.where(valid[None, :], scores, neg_large)

        m_block = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, m_block)
        alpha = tl.exp2(m_i - m_new)
        p = tl.exp2(scores - m_new[:, None])
        l_new = l_i * alpha + tl.sum(p, axis=1)

        if USE_FP8_DOT:
            p_dot = (p * fp8_max).to(input_type)
        else:
            p_dot = p.to(input_type)
        acc0 = acc0 * alpha[:, None] + tl.dot(p_dot, kv0).to(tl.float32) * p_dot_scale
        if NUM_GROUPS >= 2:
            acc1 = (
                acc1 * alpha[:, None] + tl.dot(p_dot, kv1).to(tl.float32) * p_dot_scale
            )
        if NUM_GROUPS >= 3:
            acc2 = (
                acc2 * alpha[:, None] + tl.dot(p_dot, kv2).to(tl.float32) * p_dot_scale
            )
        if NUM_GROUPS >= 4:
            acc3 = (
                acc3 * alpha[:, None] + tl.dot(p_dot, kv3).to(tl.float32) * p_dot_scale
            )
        m_i = m_new
        l_i = l_new

    neg_large = -3.4028234663852886e38
    denom = tl.maximum(l_i, 1.0e-30)
    inv_denom = 1.0 / denom
    has_data = l_i > 0.0
    acc0 = tl.where(has_data[:, None], acc0 * inv_denom[:, None], 0.0)
    if NUM_GROUPS >= 2:
        acc1 = tl.where(has_data[:, None], acc1 * inv_denom[:, None], 0.0)
    if NUM_GROUPS >= 3:
        acc2 = tl.where(has_data[:, None], acc2 * inv_denom[:, None], 0.0)
    if NUM_GROUPS >= 4:
        acc3 = tl.where(has_data[:, None], acc3 * inv_denom[:, None], 0.0)

    lse = tl.where(has_data, tl.log2(l_i) + m_i, neg_large)

    H_padded = tl.cdiv(H, BLOCK_H) * BLOCK_H
    lse_base = t * KV_SPLITS * H_padded + pid_k * H_padded
    tl.store(lse_partial_ptr + lse_base + h_offs, lse, mask=h_mask)

    ap_base = t * KV_SPLITS * H_padded * D_V + pid_k * H_padded * D_V
    tl.store(
        acc_partial_ptr + ap_base + h_offs[:, None] * D_V + g[None, :],
        acc0.to(tl.bfloat16),
        mask=h_mask[:, None],
    )
    if NUM_GROUPS >= 2:
        tl.store(
            acc_partial_ptr + ap_base + h_offs[:, None] * D_V + (_G + g)[None, :],
            acc1.to(tl.bfloat16),
            mask=h_mask[:, None],
        )
    if NUM_GROUPS >= 3:
        tl.store(
            acc_partial_ptr + ap_base + h_offs[:, None] * D_V + (2 * _G + g)[None, :],
            acc2.to(tl.bfloat16),
            mask=h_mask[:, None],
        )
    if NUM_GROUPS >= 4:
        tl.store(
            acc_partial_ptr + ap_base + h_offs[:, None] * D_V + (3 * _G + g)[None, :],
            acc3.to(tl.bfloat16),
            mask=h_mask[:, None],
        )


@triton.jit
def _sparse_mla_reduce_kernel(
    lse_partial_ptr,
    acc_partial_ptr,
    out_ptr,
    H: tl.constexpr,
    D_V: tl.constexpr,
    KV_SPLITS: tl.constexpr,
    ACTIVE_SPLITS: tl.constexpr,
    ACTIVE_SPLITS_POW2: tl.constexpr,
    D_CHUNK: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Reduce split-K partials via log-space combine. grid=(seq, H, d_v_chunks)."""
    t = tl.program_id(0)
    h = tl.program_id(1)
    dc = tl.program_id(2)

    d_offs = dc * D_CHUNK + tl.arange(0, D_CHUNK)
    # tl.arange needs a power-of-two extent, but ACTIVE_SPLITS is only a power
    # of two when topk // BLOCK_K is. Iterate over the padded range and mask the
    # tail: -3.4e38 drives exp2() to 0 without the NaN an -inf would produce.
    k_offs = tl.arange(0, ACTIVE_SPLITS_POW2)
    k_mask = k_offs < ACTIVE_SPLITS
    d_mask = d_offs < D_V

    H_padded = tl.cdiv(H, 16) * 16

    lse_base = t * KV_SPLITS * H_padded
    lse_p = tl.load(
        lse_partial_ptr + lse_base + k_offs * H_padded + h,
        mask=k_mask,
        other=-3.4e38,
    )

    ap_base = t * KV_SPLITS * H_padded * D_V
    a_p = tl.load(
        acc_partial_ptr
        + ap_base
        + k_offs[:, None] * H_padded * D_V
        + h * D_V
        + d_offs[None, :],
        mask=k_mask[:, None] & d_mask[None, :],
        other=0.0,
    ).to(tl.float32)

    lse_max = tl.max(lse_p, axis=0)
    weights = tl.exp2(lse_p - lse_max)
    w_sum = tl.sum(weights, axis=0)
    scale = tl.exp2(lse_p - lse_max - tl.log2(tl.maximum(w_sum, 1.0e-30)))
    out = tl.sum(a_p * scale[:, None], axis=0)

    tl.store(
        out_ptr + t * H * D_V + h * D_V + d_offs,
        out.to(tl.bfloat16),
        mask=d_mask,
    )


def _triton_sparse_mla_fwd_splitk(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    d_v: int,
    kv_splits: int,
) -> torch.Tensor:
    """Split-K path for short sequences."""
    is_fp8 = _validate_input_dtypes(q_nope, q_rope, kv)
    use_fp8_dot = is_fp8
    seq, H, d_v_in = q_nope.shape
    assert d_v_in == d_v
    d_tail = q_rope.shape[-1]
    kv_dim = kv.shape[-1]
    topk = indices.shape[-1]
    idx_flat = indices.squeeze(1).contiguous() if indices.dim() == 3 else indices
    q_nope, stride_qn_t, stride_qn_h = _row_strides(q_nope)
    q_rope, stride_qr_t, stride_qr_h = _row_strides(q_rope)

    BLOCK_H = 16
    BLOCK_K = _sparse_mla_block_k(kv)
    n_head_blocks = (H + BLOCK_H - 1) // BLOCK_H
    h_padded = n_head_blocks * BLOCK_H

    num_groups = d_v // 128
    assert num_groups <= 4, (
        f"Triton sparse MLA supports d_v up to 512 (4 groups), got d_v={d_v}"
    )
    qk_scale = float(sm_scale) * _LOG2E

    # Preserve the established split cap when a smaller BLOCK_K is required
    # for shared-memory capacity; otherwise BF16 on a 64 KiB device would
    # double its partial-buffer traffic merely because each tile is smaller.
    max_kv_splits = max(1, topk // _PREFERRED_BLOCK_K)
    kv_splits = min(kv_splits, max_kv_splits)

    out = torch.empty(seq, H, d_v, device=q_nope.device, dtype=torch.bfloat16)

    if kv_splits == 1:
        _sparse_mla_fused_kernel[(seq, n_head_blocks)](
            q_nope,
            q_rope,
            kv,
            idx_flat,
            out,
            qk_scale,
            _FP8_MAX,
            topk=topk,
            H=H,
            KV_DIM=kv_dim,
            D_V=d_v,
            D_TAIL=d_tail,
            NUM_GROUPS=num_groups,
            STRIDE_QN_T=stride_qn_t,
            STRIDE_QN_H=stride_qn_h,
            STRIDE_QR_T=stride_qr_t,
            STRIDE_QR_H=stride_qr_h,
            USE_FP8_DOT=use_fp8_dot,
            BLOCK_H=BLOCK_H,
            BLOCK_K=BLOCK_K,
            num_warps=4,
            num_stages=2,
        )
        return out.unsqueeze(0)

    tiles_per_split = (topk + kv_splits * BLOCK_K - 1) // (kv_splits * BLOCK_K)
    active_splits = (topk + tiles_per_split * BLOCK_K - 1) // (
        tiles_per_split * BLOCK_K
    )
    active_splits = min(active_splits, kv_splits)

    lse_partial = torch.empty(
        seq, kv_splits, h_padded, dtype=torch.float32, device=q_nope.device
    )
    acc_partial = torch.empty(
        seq, kv_splits, h_padded, d_v, dtype=torch.bfloat16, device=q_nope.device
    )

    _sparse_mla_split_k_kernel[(seq, n_head_blocks, kv_splits)](
        q_nope,
        q_rope,
        kv,
        idx_flat,
        lse_partial,
        acc_partial,
        qk_scale,
        _FP8_MAX,
        topk=topk,
        H=H,
        KV_DIM=kv_dim,
        D_V=d_v,
        D_TAIL=d_tail,
        NUM_GROUPS=num_groups,
        STRIDE_QN_T=stride_qn_t,
        STRIDE_QN_H=stride_qn_h,
        STRIDE_QR_T=stride_qr_t,
        STRIDE_QR_H=stride_qr_h,
        USE_FP8_DOT=use_fp8_dot,
        KV_SPLITS=kv_splits,
        BLOCK_H=BLOCK_H,
        BLOCK_K=BLOCK_K,
        num_warps=4,
        num_stages=2,
    )

    D_CHUNK = 64
    _sparse_mla_reduce_kernel[(seq, H, (d_v + D_CHUNK - 1) // D_CHUNK)](
        lse_partial,
        acc_partial,
        out,
        H=H,
        D_V=d_v,
        KV_SPLITS=kv_splits,
        ACTIVE_SPLITS=active_splits,
        ACTIVE_SPLITS_POW2=_next_pow2(active_splits),
        D_CHUNK=D_CHUNK,
        BLOCK_K=BLOCK_K,
        num_warps=4,
    )
    return out.unsqueeze(0)


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------


def triton_sparse_mla_fwd(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    d_v: int = 512,
) -> torch.Tensor:
    """Unified sparse MLA forward. Auto-selects single-pass vs split-K.

    q_nope: [seq, H, d_v] fp8/bf16, q_rope: [seq, H, dim-d_v] fp8/bf16,
    kv: [num_pages, 1, dim] fp8/bf16, indices: [seq, 1, topk].

    Returns [1, seq, H, d_v] bf16 to match tilelang_sparse_fwd.
    """
    seq = q_nope.shape[0]
    H = q_nope.shape[1]
    num_cu = _cu_count()
    BLOCK_H = 16
    BLOCK_K = _sparse_mla_block_k(kv)
    topk = indices.shape[-1]
    max_kv_splits = max(1, topk // _PREFERRED_BLOCK_K)
    head_blocks = max(1, (H + BLOCK_H - 1) // BLOCK_H)
    base_ctas = seq * head_blocks
    if base_ctas > num_cu:
        return _triton_sparse_mla_fwd_single(q_nope, q_rope, kv, indices, sm_scale, d_v)
    kv_splits = min(
        _kv_splits_heuristic(
            seq, H, BLOCK_H, target_wg_per_cu=1.0, max_kv_splits=max_kv_splits
        ),
        max_kv_splits,
    )
    return _triton_sparse_mla_fwd_splitk(
        q_nope, q_rope, kv, indices, sm_scale, d_v, kv_splits
    )
