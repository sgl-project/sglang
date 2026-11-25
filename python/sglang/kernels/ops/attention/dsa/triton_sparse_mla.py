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

from sglang.kernels.arch import current_arch
from sglang.kernels.ops.attention.metadata import q8kv8_topk_length_from_indices
from sglang.kernels.ops.quantization.dsa_quant_k_cache import fp8_ds_mla_row_width
from sglang.kernels.ops.quantization.fp8_kernel import is_fp8_fnuz
from sglang.srt.utils import get_device_core_count, is_gfx95_supported, print_info_once

_ASYNC_COPY_OFF_ARCHES = frozenset({"gfx950"})
_IS_FNUZ = is_fp8_fnuz()
_FP8_MAX = 240.0 if _IS_FNUZ else 448.0
_LOG2E = 1.4426950408889634
_LN2 = tl.constexpr(0.6931471805599453)
_G = tl.constexpr(128)
_PREFERRED_BLOCK_K = 64
_MIN_BLOCK_K = 16
_INDEX_ELEMENT_SIZE = 4
_I32_MAX = (1 << 31) - 1
# Measured on MI355X, 64 heads, topk 2048: the decode plan with more splits (1-14 rows)
# ran 1.05-2.5x slower than the BLOCK_H=16 path; at 16 splits it is at parity.
_DECODE_MAX_SPLITS = 16

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


def _as_scaled_fp8_kv(
    kv: torch.Tensor, d_v: int, d_tail: int
) -> tuple[torch.Tensor, bool]:
    """Return ``kv`` with an FP8 element type and whether its rows are scaled.

    Scaled rows use the fp8_ds_mla layout: FP8 NoPE, one FP32 scale per 128 NoPE
    values, then a BF16 RoPE tail (656 B for D_V=512, D_TAIL=64). The kernels
    dequantize them to BF16, so the dots run in BF16 like FlashMLA's FP8 path.
    """
    if kv.dtype == torch.uint8:
        kv = kv.view(torch.float8_e4m3fn)
    kv_dim = kv.shape[-1]
    if kv.dtype == torch.bfloat16 or kv_dim == d_v + d_tail:
        return kv, False
    scaled_dim = fp8_ds_mla_row_width(d_v, d_tail)
    if kv_dim != scaled_dim:
        raise ValueError(
            f"Triton sparse MLA expects a {d_v + d_tail}-wide raw or "
            f"{scaled_dim}-byte scaled FP8 KV row; got {kv_dim}."
        )
    return kv, True


@triton.jit
def _load_kv_group(
    kv_base,
    valid,
    GROUP: tl.constexpr,
    D_V: tl.constexpr,
    KV_SCALED: tl.constexpr,
    GW: tl.constexpr = _G,
    CACHE_MOD: tl.constexpr = "",
):
    """Load NoPE columns [GROUP * GW, GROUP * GW + GW) of [BLOCK_K] KV rows."""
    # Scaled rows carry one scale per 128 NoPE values, so a group must not span two.
    tl.static_assert(not KV_SCALED or GW == _G)
    g = tl.arange(0, GW)
    kv = tl.load(
        kv_base + (GROUP * GW + g)[None, :],
        mask=valid[:, None],
        other=0.0,
        cache_modifier=CACHE_MOD,
    )
    if KV_SCALED:
        scale_ptr = (kv_base + D_V).to(tl.pointer_type(tl.float32)) + GROUP
        scale = tl.load(scale_ptr, mask=valid[:, None], other=0.0)
        kv = kv.to(tl.float32) * scale
    return kv


@triton.jit
def _load_kv_tail(
    kv_base,
    valid,
    D_V: tl.constexpr,
    D_TAIL: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    KV_SCALED: tl.constexpr,
    CACHE_MOD: tl.constexpr = "",
):
    dt = tl.arange(0, D_TAIL)
    if KV_SCALED:
        # The BF16 tail follows the FP8 NoPE bytes and one FP32 scale per group.
        tail_base = (kv_base + D_V + 4 * NUM_GROUPS).to(tl.pointer_type(tl.bfloat16))
        tail_ptr = tail_base + dt[None, :]
    else:
        tail_ptr = kv_base + (D_V + dt)[None, :]
    return tl.load(tail_ptr, mask=valid[:, None], other=0.0, cache_modifier=CACHE_MOD)


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

    hip = current_arch().hip
    return hip is not None and hip.gcn_arch in _ASYNC_COPY_OFF_ARCHES


def _no_async_copy_scope_is_effective() -> bool:
    """Whether launches inside :func:`_no_async_copy` lower without async copies.

    False when an explicit TRITON_HIP_USE_ASYNC_COPY=1 overrides the scope, or
    when this Triton exposes no knob to check (be conservative).
    """
    try:
        from triton import knobs
    except ImportError:
        return False

    amd_knobs = getattr(knobs, "amd", None)
    if not hasattr(amd_knobs, "use_async_copy"):
        return False
    if amd_knobs.use_async_copy is not None:
        return not amd_knobs.use_async_copy
    return _async_copy_is_harmful()


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


def _device_index(device: torch.device | int | None = None) -> int:
    if isinstance(device, torch.device):
        device = device.index
    if device is None:
        return torch.cuda.current_device()
    return device


@functools.lru_cache(maxsize=None)
def _cu_count_for_device(device_index: int) -> int:
    return get_device_core_count(device_index)


def _cu_count(device: torch.device | int | None = None) -> int:
    return _cu_count_for_device(_device_index(device))


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


def _scaled_kv_block_k(kv: torch.Tensor, kv_scaled: bool) -> int:
    # gfx950 with scaled-FP8 (fp8_ds_mla) KV: 64-key tiles spill scratch, 32-key tiles
    # do not (1.4-2.1x faster at 8-384 rows, measured on MI350X, K3.1).
    block_k = _sparse_mla_block_k(kv)
    if kv_scaled and is_gfx95_supported():
        return min(block_k, 32)
    return block_k


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


def _is_gfx950_sparse_mla_fp8(
    kv_dtype: torch.dtype,
    H: int,
    d_v: int,
    d_tail: int,
    kv_dim: int,
) -> bool:
    """Gate for gfx950 FP8 prefill tuning in the TP4/TP8 DSA shapes."""
    return (
        kv_dtype != torch.bfloat16
        and H in (8, 16)
        and d_v == 512
        and d_tail == 64
        and kv_dim == 576
        and is_gfx95_supported()
    )


# Tuning constants below were measured on MI355X (TP4/EP4, GLM-5.2, FP8 KV).
def _gfx950_sparse_mla_kv_splits(
    base_ctas: int,
    topk: int,
    block_k: int,
    num_cu: int,
    kv_splits: int,
    max_kv_splits: int,
) -> int:
    tiles_per_split = (topk + kv_splits * block_k - 1) // (kv_splits * block_k)
    if 4 < base_ctas <= 8 and kv_splits == 32 and tiles_per_split == 1:
        return kv_splits // 2

    tokens_per_split = (topk + kv_splits - 1) // kv_splits
    if (
        topk >= 2048
        and base_ctas <= num_cu
        and tokens_per_split >= 1024
        and kv_splits < max_kv_splits
    ):
        return min(kv_splits * 2, max_kv_splits)

    return kv_splits


def _is_gfx950_decode_fp8_64h(
    kv_dtype: torch.dtype,
    H: int,
    d_v: int,
    d_tail: int,
    kv_dim: int,
) -> bool:
    """Gate for the gfx950 FP8 decode plan: 64 heads on raw 576 B rows (GLM-5.2)."""
    return (
        kv_dtype != torch.bfloat16
        and H == 64
        and d_v == 512
        and d_tail == 64
        and kv_dim == 576
        and is_gfx95_supported()
    )


def _decode_splitk_splits(seq: int, topk: int, block_k: int, num_cu: int) -> int | None:
    """KV splits for the gfx950 FP8 decode plan at BLOCK_H == H, or None.

    The partial grid is ``(seq, 1, splits)`` at one CTA per CU, so a split past
    ``num_cu // seq`` adds a whole round; ``_kv_splits_heuristic`` rounds to a
    power of two and cannot express that. None below two splits (the launcher
    would take the fused kernel, slower at BLOCK_H == H) and above
    ``_DECODE_MAX_SPLITS`` (the BLOCK_H=16 path is faster there).
    """
    splits = min(
        num_cu // max(seq, 1),
        (topk + block_k - 1) // block_k,
        max(1, topk // _PREFERRED_BLOCK_K),
    )
    return splits if 2 <= splits <= _DECODE_MAX_SPLITS else None


def _gfx950_sparse_mla_num_warps(
    base_ctas: int, active_splits: int, num_cu: int
) -> int:
    return 2 if base_ctas * active_splits > num_cu else 4


def _page_offsets_fit_i32(num_pages: int, kv_dim: int) -> bool:
    return num_pages * kv_dim <= _I32_MAX


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
    if kwargs["KV_SCALED"] and is_gfx95_supported():
        # num_warps=1 spills ~4 KB of scratch a lane on gfx950 (16 head rows x 576
        # fp32), and launching it in the serving autotune pins that scratch on
        # every queue.
        candidates = [c for c in candidates if c.num_warps >= 2]
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
    key=["topk", "H", "USE_FP8_DOT", "KV_SCALED", "SEQ_BUCKET"],
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
    STRIDE_QN_T,
    STRIDE_QN_H,
    STRIDE_QR_T,
    STRIDE_QR_H,
    USE_FP8_DOT: tl.constexpr,
    KV_SCALED: tl.constexpr,
    SEQ_BUCKET: tl.constexpr,
    BLOCK_N: tl.constexpr,
    lse_out_ptr=None,
    STORE_LSE: tl.constexpr = False,
):
    s_i = tl.program_id(0).to(tl.int64)

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

        kv0 = _load_kv_group(kbase, valid, 0, D_V, KV_SCALED).to(input_type)
        if NUM_GROUPS >= 2:
            kv1 = _load_kv_group(kbase, valid, 1, D_V, KV_SCALED).to(input_type)
        if NUM_GROUPS >= 3:
            kv2 = _load_kv_group(kbase, valid, 2, D_V, KV_SCALED).to(input_type)
        if NUM_GROUPS >= 4:
            kv3 = _load_kv_group(kbase, valid, 3, D_V, KV_SCALED).to(input_type)
        kv_tail = _load_kv_tail(kbase, valid, D_V, D_TAIL, NUM_GROUPS, KV_SCALED).to(
            input_type
        )

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
    if STORE_LSE:
        # A row whose keys are all masked keeps m_i at the -3.4e38 sentinel.
        lse = tl.where(m_i > -1.0e38, (m_i + tl.log2(l_safe)) * _LN2, float("-inf"))
        tl.store(lse_out_ptr + s_i * H + h, lse)
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
    return_lse: bool = False,
    bf16_dot: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Single-pass prefill: grid=(seq,), loops over all topk per CTA."""
    kv, kv_scaled = _as_scaled_fp8_kv(kv, d_v, q_rope.shape[-1])
    is_fp8 = _validate_input_dtypes(q_nope, q_rope, kv)
    use_fp8_dot = is_fp8 and not kv_scaled and not bf16_dot
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
        lse = (
            torch.empty(seq, H_pad, device=q_nope.device, dtype=torch.float32)
            if return_lse
            else None
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
            KV_SCALED=kv_scaled,
            SEQ_BUCKET=seq_bucket,
            lse_out_ptr=lse,
            STORE_LSE=return_lse,
        )
        out = out_pad[:, :H, :].contiguous()
        if return_lse:
            lse = lse[:, :H].contiguous()
    else:
        lse = (
            torch.empty(seq, H, device=q_nope.device, dtype=torch.float32)
            if return_lse
            else None
        )
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
            KV_SCALED=kv_scaled,
            SEQ_BUCKET=seq_bucket,
            lse_out_ptr=lse,
            STORE_LSE=return_lse,
        )
    return (out.unsqueeze(0), lse) if return_lse else out.unsqueeze(0)


def _prev_pow2(n: int) -> int:
    if n < 1:
        return 1
    return 1 << (n.bit_length() - 1)


def _next_pow2(n: int) -> int:
    if n < 1:
        return 1
    return 1 << (n - 1).bit_length()


def _reduce_d_chunk(active_splits: int, output_rows: int = 0) -> int:
    if active_splits <= 4:
        return 512 if output_rows >= 1536 else 128
    return 64


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
    topk_length_ptr,
    out_ptr,
    qk_scale,
    fp8_max,
    topk: tl.constexpr,
    H: tl.constexpr,
    KV_DIM: tl.constexpr,
    D_V: tl.constexpr,
    D_TAIL: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    STRIDE_QN_T,
    STRIDE_QN_H,
    STRIDE_QR_T,
    STRIDE_QR_H,
    USE_FP8_DOT: tl.constexpr,
    USE_TOPK_LENGTH: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
    USE_I64_PAGE: tl.constexpr = True,
    PIPE_STAGES: tl.constexpr = 3,
    lse_out_ptr=None,
    STORE_LSE: tl.constexpr = False,
    NATURAL_LSE: tl.constexpr = False,
    # Set only when 0 <= topk_length <= topk and every index past it is -1,
    # as transform_index_page_table_dcp writes the DCP-localized table: the
    # tile count skips the clamp and the index load masks on the constexpr
    # topk, leaving the -1 tail to slot >= 0.
    TOPK_TAIL_INVALID: tl.constexpr = False,
    KV_SCALED: tl.constexpr = False,
):
    """Single-pass with head-block splitting. grid=(seq, head_blocks)."""
    t = tl.program_id(0).to(tl.int64)
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
    valid_topk = topk
    if TOPK_TAIL_INVALID:
        num_tiles = tl.cdiv(tl.load(topk_length_ptr + t), BLOCK_K)
    else:
        if USE_TOPK_LENGTH:
            valid_topk = tl.minimum(tl.maximum(tl.load(topk_length_ptr + t), 0), topk)
        num_tiles = tl.cdiv(valid_topk, BLOCK_K)

    for j in tl.range(0, num_tiles, num_stages=PIPE_STAGES):
        k_start = j * BLOCK_K
        k_pos = k_start + k_offs
        valid = k_pos < valid_topk

        slot = tl.load(idx_ptr + t * topk + k_pos, mask=valid, other=0)
        valid = valid & (slot >= 0)
        page = tl.where(valid, slot, 0)
        if USE_I64_PAGE:
            page = page.to(tl.int64)

        kv_base = kv_ptr + page[:, None] * KV_DIM
        kv0 = _load_kv_group(kv_base, valid, 0, D_V, KV_SCALED).to(input_type)
        if NUM_GROUPS >= 2:
            kv1 = _load_kv_group(kv_base, valid, 1, D_V, KV_SCALED).to(input_type)
        if NUM_GROUPS >= 3:
            kv2 = _load_kv_group(kv_base, valid, 2, D_V, KV_SCALED).to(input_type)
        if NUM_GROUPS >= 4:
            kv3 = _load_kv_group(kv_base, valid, 3, D_V, KV_SCALED).to(input_type)
        kv_tail = _load_kv_tail(kv_base, valid, D_V, D_TAIL, NUM_GROUPS, KV_SCALED).to(
            input_type
        )

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

    if STORE_LSE:
        if NATURAL_LSE:
            # A row whose keys are all masked keeps m_i at the -3.4e38 sentinel.
            lse = tl.where(m_i > -1.0e38, (m_i + tl.log2(denom)) * _LN2, float("-inf"))
        else:
            # Finite floor: the DCP merge weights an empty rank by exp2(lse - max),
            # so an all-masked row must not come back as -inf.
            lse = tl.where(l_i > 0.0, tl.log2(denom) + m_i, -1073741824.0)
        tl.store(lse_out_ptr + t * H + h_offs, lse, mask=h_mask)

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
    STRIDE_QN_T,
    STRIDE_QN_H,
    STRIDE_QR_T,
    STRIDE_QR_H,
    USE_FP8_DOT: tl.constexpr,
    KV_SPLITS: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
    H_PADDED: tl.constexpr = 0,
    USE_I64_PAGE: tl.constexpr = True,
    DENSE_TILES: tl.constexpr = False,
    PIPE_STAGES: tl.constexpr = 3,
    GROUP_W: tl.constexpr = 0,
    ACC_INPLACE: tl.constexpr = False,
    KV_CACHE_MOD: tl.constexpr = "",
    DECOUPLE_DOTS: tl.constexpr = False,
    PARTIAL_DQ: tl.constexpr = False,
    LEAN_EPILOGUE: tl.constexpr = False,
    PARTIAL_STORE_MOD: tl.constexpr = "",
    FIXED_TRIP: tl.constexpr = False,
    topk_length_ptr=None,
    USE_TOPK_LENGTH: tl.constexpr = False,
    KV_SCALED: tl.constexpr = False,
):
    """Split-K partial kernel. grid=(seq, head_blocks, kv_splits).

    The decode plan (``_triton_sparse_mla_fwd_splitk(decode_tuned=True)``) turns
    on the specialisations below; every other caller keeps their defaults.

    - ``GROUP_W``: width of one value-dimension group (0 = 128); wider means
      fewer loads per gathered KV row.
    - ``DECOUPLE_DOTS`` (requires ``ACC_INPLACE``): puts the P@V dots in their
      own region so Triton's AMD dot layout does not treat Q@K^T -> P@V as an
      attention chain, which at BLOCK_H = 64 computes Q@K^T twice per CTA. Must
      compile without direct-to-LDS async copies (see ``split_async_scope``).
    - ``PARTIAL_DQ``: stores each (row, split) slab of ``acc_partial`` as
      ``[d // 4][h][d % 4]`` (read by :func:`_sparse_mla_reduce_dq_kernel`) so a
      store covers whole cache lines; requires ``BLOCK_H == H``.
    - ``LEAN_EPILOGUE``: drops the ``l_i > 0`` select; every CTA runs at least
      one tile and each tile adds at least 1 to ``l_i``.
    - ``PARTIAL_STORE_MOD``: cache modifier of the partial stores (``".wt"``
      leaves no dirty lines for the end-of-kernel release to write back).
    - ``FIXED_TRIP``: a compile-time tile count; tiles past ``topk`` read slot
      -1 and add nothing.
    """
    # FIXED_TRIP's compile-time trip count covers the whole topk; a per-row
    # topk_length bound is only honoured by the runtime trip count.
    tl.static_assert(not (FIXED_TRIP and USE_TOPK_LENGTH))
    t = tl.program_id(0).to(tl.int64)
    pid_h = tl.program_id(1)
    pid_k = tl.program_id(2)

    h_offs = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    h_mask = h_offs < H
    dt = tl.arange(0, D_TAIL)
    GW: tl.constexpr = _G if GROUP_W == 0 else GROUP_W
    g = tl.arange(0, GW)

    input_type = kv_ptr.dtype.element_ty if USE_FP8_DOT else tl.bfloat16
    if USE_FP8_DOT:
        p_dot_scale = 1.0 / fp8_max
    else:
        p_dot_scale = 1.0

    qn_row = q_nope_ptr + t * STRIDE_QN_T + h_offs[:, None] * STRIDE_QN_H
    q0 = tl.load(qn_row + g[None, :], mask=h_mask[:, None], other=0.0).to(input_type)
    if NUM_GROUPS >= 2:
        q1 = tl.load(
            qn_row + (GW + g)[None, :],
            mask=h_mask[:, None],
            other=0.0,
        ).to(input_type)
    if NUM_GROUPS >= 3:
        q2 = tl.load(
            qn_row + (2 * GW + g)[None, :],
            mask=h_mask[:, None],
            other=0.0,
        ).to(input_type)
    if NUM_GROUPS >= 4:
        q3 = tl.load(
            qn_row + (3 * GW + g)[None, :],
            mask=h_mask[:, None],
            other=0.0,
        ).to(input_type)
    q_tail = tl.load(
        q_rope_ptr + t * STRIDE_QR_T + h_offs[:, None] * STRIDE_QR_H + dt[None, :],
        mask=h_mask[:, None],
        other=0.0,
    ).to(input_type)

    if USE_TOPK_LENGTH:
        # The reduce reads all KV_SPLITS partials, so a split left with no tiles
        # must fall through and store an empty one rather than return early.
        valid_topk = tl.minimum(tl.maximum(tl.load(topk_length_ptr + t), 0), topk)
        num_tiles = tl.cdiv(valid_topk, BLOCK_K)
        tiles_per_segment = tl.cdiv(num_tiles, KV_SPLITS)
        tile_start = pid_k * tiles_per_segment
        tile_end = tl.minimum(tile_start + tiles_per_segment, num_tiles)
    else:
        tiles_per_segment = tl.cdiv(topk, KV_SPLITS * BLOCK_K)
        if pid_k * tiles_per_segment * BLOCK_K >= topk:
            return
        num_tiles = tl.cdiv(topk, BLOCK_K)
        tile_start = pid_k * tiles_per_segment
        tile_end = tl.minimum((pid_k + 1) * tiles_per_segment, num_tiles)

    neg_large = -3.4028234663852886e38
    m_i = tl.full((BLOCK_H,), neg_large, dtype=tl.float32)
    l_i = tl.zeros((BLOCK_H,), dtype=tl.float32)
    acc0 = tl.zeros((BLOCK_H, GW), dtype=tl.float32)
    if NUM_GROUPS >= 2:
        acc1 = tl.zeros((BLOCK_H, GW), dtype=tl.float32)
    if NUM_GROUPS >= 3:
        acc2 = tl.zeros((BLOCK_H, GW), dtype=tl.float32)
    if NUM_GROUPS >= 4:
        acc3 = tl.zeros((BLOCK_H, GW), dtype=tl.float32)

    k_offs = tl.arange(0, BLOCK_K)
    idx_row = idx_ptr + t * topk
    # FIXED_TRIP (see the docstring): a compile-time trip count, with the
    # tiles past ``topk`` masked through the index load below.
    if FIXED_TRIP:
        loop_lo = 0
        loop_hi: tl.constexpr = (topk + KV_SPLITS * BLOCK_K - 1) // (
            KV_SPLITS * BLOCK_K
        )
    else:
        loop_lo = tile_start
        loop_hi = tile_end
    for jj in tl.range(loop_lo, loop_hi, num_stages=PIPE_STAGES):
        if FIXED_TRIP:
            j = tile_start + jj
        else:
            j = jj
        k_start = j * BLOCK_K
        k_pos = k_start + k_offs

        if DENSE_TILES and not FIXED_TRIP:
            slot = tl.load(idx_row + k_pos)
        else:
            # Under FIXED_TRIP the last split may run past ``topk``: its
            # extra tiles read slot -1, i.e. fully masked.
            slot = tl.load(idx_row + k_pos, mask=k_pos < topk, other=-1)

        if DENSE_TILES:
            valid = slot >= 0
        else:
            valid = (k_pos < topk) & (slot >= 0)
        page = tl.where(valid, slot, 0)
        if USE_I64_PAGE:
            kv_base = kv_ptr + page[:, None].to(tl.int64) * KV_DIM
        else:
            kv_base = kv_ptr + page[:, None] * KV_DIM

        kv0 = _load_kv_group(kv_base, valid, 0, D_V, KV_SCALED, GW, KV_CACHE_MOD).to(
            input_type
        )
        if NUM_GROUPS >= 2:
            kv1 = _load_kv_group(
                kv_base, valid, 1, D_V, KV_SCALED, GW, KV_CACHE_MOD
            ).to(input_type)
        if NUM_GROUPS >= 3:
            kv2 = _load_kv_group(
                kv_base, valid, 2, D_V, KV_SCALED, GW, KV_CACHE_MOD
            ).to(input_type)
        if NUM_GROUPS >= 4:
            kv3 = _load_kv_group(
                kv_base, valid, 3, D_V, KV_SCALED, GW, KV_CACHE_MOD
            ).to(input_type)
        kv_tail = _load_kv_tail(
            kv_base, valid, D_V, D_TAIL, NUM_GROUPS, KV_SCALED, KV_CACHE_MOD
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
        if ACC_INPLACE:
            # D = A*B + C folds the rescaled accumulator straight into the
            # MFMA's C operand; the uniform 1/fp8_max factor is hoisted into
            # the epilogue (algebraically exact -- every increment carries it,
            # and the alpha rescale is linear).
            if DECOUPLE_DOTS:
                # Region fence for the P@V dots (see DECOUPLE_DOTS in the
                # docstring). fp8_max is always 240 or 448 and both arms are
                # identical, so this is the plain in-place update; the fence
                # only steers Triton's AMD dot-layout choice.
                if fp8_max > 0.0:
                    acc0 = tl.dot(p_dot, kv0, acc0 * alpha[:, None])
                    if NUM_GROUPS >= 2:
                        acc1 = tl.dot(p_dot, kv1, acc1 * alpha[:, None])
                    if NUM_GROUPS >= 3:
                        acc2 = tl.dot(p_dot, kv2, acc2 * alpha[:, None])
                    if NUM_GROUPS >= 4:
                        acc3 = tl.dot(p_dot, kv3, acc3 * alpha[:, None])
                else:
                    acc0 = tl.dot(p_dot, kv0, acc0 * alpha[:, None])
                    if NUM_GROUPS >= 2:
                        acc1 = tl.dot(p_dot, kv1, acc1 * alpha[:, None])
                    if NUM_GROUPS >= 3:
                        acc2 = tl.dot(p_dot, kv2, acc2 * alpha[:, None])
                    if NUM_GROUPS >= 4:
                        acc3 = tl.dot(p_dot, kv3, acc3 * alpha[:, None])
            else:
                acc0 = tl.dot(p_dot, kv0, acc0 * alpha[:, None])
                if NUM_GROUPS >= 2:
                    acc1 = tl.dot(p_dot, kv1, acc1 * alpha[:, None])
                if NUM_GROUPS >= 3:
                    acc2 = tl.dot(p_dot, kv2, acc2 * alpha[:, None])
                if NUM_GROUPS >= 4:
                    acc3 = tl.dot(p_dot, kv3, acc3 * alpha[:, None])
        else:
            acc0 = (
                acc0 * alpha[:, None] + tl.dot(p_dot, kv0).to(tl.float32) * p_dot_scale
            )
            if NUM_GROUPS >= 2:
                acc1 = (
                    acc1 * alpha[:, None]
                    + tl.dot(p_dot, kv1).to(tl.float32) * p_dot_scale
                )
            if NUM_GROUPS >= 3:
                acc2 = (
                    acc2 * alpha[:, None]
                    + tl.dot(p_dot, kv2).to(tl.float32) * p_dot_scale
                )
            if NUM_GROUPS >= 4:
                acc3 = (
                    acc3 * alpha[:, None]
                    + tl.dot(p_dot, kv3).to(tl.float32) * p_dot_scale
                )
        m_i = m_new
        l_i = l_new

    neg_large = -3.4028234663852886e38
    denom = tl.maximum(l_i, 1.0e-30)
    inv_denom = 1.0 / denom
    if ACC_INPLACE:
        out_scale = inv_denom * p_dot_scale
    else:
        out_scale = inv_denom
    has_data = l_i > 0.0
    if LEAN_EPILOGUE:
        acc0 = acc0 * out_scale[:, None]
        if NUM_GROUPS >= 2:
            acc1 = acc1 * out_scale[:, None]
        if NUM_GROUPS >= 3:
            acc2 = acc2 * out_scale[:, None]
        if NUM_GROUPS >= 4:
            acc3 = acc3 * out_scale[:, None]
    else:
        acc0 = tl.where(has_data[:, None], acc0 * out_scale[:, None], 0.0)
        if NUM_GROUPS >= 2:
            acc1 = tl.where(has_data[:, None], acc1 * out_scale[:, None], 0.0)
        if NUM_GROUPS >= 3:
            acc2 = tl.where(has_data[:, None], acc2 * out_scale[:, None], 0.0)
        if NUM_GROUPS >= 4:
            acc3 = tl.where(has_data[:, None], acc3 * out_scale[:, None], 0.0)

    lse = tl.where(has_data, tl.log2(l_i) + m_i, neg_large)

    # Head stride of the partial buffers; 0 means cdiv(H, BLOCK_H) * BLOCK_H.
    H_padded: tl.constexpr = (
        tl.cdiv(H, BLOCK_H) * BLOCK_H if H_PADDED == 0 else H_PADDED
    )
    lse_base = t * KV_SPLITS * H_padded + pid_k * H_padded
    tl.store(
        lse_partial_ptr + lse_base + h_offs,
        lse,
        mask=h_mask,
        cache_modifier=PARTIAL_STORE_MOD,
    )

    partial_ty: tl.constexpr = tl.bfloat16
    ap_base = t * KV_SPLITS * H_padded * D_V + pid_k * H_padded * D_V
    if PARTIAL_DQ:
        tl.static_assert(H_padded == H)
        tl.static_assert(BLOCK_H == H)
        # [d // 4][h][d % 4]: see PARTIAL_DQ in the docstring.
        hq = h_offs[:, None] * 4
        gq = (g // 4)[None, :] * (4 * H_padded) + (g % 4)[None, :]
        GQ: tl.constexpr = (GW // 4) * (4 * H_padded)
        tl.store(
            acc_partial_ptr + ap_base + gq + hq,
            acc0.to(partial_ty),
            cache_modifier=PARTIAL_STORE_MOD,
        )
        if NUM_GROUPS >= 2:
            tl.store(
                acc_partial_ptr + ap_base + GQ + gq + hq,
                acc1.to(partial_ty),
                cache_modifier=PARTIAL_STORE_MOD,
            )
        if NUM_GROUPS >= 3:
            tl.store(
                acc_partial_ptr + ap_base + 2 * GQ + gq + hq,
                acc2.to(partial_ty),
                cache_modifier=PARTIAL_STORE_MOD,
            )
        if NUM_GROUPS >= 4:
            tl.store(
                acc_partial_ptr + ap_base + 3 * GQ + gq + hq,
                acc3.to(partial_ty),
                cache_modifier=PARTIAL_STORE_MOD,
            )
        return
    tl.store(
        acc_partial_ptr + ap_base + h_offs[:, None] * D_V + g[None, :],
        acc0.to(partial_ty),
        mask=h_mask[:, None],
        cache_modifier=PARTIAL_STORE_MOD,
    )
    if NUM_GROUPS >= 2:
        tl.store(
            acc_partial_ptr + ap_base + h_offs[:, None] * D_V + (GW + g)[None, :],
            acc1.to(partial_ty),
            mask=h_mask[:, None],
            cache_modifier=PARTIAL_STORE_MOD,
        )
    if NUM_GROUPS >= 3:
        tl.store(
            acc_partial_ptr + ap_base + h_offs[:, None] * D_V + (2 * GW + g)[None, :],
            acc2.to(partial_ty),
            mask=h_mask[:, None],
            cache_modifier=PARTIAL_STORE_MOD,
        )
    if NUM_GROUPS >= 4:
        tl.store(
            acc_partial_ptr + ap_base + h_offs[:, None] * D_V + (3 * GW + g)[None, :],
            acc3.to(partial_ty),
            mask=h_mask[:, None],
            cache_modifier=PARTIAL_STORE_MOD,
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
    lse_out_ptr=None,
    STORE_LSE: tl.constexpr = False,
    NATURAL_LSE: tl.constexpr = False,
):
    """Reduce split-K partials via log-space combine. grid=(seq, H, d_v_chunks)."""
    t = tl.program_id(0).to(tl.int64)
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
    if STORE_LSE:
        if dc == 0:
            lse = lse_max + tl.log2(tl.maximum(w_sum, 1.0e-30))
            if NATURAL_LSE:
                # Every split empty leaves lse_max at the -3.4e38 sentinel.
                lse = tl.where(lse_max > -1.0e38, lse * _LN2, float("-inf"))
            tl.store(lse_out_ptr + t * H + h, lse)


@triton.jit
def _sparse_mla_reduce_dq_kernel(
    lse_partial_ptr,
    acc_partial_ptr,
    out_ptr,
    H: tl.constexpr,
    D_V: tl.constexpr,
    KV_SPLITS: tl.constexpr,
    ACTIVE_SPLITS: tl.constexpr,
    ACTIVE_SPLITS_POW2: tl.constexpr,
    D_CHUNK: tl.constexpr,
    H_PADDED: tl.constexpr,
    OUT_STORE_MOD: tl.constexpr = "",
):
    """Reduce split-K partials stored in the ``PARTIAL_DQ`` layout.

    grid=(seq, D_V // D_CHUNK). One CTA owns every head of one ``D_CHUNK``-wide
    slice of one row; per split it reads one contiguous
    ``(D_CHUNK // 4) x H_PADDED x 4`` block, indexed in memory order
    ``[k][d // 4][h][d % 4]`` so a wave's loads walk whole 256 B runs. The
    log-space combine is the same expression, in the same order, as
    :func:`_sparse_mla_reduce_kernel`. ``OUT_STORE_MOD`` is the cache modifier
    of the output store (``".wt"`` on the decode plan: see PARTIAL_STORE_MOD
    of :func:`_sparse_mla_split_k_kernel`).
    """
    t = tl.program_id(0).to(tl.int64)
    c = tl.program_id(1)
    DQ: tl.constexpr = D_CHUNK // 4
    k_offs = tl.arange(0, ACTIVE_SPLITS_POW2)
    k_mask = k_offs < ACTIVE_SPLITS
    h_offs = tl.arange(0, H_PADDED)
    h_mask = h_offs < H
    q_offs = tl.arange(0, DQ)
    e_offs = tl.arange(0, 4)

    lse_base = t * KV_SPLITS * H_PADDED
    ap_base = t * KV_SPLITS * H_PADDED * D_V
    lse_p = tl.load(
        lse_partial_ptr + lse_base + k_offs[:, None] * H_PADDED + h_offs[None, :],
        mask=k_mask[:, None] & h_mask[None, :],
        other=-3.4e38,
    )
    a_off = (
        k_offs[:, None, None, None] * (H_PADDED * D_V)
        + (c * DQ + q_offs)[None, :, None, None] * (4 * H_PADDED)
        + h_offs[None, None, :, None] * 4
        + e_offs[None, None, None, :]
    )
    a_p = tl.load(
        acc_partial_ptr + ap_base + a_off,
        mask=k_mask[:, None, None, None],
        other=0.0,
    ).to(tl.float32)

    lse_max = tl.max(lse_p, axis=0)
    weights = tl.exp2(lse_p - lse_max[None, :])
    w_sum = tl.sum(weights, axis=0)
    scale = tl.exp2(
        lse_p - lse_max[None, :] - tl.log2(tl.maximum(w_sum, 1.0e-30))[None, :]
    )
    out = tl.sum(a_p * scale[:, None, :, None], axis=0)
    out = tl.reshape(tl.permute(out, (1, 0, 2)), (H_PADDED, D_CHUNK))
    d_offs = c * D_CHUNK + tl.arange(0, D_CHUNK)
    tl.store(
        out_ptr + t * H * D_V + h_offs[:, None] * D_V + d_offs[None, :],
        out.to(tl.bfloat16),
        mask=h_mask[:, None],
        cache_modifier=OUT_STORE_MOD,
    )


def _triton_sparse_mla_fwd_splitk(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    d_v: int,
    kv_splits: int,
    topk_length: torch.Tensor | None = None,
    use_topk_length: bool = False,
    decode_tuned: bool = False,
    return_lse: bool = False,
    bf16_dot: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Split-K path for short sequences.

    ``decode_tuned`` selects the gfx950 FP8 decode plan: one head block per row
    (BLOCK_H == H) with the partial-loop specialisations of
    :func:`_sparse_mla_split_k_kernel` and the ``PARTIAL_DQ`` reduce, which
    stores no LSE.
    """
    assert not (decode_tuned and (return_lse or use_topk_length or kv_splits < 2))
    kv, kv_scaled = _as_scaled_fp8_kv(kv, d_v, q_rope.shape[-1])
    is_fp8 = _validate_input_dtypes(q_nope, q_rope, kv)
    use_fp8_dot = is_fp8 and not kv_scaled and not bf16_dot
    seq, H, d_v_in = q_nope.shape
    assert d_v_in == d_v
    d_tail = q_rope.shape[-1]
    kv_dim = kv.shape[-1]
    topk = indices.shape[-1]
    idx_flat = indices.squeeze(1).contiguous() if indices.dim() == 3 else indices
    if use_topk_length:
        assert topk_length is not None
        assert topk_length.dtype == torch.int32
        assert topk_length.device == q_nope.device
        assert topk_length.numel() >= seq
        assert topk_length.is_contiguous()
        topk_length_ptr = topk_length.reshape(-1)
    else:
        topk_length_ptr = idx_flat
    q_nope, stride_qn_t, stride_qn_h = _row_strides(q_nope)
    q_rope, stride_qr_t, stride_qr_h = _row_strides(q_rope)

    BLOCK_H = H if decode_tuned else 16
    BLOCK_K = _scaled_kv_block_k(kv, kv_scaled)
    n_head_blocks = (H + BLOCK_H - 1) // BLOCK_H
    h_padded = n_head_blocks * BLOCK_H
    num_cu = _cu_count(q_nope.device)
    base_ctas = seq * n_head_blocks
    optimize_gfx950_fp8 = _is_gfx950_sparse_mla_fp8(kv.dtype, H, d_v, d_tail, kv_dim)

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
    if use_topk_length:
        kv_splits = 1

    out = torch.empty(seq, H, d_v, device=q_nope.device, dtype=torch.bfloat16)
    lse = (
        torch.empty(seq, H, device=q_nope.device, dtype=torch.float32)
        if return_lse
        else None
    )

    if kv_splits == 1:
        fused_num_warps = 4
        if optimize_gfx950_fp8:
            fused_num_warps = _gfx950_sparse_mla_num_warps(base_ctas, 1, num_cu)
        # i32 page offsets are correct whenever they fit; the rest is a tuning
        # gate, since narrowing only paid off in this configuration.
        i32_page_safe = _page_offsets_fit_i32(kv.shape[0], kv_dim)
        i32_page_tuned = optimize_gfx950_fp8 and fused_num_warps == 2 and topk >= 2048
        fused_pipe_stages = 1 if use_topk_length else 3
        # Direct-to-LDS staging of the gathered KV operand is a loss for the
        # deep-prologue fused variant on an fp8 cache (measured); the shallower
        # pipelines are neutral, and a bf16 cache measured 1.7x slower without
        # it, so those keep their compile-cache keys. Scaled-FP8 rows were not
        # measured either way and keep the backend default.
        fused_async_scope = (
            _no_async_copy()
            if fused_pipe_stages >= 3 and is_fp8 and not kv_scaled
            else contextlib.nullcontext()
        )
        with fused_async_scope:
            _sparse_mla_fused_kernel[(seq, n_head_blocks)](
                q_nope,
                q_rope,
                kv,
                idx_flat,
                topk_length_ptr,
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
                USE_TOPK_LENGTH=use_topk_length,
                BLOCK_H=BLOCK_H,
                BLOCK_K=BLOCK_K,
                USE_I64_PAGE=not (i32_page_safe and i32_page_tuned),
                PIPE_STAGES=fused_pipe_stages,
                KV_SCALED=kv_scaled,
                lse_out_ptr=lse,
                STORE_LSE=return_lse,
                NATURAL_LSE=True,
                num_warps=fused_num_warps,
                num_stages=2,
            )
        return (out.unsqueeze(0), lse) if return_lse else out.unsqueeze(0)

    tiles_per_split = (topk + kv_splits * BLOCK_K - 1) // (kv_splits * BLOCK_K)
    active_splits = (topk + tiles_per_split * BLOCK_K - 1) // (
        tiles_per_split * BLOCK_K
    )
    active_splits = min(active_splits, kv_splits)
    split_num_warps = 4
    if optimize_gfx950_fp8:
        split_num_warps = _gfx950_sparse_mla_num_warps(base_ctas, active_splits, num_cu)

    # Partial-loop shape specialisations. Only the measured decode plan opts in,
    # so every legacy caller keeps its current lowering (and compile-cache key).
    split_i32_page = decode_tuned and _page_offsets_fit_i32(kv.shape[0], kv_dim)
    split_dense_tiles = decode_tuned and topk % BLOCK_K == 0
    # 256-wide value groups: fewer loads per gathered row; measured a win only
    # at a two-deep pipeline.
    split_group_w = 256 if decode_tuned and d_v % 256 == 0 else 128
    split_pipe_stages = 2 if decode_tuned else 3
    split_num_groups = d_v // split_group_w
    # Accumulator rescale folded into the MFMA C operand, 1/fp8_max hoisted to
    # the epilogue.
    split_acc_inplace = bool(decode_tuned)
    # Streaming hint on the gathered KV rows: a random walk over the pool is
    # never reused, and allocating it evicts q, the index row and the partials.
    split_kv_cache_mod = ".cs" if decode_tuned else ""
    split_extra: dict = {"matrix_instr_nonkdim": 32} if decode_tuned else {}
    # Only where the launch is guaranteed to compile without async copies (see
    # split_async_scope below).
    split_decouple_dots = (
        bool(decode_tuned) and split_acc_inplace and _no_async_copy_scope_is_effective()
    )

    lse_partial = torch.empty(
        seq, kv_splits, h_padded, dtype=torch.float32, device=q_nope.device
    )
    acc_partial = torch.empty(
        seq, kv_splits, h_padded, d_v, dtype=torch.bfloat16, device=q_nope.device
    )

    # The decode plan lets direct-to-LDS async copies stage the gathered KV,
    # except with DECOUPLE_DOTS: with Triton 3.8.0 (ROCm 10) that combination
    # was observed to corrupt heads 48-63 nondeterministically (~24 dB); 3.7.0
    # is exact. Suspected cause: https://github.com/ROCm/triton/issues/985.
    # Every other caller keeps main's staged lowering.
    split_async_scope = (
        contextlib.nullcontext()
        if decode_tuned and not split_decouple_dots
        else _no_async_copy()
    )
    split_partial_dq = bool(decode_tuned)
    split_lean_epilogue = bool(decode_tuned)
    split_store_mod = ".wt" if decode_tuned else ""
    split_fixed_trip = bool(decode_tuned)
    with split_async_scope:
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
            NUM_GROUPS=split_num_groups,
            STRIDE_QN_T=stride_qn_t,
            STRIDE_QN_H=stride_qn_h,
            STRIDE_QR_T=stride_qr_t,
            STRIDE_QR_H=stride_qr_h,
            USE_FP8_DOT=use_fp8_dot,
            KV_SPLITS=kv_splits,
            BLOCK_H=BLOCK_H,
            BLOCK_K=BLOCK_K,
            H_PADDED=h_padded,
            USE_I64_PAGE=not split_i32_page,
            DENSE_TILES=split_dense_tiles,
            PIPE_STAGES=split_pipe_stages,
            GROUP_W=split_group_w,
            ACC_INPLACE=split_acc_inplace,
            KV_CACHE_MOD=split_kv_cache_mod,
            DECOUPLE_DOTS=split_decouple_dots,
            PARTIAL_DQ=split_partial_dq,
            LEAN_EPILOGUE=split_lean_epilogue,
            PARTIAL_STORE_MOD=split_store_mod,
            FIXED_TRIP=split_fixed_trip,
            KV_SCALED=kv_scaled,
            num_warps=split_num_warps,
            num_stages=2,
            **split_extra,
        )

    if split_partial_dq:
        # 64-wide d slices x every head, 8 warps: the best of a 4x4
        # (D_CHUNK x num_warps) screen, all within 0.3 us of each other.
        dq_chunk = 64 if d_v % 64 == 0 else d_v
        with _no_async_copy():
            _sparse_mla_reduce_dq_kernel[(seq, d_v // dq_chunk)](
                lse_partial,
                acc_partial,
                out,
                H=H,
                D_V=d_v,
                KV_SPLITS=kv_splits,
                ACTIVE_SPLITS=active_splits,
                ACTIVE_SPLITS_POW2=_next_pow2(active_splits),
                D_CHUNK=dq_chunk,
                H_PADDED=h_padded,
                OUT_STORE_MOD=split_store_mod,
                num_warps=8,
            )
        return out.unsqueeze(0)

    D_CHUNK = _reduce_d_chunk(active_splits, seq * H) if optimize_gfx950_fp8 else 64
    with _no_async_copy():
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
            lse_out_ptr=lse,
            STORE_LSE=return_lse,
            NATURAL_LSE=True,
            num_warps=4,
        )
    return (out.unsqueeze(0), lse) if return_lse else out.unsqueeze(0)


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------


# 16..512 verify rows take the gfx950 HIP split-K (faster on MI355X); fewer stay on Triton split-K
_HIP_SPLIT_MIN_ROWS = 16
_HIP_SPLIT_MAX_ROWS = 512


def triton_sparse_mla_fwd(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    d_v: int = 512,
    topk_length: torch.Tensor | None = None,
    max_topk_length: int | None = None,
    return_lse: bool = False,
    bf16_dot: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Unified sparse MLA forward. Auto-selects single-pass vs split-K.

    q_nope: [seq, H, d_v] fp8/bf16, q_rope: [seq, H, dim-d_v] fp8/bf16,
    kv: [num_pages, 1, dim] fp8/bf16, indices: [seq, 1, topk].
    topk_length optionally gives a per-row upper bound on the last valid index;
    max_topk_length is its host-known batch maximum.
    bf16_dot: with a raw (unscaled) FP8 KV and BF16 q, upcast the KV and run the
    dots in BF16 instead of casting q and the probabilities to FP8.

    Returns [1, seq, H, d_v] bf16 to match tilelang_sparse_fwd, and with
    return_lse also its [seq, H] fp32 natural-log LSE (-inf for rows with no key).
    """
    seq = q_nope.shape[0]
    H = q_nope.shape[1]
    num_cu = _cu_count(q_nope.device)
    BLOCK_H = 16
    BLOCK_K = _sparse_mla_block_k(kv)
    topk = indices.shape[-1]
    max_kv_splits = max(1, topk // _PREFERRED_BLOCK_K)
    head_blocks = max(1, (H + BLOCK_H - 1) // BLOCK_H)
    base_ctas = seq * head_blocks
    optimize_gfx950_fp8 = _is_gfx950_sparse_mla_fp8(
        kv.dtype, H, d_v, q_rope.shape[-1], kv.shape[-1]
    )
    use_topk_length = (
        optimize_gfx950_fp8
        and topk >= 2048
        and topk % 128 == 0
        and max_topk_length is not None
        and max_topk_length <= 512
    )
    if use_topk_length:
        if topk_length is None:
            indices_2d = indices.squeeze(1) if indices.dim() == 3 else indices
            if indices_2d.stride(1) != 1:
                indices_2d = indices_2d.contiguous()
            topk_length = q8kv8_topk_length_from_indices(indices_2d)
        with _no_async_copy():
            return _triton_sparse_mla_fwd_splitk(
                q_nope,
                q_rope,
                kv,
                indices,
                sm_scale,
                d_v,
                kv_splits=1,
                topk_length=topk_length,
                use_topk_length=True,
                return_lse=return_lse,
                bf16_dot=bf16_dot,
            )
    # The decode plan was measured on the FP8-dot, output-only path.
    decode_fp8_64h = (
        not return_lse
        and not bf16_dot
        and _is_gfx950_decode_fp8_64h(kv.dtype, H, d_v, q_rope.shape[-1], kv.shape[-1])
    )
    if decode_fp8_64h:
        # Decode plan: one head block per query row, parallelism bought with KV
        # splits instead. Shapes only -- cudagraph-safe, no tensor reads.
        decode_splits = _decode_splitk_splits(seq, topk, BLOCK_K, num_cu)
        if decode_splits is not None:
            return _triton_sparse_mla_fwd_splitk(
                q_nope,
                q_rope,
                kv,
                indices,
                sm_scale,
                d_v,
                kv_splits=decode_splits,
                decode_tuned=True,
            )
    if _HIP_SPLIT_MIN_ROWS <= seq <= _HIP_SPLIT_MAX_ROWS and is_gfx95_supported():
        from sglang.kernels.ops.attention.dsa.sparse_mla_prefill_gfx950 import (
            sparse_mla_decode_splits,
            sparse_mla_split_gfx950,
            supports_sparse_mla_prefill_gfx950,
        )

        if supports_sparse_mla_prefill_gfx950(q_nope, q_rope, kv, d_v):
            return sparse_mla_split_gfx950(
                q_nope,
                q_rope,
                kv,
                indices,
                sm_scale,
                sparse_mla_decode_splits(seq),
                d_v=d_v,
                return_lse=return_lse,
            )
    if base_ctas > num_cu and is_gfx95_supported():
        from sglang.kernels.ops.attention.dsa.sparse_mla_prefill_gfx950 import (
            sparse_mla_prefill_gfx950,
            supports_sparse_mla_prefill_gfx950,
        )

        if supports_sparse_mla_prefill_gfx950(q_nope, q_rope, kv, d_v):
            return sparse_mla_prefill_gfx950(
                q_nope, q_rope, kv, indices, sm_scale, d_v, return_lse=return_lse
            )
        print_info_once(
            "gfx950 sparse MLA prefill takes only 656-byte fp8_ds_mla rows, at most 16 "
            "heads and a KV pool within 31-bit offsets; long prefill stays on Triton"
        )
    if base_ctas > num_cu:
        # The single-pass kernel needs H <= 16 or a power of two (DCP decode
        # gathers q from the group, e.g. 96 heads); the fused kernel masks heads.
        if optimize_gfx950_fp8 or (H > 16 and H & (H - 1)):
            with _no_async_copy():
                return _triton_sparse_mla_fwd_splitk(
                    q_nope,
                    q_rope,
                    kv,
                    indices,
                    sm_scale,
                    d_v,
                    kv_splits=1,
                    return_lse=return_lse,
                    bf16_dot=bf16_dot,
                )
        return _triton_sparse_mla_fwd_single(
            q_nope,
            q_rope,
            kv,
            indices,
            sm_scale,
            d_v,
            return_lse=return_lse,
            bf16_dot=bf16_dot,
        )
    kv_splits = min(
        _kv_splits_heuristic(
            seq, H, BLOCK_H, target_wg_per_cu=1.0, max_kv_splits=max_kv_splits
        ),
        max_kv_splits,
    )
    if optimize_gfx950_fp8:
        kv_splits = _gfx950_sparse_mla_kv_splits(
            base_ctas,
            topk,
            BLOCK_K,
            num_cu,
            kv_splits,
            max_kv_splits,
        )
    return _triton_sparse_mla_fwd_splitk(
        q_nope,
        q_rope,
        kv,
        indices,
        sm_scale,
        d_v,
        kv_splits,
        return_lse=return_lse,
        bf16_dot=bf16_dot,
    )
