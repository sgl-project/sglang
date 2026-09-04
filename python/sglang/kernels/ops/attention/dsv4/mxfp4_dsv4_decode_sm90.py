"""JIT-compiled MXFP4 KV-cache decode attention for DeepSeek-V4 on SM90.

Fused three-stage split-KV decode over the MXFP4 (E8M0 block-32 scale + E2M1
mantissa) KV cache: a one-CTA scheduler-metadata kernel, a persistent WGMMA
main kernel, and the combine kernel. One call covers the SWA source plus an
optional compressed (C4/C128) source and the attention sink in a single
online softmax.

The kernel is a JIT adaptation of the FlashMLA-style split-KV sparse decode
(from the SGLang reference PR #31269, ported to MXFP4). It requires SM90
(Hopper) and bf16 queries with head dim 512.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

import msgspec
import torch

from sglang.kernels.jit.utils import cache_once, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module


# ---------------------------------------------------------------------------
# Build flags
# ---------------------------------------------------------------------------


def _mxfp4_cuda_flags() -> list[str]:
    # Mirrors the verified AOT flag set for this FlashMLA-derived kernel
    # (FlashMLA setup.py + the DSV4 build): --use_fast_math only. The Q8KV8
    # prefill kernel's -DCUTE_USE_PACKED_TUPLE=1 was ablated here and caused
    # a measurable slowdown (~1.3-1.7x), so it is deliberately omitted.
    return [
        "-O3",
        "-DNDEBUG",
        "--expt-extended-lambda",
        "--use_fast_math",
    ]


@cache_once
def _jit_mxfp4_dsv4_decode_module() -> Module:
    """Compile and cache the MXFP4 DSV4 decode module (one translation unit)."""
    if torch.cuda.get_device_capability()[0] != 9:
        raise RuntimeError("MXFP4 DSV4 decode requires SM90 (Hopper)")
    return load_jit(
        "mxfp4_dsv4_decode_sm90",
        cuda_files=["deepseek_v4/mxfp4_dsv4_decode_sm90/entry.cuh"],
        cuda_wrappers=[("mxfp4_dsv4_decode_dispatch", "mxfp4_dsv4_decode_dispatch")],
        extra_cuda_cflags=_mxfp4_cuda_flags(),
        extra_dependencies=["cutlass"],
    )


# Pre-resolve the FFI entry point once; per-call module attribute lookup is
# measurable at decode-step granularity.
_resolved_dispatch: Optional[callable] = None


def _get_dispatch() -> callable:
    global _resolved_dispatch
    if _resolved_dispatch is None:
        _resolved_dispatch = _jit_mxfp4_dsv4_decode_module().mxfp4_dsv4_decode_dispatch
    return _resolved_dispatch


# Optional tensors are passed as empty tensors; cache them per device to avoid
# a torch.empty call on every decode step.
_EMPTY_I32: dict[int, torch.Tensor] = {}
_EMPTY_F32: dict[int, torch.Tensor] = {}


def _empty_i32(device: torch.device) -> torch.Tensor:
    cached = _EMPTY_I32.get(device.index)
    if cached is None:
        cached = torch.empty(0, dtype=torch.int32, device=device)
        _EMPTY_I32[device.index] = cached
    return cached


def _empty_f32(device: torch.device) -> torch.Tensor:
    cached = _EMPTY_F32.get(device.index)
    if cached is None:
        cached = torch.empty(0, dtype=torch.float32, device=device)
        _EMPTY_F32[device.index] = cached
    return cached


# Split-K accumulators are per-step internal scratch. Reuse them across calls
# with the same geometry: fixed addresses are also CUDA-graph friendly.
#
# This module-global cache is only a fallback for standalone callers (tests,
# benchmarks): it keeps one (b + num_sm_parts)-row FP32 pair per geometry
# forever, which across a CUDA-graph capture sweep (one entry per captured
# batch size) holds hundreds of MiB of dead weight, and two runners sharing
# a geometry would race on the same tensors.  Production callers (the DSV4
# attention backend) pass their own per-runner workspace via
# ``split_accum_buffers`` instead.
_SCRATCH: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}


def _get_scratch(b: int, s_q: int, h_q: int, device: torch.device, head_dim_v: int):
    key = (b, s_q, h_q, device.index, head_dim_v)
    cached = _SCRATCH.get(key)
    if cached is None:
        total_num_splits = b + _num_sm_parts(b, s_q, h_q, device)
        lse_accum = torch.empty(
            (total_num_splits, s_q, h_q), dtype=torch.float32, device=device
        )
        o_accum = torch.empty(
            (total_num_splits, s_q, h_q, head_dim_v),
            dtype=torch.float32,
            device=device,
        )
        cached = (lse_accum, o_accum)
        _SCRATCH[key] = cached
    return cached


def _slice_accum_buffers(
    lse_arena: torch.Tensor,
    o_arena: torch.Tensor,
    total_num_splits: int,
    b: int,
    s_q: int,
    h_q: int,
    head_dim_v: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Prefix-slice a caller-owned split-K workspace for one call.

    Every batch size only ever touches accumulator rows ``[0, b + parts)``, so
    a single arena sized for the largest batch serves all smaller batches as a
    prefix slice — one stable allocation instead of one per geometry.
    """
    if lse_arena.dtype != torch.float32 or o_arena.dtype != torch.float32:
        raise ValueError("split_accum_buffers must be float32 tensors")
    if lse_arena.device != device or o_arena.device != device:
        raise ValueError("split_accum_buffers must live on the query device")
    if lse_arena.ndim != 3 or o_arena.ndim != 4:
        raise ValueError(
            "split_accum_buffers must be (rows, s_q, h_q) and "
            "(rows, s_q, h_q, head_dim_v)"
        )
    if lse_arena.shape[0] < total_num_splits or o_arena.shape[0] < total_num_splits:
        raise ValueError(
            f"split_accum_buffers need at least {total_num_splits} rows "
            f"(batch {b} + sm parts), got {lse_arena.shape[0]}"
        )
    if lse_arena.shape[1:] != (s_q, h_q) or o_arena.shape[1:] != (s_q, h_q, head_dim_v):
        raise ValueError(
            f"split_accum_buffers trailing dims must be ({s_q}, {h_q}) and "
            f"({s_q}, {h_q}, {head_dim_v}), got {lse_arena.shape[1:]} and "
            f"{o_arena.shape[1:]}"
        )
    return lse_arena[:total_num_splits], o_arena[:total_num_splits]


# ---------------------------------------------------------------------------
# Scheduler metadata container (shape-keyed, replayable across CUDA graphs)
# ---------------------------------------------------------------------------


class FlashMLASchedMeta(msgspec.Struct):
    """Tile scheduler metadata for the DSV4 MXFP4 decode.

    Holds the split-K scheduler tensors across calls. A single instance is
    only valid for one (batch, cache-page, top-k) geometry; reuse a matching
    instance to let CUDA graphs replay the captured scheduler kernels.
    """

    class Config(msgspec.Struct):
        b: int
        s_q: int
        h_q: int
        page_block_size: int
        h_k: int
        causal: bool
        is_fp8_kvcache: bool
        topk: Optional[int]
        extra_page_block_size: Optional[int]
        extra_topk: Optional[int]

    have_initialized: bool = False
    config: Optional[Config] = None
    tile_scheduler_metadata: Optional[torch.Tensor] = None
    num_splits: Optional[torch.Tensor] = None


def _num_sm_parts(b: int, s_q: int, h_q: int, device: torch.device) -> int:
    """Split-K partition count, matching the native FlashMLA layout."""
    mpc = torch.cuda.get_device_properties(device).multi_processor_count
    return max(mpc // s_q // (h_q // 64), 1)


def _validate_core_inputs(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    indices: torch.Tensor,
    head_dim_v: int,
) -> None:
    """Reject inputs the SM90 kernel cannot consume safely.

    The kernel hardcodes head dim 512 (QK and V), int32 indices, contiguous
    row-major layouts, and 128-bit vector loads of the 368-byte cache rows;
    every deviation must fail here as a Python exception rather than read
    out of bounds on the device.
    """
    if q.dtype != torch.bfloat16:
        raise RuntimeError("q must have dtype bfloat16")
    if q.ndim != 4 or k_cache.ndim != 4 or indices.ndim != 3:
        raise ValueError(
            "Expected q [B, S_Q, H_Q, D], cache [pages, page_size, H_K, 368], "
            f"and indices [B, S_Q, topk], got {q.shape=}, {k_cache.shape=}, "
            f"{indices.shape=}"
        )
    b, s_q, h_q, d_qk = q.shape
    if b <= 0 or s_q <= 0:
        raise ValueError(
            f"q batch and sequence dimensions must be positive, got {b=} {s_q=}"
        )
    if h_q not in (64, 128):
        raise ValueError(f"q must contain 64 or 128 heads, got {h_q}")
    if d_qk != 512 or head_dim_v != 512:
        raise ValueError(
            "Query head dim and head_dim_v must both be 512, got "
            f"d_qk={d_qk}, head_dim_v={head_dim_v}"
        )
    if indices.shape[:2] != (b, s_q):
        raise ValueError(
            f"indices must have shape [{b}, {s_q}, topk], got {tuple(indices.shape)}"
        )
    if indices.shape[-1] % 64 != 0:
        raise RuntimeError(
            f"indices top-k width must be a multiple of 64, got {indices.shape[-1]}"
        )
    if indices.dtype != torch.int32:
        raise ValueError(f"indices must be int32, got {indices.dtype}")
    if k_cache.shape[2] != 1 or k_cache.shape[3] != 368:
        raise ValueError(
            "DeepSeek V4 MXFP4 cache must have shape "
            f"[pages, page_size, 1, 368], got {tuple(k_cache.shape)}"
        )
    # 368 = 23 * 16, so per-row strides preserve 128-bit vector-load
    # alignment only when the base pointer itself is 16-byte aligned.
    if k_cache.data_ptr() % 16 != 0:
        raise RuntimeError("k_cache must be 16-byte aligned for 128-bit vector loads")
    if k_cache.device != q.device or indices.device != q.device:
        raise ValueError(
            f"All tensors must live on the query device {q.device}, got "
            f"k_cache {k_cache.device} and indices {indices.device}"
        )
    if (
        not q.is_contiguous()
        or not k_cache.is_contiguous()
        or not indices.is_contiguous()
    ):
        raise ValueError("q, k_cache, and indices must be contiguous")


def _validate_lengths_vector(
    t: torch.Tensor, name: str, b: int, device: torch.device
) -> None:
    """Validate a per-request top-k length vector (int32 [B] on-device)."""
    if t.dtype != torch.int32:
        raise ValueError(f"{name} must be int32, got {t.dtype}")
    if t.shape != (b,):
        raise ValueError(
            f"{name} must have shape [{b}] (one entry per request), got "
            f"{tuple(t.shape)}"
        )
    if t.device != device:
        raise ValueError(f"{name} must live on the query device, got {t.device}")
    if not t.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def flash_mla_with_kvcache_dsv4_mxfp4(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    indices: torch.Tensor,
    topk_length: Optional[torch.Tensor],
    attn_sink: Optional[torch.Tensor],
    tile_scheduler_metadata: FlashMLASchedMeta,
    head_dim_v: int = 512,
    softmax_scale: Optional[float] = None,
    extra_k_cache: Optional[torch.Tensor] = None,
    extra_indices_in_kvcache: Optional[torch.Tensor] = None,
    extra_topk_length: Optional[torch.Tensor] = None,
    split_accum_buffers: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run fused SM90 DeepSeek-V4 sparse decode on an MXFP4 KV cache.

    ``k_cache`` stores one 368-byte DeepSeek-V4 entry per token and has the
    page-major shape ``[num_pages, page_size, 1, 368]``. C0 uses only this
    source. C4/C128 additionally pass a compressed cache, its indices, and
    valid top-k lengths. MXFP4 rows carry their own E8M0 block-32 scales, so
    no global FP32 scale is required (unlike the NVFP4 path).

    Scheduler tensors are allocated on first call and cached in
    ``tile_scheduler_metadata`` for CUDA-graph replay.

    ``split_accum_buffers`` optionally supplies a caller-owned split-K
    workspace ``(lse_accum_rows, o_accum_rows)`` sized for the caller's
    largest batch (``batch + num_sm_parts`` rows); each call prefix-slices
    it.  Passing it avoids the module-global per-geometry scratch cache —
    required from long-lived runners so a CUDA-graph capture sweep does not
    pin one accumulator pair per captured batch size, and so two runners
    never share scratch tensors.
    """
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)

    if not isinstance(tile_scheduler_metadata, FlashMLASchedMeta):
        raise TypeError(
            "DeepSeek V4 MXFP4 decode requires FlashMLASchedMeta, got "
            f"{type(tile_scheduler_metadata).__name__}"
        )
    _validate_core_inputs(q, k_cache, indices, head_dim_v)
    if topk_length is not None:
        _validate_lengths_vector(topk_length, "topk_length", q.shape[0], q.device)
    if attn_sink is not None:
        if attn_sink.dtype != torch.float32:
            raise ValueError(f"attn_sink must be float32, got {attn_sink.dtype}")
        if attn_sink.device != q.device:
            raise ValueError(
                f"attn_sink must live on the query device, got {attn_sink.device}"
            )
        if not attn_sink.is_contiguous():
            raise ValueError("attn_sink must be contiguous")

    have_extra_cache = extra_k_cache is not None
    have_extra_indices = extra_indices_in_kvcache is not None
    have_extra_length = extra_topk_length is not None
    if not (have_extra_cache == have_extra_indices == have_extra_length):
        raise ValueError(
            "extra_k_cache, extra_indices_in_kvcache, and extra_topk_length "
            "must be provided together"
        )
    if have_extra_cache:
        assert extra_indices_in_kvcache is not None
        if extra_k_cache.ndim != 4 or extra_indices_in_kvcache.ndim != 3:
            raise ValueError(
                "Expected extra cache [pages, page_size, H_K, 368] and extra "
                f"indices [B, S_Q, topk], got {extra_k_cache.shape=} and "
                f"{extra_indices_in_kvcache.shape=}"
            )
        if extra_k_cache.shape[2] != 1 or extra_k_cache.shape[3] != 368:
            raise ValueError(
                "DeepSeek V4 extra MXFP4 cache must have shape "
                "[pages, page_size, 1, 368], got "
                f"{tuple(extra_k_cache.shape)}"
            )
        if extra_k_cache.data_ptr() % 16 != 0:
            raise RuntimeError(
                "extra_k_cache must be 16-byte aligned for 128-bit vector loads"
            )
        if extra_indices_in_kvcache.shape[:2] != (q.shape[0], q.shape[1]):
            raise ValueError(
                "extra_indices_in_kvcache must have shape "
                f"[{q.shape[0]}, {q.shape[1]}, topk], got "
                f"{tuple(extra_indices_in_kvcache.shape)}"
            )
        if extra_indices_in_kvcache.shape[-1] % 64 != 0:
            raise RuntimeError(
                "extra indices top-k width must be a multiple of 64, got "
                f"{extra_indices_in_kvcache.shape[-1]}"
            )
        if extra_indices_in_kvcache.dtype != torch.int32:
            raise ValueError(
                f"extra_indices_in_kvcache must be int32, got "
                f"{extra_indices_in_kvcache.dtype}"
            )
        if (
            extra_k_cache.device != q.device
            or extra_indices_in_kvcache.device != q.device
        ):
            raise ValueError(
                f"Extra-source tensors must live on the query device {q.device}, "
                f"got cache {extra_k_cache.device} and indices "
                f"{extra_indices_in_kvcache.device}"
            )
        if (
            not extra_k_cache.is_contiguous()
            or not extra_indices_in_kvcache.is_contiguous()
        ):
            raise ValueError("extra_k_cache and extra_indices must be contiguous")
        _validate_lengths_vector(
            extra_topk_length, "extra_topk_length", q.shape[0], q.device
        )

    sched_meta = tile_scheduler_metadata
    topk = indices.shape[-1]
    extra_page_block_size = (
        extra_k_cache.shape[1] if extra_k_cache is not None else None
    )
    extra_topk = (
        extra_indices_in_kvcache.shape[-1]
        if extra_indices_in_kvcache is not None
        else None
    )
    config = FlashMLASchedMeta.Config(
        b=q.shape[0],
        s_q=q.shape[1],
        h_q=q.shape[2],
        page_block_size=k_cache.shape[1],
        h_k=k_cache.shape[2],
        causal=False,
        # The scheduler geometry is shared with the FP8 path; MXFP4 only
        # differs in the per-row storage format.
        is_fp8_kvcache=True,
        topk=topk,
        extra_page_block_size=extra_page_block_size,
        extra_topk=extra_topk,
    )
    if not sched_meta.have_initialized:
        sched_meta.have_initialized = True
        sched_meta.config = config
    else:
        helper_msg = (
            " Input arguments are inconsistent with FlashMLASchedMeta. Reuse a "
            "scheduler only for matching tensor shapes and sparse settings."
        )
        assert sched_meta.config == config, helper_msg

    device = q.device
    b, s_q, h_q = q.shape[0], q.shape[1], q.shape[2]
    need_buffer = sched_meta.tile_scheduler_metadata is None
    if need_buffer:
        num_sm_parts = _num_sm_parts(b, s_q, h_q, device)
        # DecodingSchedMeta is 8 int32 (32 bytes, 4*8 aligned).
        sched_meta.tile_scheduler_metadata = torch.empty(
            (num_sm_parts, 8), dtype=torch.int32, device=device
        )
        sched_meta.num_splits = torch.empty((b + 1,), dtype=torch.int32, device=device)
    num_sm_parts = sched_meta.tile_scheduler_metadata.shape[0]
    # The split assignment depends on the per-call top-k lengths, which in
    # eager mode change every step (a growing sequence crosses 64-token block
    # boundaries), so the scheduler re-runs on every eager call with the
    # buffers reused. During CUDA-graph capture the caller supplies a fresh
    # buffer, so the generation kernel is recorded once and re-executed on
    # every replay with the replayed lengths.
    generate_sched_meta = need_buffer or not torch.cuda.is_current_stream_capturing()

    # Outputs are per-call (returned to the caller); split-K accumulators are
    # internal scratch reused across calls of the same geometry. The scheduler
    # metadata kernels run inside CUDA-graph capture, so address stability is
    # what matters — the cached scratch and metadata buffers provide it.
    out = torch.empty((b, s_q, h_q, head_dim_v), dtype=torch.bfloat16, device=device)
    lse = torch.empty((b, s_q, h_q), dtype=torch.float32, device=device)
    total_num_splits = b + num_sm_parts
    if split_accum_buffers is not None:
        lse_accum, o_accum = _slice_accum_buffers(
            split_accum_buffers[0],
            split_accum_buffers[1],
            total_num_splits,
            b,
            s_q,
            h_q,
            head_dim_v,
            device,
        )
    else:
        lse_accum, o_accum = _get_scratch(b, s_q, h_q, device, head_dim_v)

    _get_dispatch()(
        q,
        k_cache,
        indices,
        topk_length if topk_length is not None else _empty_i32(device),
        attn_sink if attn_sink is not None else _empty_f32(device),
        sched_meta.tile_scheduler_metadata,
        sched_meta.num_splits,
        extra_k_cache if extra_k_cache is not None else _empty_i32(device),
        (
            extra_indices_in_kvcache
            if extra_indices_in_kvcache is not None
            else _empty_i32(device)
        ),
        extra_topk_length if extra_topk_length is not None else _empty_i32(device),
        lse_accum,
        o_accum,
        out,
        lse,
        head_dim_v,
        softmax_scale,
        generate_sched_meta,
    )
    return out, lse.transpose(1, 2)
