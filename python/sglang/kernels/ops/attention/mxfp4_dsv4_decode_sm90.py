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
        raise RuntimeError("MXFP4 DSV4 decode requires SM90 (Hopper) or later")
    return load_jit(
        "mxfp4_dsv4_decode_sm90",
        cuda_files=["mxfp4_dsv4_decode_sm90/entry.cuh"],
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
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run fused SM90 DeepSeek-V4 sparse decode on an MXFP4 KV cache.

    ``k_cache`` stores one 368-byte DeepSeek-V4 entry per token and has the
    page-major shape ``[num_pages, page_size, 1, 368]``. C0 uses only this
    source. C4/C128 additionally pass a compressed cache, its indices, and
    valid top-k lengths. MXFP4 rows carry their own E8M0 block-32 scales, so
    no global FP32 scale is required (unlike the NVFP4 path).

    Scheduler tensors are allocated on first call and cached in
    ``tile_scheduler_metadata`` for CUDA-graph replay.
    """
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)

    if not isinstance(tile_scheduler_metadata, FlashMLASchedMeta):
        raise TypeError(
            "DeepSeek V4 MXFP4 decode requires FlashMLASchedMeta, got "
            f"{type(tile_scheduler_metadata).__name__}"
        )
    if q.dtype != torch.bfloat16:
        raise RuntimeError("q must have dtype bfloat16")
    if q.ndim != 4 or k_cache.ndim != 4 or indices.ndim != 3:
        raise ValueError(
            "Expected q [B, S_Q, H_Q, D], cache [pages, page_size, H_K, 368], "
            f"and indices [B, S_Q, topk], got {q.shape=}, {k_cache.shape=}, "
            f"{indices.shape=}"
        )
    if indices.shape[-1] % 64 != 0:
        raise RuntimeError(
            f"indices top-k width must be a multiple of 64, got {indices.shape[-1]}"
        )
    if k_cache.data_ptr() % 4 != 0:
        raise RuntimeError("k_cache must be 4-byte aligned")
    if k_cache.shape[2] != 1 or k_cache.shape[3] != 368:
        raise ValueError(
            "DeepSeek V4 MXFP4 cache must have shape "
            f"[pages, page_size, 1, 368], got {tuple(k_cache.shape)}"
        )

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
        extra_indices_in_kvcache if extra_indices_in_kvcache is not None else _empty_i32(device),
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
