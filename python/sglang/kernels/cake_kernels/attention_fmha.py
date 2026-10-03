"""Cake dense FMHA / GQA decode and context kernels via FlashInfer.

FlashInfer entries (all at FlashInfer ``46340689a5ab``):

* ``flashinfer.decode.trtllm_batch_decode_with_kv_cache(backend="cake")``
  (= ``flashinfer.cake_fmha.cake_batch_decode_with_kv_cache``), module
  ``flashinfer.cake_fmha``, JIT ``flashinfer.jit.cake_fmha``. Same ABI as the
  trtllm-gen paged decode: Q ``[B * q_len, Hq, D]``, paged K/V 4-D (HND
  default; NHD only for the fp16 NHD route), ``D in {64, 128, 256, 512}``,
  group ratio 1..8 (1..16 for the hd64 balanced route), page size 16/32/64,
  ``q_len_per_req`` 1 (quantized / fp16 routes) up to MTP 3..8 (balanced
  BF16/FP16/hd256 routes); dtypes BF16, FP16, FP8 e4m3, BF16/FP16 Q with FP8
  KV, FP8 Q with NVFP4 (uint8) KV plus e4m3 block scales (``kv_cache_sf``).
  Built for sm_100a / sm_103a only; a route without an exact generated
  component runs the authenticated portable compat module. Explicit selection
  only: FlashInfer never auto-selects Cake.
* ``flashinfer.prefill.trtllm_batch_context_with_kv_cache(backend="cake")``
  (= ``cake_batch_context_with_kv_cache``): routes context_bf16 (HND hd128),
  context_fp16_hd256 (NHD), context_fp8 (HND hd128), context_fp8_hd256 (NHD),
  context_nvfp4 (HND hd128); page size a power of two in 16..1024;
  ``window_left == -1`` only; host scalar ``bmm1_scale`` / ``bmm2_scale``.
* ``flashinfer.cake_fmha.plan_cake_fmha_request_ordered_paged_decode`` (SM103
  only): device-free frozen plan consumed through
  ``trtllm_batch_decode_with_kv_cache(request_order=..., request_order_plan=...,
  backend="cake")`` for BF16 Q/O, FP8 e4m3 HND KV, hd256, page 64, Hq=8,
  Hkv=1, ``q_len in {1, 6}``, ``enable_pdl=True``, ``o_scale=1``.
* ``flashinfer.cake_dcp.run_dcp_spec_decode`` reached through
  ``trtllm_batch_decode_with_kv_cache(causal_seqlens_kv_global=..., cp_world=,
  cp_rank=, backend="cake")``, JIT ``flashinfer.jit.cake_dcp``: DCP speculative
  decode. BF16 query ``[B * q_len, Hq, D]``; HND K/V ``[pages, Hkv, page, D]``;
  profiles ``bf16_p16`` (BF16 KV, page 16, D=128, q_len in {1,2,3,4,5,6,8}),
  ``fp8_p64`` (e4m3 KV, page 64, D=128, same q_lens) and ``fp8_p64_d256``
  (e4m3, page 64, D=256, q_len 1..8, Hq=16, Hkv=1, cp_world in {1, 4});
  group ratio 1..8 for D=128; ``cp_world in {1, 2, 4, 8}``; int32
  ``block_tables [B, max_pages]``, ``seq_lens [B]`` (local lengths) and
  ``causal_seqlens_kv_global [B]``; BF16 out, FP32 ``lse [tokens, Hq]``; host
  scalar finite ``bmm1_scale`` / ``bmm2_scale`` (BF16 KV needs
  ``bmm2_scale == 1``); ``o_scale == 1``; no PDL, no sinks, no window, no
  ragged query lengths. Compute capability 10.0 (CUDA >= 12.8), 10.3 and 10.7
  (sm100f, CUDA >= 12.9). ``route in {"auto", "static", "balanced"}``.
* ``flashinfer.decode.prepare_balanced_batch_decode_with_kv_cache`` (impl
  ``flashinfer.experimental.balanced_gqa_decode.cake_backend``): on-device
  load-balanced BF16 paged GQA decode. Query ``[B * q_len, Hq, 128]``,
  K/V tuple each ``[pages, Hkv, 16, 128]`` HND, int32 ``block_tables``,
  int32 ``seq_lens`` (device, never read on host), exactly 8 Q heads per KV
  head, ``B <= 1024``, ``q_len_per_req`` 1..8 (3..8 = packed-row MTP
  program, causal), uint8 workspace >= ``balanced_gqa_decode_workspace_size``.
  No sinks / window / LSE. SM100 / SM103 only.
* ``flashinfer.decode.sm110_gqa_decode`` / ``prepare_sm110_gqa_decode`` /
  ``launch_sm110_gqa_decode_prepared`` (impl
  ``flashinfer.experimental.sm110_gqa_decode``): Jetson AGX Thor (compute
  capability 11.0, CUDA >= 13.0) FP16 GQA decode with fixed 32 Q heads / 8 KV
  heads, head_dim 128, one query token: ``q [B, 32, 128]``,
  ``kv [B, 2, 8, capacity, 128]`` (K at index 0, V at index 1), int32
  ``sequence_lengths [B]`` in ``[1, capacity]``.
* ``flashinfer.sm110_xqa.prepare`` / ``attention`` (impl
  ``flashinfer.experimental.sm110_xqa.backend``): Thor XQA family; FP16 Q;
  decode D=128 (``q [B, Hq, 128]``, ``kv [B, 2, Hkv, C, 128]``, Hq/Hkv in
  {4, 8, 16}) and tree D=512 (packed Q + ``q_cu_seq_lens`` + ``mask``, FP16 or
  e4m3 KV, optional paging with ``page_size=128``).

CUDA graphs: Cake FMHA needs one eager warm run per tensor/layout binding
before capture (TMA descriptors, JIT load). Split-KV routes take the caller's
``workspace_buffer`` and a zeroed, reusable ``multi_ctas_kv_counter_buffer``
(self-resetting tickets); without one FlashInfer allocates a fresh zeroed
buffer per call, so steady-state callers pass their own. The request-ordered
plan is built before capture and the int32 ``request_order`` contents may be
updated in place between replays. DCP spec decode: caller-owned workspace
(``get_dcp_spec_workspace_size_bytes`` / balanced sizes) plus an optional
zero-initialized ``completion_buffer`` (passed as
``multi_ctas_kv_counter_buffer``), ``return_lse=True`` or a caller ``lse``,
host-scalar scales, ``enable_pdl`` must not be True; prewarm before capture.
Balanced GQA decode: ``prepare`` validates, zeroes the counters once and binds
the tensors; the runner's ``launch()`` is allocation-free and one captured
graph replays for any KV-length distribution (the program is fixed by
``q_len_per_req``); never share one workspace between two live runners. SM110
prepared GQA decode owns its workspace across replays; use separate prepared
instances per stream / graph.

Not supported here (keep the existing SGLang path): SM90 / SM12x devices for
the FMHA, DCP and balanced routes; ``window_left != -1`` for context; ragged
query lengths, sinks, skip-softmax or block-sparse masks for DCP; FP8/NVFP4
or non-HND caches for balanced decode; anything but FP16 on Thor.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Sequence
from typing import Tuple, Union

from sglang.kernels.cake_kernels.attention_common import (
    SM100,
    SM103,
    SM107,
    SM110,
    archs_in,
    flashinfer_module_available,
)

if TYPE_CHECKING:
    import torch

FI_MODULE = "flashinfer.cake_fmha"
FI_JIT_MODULE = "flashinfer.jit.cake_fmha"
FI_REQUEST_ORDERED_JIT_MODULE = "flashinfer.jit.cake_fmha_request_ordered"
FI_DCP_MODULE = "flashinfer.cake_dcp"
FI_DCP_JIT_MODULE = "flashinfer.jit.cake_dcp"
FI_BALANCED_MODULE = "flashinfer.experimental.balanced_gqa_decode.cake_backend"
FI_SM110_GQA_MODULE = "flashinfer.experimental.sm110_gqa_decode.backend"
FI_SM110_GQA_PREPARED_MODULE = "flashinfer.experimental.sm110_gqa_decode.prepared"
FI_SM110_XQA_MODULE = "flashinfer.sm110_xqa"

ARCHS = (SM100, SM103)
DCP_ARCHS = (SM100, SM103, SM107)
BALANCED_ARCHS = (SM100, SM103)
SM110_ARCHS = (SM110,)

FMHA_HEAD_DIMS = (64, 128, 256, 512)
FMHA_DECODE_PAGE_SIZES = (16, 32, 64)
FMHA_CONTEXT_PAGE_SIZES = tuple(16 << i for i in range(7))  # 16 .. 1024
DCP_CP_WORLDS = (1, 2, 4, 8)
DCP_Q_LENS_D128 = (1, 2, 3, 4, 5, 6, 8)
DCP_Q_LENS_D256 = (1, 2, 3, 4, 5, 6, 7, 8)
BALANCED_HEAD_DIM = 128
BALANCED_PAGE_SIZE = 16
BALANCED_GROUP_RATIO = 8
BALANCED_MAX_REQUESTS = 1024
BALANCED_MAX_Q_LEN = 8
SM110_GQA_NUM_Q_HEADS = 32
SM110_GQA_NUM_KV_HEADS = 8
SM110_GQA_HEAD_DIM = 128
SM110_XQA_HEADS = (2, 4, 8, 16)


def _fmha_dtypes():
    import torch

    return (torch.bfloat16, torch.float16, torch.float8_e4m3fn)


def _split_kv(kv_cache):
    if isinstance(kv_cache, (tuple, list)):
        if len(kv_cache) != 2:
            return None
        return kv_cache[0], kv_cache[1]
    if kv_cache.ndim == 5 and kv_cache.shape[1] == 2:
        return kv_cache[:, 0], kv_cache[:, 1]
    return None


def _kv_shape(k_cache, kv_layout: str):
    """``(num_kv_heads, page_size, head_dim)`` of a 4-D paged cache."""
    if k_cache.ndim != 4:
        return None
    if kv_layout == "HND":
        return int(k_cache.shape[1]), int(k_cache.shape[2]), int(k_cache.shape[3])
    if kv_layout == "NHD":
        return int(k_cache.shape[2]), int(k_cache.shape[1]), int(k_cache.shape[3])
    return None


# --------------------------------------------------------------------------
# Cake FMHA paged decode (trtllm_batch_decode_with_kv_cache backend="cake")
# --------------------------------------------------------------------------


def supports_batch_decode_with_kv_cache(
    query: torch.Tensor,
    kv_cache: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    *,
    kv_layout: str = "HND",
    causal_seqlens_kv_global: Optional[torch.Tensor] = None,
) -> bool:
    """Admission check mirroring the public ABI; never raises.

    FlashInfer's ``select_cake_fmha_decode_route`` performs the exact per-route
    admission; routes without a generated component run the compat module, so
    this check only mirrors the ABI-level contract (device, dtypes, head_dim,
    page size, group ratio).
    """
    try:
        import torch

        if causal_seqlens_kv_global is not None:
            return False  # DCP speculative decode has its own adapter.
        kv = _split_kv(kv_cache)
        if kv is None or not archs_in(ARCHS, query, kv[0], kv[1]):
            return False
        k_cache, v_cache = kv
        shape = _kv_shape(k_cache, kv_layout)
        if shape is None or k_cache.shape != v_cache.shape:
            return False
        num_kv_heads, page_size, head_dim = shape
        nvfp4_kv = k_cache.dtype == torch.uint8
        return (
            flashinfer_module_available(FI_MODULE, FI_JIT_MODULE)
            and query.ndim == 3
            and query.dtype in _fmha_dtypes()
            and (
                (nvfp4_kv and query.dtype == torch.float8_e4m3fn)
                or k_cache.dtype in _fmha_dtypes()
            )
            and k_cache.dtype == v_cache.dtype
            and int(query.shape[2]) in FMHA_HEAD_DIMS
            and (nvfp4_kv or head_dim == int(query.shape[2]))
            and page_size in FMHA_DECODE_PAGE_SIZES
            and num_kv_heads > 0
            and int(query.shape[1]) % num_kv_heads == 0
            and 1 <= int(query.shape[1]) // num_kv_heads <= 16
        )
    except Exception:
        return False


def batch_decode_with_kv_cache(
    query: torch.Tensor,
    kv_cache: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    workspace_buffer: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    max_seq_len: int,
    bmm1_scale: Union[float, torch.Tensor] = 1.0,
    bmm2_scale: Union[float, torch.Tensor] = 1.0,
    window_left: int = -1,
    out: Optional[Any] = None,
    out_dtype: Optional[Union[torch.dtype, str]] = None,
    o_sf_scale: Optional[float] = None,
    o_sf_vec_size: Optional[int] = None,
    sinks: Optional[List[torch.Tensor]] = None,
    kv_layout: str = "HND",
    enable_pdl: Optional[bool] = None,
    q_len_per_req: Optional[int] = 1,
    o_scale: Optional[float] = 1.0,
    mask: Optional[torch.Tensor] = None,
    max_q_len: Optional[int] = None,
    cum_seq_lens_q: Optional[torch.Tensor] = None,
    skip_softmax_threshold_scale_factor: Optional[float] = None,
    kv_cache_sf: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    uses_shared_paged_kv_idx: bool = True,
    lse: Optional[torch.Tensor] = None,
    return_lse: bool = False,
    bmm1_scale_log2: Optional[torch.Tensor] = None,
    multi_ctas_kv_counter_buffer: Optional[torch.Tensor] = None,
    enable_block_sparse_attention: bool = False,
    bf16q_fp8kv_transform_mode: Optional[Literal["k_only", "separate_kv"]] = None,
    request_order: Optional[torch.Tensor] = None,
    request_order_plan: Optional[Any] = None,
):
    """Forward to ``trtllm_batch_decode_with_kv_cache(..., backend="cake")``.

    Returns ``out`` (or ``(out, lse)`` with ``return_lse=True``), exactly like
    the FlashInfer entry. ``bmm1_scale`` is the fused QK + softmax scale and
    defaults to ``1.0`` (pass ``1 / sqrt(head_dim)`` for standard attention).
    """
    from flashinfer.decode import trtllm_batch_decode_with_kv_cache

    return trtllm_batch_decode_with_kv_cache(
        query,
        kv_cache,
        workspace_buffer,
        block_tables,
        seq_lens,
        max_seq_len,
        bmm1_scale=bmm1_scale,
        bmm2_scale=bmm2_scale,
        window_left=window_left,
        out=out,
        out_dtype=out_dtype,
        o_sf_scale=o_sf_scale,
        o_sf_vec_size=o_sf_vec_size,
        sinks=sinks,
        kv_layout=kv_layout,
        enable_pdl=enable_pdl,
        backend="cake",
        q_len_per_req=q_len_per_req,
        o_scale=o_scale,
        mask=mask,
        max_q_len=max_q_len,
        cum_seq_lens_q=cum_seq_lens_q,
        skip_softmax_threshold_scale_factor=skip_softmax_threshold_scale_factor,
        kv_cache_sf=kv_cache_sf,
        uses_shared_paged_kv_idx=uses_shared_paged_kv_idx,
        lse=lse,
        return_lse=return_lse,
        bmm1_scale_log2=bmm1_scale_log2,
        multi_ctas_kv_counter_buffer=multi_ctas_kv_counter_buffer,
        enable_block_sparse_attention=enable_block_sparse_attention,
        bf16q_fp8kv_transform_mode=bf16q_fp8kv_transform_mode,
        request_order=request_order,
        request_order_plan=request_order_plan,
    )


def plan_request_ordered_paged_decode(
    kv_lens: Sequence[int],
    q_len: int,
    *,
    request_order_case: Literal["identity", "length_desc"] = "length_desc",
    real_batch_size: Optional[int] = None,
    write_lse: bool = False,
):
    """Host-only forward of ``plan_cake_fmha_request_ordered_paged_decode``.

    Build the frozen plan before CUDA-graph capture and pass it as
    ``request_order_plan`` (with the int32 ``request_order[B]`` tensor) to
    :func:`batch_decode_with_kv_cache`. SM103 only.
    """
    from flashinfer.cake_fmha import plan_cake_fmha_request_ordered_paged_decode

    return plan_cake_fmha_request_ordered_paged_decode(
        kv_lens,
        q_len,
        request_order_case=request_order_case,
        real_batch_size=real_batch_size,
        write_lse=write_lse,
    )


def balanced_counter_bytes(sm_count: int, q_len: int = 8, head_dim: int = 128) -> int:
    """Host-only forward of ``cake_fmha_balanced_counter_bytes``."""
    from flashinfer.cake_fmha import cake_fmha_balanced_counter_bytes

    return cake_fmha_balanced_counter_bytes(sm_count, q_len, head_dim)


def balanced_workspace_bytes(sm_count: int, q_len: int, head_dim: int = 128) -> int:
    """Host-only forward of ``cake_fmha_balanced_workspace_bytes``."""
    from flashinfer.cake_fmha import cake_fmha_balanced_workspace_bytes

    return cake_fmha_balanced_workspace_bytes(sm_count, q_len, head_dim)


# --------------------------------------------------------------------------
# Cake FMHA paged context (trtllm_batch_context_with_kv_cache backend="cake")
# --------------------------------------------------------------------------


def supports_batch_context_with_kv_cache(
    query: torch.Tensor,
    kv_cache: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    *,
    kv_layout: str = "HND",
    window_left: int = -1,
) -> bool:
    """Admission check mirroring the public ABI; never raises."""
    try:
        import torch

        kv = _split_kv(kv_cache)
        if kv is None or not archs_in(ARCHS, query, kv[0], kv[1]):
            return False
        k_cache, v_cache = kv
        shape = _kv_shape(k_cache, kv_layout)
        if shape is None or k_cache.shape != v_cache.shape:
            return False
        num_kv_heads, page_size, head_dim = shape
        nvfp4_kv = k_cache.dtype == torch.uint8
        return (
            flashinfer_module_available(FI_MODULE, FI_JIT_MODULE)
            and window_left == -1
            and query.ndim == 3
            and query.dtype in _fmha_dtypes()
            and (
                (nvfp4_kv and query.dtype == torch.float8_e4m3fn)
                or k_cache.dtype in _fmha_dtypes()
            )
            and k_cache.dtype == v_cache.dtype
            and int(query.shape[2]) in (128, 256)
            and (nvfp4_kv or head_dim == int(query.shape[2]))
            and page_size in FMHA_CONTEXT_PAGE_SIZES
            and num_kv_heads > 0
            and int(query.shape[1]) % num_kv_heads == 0
        )
    except Exception:
        return False


def batch_context_with_kv_cache(
    query: torch.Tensor,
    kv_cache: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    workspace_buffer: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    max_q_len: int,
    max_kv_len: int,
    bmm1_scale: Union[float, torch.Tensor],
    bmm2_scale: Union[float, torch.Tensor],
    batch_size: int,
    cum_seq_lens_q: torch.Tensor,
    cum_seq_lens_kv: torch.Tensor,
    window_left: int = -1,
    out: Optional[Any] = None,
    out_dtype: Optional[Union[torch.dtype, str]] = None,
    o_sf_scale: Optional[float] = None,
    o_sf_vec_size: Optional[int] = None,
    kv_layout: str = "HND",
    enable_pdl: Optional[bool] = None,
    sinks: Optional[List[torch.Tensor]] = None,
    kv_cache_sf: Optional[Any] = None,
    skip_softmax_threshold_scale_factor: Optional[float] = None,
    uses_shared_paged_kv_idx: bool = True,
    causal: bool = True,
    lse: Optional[torch.Tensor] = None,
    return_lse: bool = False,
    multi_ctas_kv_counter_buffer: Optional[torch.Tensor] = None,
    use_fp16_softmax: Optional[bool] = None,
    uses_spcompress: Optional[bool] = None,
):
    """Forward to ``trtllm_batch_context_with_kv_cache(..., backend="cake")``."""
    from flashinfer.prefill import trtllm_batch_context_with_kv_cache

    return trtllm_batch_context_with_kv_cache(
        query,
        kv_cache,
        workspace_buffer,
        block_tables,
        seq_lens,
        max_q_len,
        max_kv_len,
        bmm1_scale,
        bmm2_scale,
        batch_size,
        cum_seq_lens_q,
        cum_seq_lens_kv,
        window_left=window_left,
        out=out,
        out_dtype=out_dtype,
        o_sf_scale=o_sf_scale,
        o_sf_vec_size=o_sf_vec_size,
        kv_layout=kv_layout,
        enable_pdl=enable_pdl,
        sinks=sinks,
        kv_cache_sf=kv_cache_sf,
        skip_softmax_threshold_scale_factor=skip_softmax_threshold_scale_factor,
        uses_shared_paged_kv_idx=uses_shared_paged_kv_idx,
        causal=causal,
        lse=lse,
        return_lse=return_lse,
        multi_ctas_kv_counter_buffer=multi_ctas_kv_counter_buffer,
        use_fp16_softmax=use_fp16_softmax,
        uses_spcompress=uses_spcompress,
        backend="cake",
    )


# --------------------------------------------------------------------------
# DCP speculative decode (cake_dcp.run_dcp_spec_decode via the public decode)
# --------------------------------------------------------------------------


def supports_dcp_spec_decode(
    query: torch.Tensor,
    kv_cache: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    *,
    q_len_per_req: int,
    cp_world: int,
    kv_layout: str = "HND",
) -> bool:
    """Admission check mirroring the DCP profiles; never raises."""
    try:
        import torch

        kv = _split_kv(kv_cache)
        if kv is None or not archs_in(DCP_ARCHS, query, kv[0], kv[1]):
            return False
        k_cache, v_cache = kv
        if kv_layout != "HND" or k_cache.ndim != 4 or k_cache.shape != v_cache.shape:
            return False
        if not flashinfer_module_available(FI_DCP_MODULE, FI_DCP_JIT_MODULE):
            return False
        if query.ndim != 3 or query.dtype != torch.bfloat16:
            return False
        if k_cache.dtype != v_cache.dtype or cp_world not in DCP_CP_WORLDS:
            return False
        num_q_heads, head_dim = int(query.shape[1]), int(query.shape[2])
        num_kv_heads, page_size = int(k_cache.shape[1]), int(k_cache.shape[2])
        if int(k_cache.shape[3]) != head_dim or num_q_heads % num_kv_heads:
            return False
        ratio = num_q_heads // num_kv_heads
        if k_cache.dtype == torch.bfloat16:
            return (
                head_dim == 128
                and page_size == 16
                and 1 <= ratio <= 8
                and q_len_per_req in DCP_Q_LENS_D128
            )
        if k_cache.dtype == torch.float8_e4m3fn:
            if head_dim == 128:
                return (
                    page_size == 64
                    and 1 <= ratio <= 8
                    and q_len_per_req in DCP_Q_LENS_D128
                )
            if head_dim == 256:
                return (
                    page_size == 64
                    and num_q_heads == 16
                    and num_kv_heads == 1
                    and cp_world in (1, 4)
                    and q_len_per_req in DCP_Q_LENS_D256
                )
        return False
    except Exception:
        return False


def dcp_spec_decode(
    query: torch.Tensor,
    kv_cache: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    workspace_buffer: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    max_seq_len: int,
    causal_seqlens_kv_global: torch.Tensor,
    *,
    bmm1_scale: float = 1.0,
    bmm2_scale: float = 1.0,
    cp_world: int = 1,
    cp_rank: int = 0,
    q_len_per_req: int = 1,
    out: Optional[torch.Tensor] = None,
    lse: Optional[torch.Tensor] = None,
    return_lse: bool = True,
    multi_ctas_kv_counter_buffer: Optional[torch.Tensor] = None,
    kv_layout: str = "HND",
):
    """Forward the DCP speculative decode through the public decode entry.

    ``seq_lens`` and ``max_seq_len`` describe the rank-local paged cache;
    ``causal_seqlens_kv_global[B]`` the global causal bound. The result is
    ``(out, lse)`` with ``return_lse=True`` (default; otherwise a caller
    ``lse`` buffer is required), with FP32 ``lse`` in base 2 like trtllm-gen.
    """
    from flashinfer.decode import trtllm_batch_decode_with_kv_cache

    return trtllm_batch_decode_with_kv_cache(
        query,
        kv_cache,
        workspace_buffer,
        block_tables,
        seq_lens,
        max_seq_len,
        bmm1_scale=bmm1_scale,
        bmm2_scale=bmm2_scale,
        out=out,
        kv_layout=kv_layout,
        backend="cake",
        q_len_per_req=q_len_per_req,
        lse=lse,
        return_lse=return_lse,
        multi_ctas_kv_counter_buffer=multi_ctas_kv_counter_buffer,
        cp_world=cp_world,
        cp_rank=cp_rank,
        causal_seqlens_kv_global=causal_seqlens_kv_global,
    )


def dcp_spec_workspace_size_bytes(
    batch_size: int,
    q_len_per_req: int,
    num_qo_heads: int,
    num_split: int = 16,
    *,
    head_dim: int = 128,
) -> int:
    """Host-only forward of ``get_dcp_spec_workspace_size_bytes``."""
    from flashinfer.cake_dcp import get_dcp_spec_workspace_size_bytes

    return get_dcp_spec_workspace_size_bytes(
        batch_size, q_len_per_req, num_qo_heads, num_split, head_dim=head_dim
    )


def dcp_spec_counter_bytes(batch_size: int, q_len_per_req: int, num_kv_heads: int):
    """Host-only forward of ``get_dcp_spec_counter_bytes`` (zero once)."""
    from flashinfer.cake_dcp import get_dcp_spec_counter_bytes

    return get_dcp_spec_counter_bytes(batch_size, q_len_per_req, num_kv_heads)


def dcp_spec_balanced_workspace_bytes(sm_count: int, head_dim: int = 128) -> int:
    """Host-only forward of ``get_dcp_spec_balanced_workspace_bytes``."""
    from flashinfer.cake_dcp import get_dcp_spec_balanced_workspace_bytes

    return get_dcp_spec_balanced_workspace_bytes(sm_count, head_dim)


def dcp_spec_balanced_counter_bytes(sm_count: int) -> int:
    """Host-only forward of ``get_dcp_spec_balanced_counter_bytes``."""
    from flashinfer.cake_dcp import get_dcp_spec_balanced_counter_bytes

    return get_dcp_spec_balanced_counter_bytes(sm_count)


# --------------------------------------------------------------------------
# On-device load-balanced BF16 paged GQA decode (prepared runner)
# --------------------------------------------------------------------------


def supports_balanced_batch_decode_with_kv_cache(
    query: torch.Tensor,
    kv_cache: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    q_len_per_req: int = 1,
    kv_layout: str = "HND",
) -> bool:
    """Admission check mirroring ``validate_balanced_gqa_decode_inputs``."""
    try:
        import torch

        if kv_layout != "HND" or not isinstance(kv_cache, (tuple, list)):
            return False
        if len(kv_cache) != 2:
            return False
        k_cache, v_cache = kv_cache
        if not archs_in(BALANCED_ARCHS, query, k_cache, v_cache, block_tables):
            return False
        if not flashinfer_module_available(FI_BALANCED_MODULE):
            return False
        if query.ndim != 3 or k_cache.ndim != 4 or k_cache.shape != v_cache.shape:
            return False
        batch = int(block_tables.shape[0]) if block_tables.ndim == 2 else 0
        num_q_heads = int(query.shape[1])
        num_kv_heads = int(k_cache.shape[1])
        return (
            query.dtype == torch.bfloat16
            and k_cache.dtype == torch.bfloat16
            and v_cache.dtype == torch.bfloat16
            and block_tables.dtype == torch.int32
            and seq_lens.dtype == torch.int32
            and int(query.shape[2]) == BALANCED_HEAD_DIM
            and int(k_cache.shape[2]) == BALANCED_PAGE_SIZE
            and int(k_cache.shape[3]) == BALANCED_HEAD_DIM
            and num_q_heads == BALANCED_GROUP_RATIO * num_kv_heads
            and 1 <= q_len_per_req <= BALANCED_MAX_Q_LEN
            and 1 <= batch <= BALANCED_MAX_REQUESTS
            and int(query.shape[0]) == batch * q_len_per_req
            and int(seq_lens.numel()) == batch
        )
    except Exception:
        return False


def prepare_balanced_batch_decode_with_kv_cache(
    query: torch.Tensor,
    kv_cache: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    workspace_buffer: torch.Tensor,
    *,
    sm_scale: Optional[float] = None,
    q_len_per_req: int = 1,
    out: Optional[torch.Tensor] = None,
    kv_layout: str = "HND",
):
    """Forward to FlashInfer; returns a ``BalancedGQADecodeRunner``.

    ``runner.launch()`` (or ``runner()``) writes and returns the bound BF16
    output without allocating; tensor contents may change between launches.
    """
    from flashinfer.decode import prepare_balanced_batch_decode_with_kv_cache

    return prepare_balanced_batch_decode_with_kv_cache(
        query,
        kv_cache,
        block_tables,
        seq_lens,
        workspace_buffer,
        sm_scale=sm_scale,
        q_len_per_req=q_len_per_req,
        out=out,
        kv_layout=kv_layout,
        backend="cake",
    )


def balanced_gqa_decode_workspace_size(
    device=None, *, num_sms: Optional[int] = None, batch: int = 1024, max_pages: int = 0
) -> int:
    """Host-only forward of ``balanced_gqa_decode_workspace_size``."""
    from flashinfer.experimental.balanced_gqa_decode.cake_backend import (
        balanced_gqa_decode_workspace_size,
    )

    return balanced_gqa_decode_workspace_size(
        device, num_sms=num_sms, batch=batch, max_pages=max_pages
    )


# --------------------------------------------------------------------------
# SM110 (Thor) GQA decode and XQA
# --------------------------------------------------------------------------


def supports_sm110_gqa_decode(
    q: torch.Tensor, kv: torch.Tensor, sequence_lengths: torch.Tensor
) -> bool:
    """Admission check for the Thor FP16 32/8-head GQA decode; never raises."""
    try:
        import torch

        return (
            archs_in(SM110_ARCHS, q, kv, sequence_lengths)
            and flashinfer_module_available(FI_SM110_GQA_MODULE)
            and q.dtype == torch.float16
            and kv.dtype == torch.float16
            and sequence_lengths.dtype == torch.int32
            and q.ndim == 3
            and kv.ndim == 5
            and int(q.shape[1]) == SM110_GQA_NUM_Q_HEADS
            and int(q.shape[2]) == SM110_GQA_HEAD_DIM
            and int(kv.shape[0]) == int(q.shape[0])
            and int(kv.shape[1]) == 2
            and int(kv.shape[2]) == SM110_GQA_NUM_KV_HEADS
            and int(kv.shape[4]) == SM110_GQA_HEAD_DIM
            and int(sequence_lengths.numel()) == int(q.shape[0])
        )
    except Exception:
        return False


def sm110_gqa_decode(
    q: torch.Tensor,
    kv: torch.Tensor,
    sequence_lengths: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None,
    q_scale: float = 1.0,
) -> torch.Tensor:
    """Forward to ``flashinfer.decode.sm110_gqa_decode`` (one-shot)."""
    from flashinfer.decode import sm110_gqa_decode

    return sm110_gqa_decode(q, kv, sequence_lengths, out=out, q_scale=q_scale)


def prepare_sm110_gqa_decode(
    inputs: Dict[str, Any], num_splits: Optional[int] = None
) -> Dict[str, Any]:
    """Forward to ``flashinfer.decode.prepare_sm110_gqa_decode``.

    ``inputs`` holds ``Q`` ``[B, 32, 128]`` FP16, ``KV`` ``[B, 2, 8, C, 128]``
    FP16, caller-owned ``O`` ``[B, 32, 128]`` FP16 (not aliasing ``Q``) and
    int32 ``sequence_lengths [B]``; the returned dict owns the split workspace.
    """
    from flashinfer.decode import prepare_sm110_gqa_decode

    return prepare_sm110_gqa_decode(inputs, num_splits=num_splits)


def launch_sm110_gqa_decode_prepared(prepared: Dict[str, Any]) -> torch.Tensor:
    """Forward to ``flashinfer.decode.launch_sm110_gqa_decode_prepared``."""
    from flashinfer.decode import launch_sm110_gqa_decode_prepared

    return launch_sm110_gqa_decode_prepared(prepared)


def supports_sm110_xqa(q: torch.Tensor, kv: torch.Tensor) -> bool:
    """Coarse admission check for the Thor XQA family; never raises."""
    try:
        import torch

        if not archs_in(SM110_ARCHS, q, kv):
            return False
        if not flashinfer_module_available(FI_SM110_XQA_MODULE):
            return False
        if q.dtype != torch.float16:
            return False
        head_dim = int(q.shape[-1])
        if head_dim == 128:
            return kv.dtype == torch.float16 and kv.shape[-1] == 128
        if head_dim == 512:
            return kv.dtype in (torch.float16, torch.float8_e4m3fn) and (
                kv.shape[-1] == 512
            )
        return False
    except Exception:
        return False


def sm110_xqa_prepare(
    q: torch.Tensor,
    kv: torch.Tensor,
    sequence_lengths: torch.Tensor,
    *,
    mask: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    page_table: Optional[torch.Tensor] = None,
    page_size: int = 0,
    q_cu_seq_lens: Optional[torch.Tensor] = None,
    max_q_len: Optional[int] = None,
    sm_scale: Optional[float] = None,
    k_scale: float = 1.0,
    v_scale: float = 1.0,
    workspace: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    partition_tokens: Optional[int] = None,
    kernel: str = "tcgen05",
):
    """Forward to ``flashinfer.sm110_xqa.prepare``; returns a ``PreparedAttention``.

    ``prepared.run()`` submits the whole kernel chain on the current stream
    without allocating; call ``prepare`` before timing or graph capture.
    """
    from flashinfer.sm110_xqa import prepare

    return prepare(
        q,
        kv,
        sequence_lengths,
        mask=mask,
        out=out,
        page_table=page_table,
        page_size=page_size,
        q_cu_seq_lens=q_cu_seq_lens,
        max_q_len=max_q_len,
        sm_scale=sm_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        workspace=workspace,
        partition_tokens=partition_tokens,
        kernel=kernel,
    )


def sm110_xqa_attention(
    q: torch.Tensor, kv: torch.Tensor, sequence_lengths: torch.Tensor, **kwargs
) -> torch.Tensor:
    """Forward to ``flashinfer.sm110_xqa.attention`` (prepare + run once)."""
    from flashinfer.sm110_xqa import attention

    return attention(q, kv, sequence_lengths, **kwargs)
