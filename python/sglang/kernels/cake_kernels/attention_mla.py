"""Cake MLA (DeepSeek / Kimi-K3) decode kernels via FlashInfer.

FlashInfer entries (all at FlashInfer ``46340689a5ab``):

* ``flashinfer.mla.trtllm_batch_decode_sparse_mla_dsv4(backend="cake")``.
  On SM100 / SM103 (``flashinfer.mla.cake_dsv4.run_cake_dsv4``, JIT
  ``flashinfer.jit.cake_dsv4``): DeepSeek-V4 sparse MLA decode with BF16 or FP8
  e4m3 ``query [B, Q, H, 512]`` (or ragged ``[sum_q, H, 512]`` with
  ``cum_seq_lens_q`` / ``max_q_len``), ``H in {8, 16, 32, 64, 128}``, dense
  ``swa_kv_cache`` and ``compressed_kv_cache`` pools ``[..., 512]`` of the
  query dtype (compressed page 64 for the topk4x profile, 2 for topk128x),
  int32 ``sparse_indices`` either combined ``[T, sparse_topk]`` (first 128
  columns are the SWA table) or separate ``[T, 128]`` + ``extra_sparse_indices
  [T, topk_c]``, lengths through ``sparse_topk_lens`` (combined convention) or
  ``extra_sparse_topk_lens`` (compressed slots only, implies an offset of 128),
  int32 ``seq_lens [B]`` required, BF16 contiguous ``out``, optional FP32
  ``sinks [H]``, float or FP32 device scalar ``bmm1_scale`` / ``bmm2_scale``.
  Metadata rows ``T`` may be fewer than the query rows (padded rows are not
  touched). ``sparse_topk`` is a multiple of 4 and >= 128; ``enable_pdl`` must
  be falsy; ``hca_*`` arguments are rejected.
  On SM120 / SM121 with ``kv_cache_format="nvfp4"``
  (``flashinfer.mla._sparse_mla_sm120._cake_dsv4_nvfp4``, JIT
  ``flashinfer.jit.cake_sparse_mla_sm120_dsv4_nvfp4``): BF16 ``q [T, H, 512]``
  (448 NoPE + 64 RoPE), ``H in {8, 16, 32, 48, 64, 80, 96, 112, 128}``, packed
  NVFP4 uint8 cache (384 B per token) as ``[P, page, 384]``, HND
  ``[P, 1, page, 384]`` or NHD ``[P, page, 1, 384]`` with any page size,
  int32 ``indices [T, topk]`` with ``-1`` masks, int32 lengths ``[T]``,
  ``bmm2_scale == 1.0``, float ``bmm1_scale`` only, BF16 ``out``, base-2 FP32
  LSE. The decode-vs-prefill crossover is picked per call from the SM count.
* ``flashinfer.mla.SparseMLASm120Wrapper(backend="cake",
  kv_cache_format="nvfp4")`` and the direct
  ``cake_sparse_mla_sm120_dsv4_nvfp4_decode`` / ``_prefill`` entries with the
  same tensor contract (prefill: ``H`` a multiple of 16 in 16..128 and
  ``topk + extra_topk <= 1024``).
* ``flashinfer.mla.trtllm_batch_decode_with_kv_cache_mla(backend="cake")``
  answers two Cake families. Route A
  (``flashinfer.mla.cake_trtllm_mla_blackwell.trtllm_mla_blackwell_decode``;
  sm_100a / sm_103a builds for 148- and 152-SM parts, CUDA >= 12.9) for the
  generated dimension tuples ``(qk_nope, kv_lora, qk_rope, H)``: dense
  (128,512,64,128), (128,512,64,64), (64,256,64,32), (512,512,64,128) and
  top-k (128,512,64,{128,64}), (192,512,64,{128,64}); BF16 or FP8 e4m3 query
  ``[B, Q, H, D]`` or compact ``[T, H, D]`` + ``cum_seq_lens_q`` /
  ``max_q_len``; paged KV of the query dtype, page size 32 or 64; int32
  ``block_tables`` (dense ``[B, w]`` or sparse ``[B, Q, topk]`` / ``[T,
  topk]``); int32 ``seq_lens [B]`` required. Route B
  (``flashinfer.mla.cake_kimi_k3_mla.run_cake_kimi_k3_mla_fp8_paged_attention``,
  JIT ``flashinfer.jit.cake_kimi_k3_mla``) for every other tuple: FP8 e4m3
  query and KV only, ``kv_lora_rank=512``, ``qk_rope_head_dim=64`` (D=576),
  page size 64 (``[pages, 64, 576]`` or ``[pages, 1, 64, 576]``), host float
  scales, BF16 ``out`` of shape ``query.shape[:-1] + (512,)``; rejects
  ``sparse_mla_top_k``, sinks, LSE, DCP, skip-softmax, PDL.
* ``flashinfer.mla.KimiK3MlaFp8PagedAttention`` (prepared; all planning at
  construction, ``launch()`` allocates nothing) and
  ``flashinfer.mla.run_cake_kimi_k3_mla_fp8_paged_attention`` (one-shot).
* ``flashinfer.mla.prepare_nvfp4_batch_decode_with_kv_cache_mla`` (impl
  ``flashinfer.experimental.nvfp4_mla_decode.cake_backend``; SM100 / SM103):
  NVFP4 DeepSeek-V4 decode. uint8 ``query [B * q_len, H, 256]`` (packed E2M1)
  + uint8 ``query_scale [B * q_len, H, 32]`` (UE4M3 block-16), uint8
  ``kv_cache [pages, 64, 256]`` + ``kv_scale [pages, 64, 32]`` (one 512-wide
  latent row serves as K and V), int32 ``block_tables`` / ``seq_lens``,
  optional FP32 ``sinks [H]``, BF16 ``out [B * q_len, H, 512]``, natural-log
  FP32 ``lse``; ``q_len`` derived from the rows (6 for DSv4), causal within
  the block, ``B * q_len * H >= 128``, ``sm_scale`` required; workspace >=
  ``nvfp4_mla_decode_workspace_size`` / ``max_nvfp4_mla_decode_workspace_size``.
* ``flashinfer.mla.cake_mla_varq_dcp_decode`` /
  ``prepare_cake_mla_varq_dcp_decode`` (impl
  ``flashinfer.experimental.cake_mla_varq_dcp_decode.cake_backend``; SM100 /
  SM103): variable-q MLA decode with optional cyclic DCP. BF16 or FP8 e4m3
  compact ``query [total_q, H, 576]`` (``1 <= H <= 128``), ``kv_cache [pages,
  page, 576]`` (or ``[pages, 1, page, 576]``) of the query dtype with page
  size 32 / 64 / 128, int32 ``block_tables [B, max_pages]``, ``seq_lens [B]``
  (local), ``cum_seq_lens_q [B + 1]``, host ``max_q_len`` / ``max_seq_len``;
  DCP rank ``r`` holds global positions ``W * k + r`` and
  ``causal_seqlens_kv_global [B]`` is required when ``cp_world > 1``;
  scheduler cap ``B * ceil(max_q_len * H / 128) <= 512`` items; BF16
  ``out [total_q, H, 512]``, natural-log FP32 ``lse [total_q, H]`` always
  produced (``-inf`` / zero rows without visible keys).
* ``flashinfer.concat_ops.concat_mla_k(backend="cake")`` (JIT
  ``flashinfer.jit.cake_concat_mla_k``; sm_100f on cc 10.0 with CUDA >= 12.9,
  exact sm_103a): in-place ``k [T, 128, 192] = cat(k_nope [T, 128, 128],
  k_rope [T, 1, 64])`` for BF16 / FP16 / FP8 e4m3 / e5m2 (all equal); ``k``
  strides exactly ``(24576, 192, 1)`` or padded ``(32768, 256, 1)``;
  ``k_nope`` / ``k_rope`` strides one of the proven contiguous, nope-strided
  or both-strided profiles; no storage overlap. Kept under ``attention`` next
  to SGLang's existing ``attention.concat_mla_k`` Triton registration.

CUDA graphs: the SM100/103 DSv4 Cake route carves the caller's
``workspace_buffer`` deterministically (size with
``get_cake_dsv4_workspace_bytes``) and needs its split-merge counters zeroed
once with ``cake_dsv4_workspace_reset`` (or one eager call) before capture; the
kernels self-reset so replays need no host state, and it raises when the
counters are unprimed while capturing. The SM120 wrapper's split scratch is
grow-only: warm every shape before capture. The trtllm-style Blackwell MLA
decode (route A) copies ``seq_lens`` / ``cum_seq_lens_q`` to the host on every
call, so it is NOT replayable as-is; the Kimi-K3 route (B) plans on the host
from shapes only and its launch allocates nothing. NVFP4 MLA decode and the
varq DCP decode prepare on the host (one D2H copy of ``seq_lens`` for NVFP4
unless ``seq_lens_cpu`` is given) and launch without allocation; re-prepare
when shapes, ``max_seq_len`` or bindings change and never share one varq
workspace between two live runners. ``concat_mla_k`` has no workspace and no
host sync.

Not supported here (keep the existing SGLang path): SM90 devices; DSv4 with
``hca_*`` / RopeQuant arguments; MLA decode dimension tuples outside the
generated set with non-FP8 operands; NVFP4 MLA decode with fewer than 128
packed query rows; ``concat_mla_k`` for other head counts or strides.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Literal, Optional, Sequence, Union

from sglang.kernels.cake_kernels.attention_common import (
    SM100,
    SM103,
    SM120,
    SM121,
    archs_in,
    cuda_tensor_on,
    device_sm_count,
    flashinfer_module_available,
)

if TYPE_CHECKING:
    import torch

FI_MLA_MODULE = "flashinfer.mla"
FI_DSV4_MODULE = "flashinfer.mla.cake_dsv4"
FI_DSV4_JIT_MODULE = "flashinfer.jit.cake_dsv4"
FI_SM120_DSV4_MODULE = "flashinfer.mla._sparse_mla_sm120._cake_dsv4_nvfp4"
FI_SM120_DSV4_JIT_MODULE = "flashinfer.jit.cake_sparse_mla_sm120_dsv4_nvfp4"
FI_TRTLLM_MLA_MODULE = "flashinfer.mla.cake_trtllm_mla_blackwell"
FI_KIMI_K3_MLA_MODULE = "flashinfer.mla.cake_kimi_k3_mla"
FI_KIMI_K3_MLA_JIT_MODULE = "flashinfer.jit.cake_kimi_k3_mla"
FI_NVFP4_MLA_MODULE = "flashinfer.experimental.nvfp4_mla_decode.cake_backend"
FI_VARQ_DCP_MODULE = "flashinfer.experimental.cake_mla_varq_dcp_decode.cake_backend"
FI_CONCAT_MODULE = "flashinfer.concat_ops"
FI_CONCAT_JIT_MODULE = "flashinfer.jit.cake_concat_mla_k"

ARCHS = (SM100, SM103)
SM120_ARCHS = (SM120, SM121)
TRTLLM_MLA_SM_COUNTS = (148, 152)

DSV4_HEAD_DIM = 512
DSV4_HEADS = (8, 16, 32, 64, 128)
DSV4_SWA_TOPK = 128
SM120_DSV4_HEADS = (8, 16, 32, 48, 64, 80, 96, 112, 128)
SM120_DSV4_BYTES_PER_TOKEN = 384
TRTLLM_MLA_DENSE_TUPLES = frozenset(
    {(128, 512, 64, 128), (128, 512, 64, 64), (64, 256, 64, 32), (512, 512, 64, 128)}
)
TRTLLM_MLA_TOPK_TUPLES = frozenset(
    {(128, 512, 64, 128), (128, 512, 64, 64), (192, 512, 64, 128), (192, 512, 64, 64)}
)
TRTLLM_MLA_PAGE_SIZES = (32, 64)
KIMI_K3_LATENT = 512
KIMI_K3_ROPE = 64
KIMI_K3_QK_DIM = KIMI_K3_LATENT + KIMI_K3_ROPE
KIMI_K3_PAGE_SIZE = 64
NVFP4_MLA_PACKED_ROW = 256
NVFP4_MLA_SCALE_ROW = 32
NVFP4_MLA_PAGE_SIZE = 64
NVFP4_MLA_MIN_ROWS = 128
VARQ_HEAD_DIM_QK = 576
VARQ_MAX_HEADS = 128
VARQ_PAGE_SIZES = (32, 64, 128)
VARQ_MAX_ITEMS = 512
CONCAT_NUM_HEADS = 128
CONCAT_NOPE_DIM = 128
CONCAT_ROPE_DIM = 64
CONCAT_OUTPUT_STRIDES = {(24576, 192, 1), (32768, 256, 1)}
CONCAT_INPUT_STRIDE_PROFILES = {
    ((16384, 128, 1), (64, 64, 1)),
    ((32768, 256, 1), (64, 64, 1)),
    ((32768, 256, 1), (192, 192, 1)),
}


def _mla_dtypes():
    import torch

    return (torch.bfloat16, torch.float8_e4m3fn)


# --------------------------------------------------------------------------
# DeepSeek-V4 sparse MLA decode (trtllm_batch_decode_sparse_mla_dsv4 cake)
# --------------------------------------------------------------------------


def supports_trtllm_batch_decode_sparse_mla_dsv4(
    query: torch.Tensor,
    swa_kv_cache: torch.Tensor,
    *,
    kv_cache_format: str = "fp8",
    compressed_kv_cache: Optional[torch.Tensor] = None,
    enable_pdl: Optional[bool] = None,
) -> bool:
    """Admission check for both Cake DSv4 routes; never raises.

    SM100 / SM103 serve ``kv_cache_format="fp8"`` (dense 512-wide BF16 / FP8
    pools); SM120 / SM121 serve ``kv_cache_format="nvfp4"`` (packed uint8).
    """
    try:
        import torch

        if enable_pdl:
            return False
        if cuda_tensor_on(query, ARCHS):
            if kv_cache_format != "fp8":
                return False
            if not flashinfer_module_available(FI_DSV4_MODULE, FI_DSV4_JIT_MODULE):
                return False
            if not archs_in(ARCHS, query, swa_kv_cache):
                return False
            if compressed_kv_cache is not None and not archs_in(
                ARCHS, query, compressed_kv_cache
            ):
                return False
            heads = int(query.shape[-2]) if query.ndim in (3, 4) else 0
            return (
                query.dtype in _mla_dtypes()
                and swa_kv_cache.dtype == query.dtype
                and (
                    compressed_kv_cache is None
                    or compressed_kv_cache.dtype == query.dtype
                )
                and int(query.shape[-1]) == DSV4_HEAD_DIM
                and int(swa_kv_cache.shape[-1]) == DSV4_HEAD_DIM
                and heads in DSV4_HEADS
            )
        if cuda_tensor_on(query, SM120_ARCHS):
            if kv_cache_format != "nvfp4":
                return False
            if not flashinfer_module_available(
                FI_SM120_DSV4_MODULE, FI_SM120_DSV4_JIT_MODULE
            ):
                return False
            if not archs_in(SM120_ARCHS, query, swa_kv_cache):
                return False
            heads = int(query.shape[-2]) if query.ndim in (3, 4) else 0
            return (
                query.dtype == torch.bfloat16
                and int(query.shape[-1]) == DSV4_HEAD_DIM
                and heads in SM120_DSV4_HEADS
                and swa_kv_cache.dtype == torch.uint8
                and int(swa_kv_cache.shape[-1]) == SM120_DSV4_BYTES_PER_TOKEN
            )
        return False
    except Exception:
        return False


def trtllm_batch_decode_sparse_mla_dsv4(
    query: torch.Tensor,
    swa_kv_cache: torch.Tensor,
    workspace_buffer: torch.Tensor,
    sparse_indices: Optional[torch.Tensor] = None,
    compressed_kv_cache: Optional[torch.Tensor] = None,
    sparse_topk_lens: Optional[torch.Tensor] = None,
    seq_lens: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    bmm1_scale: Union[float, torch.Tensor] = 1.0,
    bmm2_scale: Union[float, torch.Tensor] = 1.0,
    sinks: Optional[torch.Tensor] = None,
    kv_layout: Literal["HND", "NHD"] = "HND",
    cum_seq_lens_q: Optional[torch.Tensor] = None,
    max_q_len: Optional[int] = None,
    enable_pdl: Optional[bool] = None,
    swa_topk_lens: Optional[torch.Tensor] = None,
    extra_sparse_indices: Optional[torch.Tensor] = None,
    extra_sparse_topk_lens: Optional[torch.Tensor] = None,
    sparse_topk_lens_offset: int = 0,
    *,
    kv_cache_format: Literal["fp8", "nvfp4"] = "fp8",
):
    """Forward to ``trtllm_batch_decode_sparse_mla_dsv4(..., backend="cake")``."""
    from flashinfer.mla import trtllm_batch_decode_sparse_mla_dsv4

    return trtllm_batch_decode_sparse_mla_dsv4(
        query,
        swa_kv_cache,
        workspace_buffer,
        sparse_indices=sparse_indices,
        compressed_kv_cache=compressed_kv_cache,
        sparse_topk_lens=sparse_topk_lens,
        seq_lens=seq_lens,
        out=out,
        bmm1_scale=bmm1_scale,
        bmm2_scale=bmm2_scale,
        sinks=sinks,
        kv_layout=kv_layout,
        cum_seq_lens_q=cum_seq_lens_q,
        max_q_len=max_q_len,
        enable_pdl=enable_pdl,
        swa_topk_lens=swa_topk_lens,
        extra_sparse_indices=extra_sparse_indices,
        extra_sparse_topk_lens=extra_sparse_topk_lens,
        backend="cake",
        sparse_topk_lens_offset=sparse_topk_lens_offset,
        kv_cache_format=kv_cache_format,
    )


def get_cake_dsv4_workspace_bytes(
    num_query_tokens: int,
    num_heads: int,
    sparse_topk: int,
    dtype: torch.dtype,
    *,
    num_splits: Optional[int] = None,
) -> int:
    """Host-only forward of ``flashinfer.mla.get_cake_dsv4_workspace_bytes``."""
    from flashinfer.mla import get_cake_dsv4_workspace_bytes

    return get_cake_dsv4_workspace_bytes(
        num_query_tokens, num_heads, sparse_topk, dtype, num_splits=num_splits
    )


def cake_dsv4_workspace_reset(workspace_buffer: torch.Tensor) -> None:
    """Zero the split-merge counters once (before CUDA-graph capture)."""
    from flashinfer.mla import cake_dsv4_workspace_reset

    cake_dsv4_workspace_reset(workspace_buffer)


# --------------------------------------------------------------------------
# SM120 NVFP4 DSv4 sparse MLA: wrapper + direct decode / prefill entries
# --------------------------------------------------------------------------


def supports_sparse_mla_sm120_dsv4_nvfp4(
    q: torch.Tensor, kv_cache: torch.Tensor, indices: torch.Tensor
) -> bool:
    """Admission check for the direct SM120 NVFP4 entries; never raises."""
    try:
        import torch

        return (
            archs_in(SM120_ARCHS, q, kv_cache, indices)
            and flashinfer_module_available(
                FI_SM120_DSV4_MODULE, FI_SM120_DSV4_JIT_MODULE
            )
            and q.dtype == torch.bfloat16
            and q.ndim == 3
            and int(q.shape[1]) in SM120_DSV4_HEADS
            and int(q.shape[2]) == DSV4_HEAD_DIM
            and kv_cache.dtype == torch.uint8
            and int(kv_cache.shape[-1]) == SM120_DSV4_BYTES_PER_TOKEN
            and indices.dtype == torch.int32
            and int(indices.shape[0]) == int(q.shape[0])
        )
    except Exception:
        return False


def create_sparse_mla_sm120_wrapper(
    max_num_tokens: Optional[int] = None,
    max_num_heads: Optional[int] = None,
    *,
    device=None,
):
    """``SparseMLASm120Wrapper(backend="cake", kv_cache_format="nvfp4")``.

    ``max_num_tokens`` and ``max_num_heads`` must be given together; the
    wrapper's grow-only scratch means every shape must be warmed before
    CUDA-graph capture. ``wrapper.run(...)`` keeps FlashInfer's contract.
    """
    from flashinfer.mla import SparseMLASm120Wrapper

    return SparseMLASm120Wrapper(
        max_num_tokens=max_num_tokens,
        max_num_heads=max_num_heads,
        kv_cache_format="nvfp4",
        device=device,
        backend="cake",
    )


def sparse_mla_sm120_dsv4_nvfp4_decode(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    indices: torch.Tensor,
    output: torch.Tensor,
    out_lse: torch.Tensor,
    sm_scale: float,
    *,
    topk_length: Optional[torch.Tensor] = None,
    attn_sink: Optional[torch.Tensor] = None,
    extra_kv_cache: Optional[torch.Tensor] = None,
    extra_indices: Optional[torch.Tensor] = None,
    extra_topk_length: Optional[torch.Tensor] = None,
    mid_out: Optional[torch.Tensor] = None,
    mid_lse: Optional[torch.Tensor] = None,
    lse_scale: float = 1.0,
    num_splits: Optional[int] = None,
    max_splits: int = 16,
    head_tiles: Optional[int] = None,
) -> Dict[str, int]:
    """Forward to ``flashinfer.mla.cake_sparse_mla_sm120_dsv4_nvfp4_decode``.

    Returns the resolved plan ``{head_tiles, num_splits, chunks_per_block}``;
    ``mid_out [T, H, S, 512]`` / ``mid_lse [T, H, S]`` are required when the
    plan splits (``num_splits > 1``).
    """
    from flashinfer.mla import cake_sparse_mla_sm120_dsv4_nvfp4_decode

    return cake_sparse_mla_sm120_dsv4_nvfp4_decode(
        q,
        kv_cache,
        indices,
        output,
        out_lse,
        sm_scale,
        topk_length=topk_length,
        attn_sink=attn_sink,
        extra_kv_cache=extra_kv_cache,
        extra_indices=extra_indices,
        extra_topk_length=extra_topk_length,
        mid_out=mid_out,
        mid_lse=mid_lse,
        lse_scale=lse_scale,
        num_splits=num_splits,
        max_splits=max_splits,
        head_tiles=head_tiles,
    )


def sparse_mla_sm120_dsv4_nvfp4_prefill(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    indices: torch.Tensor,
    output: torch.Tensor,
    out_lse: torch.Tensor,
    sm_scale: float,
    *,
    topk_length: Optional[torch.Tensor] = None,
    attn_sink: Optional[torch.Tensor] = None,
    extra_kv_cache: Optional[torch.Tensor] = None,
    extra_indices: Optional[torch.Tensor] = None,
    extra_topk_length: Optional[torch.Tensor] = None,
    lse_scale: float = 1.0,
    head_tiles: Optional[int] = None,
) -> Dict[str, int]:
    """Forward to ``flashinfer.mla.cake_sparse_mla_sm120_dsv4_nvfp4_prefill``."""
    from flashinfer.mla import cake_sparse_mla_sm120_dsv4_nvfp4_prefill

    return cake_sparse_mla_sm120_dsv4_nvfp4_prefill(
        q,
        kv_cache,
        indices,
        output,
        out_lse,
        sm_scale,
        topk_length=topk_length,
        attn_sink=attn_sink,
        extra_kv_cache=extra_kv_cache,
        extra_indices=extra_indices,
        extra_topk_length=extra_topk_length,
        lse_scale=lse_scale,
        head_tiles=head_tiles,
    )


def sparse_mla_sm120_dsv4_nvfp4_scratch_bytes(
    num_tokens: int, num_heads: int, topk: int, extra_topk: int = 0
) -> int:
    """Host-only forward of ``cake_sparse_mla_sm120_dsv4_nvfp4_scratch_bytes``."""
    from flashinfer.mla import cake_sparse_mla_sm120_dsv4_nvfp4_scratch_bytes

    return cake_sparse_mla_sm120_dsv4_nvfp4_scratch_bytes(
        num_tokens, num_heads, topk, extra_topk
    )


def sparse_mla_sm120_dsv4_nvfp4_select_kernel(
    *, num_tokens: int, num_heads: int, topk: int, extra_topk: int = 0, num_sms: int
) -> str:
    """Host-only forward of ``cake_sparse_mla_sm120_dsv4_nvfp4_select_kernel``."""
    from flashinfer.mla import cake_sparse_mla_sm120_dsv4_nvfp4_select_kernel

    return cake_sparse_mla_sm120_dsv4_nvfp4_select_kernel(
        num_tokens=num_tokens,
        num_heads=num_heads,
        topk=topk,
        extra_topk=extra_topk,
        num_sms=num_sms,
    )


# --------------------------------------------------------------------------
# Paged MLA decode (trtllm_batch_decode_with_kv_cache_mla backend="cake")
# --------------------------------------------------------------------------


def _kv_page_size(kv_cache) -> int:
    if kv_cache.ndim == 3:
        return int(kv_cache.shape[1])
    if kv_cache.ndim == 4:
        return int(kv_cache.shape[2])
    return 0


def supports_trtllm_batch_decode_with_kv_cache_mla(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    *,
    qk_nope_head_dim: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    sparse_mla_top_k: int = 0,
    sinks: Optional[List[torch.Tensor]] = None,
    return_lse: bool = False,
    enable_dcp: bool = False,
) -> bool:
    """Admission check for the union of the two Cake MLA routes; never raises."""
    try:
        import torch

        if not archs_in(ARCHS, query, kv_cache):
            return False
        if enable_dcp or query.ndim not in (3, 4):
            return False
        if query.dtype not in _mla_dtypes() or kv_cache.dtype != query.dtype:
            return False
        num_heads = int(query.shape[-2])
        head_dim = int(query.shape[-1])
        if head_dim != kv_lora_rank + qk_rope_head_dim:
            return False
        page_size = _kv_page_size(kv_cache)
        key = (qk_nope_head_dim, kv_lora_rank, qk_rope_head_dim, num_heads)
        tuples = (
            TRTLLM_MLA_TOPK_TUPLES
            if sparse_mla_top_k > 0
            else (TRTLLM_MLA_DENSE_TUPLES)
        )
        if key in tuples:
            return (
                flashinfer_module_available(FI_TRTLLM_MLA_MODULE)
                and device_sm_count(query.device.index) in TRTLLM_MLA_SM_COUNTS
                and page_size in TRTLLM_MLA_PAGE_SIZES
            )
        return (
            flashinfer_module_available(
                FI_KIMI_K3_MLA_MODULE, FI_KIMI_K3_MLA_JIT_MODULE
            )
            and query.dtype == torch.float8_e4m3fn
            and kv_lora_rank == KIMI_K3_LATENT
            and qk_rope_head_dim == KIMI_K3_ROPE
            and page_size == KIMI_K3_PAGE_SIZE
            and sparse_mla_top_k == 0
            and sinks is None
            and not return_lse
        )
    except Exception:
        return False


def trtllm_batch_decode_with_kv_cache_mla(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    workspace_buffer: torch.Tensor,
    qk_nope_head_dim: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    block_tables: torch.Tensor,
    seq_lens: Optional[torch.Tensor],
    max_seq_len: int,
    sparse_mla_top_k: int = 0,
    out: Optional[torch.Tensor] = None,
    bmm1_scale: Union[float, torch.Tensor] = 1.0,
    bmm2_scale: Union[float, torch.Tensor] = 1.0,
    sinks: Optional[List[torch.Tensor]] = None,
    skip_softmax_threshold_scale_factor: Optional[float] = None,
    enable_pdl: Optional[bool] = None,
    is_var_seq: bool = True,
    uses_shared_paged_kv_idx: bool = True,
    lse: Optional[torch.Tensor] = None,
    return_lse: bool = False,
    cum_seq_lens_q: Optional[torch.Tensor] = None,
    max_q_len: Optional[int] = None,
    multi_ctas_kv_counter_buffer: Optional[torch.Tensor] = None,
    sparse_mla_top_k_lens: Optional[torch.Tensor] = None,
    enable_dcp: bool = False,
    cp_world: int = 1,
    cp_rank: int = 0,
    causal_seqlens_kv_global: Optional[torch.Tensor] = None,
    use_fp16_softmax: Optional[bool] = None,
    return_lse_base: Optional[Literal["basee", "base2"]] = None,
):
    """Forward to ``trtllm_batch_decode_with_kv_cache_mla(..., backend="cake")``."""
    from flashinfer.mla import trtllm_batch_decode_with_kv_cache_mla

    return trtllm_batch_decode_with_kv_cache_mla(
        query,
        kv_cache,
        workspace_buffer,
        qk_nope_head_dim,
        kv_lora_rank,
        qk_rope_head_dim,
        block_tables,
        seq_lens,
        max_seq_len,
        sparse_mla_top_k=sparse_mla_top_k,
        out=out,
        bmm1_scale=bmm1_scale,
        bmm2_scale=bmm2_scale,
        sinks=sinks,
        skip_softmax_threshold_scale_factor=skip_softmax_threshold_scale_factor,
        enable_pdl=enable_pdl,
        backend="cake",
        is_var_seq=is_var_seq,
        uses_shared_paged_kv_idx=uses_shared_paged_kv_idx,
        lse=lse,
        return_lse=return_lse,
        cum_seq_lens_q=cum_seq_lens_q,
        max_q_len=max_q_len,
        multi_ctas_kv_counter_buffer=multi_ctas_kv_counter_buffer,
        sparse_mla_top_k_lens=sparse_mla_top_k_lens,
        enable_dcp=enable_dcp,
        cp_world=cp_world,
        cp_rank=cp_rank,
        causal_seqlens_kv_global=causal_seqlens_kv_global,
        use_fp16_softmax=use_fp16_softmax,
        return_lse_base=return_lse_base,
    )


# --------------------------------------------------------------------------
# Kimi-K3 FP8 paged MLA attention (direct Cake entries)
# --------------------------------------------------------------------------


def supports_kimi_k3_mla_fp8_paged_attention(
    query: torch.Tensor, kv_cache: torch.Tensor
) -> bool:
    """Admission check for the Kimi-K3 FP8 MLA route; never raises."""
    try:
        import torch

        return (
            archs_in(ARCHS, query, kv_cache)
            and flashinfer_module_available(
                FI_KIMI_K3_MLA_MODULE, FI_KIMI_K3_MLA_JIT_MODULE
            )
            and query.dtype == torch.float8_e4m3fn
            and kv_cache.dtype == torch.float8_e4m3fn
            and query.ndim in (3, 4)
            and int(query.shape[-1]) == KIMI_K3_QK_DIM
            and int(kv_cache.shape[-1]) == KIMI_K3_QK_DIM
            and _kv_page_size(kv_cache) == KIMI_K3_PAGE_SIZE
        )
    except Exception:
        return False


def kimi_k3_mla_fp8_paged_attention(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    out: torch.Tensor,
    workspace_buffer: torch.Tensor,
    *,
    bmm1_scale: float,
    bmm2_scale: float = 1.0,
    cum_seq_lens_q: Optional[torch.Tensor] = None,
    max_q_len: Optional[int] = None,
    max_seq_len: Optional[int] = None,
) -> torch.Tensor:
    """Forward to ``flashinfer.mla.run_cake_kimi_k3_mla_fp8_paged_attention``."""
    from flashinfer.mla import run_cake_kimi_k3_mla_fp8_paged_attention

    return run_cake_kimi_k3_mla_fp8_paged_attention(
        query,
        kv_cache,
        block_tables,
        seq_lens,
        out,
        workspace_buffer,
        bmm1_scale=bmm1_scale,
        bmm2_scale=bmm2_scale,
        cum_seq_lens_q=cum_seq_lens_q,
        max_q_len=max_q_len,
        max_seq_len=max_seq_len,
    )


def prepare_kimi_k3_mla_fp8_paged_attention(
    *,
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    out: torch.Tensor,
    workspace_buffer: torch.Tensor,
    bmm1_scale: float,
    bmm2_scale: float = 1.0,
    cum_seq_lens_q: Optional[torch.Tensor] = None,
    max_q_len: Optional[int] = None,
    max_seq_len: Optional[int] = None,
    num_split: Optional[int] = None,
):
    """Construct ``flashinfer.mla.KimiK3MlaFp8PagedAttention`` (prepared).

    All planning happens here; ``prepared.launch()`` allocates nothing and
    is CUDA-graph capturable. ``workspace_buffer`` must hold at least
    :func:`kimi_k3_mla_workspace_bytes` bytes.
    """
    from flashinfer.mla import KimiK3MlaFp8PagedAttention

    return KimiK3MlaFp8PagedAttention(
        query=query,
        kv_cache=kv_cache,
        block_tables=block_tables,
        seq_lens=seq_lens,
        out=out,
        workspace_buffer=workspace_buffer,
        bmm1_scale=bmm1_scale,
        bmm2_scale=bmm2_scale,
        cum_seq_lens_q=cum_seq_lens_q,
        max_q_len=max_q_len,
        max_seq_len=max_seq_len,
        num_split=num_split,
    )


def kimi_k3_mla_workspace_bytes(rows_max: int, num_split: int) -> int:
    """Host-only forward of ``cake_kimi_k3_mla.workspace_bytes``."""
    from flashinfer.mla.cake_kimi_k3_mla import workspace_bytes

    return workspace_bytes(rows_max, num_split)


# --------------------------------------------------------------------------
# NVFP4 DeepSeek-V4 MLA decode (prepared runner)
# --------------------------------------------------------------------------


def supports_nvfp4_batch_decode_with_kv_cache_mla(
    query: torch.Tensor,
    query_scale: torch.Tensor,
    kv_cache: torch.Tensor,
    kv_scale: torch.Tensor,
) -> bool:
    """Admission check mirroring ``validate_nvfp4_mla_decode_inputs``."""
    try:
        import torch

        return (
            archs_in(ARCHS, query, query_scale, kv_cache, kv_scale)
            and flashinfer_module_available(FI_NVFP4_MLA_MODULE)
            and query.dtype == torch.uint8
            and query_scale.dtype == torch.uint8
            and kv_cache.dtype == torch.uint8
            and kv_scale.dtype in (torch.uint8, torch.float8_e4m3fn)
            and query.ndim == 3
            and int(query.shape[2]) == NVFP4_MLA_PACKED_ROW
            and tuple(query_scale.shape)
            == (int(query.shape[0]), int(query.shape[1]), NVFP4_MLA_SCALE_ROW)
            and kv_cache.ndim == 3
            and int(kv_cache.shape[1]) == NVFP4_MLA_PAGE_SIZE
            and int(kv_cache.shape[2]) == NVFP4_MLA_PACKED_ROW
            and tuple(kv_scale.shape)
            == (int(kv_cache.shape[0]), NVFP4_MLA_PAGE_SIZE, NVFP4_MLA_SCALE_ROW)
            and int(query.shape[0]) * int(query.shape[1]) >= NVFP4_MLA_MIN_ROWS
        )
    except Exception:
        return False


def prepare_nvfp4_batch_decode_with_kv_cache_mla(
    query: torch.Tensor,
    query_scale: torch.Tensor,
    kv_cache: torch.Tensor,
    kv_scale: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    workspace_buffer: torch.Tensor,
    *,
    sm_scale: float,
    sinks: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    lse: Optional[torch.Tensor] = None,
    return_lse: bool = False,
    seq_lens_cpu: Optional[torch.Tensor] = None,
):
    """Forward to FlashInfer; returns an ``NVFP4MLADecodeRunner``.

    ``runner.launch()`` (or ``runner()``) returns ``out`` or ``(out, lse)``
    with ``return_lse=True``; it never allocates or synchronizes. Re-prepare
    when ``seq_lens`` contents or tensor bindings change.
    """
    from flashinfer.mla import prepare_nvfp4_batch_decode_with_kv_cache_mla

    return prepare_nvfp4_batch_decode_with_kv_cache_mla(
        query,
        query_scale,
        kv_cache,
        kv_scale,
        block_tables,
        seq_lens,
        workspace_buffer,
        sm_scale=sm_scale,
        sinks=sinks,
        out=out,
        lse=lse,
        return_lse=return_lse,
        seq_lens_cpu=seq_lens_cpu,
        backend="cake",
    )


def nvfp4_mla_decode_workspace_size(
    seq_lens: Sequence[int],
    num_heads: int,
    *,
    num_sms: int,
    q_len: int = 6,
    enable_sink: bool = False,
    schedule: str = "auto",
) -> int:
    """Host-only forward of ``nvfp4_mla_decode_workspace_size``."""
    from flashinfer.experimental.nvfp4_mla_decode.cake_backend import (
        nvfp4_mla_decode_workspace_size,
    )

    return nvfp4_mla_decode_workspace_size(
        seq_lens,
        num_heads,
        num_sms=num_sms,
        q_len=q_len,
        enable_sink=enable_sink,
        schedule=schedule,
    )


def max_nvfp4_mla_decode_workspace_size(
    batch: int, num_heads: int, *, q_len: int = 6, max_splits: int = 64
) -> int:
    """Host-only forward of ``max_nvfp4_mla_decode_workspace_size``."""
    from flashinfer.experimental.nvfp4_mla_decode.cake_backend import (
        max_nvfp4_mla_decode_workspace_size,
    )

    return max_nvfp4_mla_decode_workspace_size(
        batch, num_heads, q_len=q_len, max_splits=max_splits
    )


def quantize_nvfp4(x: torch.Tensor):
    """Forward of the backend's ``quantize_nvfp4`` (packed bytes + scales)."""
    from flashinfer.experimental.nvfp4_mla_decode.cake_backend import quantize_nvfp4

    return quantize_nvfp4(x)


# --------------------------------------------------------------------------
# Variable-q MLA decode with cyclic DCP (one-shot and prepared)
# --------------------------------------------------------------------------


def supports_mla_varq_dcp_decode(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    *,
    batch_size: int,
    max_q_len: int,
    cp_world: int = 1,
    causal_seqlens_kv_global: Optional[torch.Tensor] = None,
) -> bool:
    """Admission check mirroring ``validate_cake_mla_varq_dcp_decode_inputs``."""
    try:
        if not archs_in(ARCHS, query, kv_cache):
            return False
        if not flashinfer_module_available(FI_VARQ_DCP_MODULE):
            return False
        if query.ndim != 3 or query.dtype not in _mla_dtypes():
            return False
        if kv_cache.dtype != query.dtype:
            return False
        num_heads = int(query.shape[1])
        if cp_world > 1 and causal_seqlens_kv_global is None:
            return False
        items = batch_size * (-(-(max_q_len * num_heads) // 128))
        return (
            1 <= num_heads <= VARQ_MAX_HEADS
            and int(query.shape[2]) == VARQ_HEAD_DIM_QK
            and int(kv_cache.shape[-1]) == VARQ_HEAD_DIM_QK
            and _kv_page_size(kv_cache) in VARQ_PAGE_SIZES
            and max_q_len >= 1
            and int(query.shape[0]) <= batch_size * max_q_len
            and items <= VARQ_MAX_ITEMS
        )
    except Exception:
        return False


def mla_varq_dcp_decode(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    workspace_buffer: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    max_seq_len: int,
    softmax_scale: float,
    *,
    cum_seq_lens_q: torch.Tensor,
    max_q_len: int,
    enable_dcp: bool = False,
    cp_world: int = 1,
    cp_rank: int = 0,
    causal_seqlens_kv_global: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    lse: Optional[torch.Tensor] = None,
    return_lse: bool = True,
):
    """Forward to ``flashinfer.mla.cake_mla_varq_dcp_decode`` (one-shot)."""
    from flashinfer.mla import cake_mla_varq_dcp_decode

    return cake_mla_varq_dcp_decode(
        query,
        kv_cache,
        workspace_buffer,
        block_tables,
        seq_lens,
        max_seq_len,
        softmax_scale,
        cum_seq_lens_q=cum_seq_lens_q,
        max_q_len=max_q_len,
        enable_dcp=enable_dcp,
        cp_world=cp_world,
        cp_rank=cp_rank,
        causal_seqlens_kv_global=causal_seqlens_kv_global,
        out=out,
        lse=lse,
        return_lse=return_lse,
        backend="cake",
    )


def prepare_mla_varq_dcp_decode(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    page_table: torch.Tensor,
    seq_lens: torch.Tensor,
    cum_seq_lens_q: torch.Tensor,
    max_q_len: int,
    *,
    max_seq_len: int,
    softmax_scale: float,
    workspace_buffer: torch.Tensor,
    causal_seqlens_kv_global: Optional[torch.Tensor] = None,
    cp_world: int = 1,
    cp_rank: int = 0,
    out: Optional[torch.Tensor] = None,
    lse: Optional[torch.Tensor] = None,
):
    """Forward to ``flashinfer.mla.prepare_cake_mla_varq_dcp_decode``.

    Returns a ``CakeMLAVarQDcpDecodeRunner``; ``runner.launch()`` (or
    ``runner()``) returns ``(out, lse)`` without allocation or host sync.
    """
    from flashinfer.mla import prepare_cake_mla_varq_dcp_decode

    return prepare_cake_mla_varq_dcp_decode(
        query,
        kv_cache,
        page_table,
        seq_lens,
        cum_seq_lens_q,
        max_q_len,
        max_seq_len=max_seq_len,
        softmax_scale=softmax_scale,
        workspace_buffer=workspace_buffer,
        causal_seqlens_kv_global=causal_seqlens_kv_global,
        cp_world=cp_world,
        cp_rank=cp_rank,
        out=out,
        lse=lse,
        backend="cake",
    )


def mla_varq_dcp_decode_workspace_size(
    *, batch_size: int, max_q_len: int, num_heads: int, max_seq_len: int, num_sms: int
) -> int:
    """Host-only forward of ``cake_mla_varq_dcp_decode_workspace_size``."""
    from flashinfer.experimental.cake_mla_varq_dcp_decode.cake_backend import (
        cake_mla_varq_dcp_decode_workspace_size,
    )

    return cake_mla_varq_dcp_decode_workspace_size(
        batch_size=batch_size,
        max_q_len=max_q_len,
        num_heads=num_heads,
        max_seq_len=max_seq_len,
        num_sms=num_sms,
    )


def max_mla_varq_dcp_decode_workspace_size(
    *, batch_size: int, max_q_len: int, num_heads: int, num_sms: int
) -> int:
    """Host-only forward of ``max_cake_mla_varq_dcp_decode_workspace_size``."""
    from flashinfer.experimental.cake_mla_varq_dcp_decode.cake_backend import (
        max_cake_mla_varq_dcp_decode_workspace_size,
    )

    return max_cake_mla_varq_dcp_decode_workspace_size(
        batch_size=batch_size,
        max_q_len=max_q_len,
        num_heads=num_heads,
        num_sms=num_sms,
    )


# --------------------------------------------------------------------------
# concat_mla_k (in-place K assembly)
# --------------------------------------------------------------------------


def supports_concat_mla_k(
    k: torch.Tensor, k_nope: torch.Tensor, k_rope: torch.Tensor
) -> bool:
    """Admission check mirroring the Cake ``concat_mla_k`` contract."""
    try:
        import torch

        if not archs_in(ARCHS, k, k_nope, k_rope):
            return False
        if not flashinfer_module_available(FI_CONCAT_MODULE, FI_CONCAT_JIT_MODULE):
            return False
        dtypes = (
            torch.bfloat16,
            torch.float16,
            torch.float8_e4m3fn,
            torch.float8_e5m2,
        )
        num_tokens = int(k.shape[0]) if k.ndim == 3 else -1
        return (
            k.dtype in dtypes
            and k_nope.dtype == k.dtype
            and k_rope.dtype == k.dtype
            and tuple(k.shape)
            == (num_tokens, CONCAT_NUM_HEADS, CONCAT_NOPE_DIM + CONCAT_ROPE_DIM)
            and tuple(k_nope.shape) == (num_tokens, CONCAT_NUM_HEADS, CONCAT_NOPE_DIM)
            and tuple(k_rope.shape) == (num_tokens, 1, CONCAT_ROPE_DIM)
            and tuple(int(s) for s in k.stride()) in CONCAT_OUTPUT_STRIDES
            and (
                tuple(int(s) for s in k_nope.stride()),
                tuple(int(s) for s in k_rope.stride()),
            )
            in CONCAT_INPUT_STRIDE_PROFILES
        )
    except Exception:
        return False


def concat_mla_k(k: torch.Tensor, k_nope: torch.Tensor, k_rope: torch.Tensor) -> None:
    """Forward to ``flashinfer.concat_ops.concat_mla_k(..., backend="cake")``.

    Writes ``k[..., :128] = k_nope`` and ``k[..., 128:] = k_rope`` in place.
    """
    from flashinfer.concat_ops import concat_mla_k

    concat_mla_k(k, k_nope, k_rope, backend="cake")
