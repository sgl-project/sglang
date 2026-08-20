# MSA drop-in for the MiniMax-M3 main sparse-attention step.
#
# Replaces only step 3 of MiniMax sparse prefill/decode. The lightning indexer
# (steps 1-2) is unchanged and still produces `topk_idx`.
# NVIDIA Blackwell (SM100/sm_103) only; callers gate on `msa_available()`.
# Prefer FlashInfer's public source-kernel API.  The standalone fmha_sm100
# package remains a compatibility fallback for installations that have not yet
# upgraded FlashInfer and for its all-FP8 Q/K/V mode.
#
# Dtypes: bf16 end-to-end, or uniform fp8_e4m3fn Q/K/V under fp8 attn-GEMM mode
# (output bf16). fmha_sm100 selects its kernel variant from q.dtype alone and
# casts k/v pointers to the same element type, so mixed bf16-q/fp8-KV is NOT
# possible on the cutlass path and e5m2 would silently dispatch the e4m3
# kernel — `_check_msa_dtypes` enforces uniformity here. The fp8 kernel
# quantizes the unnormalized softmax P to e4m3 before the PV MMA (same
# contract as the Triton fp8 path).

from __future__ import annotations

import functools
import inspect
import os
from typing import Optional

import torch

from sglang.kernels.ops.attention.minimax_sparse.common.utils import unit_scale


class MSAUnavailableError(RuntimeError):
    """Raised when no installed MSA implementation can serve this call."""


_FLASHINFER_PREFILL_PARAMS = frozenset(
    {
        "q",
        "k",
        "v",
        "q2k_indices",
        "cu_seqlens_q",
        "page_table",
        "seqused_k",
        "q_offset",
        "workspace",
    }
)
_FLASHINFER_DECODE_PARAMS = frozenset(
    {
        "q",
        "k",
        "v",
        "q2k_indices",
        "page_table",
        "seqused_k",
        "seqlen_q",
        "workspace",
    }
)


@functools.lru_cache(maxsize=1)
def _load_flashinfer_msa():
    """Load and signature-check FlashInfer's stable public MSA surface."""

    try:
        from flashinfer import msa_ops
    except Exception as err:
        raise MSAUnavailableError("flashinfer.msa_ops is not importable") from err

    prefill = getattr(msa_ops, "msa_sparse_attention", None)
    decode = getattr(msa_ops, "msa_sparse_decode_attention", None)
    topk = getattr(msa_ops, "msa_topk_select", None)
    workspace_type = getattr(msa_ops, "MSASparseAttentionWorkspace", None)
    if not all(callable(op) for op in (prefill, decode, topk)):
        raise MSAUnavailableError(
            "flashinfer.msa_ops must export callable msa_sparse_attention, "
            "msa_sparse_decode_attention, and msa_topk_select"
        )
    if workspace_type is None or not callable(workspace_type):
        raise MSAUnavailableError(
            "flashinfer.msa_ops.MSASparseAttentionWorkspace is required for "
            "SGLang CUDA-graph capture"
        )

    try:
        prefill_params = frozenset(inspect.signature(prefill).parameters)
        decode_params = frozenset(inspect.signature(decode).parameters)
    except (TypeError, ValueError) as err:
        raise MSAUnavailableError(
            "FlashInfer MSA call signatures are not inspectable"
        ) from err
    missing_prefill = _FLASHINFER_PREFILL_PARAMS - prefill_params
    missing_decode = _FLASHINFER_DECODE_PARAMS - decode_params
    if missing_prefill or missing_decode:
        raise MSAUnavailableError(
            "FlashInfer MSA API is missing required parameters: "
            f"prefill={sorted(missing_prefill)}, decode={sorted(missing_decode)}"
        )
    return prefill, decode, topk, workspace_type


class FlashInferMSAGraphState:
    """Stable inputs plus one public FlashInfer workspace for one graph call.

    FlashInfer intentionally binds a workspace to the exact tensor addresses
    warmed before capture.  Model Q and sparse indices are intermediates whose
    allocator addresses need not be stable across SGLang's warmup and capture
    forwards, so copy them into caller-owned buffers first.  The copies become
    graph nodes and replay with fresh values.
    """

    def __init__(self, device: torch.device | str):
        _, _, _, workspace_type = _load_flashinfer_msa()
        self.workspace = workspace_type(device)
        self._buffers: dict[str, torch.Tensor] = {}

    def stage(self, name: str, value: torch.Tensor) -> torch.Tensor:
        buffer = self._buffers.get(name)
        if (
            buffer is None
            or buffer.shape != value.shape
            or buffer.dtype != value.dtype
            or buffer.device != value.device
        ):
            if torch.cuda.is_current_stream_capturing():
                raise MSAUnavailableError(
                    f"FlashInfer MSA graph buffer {name!r} was not warmed before capture"
                )
            buffer = torch.empty_like(value, memory_format=torch.contiguous_format)
            self._buffers[name] = buffer
        buffer.copy_(value)
        return buffer


def make_flashinfer_msa_graph_state(
    device: torch.device | str,
) -> FlashInferMSAGraphState:
    return FlashInferMSAGraphState(device)


def _check_msa_dtypes(q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor):
    # Uniform dtype required in BOTH modes: fmha_sm100 keys its kernel variant
    # on q.dtype alone and casts the k/v pointers to the same element type, so
    # a mismatched cache would be silently reinterpreted.
    if q.dtype == torch.bfloat16:
        assert (
            k_cache.dtype == torch.bfloat16
        ), f"MSA bf16 requires a bf16 K cache, got {k_cache.dtype}"
    elif q.dtype == torch.float8_e4m3fn:
        # e5m2 is rejected here too: fmha_sm100's variant lookup falls back to
        # the e4m3 kernel for unknown dtype codes.
        assert (
            k_cache.dtype == torch.float8_e4m3fn
        ), f"MSA fp8 requires an fp8_e4m3fn K cache, got {k_cache.dtype}"
    else:
        raise AssertionError(f"MSA supports bf16 or fp8_e4m3fn Q, got {q.dtype}")
    assert v_cache.dtype == k_cache.dtype


@functools.lru_cache(maxsize=1)
def _load_fmha_sm100():
    try:
        from fmha_sm100 import fmha_sm100, fmha_sm100_plan
    except Exception as err:
        raise MSAUnavailableError(
            "fmha_sm100 or fmha_sm100_plan is not importable"
        ) from err
    if not callable(fmha_sm100) or not callable(fmha_sm100_plan):
        raise MSAUnavailableError("fmha_sm100 exports must be callable")
    return fmha_sm100, fmha_sm100_plan


def _run_fmha_sm100_plan(*args, **kwargs):
    _, fmha_sm100_plan = _load_fmha_sm100()
    try:
        return fmha_sm100_plan(*args, **kwargs)
    except (AttributeError, RuntimeError, TypeError) as err:
        raise MSAUnavailableError("fmha_sm100_plan failed") from err


def _msa_device_supported() -> bool:
    try:
        cap = torch.cuda.get_device_capability()
    except Exception:
        return False
    return cap[0] == 10 and cap[1] in (0, 3)


def _flashinfer_msa_device_supported() -> bool:
    if not _msa_device_supported():
        return False
    try:
        cap = torch.cuda.get_device_capability()
        major, minor = (int(part) for part in torch.version.cuda.split(".")[:2])
    except (AttributeError, TypeError, ValueError):
        return False
    minimum = (12, 9) if cap == (10, 3) else (12, 8)
    return (major, minor) >= minimum


@functools.lru_cache(maxsize=1)
def flashinfer_msa_available() -> bool:
    if not _flashinfer_msa_device_supported():
        return False
    try:
        _load_flashinfer_msa()
        return True
    except MSAUnavailableError:
        return False


@functools.lru_cache(maxsize=1)
def fmha_sm100_available() -> bool:
    if not _msa_device_supported():
        return False
    try:
        _load_fmha_sm100()
        return True
    except MSAUnavailableError:
        return False


def selected_msa_backend(
    kv_dtype: torch.dtype, num_kv_heads: int, topk: int
) -> Optional[str]:
    """Resolve ``flashinfer`` / ``fmha_sm100`` without probing private APIs.

    ``SGLANG_MINIMAX_MSA_BACKEND`` is an A/B and emergency control.  ``auto``
    prefers the source-distributed FlashInfer implementation.  FlashInfer's
    SM100 paged API currently requires an HND-contiguous view of SGLang's
    slot-major cache; that view is zero-copy when the TP-local KV-head count is
    one, and its MiniMax-M3 source route is fixed to top-k 16. Other layouts
    and top-k values retain the existing standalone implementation.
    """

    requested = os.environ.get("SGLANG_MINIMAX_MSA_BACKEND", "auto").lower()
    if requested not in ("auto", "flashinfer", "fmha_sm100"):
        raise ValueError(
            "SGLANG_MINIMAX_MSA_BACKEND must be auto, flashinfer, or fmha_sm100"
        )
    flashinfer_compatible = (
        kv_dtype in (torch.bfloat16, torch.float16) and num_kv_heads == 1 and topk == 16
    )
    if requested == "flashinfer":
        if not flashinfer_compatible:
            raise MSAUnavailableError(
                "FlashInfer MSA requires BF16/FP16 KV, one TP-local KV head, "
                "and top-k 16"
            )
        if not flashinfer_msa_available():
            raise MSAUnavailableError(
                "the requested FlashInfer MSA source API is unavailable"
            )
        return "flashinfer"
    if requested == "fmha_sm100":
        if not fmha_sm100_available():
            raise MSAUnavailableError(
                "the requested fmha_sm100 compatibility backend is unavailable"
            )
        return "fmha_sm100"
    if flashinfer_compatible and flashinfer_msa_available():
        return "flashinfer"
    if fmha_sm100_available():
        return "fmha_sm100"
    return None


def msa_runtime_fallback_allowed() -> bool:
    """Whether a provider failure may fall back to the Triton implementation.

    ``auto`` is the production availability policy and may recover from a
    provider-specific runtime rejection.  An explicitly selected provider is
    also an A/B-validation contract: silently falling back there would label a
    Triton run as FlashInfer/fmha_sm100 and invalidate correctness and speed
    evidence, so explicit selections fail closed.
    """

    return os.environ.get("SGLANG_MINIMAX_MSA_BACKEND", "auto").lower() == "auto"


@functools.lru_cache(maxsize=1)
def msa_available() -> bool:
    """True iff either public FlashInfer MSA or the compatibility backend loads."""

    return flashinfer_msa_available() or fmha_sm100_available()


def build_flashinfer_page_table(
    req_to_token: torch.Tensor,
    slot_ids: torch.Tensor,
    seq_lens: torch.Tensor,
    page_size: int,
    *,
    max_pages: Optional[int] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Build SGLang's logical-to-physical page table in FlashInfer's 2D form.

    When ``out`` is supplied its address and shape remain stable across CUDA
    graph replay; all contents are refreshed without a host synchronization.
    """

    batch_size = int(slot_ids.shape[0])
    if out is not None:
        if out.shape[0] != batch_size or out.dtype != torch.int32:
            raise ValueError("persistent MSA page table has incompatible shape/dtype")
        max_pages = int(out.shape[1])
    elif max_pages is None:
        max_seq_len = int(seq_lens.max().item()) if batch_size else 0
        max_pages = (max_seq_len + page_size - 1) // page_size
    assert max_pages is not None
    if out is None:
        out = torch.empty(
            (batch_size, max_pages), dtype=torch.int32, device=req_to_token.device
        )
    if max_pages == 0:
        return out
    logical_first = torch.arange(
        max_pages, dtype=torch.int64, device=req_to_token.device
    ).mul_(page_size)
    if (max_pages - 1) * page_size >= req_to_token.shape[1]:
        raise ValueError(
            f"MSA page table width {max_pages} exceeds request table capacity"
        )
    rows = slot_ids.to(torch.int64).unsqueeze(1)
    out.copy_(
        (req_to_token[rows, logical_first.unsqueeze(0)] // page_size).to(torch.int32)
    )
    return out


def _build_page_table(
    req_to_token: torch.Tensor,  # [max_reqs, max_kv_len], physical slot per logical pos
    slot_ids: torch.Tensor,  # [batch]
    seq_lens: torch.Tensor,  # [batch] total K length (prefix + chunk)
    page_size: int,
) -> torch.Tensor:
    """Flattened physical page ids per request (MSA `kv_indices`).

    sglang's paged allocator stores page_size contiguous physical slots per page, so the
    physical page of logical position p is ``req_to_token[req, p] // page_size`` and is the
    same for every p within a page. We read one slot per logical page to recover the table.

    Vectorized (no per-request Python loop): pages are packed contiguously by request in the
    same order MSA's planner expects (``kv_page_indptr = cumsum(ceil(seq_lens/page_size))``).
    ``searchsorted`` maps each packed page slot back to its request; one ``.item()`` recovers
    the total page count (this runs eagerly, outside CUDA-graph capture).
    """
    P = page_size
    n_pages = (seq_lens.to(torch.int64) + (P - 1)) // P  # [batch]
    offsets = (
        torch.cumsum(n_pages, 0) - n_pages
    )  # [batch] exclusive page offset per request
    total = int(n_pages.sum().item())
    idx = torch.arange(
        total, device=req_to_token.device
    )  # packed page slot -> (req, page)
    req = torch.searchsorted(
        offsets + n_pages, idx, right=True
    )  # request id per packed slot
    logical_first = (idx - offsets[req]) * P  # first logical position of that page
    rows = slot_ids[req].to(torch.int64)
    return (req_to_token[rows, logical_first] // P).to(torch.int32)


def _flashinfer_paged_kv(
    k_cache: torch.Tensor, v_cache: torch.Tensor, page_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    max_slots, num_kv_heads, head_dim = k_cache.shape
    if num_kv_heads != 1:
        raise MSAUnavailableError(
            "FlashInfer MSA serving integration requires one TP-local KV head"
        )
    if max_slots % page_size:
        raise MSAUnavailableError(
            f"max_slots={max_slots} is not divisible by page_size={page_size}"
        )
    num_pages = max_slots // page_size
    k_paged = k_cache.view(num_pages, page_size, num_kv_heads, head_dim).permute(
        0, 2, 1, 3
    )
    v_paged = v_cache.view(num_pages, page_size, num_kv_heads, head_dim).permute(
        0, 2, 1, 3
    )
    # Do not hide a whole-cache transpose behind this serving call.  MiniMax-M3
    # TP4 has one local KV head, for which this HND view is already contiguous.
    if not k_paged.is_contiguous() or not v_paged.is_contiguous():
        raise MSAUnavailableError(
            "FlashInfer MSA requires zero-copy contiguous HND paged K/V; "
            "use TP with one local KV head or the fmha_sm100 compatibility backend"
        )
    return k_paged, v_paged


def _flashinfer_prefill(
    *,
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    topk_idx: torch.Tensor,
    req_to_token: torch.Tensor,
    slot_ids: torch.Tensor,
    cu_seqlens: torch.Tensor,
    seq_lens: torch.Tensor,
    prefix_lens: torch.Tensor,
    block_size_k: int,
    sm_scale: Optional[float],
    page_table: Optional[torch.Tensor],
    graph_state: Optional[FlashInferMSAGraphState],
) -> torch.Tensor:
    prefill, _, _, _ = _load_flashinfer_msa()
    if topk_idx.shape[-1] != 16:
        raise MSAUnavailableError("FlashInfer MSA source kernels require top-k 16")
    if q.dtype not in (torch.bfloat16, torch.float16):
        raise MSAUnavailableError("FlashInfer source MSA requires BF16/FP16 Q")
    if k_cache.dtype != q.dtype or v_cache.dtype != q.dtype:
        raise MSAUnavailableError(
            "FlashInfer source MSA integration currently requires K/V dtype to match Q"
        )
    k_paged, v_paged = _flashinfer_paged_kv(k_cache, v_cache, block_size_k)
    if page_table is None:
        page_table = build_flashinfer_page_table(
            req_to_token, slot_ids, seq_lens, block_size_k
        )
    q = q.contiguous()
    q2k = topk_idx.contiguous().to(torch.int32)
    cu_q = cu_seqlens.to(torch.int32)
    kv_lens = seq_lens.to(torch.int32)
    q_offsets = prefix_lens.to(torch.int32)
    workspace = None
    if graph_state is not None:
        q = graph_state.stage("q", q)
        q2k = graph_state.stage("q2k_indices", q2k)
        cu_q = graph_state.stage("cu_seqlens_q", cu_q)
        kv_lens = graph_state.stage("seqused_k", kv_lens)
        q_offsets = graph_state.stage("q_offset", q_offsets)
        workspace = graph_state.workspace
    try:
        return prefill(
            q=q,
            k=k_paged,
            v=v_paged,
            q2k_indices=q2k,
            cu_seqlens_q=cu_q,
            causal=True,
            softmax_scale=sm_scale,
            page_table=page_table,
            seqused_k=kv_lens,
            q_offset=q_offsets,
            workspace=workspace,
        )
    except (AttributeError, TypeError, ValueError) as err:
        raise MSAUnavailableError("FlashInfer MSA prefill rejected the call") from err


def msa_sparse_prefill_main(
    q: torch.Tensor,  # [total_q, num_q_heads, head_dim]
    k_cache: torch.Tensor,  # [max_slots, num_kv_heads, head_dim] (slot-major NHD)
    v_cache: torch.Tensor,  # [max_slots, num_kv_heads, head_dim]
    topk_idx: torch.Tensor,  # [num_kv_heads, total_q, topk] (0-based, -1 pad) -- step1/2 output
    req_to_token: torch.Tensor,  # [max_reqs, max_kv_len]
    slot_ids: torch.Tensor,  # [batch]
    cu_seqlens: torch.Tensor,  # [batch+1] cumulative Q lengths
    seq_lens: torch.Tensor,  # [batch] total K length (prefix + chunk)
    prefix_lens: torch.Tensor,  # [batch]
    block_size_k: int,  # == page_size == 128 for M3
    sm_scale: Optional[float] = None,
    q_scale: Optional[float] = None,
    k_scale: Optional[float] = None,
    v_scale: Optional[float] = None,
    page_table: Optional[torch.Tensor] = None,
    graph_state: Optional[FlashInferMSAGraphState] = None,
    backend: Optional[str] = None,
) -> torch.Tensor:
    """Drop-in for flash_prefill_with_gqa_share_sparse using MSA fmha_sm100.

    Returns o [total_q, num_q_heads, head_dim] (bf16 for fp8 inputs).

    Scale semantics (per-tensor, None = unit): attention runs on Q*q_scale,
    K*k_scale, V*v_scale. The long-q cute path honors only sm_scale, so
    q_scale*k_scale is folded into sm_scale (exact for softmax) and v_scale is
    applied on the output; the short-q cutlass path gets the same folded values.
    """
    if backend is None:
        backend = selected_msa_backend(
            k_cache.dtype, int(k_cache.shape[1]), int(topk_idx.shape[-1])
        )
    if backend == "flashinfer":
        if any(unit_scale(scale) != 1.0 for scale in (q_scale, k_scale, v_scale)):
            raise MSAUnavailableError(
                "FlashInfer source MSA scale support is not part of the frozen SM100 contract"
            )
        return _flashinfer_prefill(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            topk_idx=topk_idx,
            req_to_token=req_to_token,
            slot_ids=slot_ids,
            cu_seqlens=cu_seqlens,
            seq_lens=seq_lens,
            prefix_lens=prefix_lens,
            block_size_k=block_size_k,
            sm_scale=sm_scale,
            page_table=page_table,
            graph_state=graph_state,
        )
    if backend != "fmha_sm100":
        raise MSAUnavailableError("no MSA prefill backend supports this input")

    fmha_sm100, _ = _load_fmha_sm100()
    _check_msa_dtypes(q, k_cache, v_cache)
    is_fp8 = q.dtype == torch.float8_e4m3fn
    v_scale = unit_scale(v_scale)

    max_slots, num_kv_heads, head_dim = k_cache.shape
    num_q_heads = q.shape[1]
    P = block_size_k
    topk = topk_idx.shape[-1]
    if max_slots % P != 0:
        raise ValueError(f"max_slots={max_slots} not divisible by page_size={P}")
    if sm_scale is None:
        sm_scale = head_dim**-0.5
    sm_scale = sm_scale * unit_scale(q_scale) * unit_scale(k_scale)

    # Whole pool as MSA paged KV: [num_phys_pages, num_kv_heads, P, head_dim].
    n_phys_pages = max_slots // P
    k_paged = k_cache.view(n_phys_pages, P, num_kv_heads, head_dim).permute(0, 2, 1, 3)
    v_paged = v_cache.view(n_phys_pages, P, num_kv_heads, head_dim).permute(0, 2, 1, 3)

    # Per-request Q lengths (extend) and physical page table.
    qo_segment_lens = (cu_seqlens[1:] - cu_seqlens[:-1]).to(torch.int32)
    kv_indices = _build_page_table(req_to_token, slot_ids, seq_lens, P)

    # topk_idx [Hkv, total_q, topk] -> kv_block_indexes [total_q, Hkv, topk].
    kv_block_indexes = topk_idx.permute(1, 0, 2).contiguous().to(torch.int32)

    plan = _run_fmha_sm100_plan(
        qo_segment_lens,
        seq_lens.to(torch.int32),
        num_q_heads,
        num_kv_heads=num_kv_heads,
        page_size=P,
        kv_block_num=topk,
        causal=True,
        qo_offset=prefix_lens.to(torch.int32),
        use_fp8_kvcache=is_fp8,
    )
    o, _ = fmha_sm100(
        q,
        k_paged,
        v_paged,
        plan,
        sm_scale=sm_scale,
        kv_indices=kv_indices,
        kv_block_indexes=kv_block_indexes,
    )
    # The cute (long-q) sparse prefill backend honors sm_scale only; apply the
    # V dequant scale on the output (exact: softmax normalization excludes V).
    if v_scale != 1.0:
        o = o * v_scale
    return o


def build_msa_decode_meta(
    k_cache: torch.Tensor,  # [max_slots, num_kv_heads, head_dim]
    req_to_token: torch.Tensor,
    slot_ids: torch.Tensor,  # [batch]
    seq_lens: torch.Tensor,  # [batch] cached KV length per request
    num_q_heads: int,
    block_size_k: int,
    topk: int,
    is_fp8: bool = False,
):
    """Per-forward MSA decode metadata (page table + fmha plan), shared across layers.

    Within one decode forward every sparse layer has the same batch, seq_lens, page size
    and topk, so the physical page table and the fmha_sm100 plan are identical for all of
    them — only ``kv_block_indexes`` (the per-layer top-k selection) changes. Building these
    once per forward instead of once per layer removes the dominant host-side overhead
    (page-table build + the host-side ``fmha_sm100_plan``) from the 57-layer decode loop.

    Used only by the standalone parity harnesses; the serving backend builds eager-decode
    metadata via ``build_msa_decode_cg_plan`` + ``update_msa_decode_cg_meta`` instead.
    """
    max_slots, num_kv_heads, _ = k_cache.shape
    P = block_size_k
    if max_slots % P != 0:
        raise ValueError(f"max_slots={max_slots} not divisible by page_size={P}")
    B = slot_ids.shape[0]
    kv_indices = _build_page_table(req_to_token, slot_ids, seq_lens, P)
    seq_lens_i32 = seq_lens.to(torch.int32)
    plan = _run_fmha_sm100_plan(
        torch.ones(B, dtype=torch.int32),
        seq_lens_i32,
        num_q_heads,
        num_kv_heads=num_kv_heads,
        page_size=P,
        kv_block_num=topk,
        causal=False,
        qo_offset=seq_lens_i32 - 1,  # decode query sits at the last cached position
        use_fp8_kvcache=is_fp8,
    )
    return kv_indices, plan


# ---------------------------------------------------------------------------
# MSA decode plan (persistent per batch size; used by eager decode AND under
# CUDA graph)
#
# History: MSA decode under CUDA graph was disabled after silently wrong
# results (~14% GSM8K on B200). Root cause (2026-07): the topk producers
# emitted block ids in score order, violating fmha_sm100's strictly-ascending
# kv_block_indexes contract — its sorted-order early-exit then mis-masked the
# partial last block for any row with seq_len > topk*block_size. The producers
# now sort ascending (minimax_decode_topk.cuh, _topk_index_merge_kernel,
# prefill _topk_index_kernel), and capture/replay of the full pipeline is
# bit-exact vs eager (see tests/repro_msa_decode_degenerate.py).
#
# The build-once / update-in-place structure below refreshes the four length
# tensors ``{kv_segment_lens, kv_segment_offsets, kv_page_indptr, qo_offset}``
# and the page table each forward; the captured graph reads the same tensor
# addresses on replay.
# ---------------------------------------------------------------------------

_MSA_CG_LEN_KEYS = (
    "kv_segment_lens",
    "kv_segment_offsets",
    "kv_page_indptr",
    "qo_offset",
)


def _check_cg_plan_layout(plan) -> None:
    """Fail fast if fmha_sm100's plan layout drifted from what replay-update
    assumes (dict at tuple index 3 holding the four length tensors) — these are
    undocumented fmha_sm100 internals."""
    if not (isinstance(plan, tuple) and len(plan) > 3 and isinstance(plan[3], dict)):
        raise MSAUnavailableError(
            "fmha_sm100_plan no longer returns a tuple with a metadata dict at index 3; "
            "the MSA CUDA-graph decode path must be revalidated against this fmha_sm100 "
            "version. Set SGLANG_DISABLE_MSA=1 to use the Triton path meanwhile."
        )
    missing = [k for k in _MSA_CG_LEN_KEYS if not torch.is_tensor(plan[3].get(k))]
    if missing:
        raise MSAUnavailableError(
            f"fmha_sm100 plan is missing length tensors {missing}; the MSA CUDA-graph "
            "decode path must be revalidated against this fmha_sm100 version. "
            "Set SGLANG_DISABLE_MSA=1 to use the Triton path meanwhile."
        )


def build_msa_decode_cg_plan(
    num_q_heads: int,
    num_kv_heads: int,
    block_size_k: int,
    topk: int,
    batch_size: int,
    device: Optional[torch.device] = None,
    is_fp8: bool = False,
):
    """Persistent fmha_sm100 decode plan for one batch size (CUDA-graph stable).

    Built once per captured batch size. The worklist is length-independent (it uses
    ``topk * page_size`` internally), so the reference KV length here only has to make
    every topk block valid; the length-dependent tensors are overwritten each step by
    ``update_msa_decode_cg_meta``. Returns the plan tuple to pass to ``fmha_sm100``.
    """
    P = block_size_k
    ref_len = topk * P  # length at which all topk blocks exist -> full worklist
    qo = torch.ones(batch_size, dtype=torch.int32)
    kv = torch.full((batch_size,), ref_len, dtype=torch.int32)
    plan = _run_fmha_sm100_plan(
        qo,
        kv,
        num_q_heads,
        num_kv_heads=num_kv_heads,
        page_size=P,
        kv_block_num=topk,
        causal=False,
        qo_offset=kv - 1,
        device=device,
        use_fp8_kvcache=is_fp8,
    )
    _check_cg_plan_layout(plan)
    return plan


def update_msa_decode_cg_meta(
    plan,
    kv_indices_buf: torch.Tensor,  # persistent page-table buffer [batch * max_pages]
    req_to_token: torch.Tensor,
    slot_ids: torch.Tensor,  # [batch]
    seq_lens: torch.Tensor,  # [batch] cached KV length per request
    block_size_k: int,
    topk: int,
    num_q_heads: int,
    num_kv_heads: int,
):
    """Refresh the persistent decode plan's length-dependent tensors + page table
    IN PLACE, entirely with device-side ops (no host<->device sync).

    Runs once per decode forward from ``init_forward_metadata_out_graph``; the
    captured graph then reads the same plan-tensor and ``kv_indices_buf``
    addresses on replay. A device sync here stalls the overlap scheduler, so the
    previous implementation — a throwaway ``fmha_sm100_plan`` build per step
    (``.tolist()``/``.item()`` syncs + a plan-kernel launch) just to copy four
    length tensors — is replaced by computing their contents directly, matching
    ``_fmha_sm100_plan``'s sparse-decode (qo_len==1, causal=False) layout:

      kv_segment_lens    = seq_lens
      kv_segment_offsets = [0, cumsum(seq_lens)]
      kv_page_indptr     = [0, cumsum(ceil(seq_lens / P))]
      qo_offset          = broadcast max(seq_lens)   (causal=False planner quirk)

    The worklist tensors stay untouched: the plan schedules from the constant
    ``topk * P`` per request, never the real lengths (see build_msa_decode_cg_plan).
    """
    P = block_size_k
    B = seq_lens.shape[0]
    if B == 0:  # idle batch: serving guards this, but keep the helper total
        return
    pd = plan[3]
    seq_lens_i32 = seq_lens.to(torch.int32)
    pd["kv_segment_lens"].copy_(seq_lens_i32)
    kv_off = pd["kv_segment_offsets"]
    kv_off[0].zero_()
    torch.cumsum(seq_lens_i32, 0, out=kv_off[1:])
    n_pages = torch.div(seq_lens_i32 + (P - 1), P, rounding_mode="floor")
    indptr = pd["kv_page_indptr"]
    indptr[0].zero_()
    torch.cumsum(n_pages, 0, out=indptr[1:])
    pd["qo_offset"].copy_(seq_lens_i32.max().expand(B))

    # Page table, sync-free: fill the WHOLE persistent buffer (fixed size, no
    # host-side total-page count). Packed slot -> (request, logical page) via
    # searchsorted; slots beyond the live page count land on clamped reads and
    # are never dereferenced (the kernel walks kv_page_indptr ranges only).
    n = kv_indices_buf.numel()
    idx = torch.arange(n, device=kv_indices_buf.device)
    ends = torch.cumsum(n_pages.to(torch.int64), 0)
    req = torch.searchsorted(ends, idx, right=True).clamp_max_(B - 1)
    starts = ends - n_pages
    logical_first = ((idx - starts[req]) * P).clamp_(0, req_to_token.shape[1] - 1)
    rows = slot_ids[req].to(torch.int64)
    kv_indices_buf.copy_((req_to_token[rows, logical_first] // P).to(torch.int32))


def _flashinfer_decode(
    *,
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    topk_idx: torch.Tensor,
    req_to_token: torch.Tensor,
    slot_ids: torch.Tensor,
    seq_lens: torch.Tensor,
    block_size_k: int,
    sm_scale: Optional[float],
    page_table: Optional[torch.Tensor],
    graph_state: Optional[FlashInferMSAGraphState],
) -> torch.Tensor:
    _, decode, _, _ = _load_flashinfer_msa()
    if topk_idx.shape[-1] != 16:
        raise MSAUnavailableError("FlashInfer MSA source kernels require top-k 16")
    if q.dtype not in (torch.bfloat16, torch.float16):
        raise MSAUnavailableError("FlashInfer source MSA requires BF16/FP16 Q")
    if k_cache.dtype != q.dtype or v_cache.dtype != q.dtype:
        raise MSAUnavailableError(
            "FlashInfer source MSA integration currently requires K/V dtype to match Q"
        )
    k_paged, v_paged = _flashinfer_paged_kv(k_cache, v_cache, block_size_k)
    if page_table is None:
        page_table = build_flashinfer_page_table(
            req_to_token, slot_ids, seq_lens, block_size_k
        )
    q = q.contiguous()
    q2k = topk_idx.contiguous().to(torch.int32)
    kv_lens = seq_lens.to(torch.int32)
    workspace = None
    if graph_state is not None:
        q = graph_state.stage("q", q)
        q2k = graph_state.stage("q2k_indices", q2k)
        kv_lens = graph_state.stage("seqused_k", kv_lens)
        workspace = graph_state.workspace
    try:
        return decode(
            q=q,
            k=k_paged,
            v=v_paged,
            q2k_indices=q2k,
            page_table=page_table,
            seqused_k=kv_lens,
            seqlen_q=1,
            causal=True,
            softmax_scale=sm_scale,
            workspace=workspace,
        )
    except (AttributeError, TypeError, ValueError) as err:
        raise MSAUnavailableError("FlashInfer MSA decode rejected the call") from err


def msa_sparse_decode_main(
    q: torch.Tensor,  # [batch, num_q_heads, head_dim] (1 query token per request)
    k_cache: torch.Tensor,  # [max_slots, num_kv_heads, head_dim] (slot-major NHD)
    v_cache: torch.Tensor,  # [max_slots, num_kv_heads, head_dim]
    topk_idx: torch.Tensor,  # [num_kv_heads, batch, topk] (0-based, -1 pad)
    req_to_token: torch.Tensor,  # [max_reqs, max_kv_len]
    slot_ids: torch.Tensor,  # [batch]
    seq_lens: torch.Tensor,  # [batch] cached KV length per request
    block_size_k: int,  # == page_size == 128
    sm_scale: Optional[float] = None,
    kv_indices: Optional[
        torch.Tensor
    ] = None,  # precomputed page table (per-forward cache)
    plan=None,  # precomputed fmha_sm100 plan (per-forward cache)
    q_scale: Optional[float] = None,
    k_scale: Optional[float] = None,
    v_scale: Optional[float] = None,
    page_table: Optional[torch.Tensor] = None,
    graph_state: Optional[FlashInferMSAGraphState] = None,
    backend: Optional[str] = None,
) -> torch.Tensor:
    """Drop-in for flash_decode_with_gqa_share_sparse using MSA fmha_sm100.

    Each request is one decode query at absolute position seq_len-1 attending to its
    cached KV through the topk selected 128-blocks. Returns o [batch, num_q_heads,
    head_dim] (bf16 for fp8 inputs).

    ``kv_indices`` / ``plan`` are shared across all sparse layers of a forward; the serving
    backend builds them once via ``build_msa_decode_cg_plan`` + ``update_msa_decode_cg_meta``
    (eager decode only) and passes them in. When omitted (only the standalone parity
    harnesses) they are built here via ``build_msa_decode_meta``.

    Scales (None = unit) are passed natively: the cutlass decode path folds
    q_scale*k_scale into the softmax scale and applies v_scale on the output
    in-kernel.
    """
    if backend is None:
        backend = selected_msa_backend(
            k_cache.dtype, int(k_cache.shape[1]), int(topk_idx.shape[-1])
        )
    if backend == "flashinfer":
        if any(unit_scale(scale) != 1.0 for scale in (q_scale, k_scale, v_scale)):
            raise MSAUnavailableError(
                "FlashInfer source MSA scale support is not part of the frozen SM100 contract"
            )
        return _flashinfer_decode(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            topk_idx=topk_idx,
            req_to_token=req_to_token,
            slot_ids=slot_ids,
            seq_lens=seq_lens,
            block_size_k=block_size_k,
            sm_scale=sm_scale,
            page_table=page_table,
            graph_state=graph_state,
        )
    if backend != "fmha_sm100":
        raise MSAUnavailableError("no MSA decode backend supports this input")

    fmha_sm100, _ = _load_fmha_sm100()
    _check_msa_dtypes(q, k_cache, v_cache)

    max_slots, num_kv_heads, head_dim = k_cache.shape
    H = q.shape[1]
    P = block_size_k
    topk = topk_idx.shape[-1]
    if max_slots % P != 0:
        raise ValueError(f"max_slots={max_slots} not divisible by page_size={P}")
    if sm_scale is None:
        sm_scale = head_dim**-0.5

    n_phys_pages = max_slots // P
    k_paged = k_cache.view(n_phys_pages, P, num_kv_heads, head_dim).permute(0, 2, 1, 3)
    v_paged = v_cache.view(n_phys_pages, P, num_kv_heads, head_dim).permute(0, 2, 1, 3)

    if kv_indices is None or plan is None:
        kv_indices, plan = build_msa_decode_meta(
            k_cache,
            req_to_token,
            slot_ids,
            seq_lens,
            H,
            P,
            topk,
            is_fp8=q.dtype == torch.float8_e4m3fn,
        )
    kv_block_indexes = topk_idx.permute(1, 0, 2).contiguous().to(torch.int32)

    o, _ = fmha_sm100(
        q,
        k_paged,
        v_paged,
        plan,
        sm_scale=sm_scale,
        q_scale=unit_scale(q_scale),
        k_scale=unit_scale(k_scale),
        v_scale=unit_scale(v_scale),
        kv_indices=kv_indices,
        kv_block_indexes=kv_block_indexes,
    )
    return o
