"""Runtime-view adapter for the isolated SM80 V4.1 sparse MLA component.

This module performs no backend dispatch and does not imply end-to-end model
support. Cache views are reinterpreted as bytes without copying their storage.
"""

import math
import numbers

import torch

from sglang.srt.environ import envs

from .kv_layout import KVLayout
from .sparse_attention_sm80 import sparse_attention_sm80


def _validate_tensor(name, tensor, shape, dtypes, device):
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.layout != torch.strided:
        raise ValueError(f"{name} must have strided layout")
    if tuple(tensor.shape) != tuple(shape):
        raise ValueError(
            f"{name} must have shape {tuple(shape)}, got {tuple(tensor.shape)}"
        )
    if tensor.dtype not in dtypes:
        raise TypeError(f"{name} must have dtype in {dtypes}, got {tensor.dtype}")
    if tensor.device != device:
        raise ValueError(f"{name} must be on {device}, got {tensor.device}")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def _indices_view(name, indices, tokens, device):
    if not isinstance(indices, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if indices.ndim != 3 or indices.shape[:2] != (tokens, 1):
        raise ValueError(f"{name} must have shape [T, 1, K] with T={tokens}")
    _validate_tensor(
        name, indices, (tokens, 1, indices.shape[2]),
        (torch.int32, torch.int64), device,
    )
    return indices.squeeze(1)


def _length_view(name, length, tokens, device):
    if length is None:
        return None
    if not isinstance(length, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tuple(length.shape) not in ((tokens,), (tokens, 1)):
        raise ValueError(f"{name} must have shape [T] or [T, 1] with T={tokens}")
    _validate_tensor(name, length, length.shape, (torch.int32, torch.int64), device)
    return length.squeeze(1) if length.ndim == 2 else length


def _raw_cache_view(name, cache, device):
    if not isinstance(cache, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if cache.layout != torch.strided:
        raise ValueError(f"{name} must have strided layout")
    if cache.ndim != 4 or cache.shape[2] != 1:
        raise ValueError(f"{name} must have shape [pages, P, 1, BPT]")
    if cache.dtype not in (torch.uint8, torch.float8_e4m3fn):
        raise TypeError(f"{name} must use a uint8 or float8_e4m3fn byte carrier")
    if cache.device != device:
        raise ValueError(f"{name} must be on {device}, got {cache.device}")
    pages, page_size, _, bytes_per_token = cache.shape
    if page_size <= 0:
        raise ValueError(f"{name} page size P must be positive")
    if bytes_per_token == KVLayout.V41.bytes_per_token:
        layout = KVLayout.V41
    elif bytes_per_token == KVLayout.V41_FP4.bytes_per_token:
        layout = KVLayout.V41_FP4
    else:
        raise ValueError(f"{name} BPT must be 528 (V41) or 288 (V41_FP4)")

    # Singleton dimensions have no observable stride; otherwise these are the
    # inner strides of raw[:, :P*BPT].view(pages, P, 1, BPT).
    for dim, expected in ((1, bytes_per_token), (2, bytes_per_token), (3, 1)):
        if cache.shape[dim] != 1 and cache.stride(dim) != expected:
            raise ValueError(
                f"{name} stride({dim}) must be {expected} for a non-singleton dim"
            )
    # Both accepted carriers have element_size == 1, so offsets/strides are bytes.
    pitch = cache.stride(0)
    if pitch < page_size * bytes_per_token:
        raise ValueError(f"{name} page byte pitch cannot hold P * BPT bytes")
    offset = cache.storage_offset()
    storage_bytes = cache.untyped_storage().nbytes()
    # A narrowed view can have all its visible bytes in bounds yet lack the last
    # page's tail padding. The full rectangular raw view must also be in bounds.
    if offset < 0 or offset + pages * pitch > storage_bytes:
        raise ValueError(
            f"{name} storage must contain pages * byte_pitch bytes starting at "
            "its storage_offset, including the final page padding"
        )
    raw = torch.as_strided(
        cache.view(torch.uint8),
        size=(pages, pitch),
        stride=(pitch, 1),
        storage_offset=offset,
    )
    return raw, page_size, layout


def _use_tiled_decode(tokens: int, multirow_max) -> bool:
    """Whether this token count may use the tiled split-K component.

    Single-row decode has always used it. Batched decode (2..8 rows) is the case
    the old `tokens == 1` gate dropped onto the serial kernel, so it is enabled by
    an explicit positive threshold. The threshold is an exact opt-in: unset,
    non-numeric and non-positive values keep every multi-row count -- including
    mixed-cache prefill chunks -- on the serial component, which is what the
    original gate was protecting.
    """
    try:
        limit = int(multirow_max or 0)
    except (TypeError, ValueError):
        limit = 0
    return tokens == 1 or 0 < tokens <= limit


def flash_mla_with_kvcache_sm80(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    head_dim_v: int,
    softmax_scale: float,
    indices: torch.Tensor,
    topk_length: torch.Tensor | None = None,
    attn_sink: torch.Tensor | None = None,
    extra_k_cache: torch.Tensor | None = None,
    extra_indices_in_kvcache: torch.Tensor | None = None,
    extra_topk_length: torch.Tensor | None = None,
    *,
    out: torch.Tensor | None = None,
    lse: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Adapt backend views to SM80 sparse attention, returning (out, unsunk_lse).

    q is contiguous BF16[T,1,H,512]; only sq=1 and head_dim_v=512 are supported.
    indices is contiguous int32/int64[T,1,K] of each pool's physical TOKEN slots,
    not page IDs. Optional contiguous int32/int64 lengths [T] or [T,1] select a
    per-query prefix: negative lengths mask everything; lengths > K use all K.
    Negative indices are masked and repeated indices count independently.

    Each cache is [pages,P,1,BPT] with BPT=528 (V41) or 288 (V41_FP4). The uint8
    or float8_e4m3fn dtype is only a one-byte storage carrier, NOT the dtype of
    every byte in the page (data and scales occupy separate regions). Page
    stride may include padding; inner strides must be (BPT,BPT,1), ignoring
    singleton strides. Nonzero storage offsets are supported. Storage must hold
    pages * byte_pitch bytes from that offset, including final-page padding.
    Main/extra pools may independently use different formats, P and byte pitches.
    V4 (584 bytes/token) and uniform layouts are not supported.

    The CALLER supplies causal/window-filtered candidates and ensures every
    active nonnegative ID is < its pool's pages * P. q, decoded BF16 KV, FP32 dot
    products, scaled scores and accumulators must be finite. Optional contiguous
    FP32 attn_sink[H] must be finite or -inf, never NaN/+inf. Values are not
    checked on the host; validation uses metadata only, without CPU copies or
    synchronization. softmax_scale must be a finite Python real scalar.

    out is contiguous BF16[T,1,H,512]; unsunk_lse is contiguous FP32[T,H,1], the
    NATURAL-LOG logsumexp of valid real keys, excluding the sink. The sink only
    contributes a zero-value entry to the output denominator. All-masked rows
    give zero output and lse=-inf. Empty T is allowed. All tensors must share
    one NVIDIA CUDA SM80 device (not SM86); the component checks architecture.

    Supplied buffers avoid output allocation; single-row split-K decode still
    allocates a graph-capturable partial workspace. Warm up compilation and
    device metadata first. Supplied out/lse
    are returned as the exact original objects. Inputs and outputs must not
    alias each other or overlap: the caller guarantees this. No cache copying,
    query/head padding, backend routing or end-to-end support is provided here.
    """
    if not isinstance(q, torch.Tensor):
        raise TypeError("q must be a torch.Tensor")
    if q.ndim != 4 or q.shape[1] != 1 or q.shape[3] != 512:
        raise ValueError("q must have shape [T, 1, H, 512] (sq=1 only)")
    if type(head_dim_v) is not int or head_dim_v != 512:
        raise ValueError("head_dim_v must be the Python int 512")
    if not q.is_cuda or torch.version.hip is not None:
        raise ValueError("flash_mla_with_kvcache_sm80 requires NVIDIA CUDA SM80")
    device = q.device
    tokens, _, heads, _ = q.shape
    _validate_tensor("q", q, (tokens, 1, heads, 512), (torch.bfloat16,), device)
    if (
        isinstance(softmax_scale, bool)
        or not isinstance(softmax_scale, numbers.Real)
        or not math.isfinite(softmax_scale)
    ):
        raise ValueError("softmax_scale must be a finite Python real scalar")

    raw_cache, page_size, layout = _raw_cache_view("k_cache", k_cache, device)
    ids = _indices_view("indices", indices, tokens, device)
    length = _length_view("topk_length", topk_length, tokens, device)
    if attn_sink is not None:
        _validate_tensor("attn_sink", attn_sink, (heads,), (torch.float32,), device)

    raw_extra = extra_ids = extra_length = extra_page_size = extra_layout = None
    if extra_k_cache is not None:
        if extra_indices_in_kvcache is None:
            raise ValueError("extra_k_cache requires extra_indices_in_kvcache")
        raw_extra, extra_page_size, extra_layout = _raw_cache_view(
            "extra_k_cache", extra_k_cache, device
        )
        extra_ids = _indices_view(
            "extra_indices_in_kvcache", extra_indices_in_kvcache, tokens, device
        )
        extra_length = _length_view(
            "extra_topk_length", extra_topk_length, tokens, device
        )
    elif extra_indices_in_kvcache is not None or extra_topk_length is not None:
        raise ValueError("extra indices/length require extra_k_cache")

    if out is not None:
        _validate_tensor(
            "out", out, (tokens, 1, heads, 512), (torch.bfloat16,), device
        )
    if lse is not None:
        _validate_tensor("lse", lse, (tokens, heads, 1), (torch.float32,), device)

    # A100 measurements favor split-K for one-row decode, but mixed-cache
    # prefill is faster with the serial component. Never promote tiled prefill
    # merely because the decode benchmark improves.
    attention = sparse_attention_sm80
    # Batched decode belongs on the tiled path too; see _use_tiled_decode for why
    # the threshold stays an exact opt-in and never promotes prefill.
    if _use_tiled_decode(
        tokens, envs.SGLANG_DSV41_SM80_TILED_MULTIROW_MAX_TOKENS.get()
    ):
        from functools import partial

        from .sparse_attention_sm80_tiled import sparse_attention_sm80_tiled

        # Exact FP32 split-K arithmetic; smaller key tiles avoid SM80 register
        # pressure. These are static shape choices, never live GPU-length reads.
        attention = partial(
            sparse_attention_sm80_tiled,
            block=8 if ids.shape[1] <= 128 and raw_extra is not None else 16,
            warps=4,
        )
    component_out, component_lse = attention(
        q.squeeze(1), raw_cache, ids, page_size, layout, softmax_scale,
        topk_length=length,
        attn_sink=attn_sink,
        extra_cache=raw_extra,
        extra_indices=extra_ids,
        extra_page_size=extra_page_size,
        extra_layout=extra_layout,
        extra_topk_length=extra_length,
        out=out.squeeze(1) if out is not None else None,
        lse=lse.squeeze(-1) if lse is not None else None,
    )
    return (
        out if out is not None else component_out.unsqueeze(1),
        lse if lse is not None else component_lse.unsqueeze(-1),
    )
