"""BF16 KV-cache helpers for the DSV4 sparse-attention path.

The regular DSV4 cache stores a packed FP8-nope/BF16-rope representation.  The
BF16 path deliberately uses a separate, plain ``(page, token, 1, 512)`` view so
that the sparse FlashMLA kernel receives the same values that were produced by
the model, without an FP8 round trip.
"""

from __future__ import annotations

from typing import Optional

import torch

BF16_KV_HEAD_DIM = 512
SPARSE_DECODE_TOPK_ALIGNMENT = 128


def _is_cuda_graph_capturing() -> bool:
    return torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()


def _normalize_indices(indices: torch.Tensor) -> torch.Tensor:
    if indices.ndim == 3:
        assert indices.shape[1] == 1, (
            f"expected one query per decode row, got {indices.shape}"
        )
        indices = indices[:, 0]
    assert indices.ndim == 2, f"expected [batch, topk] indices, got {indices.shape}"
    return indices


def _as_flat_bf16_cache(cache: torch.Tensor, page_size: int) -> torch.Tensor:
    assert cache.dtype == torch.bfloat16, f"expected BF16 cache, got {cache.dtype}"
    assert cache.is_contiguous(), "BF16 cache must be contiguous"
    if cache.ndim == 4:
        assert cache.shape[1] == page_size and cache.shape[2] == 1
        assert cache.shape[3] == BF16_KV_HEAD_DIM
        return cache.reshape(-1, 1, BF16_KV_HEAD_DIM)
    if cache.ndim == 3:
        assert cache.shape[1] == page_size
        assert cache.shape[2] == BF16_KV_HEAD_DIM
        return cache.reshape(-1, 1, BF16_KV_HEAD_DIM)
    assert cache.ndim == 2, f"expected 2D, 3D, or 4D cache, got {cache.shape}"
    assert cache.shape[1] == page_size * BF16_KV_HEAD_DIM
    return cache.reshape(-1, 1, BF16_KV_HEAD_DIM)


def gather_bf16_kv_cache_paged(
    cache: torch.Tensor,
    token_ids: torch.Tensor,
    *,
    page_size: int,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Gather physical token locations from a plain BF16 paged cache."""

    flat_cache = _as_flat_bf16_cache(cache, page_size)
    token_ids = token_ids.to(dtype=torch.int64)
    assert token_ids.ndim == 1
    # Do not convert CUDA tensors to Python bools during CUDA Graph capture.
    # The indices are produced by the scheduler and are validated on the
    # non-capture path.
    if not _is_cuda_graph_capturing():
        assert (token_ids >= 0).all() and (token_ids < flat_cache.shape[0]).all()
    gathered = flat_cache.index_select(0, token_ids)
    if out is not None:
        assert out.shape == gathered.shape and out.dtype == torch.bfloat16
        out.copy_(gathered)
        return out
    return gathered


def build_bf16_sparse_decode_inputs(
    *,
    swa_cache: torch.Tensor,
    swa_indices: torch.Tensor,
    swa_lengths: torch.Tensor,
    swa_page_size: int,
    extra_cache: Optional[torch.Tensor] = None,
    extra_indices: Optional[torch.Tensor] = None,
    extra_lengths: Optional[torch.Tensor] = None,
    extra_page_size: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build a fixed-shape BF16 workspace and sparse indices for one decode.

    Preserve the historical ``[extra region, SWA region]`` slot layout used by
    the branch whose BF16-KV precision was validated end to end. Invalid slots
    are represented by ``-1`` indices. ``topk_length`` therefore covers the
    complete aligned index width, allowing FlashMLA to visit valid SWA slots
    after a short extra region while skipping the holes.
    """

    swa_indices = _normalize_indices(swa_indices)
    assert swa_lengths.ndim == 1 and swa_lengths.shape[0] == swa_indices.shape[0]
    batch_size, swa_width = swa_indices.shape
    if not _is_cuda_graph_capturing():
        assert (swa_lengths >= 0).all() and (swa_lengths <= swa_width).all()

    has_extra = extra_cache is not None
    if has_extra:
        assert extra_indices is not None
        assert extra_lengths is not None
        assert extra_page_size is not None
        extra_indices = _normalize_indices(extra_indices)
        assert extra_indices.shape[0] == batch_size
        assert extra_lengths.ndim == 1 and extra_lengths.shape[0] == batch_size
        extra_width = extra_indices.shape[1]
        if not _is_cuda_graph_capturing():
            assert (extra_lengths >= 0).all() and (
                extra_lengths <= extra_width
            ).all()
    else:
        extra_width = 0

    def gather_region(
        cache: torch.Tensor,
        indices: torch.Tensor,
        lengths: torch.Tensor,
        page_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cols = torch.arange(indices.shape[1], device=indices.device)
        valid = (indices >= 0) & (cols[None, :] < lengths[:, None])
        safe_indices = indices.reshape(-1).masked_fill(~valid.reshape(-1), 0)
        values = gather_bf16_kv_cache_paged(
            cache,
            safe_indices,
            page_size=page_size,
        )
        return values, valid

    def workspace_indices(valid: torch.Tensor, offset: int) -> torch.Tensor:
        rows, width = valid.shape
        indices = torch.arange(
            rows * width, device=valid.device, dtype=torch.int32
        ).view(rows, width)
        return (indices + offset).masked_fill(~valid, -1)

    swa_workspace, swa_valid = gather_region(
        swa_cache,
        swa_indices,
        swa_lengths,
        swa_page_size,
    )
    if has_extra:
        extra_workspace, extra_valid = gather_region(
            extra_cache,
            extra_indices,
            extra_lengths,
            extra_page_size,
        )
        workspace = torch.cat([extra_workspace, swa_workspace], dim=0)
        combined_indices = torch.cat(
            [
                workspace_indices(extra_valid, offset=0),
                workspace_indices(swa_valid, offset=extra_workspace.shape[0]),
            ],
            dim=1,
        )
    else:
        workspace = swa_workspace
        combined_indices = workspace_indices(swa_valid, offset=0)

    padding = (-combined_indices.shape[1]) % SPARSE_DECODE_TOPK_ALIGNMENT
    if padding:
        combined_indices = torch.nn.functional.pad(
            combined_indices, (0, padding), value=-1
        )
    full_lengths = torch.full(
        (batch_size,),
        combined_indices.shape[1],
        dtype=torch.int32,
        device=combined_indices.device,
    )
    return workspace, combined_indices.unsqueeze(1), full_lengths
