"""Page-signature write — FP8-scale-aware K-cache projection.

Each KV page is compressed to one ``[num_heads_local, label_dim]`` fp16
signature row by:

1. Reading the per-tile FP8 ``nope`` bytes and the per-tile fp32 scales
   from the existing NSA ``quant_k_cache`` layout (``[nope_fp8(512) |
   scales(16)]`` per token; 4 tiles × 128 channels per token; 4 fp32
   scales × 4 bytes = 16 bytes of scale metadata at byte offset 512).
2. Dequantising each tile to bf16: ``nope_bf16[tile] = nope_fp8[tile] *
   scale[tile]``.
3. Projecting the dequantised vector through the per-(layer, head)
   channel mask: ``q_channel = nope_bf16[channel_selection[h]] *
   channel_weights[h]``.
4. Reducing per-page: mean (or sum, configurable) over the page's
   tokens.

This module ships a torch reference that runs on CUDA and CPU and matches
the kernel contract. A Triton kernel for production-grade throughput lives
in :func:`_triton_kernel` and is selected automatically on CUDA when
Triton is importable; the torch path is kept as the documented reference
and as the fallback for CPU unit tests + CUDA-graph capture debugging.

Hot-page rule (CMT-14): callers update the active in-fill page's
signature every decode step and force it (and a configurable local window
of N most-recent pages, default 1) into the selected set regardless of
score. The :func:`compute_hot_pages` helper returns the page IDs that
must be forced for each request.
"""

from __future__ import annotations

import logging
from typing import Optional, Sequence, Tuple

import torch

logger = logging.getLogger(__name__)


_NOPE_BYTES = 512
_SCALE_BYTES = 16
_NOPE_DIM = 512
_TILE_SIZE = 128
_NUM_TILES = _NOPE_DIM // _TILE_SIZE
_PAGE_NOPE_STRIDE_BYTES = _NOPE_BYTES + _SCALE_BYTES  # 528

_TRITON_AVAILABLE = False
try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except ImportError:
    _TRITON_AVAILABLE = False


def _next_pow2(n: int) -> int:
    """Smallest power of two >= ``n`` (n>=1). Triton's ``tl.arange`` extents
    must be powers of two; pad up and mask the unused tail.
    """
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


if _TRITON_AVAILABLE:

    @triton.jit
    def _page_signature_write_kernel(
        nope_fp8_ptr,  # uint8 reinterpreted as fp8 — [num_pages, page_size, 512]
        scales_ptr,  # fp32 — [num_pages, page_size, num_tiles]
        sel_ptr,  # int32 — [H_local, label_dim]
        w_ptr,  # fp32 — [H_local, label_dim]
        out_ptr,  # fp16 — [num_pages, H_local, label_dim]
        num_pages: tl.constexpr,
        page_size: tl.constexpr,
        num_heads: tl.constexpr,
        label_dim: tl.constexpr,
        tile_size: tl.constexpr,
        num_tiles: tl.constexpr,
        nope_dim: tl.constexpr,
        nope_stride_page: tl.constexpr,
        nope_stride_token: tl.constexpr,
        scales_stride_page: tl.constexpr,
        scales_stride_token: tl.constexpr,
        sel_stride_head: tl.constexpr,
        w_stride_head: tl.constexpr,
        out_stride_page: tl.constexpr,
        out_stride_head: tl.constexpr,
        LABEL_DIM_POW2: tl.constexpr,  # power-of-two >= label_dim
        NUM_TILES_POW2: tl.constexpr,  # power-of-two >= num_tiles
    ):
        page_id = tl.program_id(0)
        head_id = tl.program_id(1)

        d_offsets = tl.arange(0, LABEL_DIM_POW2)
        d_mask = d_offsets < label_dim
        tile_offsets = tl.arange(0, NUM_TILES_POW2)
        tile_mask = tile_offsets < num_tiles

        sel_offsets = head_id * sel_stride_head + d_offsets
        sel_idx = tl.load(
            sel_ptr + sel_offsets, mask=d_mask, other=0
        ).to(tl.int32)  # [LABEL_DIM_POW2]
        w = tl.load(
            w_ptr + head_id * w_stride_head + d_offsets, mask=d_mask, other=0.0
        ).to(tl.float32)

        # Derive tile id per channel index.
        tile_idx = sel_idx // tile_size  # [LABEL_DIM_POW2] in [0, num_tiles)

        acc = tl.zeros((LABEL_DIM_POW2,), dtype=tl.float32)

        for tok in range(page_size):
            # Load all scales for this token, then gather the per-channel
            # scale by tile_idx.
            token_scale_base = (
                page_id * scales_stride_page + tok * scales_stride_token
            )
            scales = tl.load(
                scales_ptr + token_scale_base + tile_offsets,
                mask=tile_mask,
                other=0.0,
            )
            # [NUM_TILES_POW2] fp32

            # Gather per-channel scale: scales[tile_idx[d]] for each d.
            # Triton trick: broadcast (LABEL_DIM_POW2, NUM_TILES_POW2) match and reduce.
            tile_match = (
                tile_idx[:, None] == tile_offsets[None, :]
            ).to(tl.float32)
            ch_scale = tl.sum(tile_match * scales[None, :], axis=1)

            # Load the per-channel FP8 nope value. Padded label-dim slots
            # use the dummy sel_idx=0 channel; their contribution is gated
            # to zero by the d_mask multiply below.
            fp8_offsets = (
                page_id * nope_stride_page
                + tok * nope_stride_token
                + sel_idx
            )
            ch_fp8 = tl.load(nope_fp8_ptr + fp8_offsets).to(tl.float32)

            acc += ch_fp8 * ch_scale * w

        acc = acc / page_size  # mean over page tokens

        out_offsets = (
            page_id * out_stride_page + head_id * out_stride_head + d_offsets
        )
        tl.store(out_ptr + out_offsets, acc.to(tl.float16), mask=d_mask)


def _page_signature_write_triton(
    nope_parts_u8: torch.Tensor,  # [num_pages, page_size, 528] uint8
    channel_selection_layer: torch.Tensor,  # [H_local, label_dim] int32
    channel_weights_layer: torch.Tensor,  # [H_local, label_dim] fp32
) -> torch.Tensor:
    """Run the Triton page_signature_write kernel. Returns [num_pages, H_local, label_dim] fp16."""

    num_pages, page_size, stride = nope_parts_u8.shape
    assert stride == _PAGE_NOPE_STRIDE_BYTES, stride
    H, label_dim = channel_selection_layer.shape

    # Reinterpret the byte tensor as fp8 (bytes 0..512) and fp32 (bytes 512..528).
    nope_u8_flat = nope_parts_u8.contiguous()
    nope_fp8 = (
        nope_u8_flat[..., :_NOPE_BYTES]
        .view(torch.float8_e4m3fn)
        .reshape(num_pages, page_size, _NOPE_DIM)
    )
    scales = (
        nope_u8_flat[..., _NOPE_BYTES:]
        .contiguous()
        .view(torch.float32)
        .reshape(num_pages, page_size, _NUM_TILES)
    )

    out = torch.empty(
        (num_pages, H, label_dim),
        dtype=torch.float16,
        device=nope_parts_u8.device,
    )
    grid = (num_pages, H)

    _page_signature_write_kernel[grid](
        nope_fp8,
        scales,
        channel_selection_layer.to(torch.int32).contiguous(),
        channel_weights_layer.to(torch.float32).contiguous(),
        out,
        num_pages=num_pages,
        page_size=page_size,
        num_heads=H,
        label_dim=label_dim,
        tile_size=_TILE_SIZE,
        num_tiles=_NUM_TILES,
        nope_dim=_NOPE_DIM,
        nope_stride_page=nope_fp8.stride(0),
        nope_stride_token=nope_fp8.stride(1),
        scales_stride_page=scales.stride(0),
        scales_stride_token=scales.stride(1),
        sel_stride_head=channel_selection_layer.stride(0),
        w_stride_head=channel_weights_layer.stride(0),
        out_stride_page=out.stride(0),
        out_stride_head=out.stride(1),
        LABEL_DIM_POW2=_next_pow2(max(label_dim, 1)),
        NUM_TILES_POW2=_next_pow2(max(_NUM_TILES, 1)),
    )
    return out


def dequant_nope_fp8_to_bf16(
    nope_part_u8: torch.Tensor,
    *,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dequantise the per-tile FP8 ``nope`` payload to bf16.

    Args:
        nope_part_u8: ``[num_tokens, 528]`` uint8, layout
            ``[nope_fp8(512) | scales_fp32(16)]`` per token.

    Returns:
        ``[num_tokens, 512]`` bf16.
    """

    if nope_part_u8.dim() == 3 and nope_part_u8.shape[1] == 1:
        nope_part_u8 = nope_part_u8.squeeze(1)
    if nope_part_u8.dim() != 2 or nope_part_u8.shape[-1] != _PAGE_NOPE_STRIDE_BYTES:
        raise ValueError(
            f"nope_part_u8 must be [num_tokens, {_PAGE_NOPE_STRIDE_BYTES}] uint8, "
            f"got shape {tuple(nope_part_u8.shape)}."
        )
    if nope_part_u8.dtype != torch.uint8:
        raise TypeError(
            f"nope_part_u8 must be torch.uint8, got {nope_part_u8.dtype}."
        )

    num_tokens = int(nope_part_u8.shape[0])

    nope_fp8_bytes = nope_part_u8[:, :_NOPE_BYTES].contiguous()
    scales_bytes = nope_part_u8[:, _NOPE_BYTES:].contiguous()

    nope_fp8 = nope_fp8_bytes.view(torch.float8_e4m3fn).reshape(
        num_tokens, _NUM_TILES, _TILE_SIZE
    )
    scales = scales_bytes.view(torch.float32).reshape(num_tokens, _NUM_TILES)

    nope_bf16 = (nope_fp8.to(torch.float32) * scales.unsqueeze(-1)).to(out_dtype)
    return nope_bf16.reshape(num_tokens, _NOPE_DIM)


def project_page_to_signature(
    nope_bf16: torch.Tensor,
    channel_selection_layer: torch.Tensor,
    channel_weights_layer: torch.Tensor,
    *,
    reduce: str = "mean",
) -> torch.Tensor:
    """Project a page's bf16 ``nope`` keys to per-head label_dim signatures.

    Args:
        nope_bf16: ``[num_tokens, num_local_heads, head_dim]`` (head_dim=512
            for V3.2 FP8 nope) OR ``[num_tokens, head_dim]`` for non-MQA
            paths (broadcast across heads via channel_selection_layer's H).
        channel_selection_layer: ``[num_local_heads, label_dim]`` int32.
        channel_weights_layer: ``[num_local_heads, label_dim]`` float32.
        reduce: ``"mean"`` (default) or ``"sum"``.

    Returns:
        ``[num_local_heads, label_dim]`` float32 signature.
    """

    if reduce not in ("mean", "sum"):
        raise ValueError(f"reduce must be 'mean' or 'sum', got {reduce!r}.")
    if channel_selection_layer.dim() != 2 or channel_weights_layer.dim() != 2:
        raise ValueError(
            "channel_selection_layer / channel_weights_layer must be [H, label_dim]."
        )
    H, label_dim = channel_selection_layer.shape

    if nope_bf16.dim() == 2:
        # broadcast across heads — same nope tensor projects through each
        # head's distinct channel mask.
        nope_bf16 = nope_bf16.unsqueeze(1).expand(-1, H, -1).contiguous()
    elif nope_bf16.dim() != 3:
        raise ValueError(
            f"nope_bf16 must be 2-D [T, D] or 3-D [T, H, D], got {tuple(nope_bf16.shape)}."
        )

    num_tokens = nope_bf16.shape[0]
    sel_idx = channel_selection_layer.long().unsqueeze(0).expand(num_tokens, -1, -1)
    gathered = torch.gather(nope_bf16, dim=-1, index=sel_idx).to(torch.float32)
    weighted = gathered * channel_weights_layer.unsqueeze(0)  # [T, H, label_dim]

    if reduce == "mean":
        return weighted.mean(dim=0)
    return weighted.sum(dim=0)


def page_signature_write(
    table_signatures: torch.Tensor,  # [L, P, H_local, label_dim]
    table_valid_mask: torch.Tensor,  # bool [L, P]
    layer_id: int,
    page_ids: Sequence[int],
    nope_parts_u8: torch.Tensor,  # [num_pages, page_size, 528] uint8
    channel_selection_layer: torch.Tensor,
    channel_weights_layer: torch.Tensor,
    *,
    reduce: str = "mean",
) -> None:
    """Populate signatures for ``page_ids`` from raw nope_part bytes.

    Mutates ``table_signatures[layer_id, page_id]`` and sets
    ``table_valid_mask[layer_id, page_id] = True`` for each entry. Designed
    to be called on prefill (one batched call per layer covering newly
    assigned pages) and on every decode step for the active in-fill page.
    """

    if nope_parts_u8.dim() == 4 and nope_parts_u8.shape[2] == 1:
        nope_parts_u8 = nope_parts_u8.squeeze(2)
    if nope_parts_u8.dim() != 3:
        raise ValueError(
            "nope_parts_u8 must be [num_pages, page_size, 528] uint8, got shape "
            f"{tuple(nope_parts_u8.shape)}."
        )
    num_pages_in, page_size, stride = nope_parts_u8.shape
    if stride != _PAGE_NOPE_STRIDE_BYTES:
        raise ValueError(
            f"nope_parts_u8 last dim must be {_PAGE_NOPE_STRIDE_BYTES}, got {stride}."
        )
    if len(page_ids) != num_pages_in:
        raise ValueError(
            f"page_ids length {len(page_ids)} does not match nope_parts batch dim "
            f"{num_pages_in}."
        )

    # Fused Triton path when on CUDA + Triton available + the reduction is
    # `mean` (the kernel currently bakes mean reduction in; sum is the
    # documented fallback to the torch reference).
    if (
        _TRITON_AVAILABLE
        and reduce == "mean"
        and nope_parts_u8.is_cuda
        and channel_selection_layer.is_cuda
        and channel_weights_layer.is_cuda
        and num_pages_in > 0
    ):
        signatures = _page_signature_write_triton(
            nope_parts_u8,
            channel_selection_layer,
            channel_weights_layer,
        )  # [num_pages_in, H, label_dim] fp16
        for batch_idx, page_id in enumerate(page_ids):
            table_signatures[layer_id, page_id] = signatures[batch_idx].to(
                table_signatures.dtype
            )
            table_valid_mask[layer_id, page_id] = True
        return

    for batch_idx, page_id in enumerate(page_ids):
        nope_u8 = nope_parts_u8[batch_idx]  # [page_size, 528]
        nope_bf16 = dequant_nope_fp8_to_bf16(nope_u8)  # [page_size, 512]
        signature = project_page_to_signature(
            nope_bf16,
            channel_selection_layer,
            channel_weights_layer,
            reduce=reduce,
        )  # [H, label_dim]
        table_signatures[layer_id, page_id] = signature.to(table_signatures.dtype)
        table_valid_mask[layer_id, page_id] = True


def update_active_page(
    table_signatures: torch.Tensor,
    table_valid_mask: torch.Tensor,
    layer_id: int,
    active_page_id: int,
    nope_part_u8: torch.Tensor,  # [page_size, 528]
    channel_selection_layer: torch.Tensor,
    channel_weights_layer: torch.Tensor,
    *,
    reduce: str = "mean",
) -> None:
    """Refresh the active in-fill page's signature every decode step.

    The active page is being filled token-by-token; without this refresh the
    freshest up-to-63 tokens are invisible to selection (CMT-14).
    """

    page_signature_write(
        table_signatures,
        table_valid_mask,
        layer_id,
        [int(active_page_id)],
        nope_part_u8.unsqueeze(0) if nope_part_u8.dim() == 2 else nope_part_u8,
        channel_selection_layer,
        channel_weights_layer,
        reduce=reduce,
    )


def compute_hot_pages(
    *,
    seq_lens: torch.Tensor,
    page_size: int,
    local_window: int = 1,
) -> list:
    """Per-row hot-page list = active in-fill page + (local_window-1) recents.

    Args:
        seq_lens: int32 ``[bs]`` per-request current sequence length.
        page_size: page size in tokens.
        local_window: number of most-recent pages to force into the selected
            set regardless of score. Default 1 = active page only.

    Returns:
        List of length ``bs``; each entry is a list of page IDs.
    """

    if local_window < 1:
        raise ValueError(f"local_window must be >= 1, got {local_window}.")
    seqs = seq_lens.detach().cpu().tolist()
    out = []
    for s in seqs:
        if s <= 0:
            out.append([])
            continue
        last_page = max(0, (int(s) - 1) // page_size)
        window = list(range(max(0, last_page - local_window + 1), last_page + 1))
        out.append(window)
    return out


def m3b_page_stability_fixture(
    selector,
    *,
    prompt_tokens: torch.Tensor,
    page_size: int,
    num_repeats: int = 2,
) -> bool:
    """M3-B page-stability fixture (cold vs warm prefix).

    Runs the same deterministic prefix through the selector twice and
    asserts that ``retrieve_topk`` produces bit-identical
    ``selected_indices`` across runs. Used by the DEC-2 radix-cache gate:
    passing this fixture grants permission for radix cache to coexist with
    Double Sparsity.

    Returns True on stability; False on divergence.
    """

    if selector is None or getattr(selector, "page_signature_table", None) is None:
        logger.warning(
            "m3b_page_stability_fixture: selector has no page_signature_table; "
            "fixture is inconclusive."
        )
        return False

    bs = 1
    seq_len = prompt_tokens.shape[-1]
    device = selector.page_signature_table.signatures.device

    req_pool_indices = torch.tensor([0], dtype=torch.int32, device=device)
    seq_lens = torch.tensor([seq_len], dtype=torch.int32, device=device)
    num_pages = (seq_len + page_size - 1) // page_size
    # sparse_mask must match the bound page-signature table width (per the
    # retrieve_topk_via_signatures contract), not the prompt length. The
    # allocator's table is typically wider than any single prompt uses.
    table_max_pages = int(selector.page_signature_table.max_pages)
    sparse_mask = torch.zeros(bs, table_max_pages, dtype=torch.int32, device=device)
    sparse_mask[:, :num_pages] = 1
    queries = torch.zeros(
        bs, selector.num_local_heads, selector.head_dim, device=device
    )

    last = None
    for run in range(num_repeats):
        indices, lengths = selector.retrieve_topk(
            queries=queries,
            layer_id=0,
            req_pool_indices=req_pool_indices,
            sparse_mask=sparse_mask,
            seq_lens=seq_lens,
        )
        if last is not None:
            if not torch.equal(indices, last["indices"]) or not torch.equal(
                lengths, last["lengths"]
            ):
                logger.warning(
                    "m3b_page_stability_fixture: cold vs warm divergence on run %d.",
                    run,
                )
                return False
        last = {"indices": indices, "lengths": lengths}
    return True
