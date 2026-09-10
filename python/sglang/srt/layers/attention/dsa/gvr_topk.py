"""FlashInfer sparse top-k and graph-safe raw-index hint bookkeeping."""

from functools import lru_cache

import torch
import triton
import triton.language as tl


@lru_cache(maxsize=None)
def gvr_available(device: torch.device) -> bool:
    if device.type != "cuda" or torch.version.hip:
        return False
    try:
        from flashinfer import top_k_varlen

        # This public API was added with hint-free GVR_2 and per-stream caches.
        from flashinfer.topk_varlen import release_gvr2_resources  # noqa: F401
    except ImportError:
        return False
    major, minor = torch.cuda.get_device_capability(device)
    return top_k_varlen.is_backend_supported("gvr_2", major * 10 + minor)


def can_use_gvr(logits, top_k: int) -> bool:
    return (
        logits.dtype == torch.float32
        and top_k in (512, 1024, 2048)
        and logits.ndim == 2
        and logits.shape[0] > 0
        and logits.shape[1] > 0
        and logits.stride(1) == 1
        and logits.stride(0) >= logits.shape[1]
        and (logits.stride(0) if logits.shape[0] > 1 else logits.shape[1]) % 4 == 0
        and logits.data_ptr() % 16 == 0
        and gvr_available(logits.device)
    )


def check_flashinfer_gvr_available(device=None) -> None:
    device = torch.device("cuda" if device is None else device)
    if not gvr_available(device):
        raise RuntimeError(
            "flashinfer-gvr requires FlashInfer with hint-free top_k_varlen "
            "backend='gvr_2' and release_gvr2_resources on a supported GPU."
        )


class GvrTopkState:
    """Raw hints owned by an attention backend, indexed by layer/request slot.

    Prefill invalidates participating slots. Concurrent attention backends
    must own separate states.
    """

    def __init__(self, *, num_layers, num_slots, top_k, device):
        self.hints = torch.full(
            (num_layers, num_slots, top_k), -1, dtype=torch.int32, device=device
        )
        self._generations = torch.zeros(num_slots, dtype=torch.int64, device="cpu")

    def sync_generations(self, generations):
        """Invalidate reused slots before eager execution or graph replay.

        The request pool owns CPU generation counters, including on decode
        workers in disaggregated serving. No hint transfer is necessary.
        """
        changed = (generations != self._generations).nonzero().flatten()
        if changed.numel():
            self.hints.index_fill_(1, changed.to(self.hints.device), -1)
            self._generations.copy_(generations)

    def reset(self, layer_id, slots):
        self.hints[layer_id].index_fill_(0, slots, -1)


@triton.jit
def _gather_hints(Hints, Slots, Lengths, Out, K: tl.constexpr, N: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, K)
    slot = tl.load(Slots + row)
    length = tl.minimum(tl.load(Lengths + row), N)
    hint = tl.load(Hints + slot * K + col)
    stale = tl.sum(((hint < 0) | (hint >= length)).to(tl.int32), 0) > 0
    # Identical sampling anchor to FlashInfer's hint-free path. Entire rows
    # are reset, avoiding duplicate indices from elementwise clamping.
    hint = tl.where(stale, col, hint)
    tl.store(Out + row * K + col, hint)


@triton.jit
def _pack_rows(
    Scores,
    Starts,
    Lengths,
    Packed,
    N: tl.constexpr,
    PADDED_N: tl.constexpr,
    STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    start = tl.load(Starts + row)
    length = tl.load(Lengths + row)
    value = tl.load(
        Scores + row * STRIDE + start + col,
        (col < N) & (col < length) & (start + col < N),
        other=-float("inf"),
    )
    tl.store(Packed + row * PADDED_N + col, value, col < PADDED_N)


@triton.jit
def _finish_topk(
    Raw,
    Lengths,
    Pages,
    Mapping,
    Offsets,
    PageOffsets,
    Out,
    Hints,
    Slots,
    K: tl.constexpr,
    N: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    PAGE_STRIDE: tl.constexpr,
    PAGE_COLS: tl.constexpr,
    OUT_STRIDE: tl.constexpr,
    HAS_MAPPING: tl.constexpr,
    RAGGED: tl.constexpr,
    HAS_PAGE_OFFSETS: tl.constexpr,
    SAVE_HINT: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.arange(0, K)
    length = tl.minimum(tl.maximum(tl.load(Lengths + row), 0), N)
    raw = tl.load(Raw + row * K + col) if N > K else col
    # Normalize trivial rows without relying on kernel surplus-slot contents.
    raw = tl.where(length <= K, col, raw)
    valid = (col < length) & (raw >= 0) & (raw < length)
    raw = tl.where(valid, raw, -1)
    tl.store(Raw + row * K + col, raw)
    if SAVE_HINT:
        slot = tl.load(Slots + row)
        # Request slot zero is reserved for graph padding; many padded rows
        # can refer to it and must not race when publishing hints.
        tl.store(Hints + slot * K + col, raw, (length > 0) & (slot > 0))
    if PAGE_SIZE > 0:
        batch = tl.load(Mapping + row) if HAS_MAPPING else row
        position = raw + tl.load(PageOffsets + row) if HAS_PAGE_OFFSETS else raw
        page = position // PAGE_SIZE
        valid = valid & (page >= 0) & (page < PAGE_COLS)
        physical = tl.load(Pages + batch * PAGE_STRIDE + page, valid, other=0)
        result = physical * PAGE_SIZE + position % PAGE_SIZE
    elif RAGGED:
        result = raw + tl.load(Offsets + row)
    else:
        result = raw
    tl.store(Out + row * OUT_STRIDE + col, tl.where(valid, result, -1))


def flashinfer_sparse_topk(
    logits,
    lengths,
    top_k,
    *,
    backend="auto",
    page_table=None,
    page_size=1,
    row_to_batch=None,
    offsets=None,
    out=None,
    raw_out=None,
    state=None,
    layer_id=0,
    req_pool_indices=None,
    row_starts=None,
    page_offsets=None,
):
    """Select raw row-local indices, optionally save hints and transform.

    Lengths are already in the logits' index space. Treat each expanded query
    as an independent row (compress_ratio=next_n=1), preserving the caller's
    causal lengths for prefill, compressed KV and speculative verification.
    """
    import flashinfer

    rows = logits.shape[0]
    lengths = lengths.to(dtype=torch.int32).contiguous()
    if row_starts is not None:
        padded_n = triton.cdiv(logits.shape[1], 4) * 4
        packed = torch.empty((rows, padded_n), dtype=logits.dtype, device=logits.device)
        _pack_rows[(rows, triton.cdiv(logits.shape[1], 1024))](
            logits,
            row_starts,
            lengths,
            packed,
            logits.shape[1],
            padded_n,
            logits.stride(0),
            1024,
        )
        logits = packed
    elif (
        not logits.is_contiguous()
        and logits.untyped_storage().nbytes()
        < (logits.storage_offset() + rows * logits.stride(0)) * logits.element_size()
    ):
        # DeepGEMM can omit the last row's padding. GVR widens the view to
        # its row pitch, so supply fully backed, aligned storage in that case.
        packed = torch.empty(
            (rows, triton.cdiv(logits.shape[1], 4) * 4),
            dtype=logits.dtype,
            device=logits.device,
        )
        packed[:, : logits.shape[1]].copy_(logits)
        logits = packed
    raw = (
        torch.empty((rows, top_k), dtype=torch.int32, device=logits.device)
        if raw_out is None
        else raw_out
    )
    hint = None
    if state is not None and logits.shape[1] > top_k:
        hint = torch.empty_like(raw)
        _gather_hints[(rows,)](
            state.hints[layer_id],
            req_pool_indices,
            lengths,
            hint,
            top_k,
            logits.shape[1],
        )
    if logits.shape[1] > top_k:
        flashinfer.top_k_varlen(
            logits,
            lengths,
            top_k,
            pre_idx=hint,
            out_indices=raw,
            backend=backend,
        )
    if out is None:
        out = (
            torch.empty_like(raw)
            if page_table is not None or offsets is not None
            else raw
        )
    _finish_topk[(rows,)](
        raw,
        lengths,
        page_table,
        row_to_batch,
        offsets,
        page_offsets,
        out,
        state.hints[layer_id] if state is not None else None,
        req_pool_indices,
        top_k,
        logits.shape[1],
        page_size if page_table is not None else 0,
        page_table.stride(0) if page_table is not None else 0,
        page_table.shape[1] if page_table is not None else 0,
        out.stride(0),
        row_to_batch is not None,
        offsets is not None,
        page_offsets is not None,
        state is not None,
    )
    return out
