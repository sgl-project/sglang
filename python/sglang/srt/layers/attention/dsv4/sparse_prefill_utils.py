"""Per-query sparse-index combiner for the FlashMLA sparse prefill path.

Adapts vllm's ``combine_topk_swa_indices`` to sglang's flat-workspace layout.
Reference:
https://github.com/vllm-project/vllm/blob/124fac10cb0ea83aee2ffeabac0b413d6b759b26/vllm/models/deepseek_v4/common/ops/cache_utils.py#L476

For each
query token in a prefill chunk, emits one row of combined indices into the
chunk's bf16 KV workspace:

    [ topk indices into compressed cache (rebased)   ]
    [ swa positional indices (rebased)               ]
    [ -1 padding up to a multiple of 128             ]

The workspace is a single flat ``(total_workspace_tokens, 512)`` tensor
formed by concatenating, per request, that request's compressed-region
gather followed by all requests' SWA-region gathers. Two per-request
offset tensors describe the layout:

  * ``compressed_base[r]`` — flat index where request r's compressed
    region begins. Topk indices ``topk_indices[token, j]`` are local to
    request r's compressed region (in ``[0, compressed_gather_len[r])``)
    and get rebased to flat space by adding ``compressed_base[r]``.
  * ``swa_base[r]`` — flat index where request r's SWA region begins.
    Per-query SWA indices are computed positionally as
    ``swa_base[r] + (pos - swa_len + 1 - gather_start) + j``.

This is the natural layout for ``flash_mla_sparse_fwd``'s ``kv: (s_kv, 1,
d_qk)`` argument, where ``s_kv`` is the total flat workspace length.

For SWA-only layers callers pass ``topk=0``, ``compressed_base = 0`` (the
compressed branch becomes a no-op) and any ``compress_ratio >= 1``.
"""

from dataclasses import dataclass, field
from typing import Optional

import torch
import triton

from sglang.kernels.ops.attention.dsv4.dequant_k_cache import DIM_NOPE, DIM_ROPE
from sglang.srt.utils import ceil_align

# FlashMLA sparse prefill asserts ``params.topk % B_TOPK == 0``. B_TOPK is 64
# for the h_q=64 kernel and 128 for h_q=128; pad to 128 to satisfy both.
SPARSE_PREFILL_TOPK_ALIGNMENT = 128
# Bf16 workspace per-token width, matching ``dequantize_k_cache_paged``'s
# output: 448 fp8 nope (dequanted) + 64 bf16 rope = 512.
WORKSPACE_DIM = DIM_NOPE + DIM_ROPE


from sglang.kernels.ops.attention.dsv4.sparse_prefill_kernels import (
    _build_swa_token_ids_kernel,
    _combine_topk_swa_indices_kernel,
)


class SparsePrefillWorkspace:
    """Backend-owned scratch storage for sparse prefill KV dequantization.

    The workspace contents are fully overwritten before every attention call,
    so token buckets and compression ratios can safely share one buffer. Sparse
    prefill executes eagerly and serially on the supported paths, which makes it
    safe to replace the scratch allocation when a larger extent is needed.
    """

    def __init__(self, device: torch.device):
        self.device = device
        self._buffer: Optional[torch.Tensor] = None

    def get(self, num_tokens: int) -> torch.Tensor:
        assert num_tokens > 0
        current_capacity = self._buffer.shape[0] if self._buffer is not None else 0
        if num_tokens > current_capacity:
            self._buffer = torch.empty(
                (num_tokens, 1, WORKSPACE_DIM),
                dtype=torch.bfloat16,
                device=self.device,
            )
        return self._buffer[:num_tokens]


def combined_topk_width(topk: int, window_size: int) -> int:
    """Width of the padded combined_indices last dim that
    ``combine_topk_swa_indices`` would produce for these args."""
    return ceil_align(topk + window_size, SPARSE_PREFILL_TOPK_ALIGNMENT)


def combine_topk_swa_indices(
    topk_indices: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    gather_lens: torch.Tensor,
    compressed_base: torch.Tensor,
    swa_base: torch.Tensor,
    window_size: int,
    compress_ratio: int,
    topk: int,
    out_indices: Optional[torch.Tensor] = None,
    out_lens: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Combine topk + SWA indices into a single ``flash_mla_sparse_fwd`` row.

    Args:
        topk_indices: (num_tokens, K) int32. Per-query indices into the
            compressed-cache region, **already in request-local space** —
            i.e. in ``[0, compressed_gather_len[r])`` for the request that
            owns each token. Pad entries can be any value; they are ignored
            beyond ``topk_len``.
        query_start_loc: (num_reqs+1,) int32. Cumulative query lengths; may
            be in global (cross-chunk) space — kernel rebases by subtracting
            ``query_start_loc[0]``.
        seq_lens: (num_reqs,) int32. Each request's full sequence length.
        gather_lens: (num_reqs,) int32. Trailing tokens dequanted into the
            SWA region for that request.
        compressed_base: (num_reqs,) int32. Flat workspace offset where
            request r's compressed region begins. Pass all-zeros (or any
            value) for SWA-only layers since topk=0 disables this branch.
        swa_base: (num_reqs,) int32. Flat workspace offset where request
            r's SWA region begins.
        window_size: SWA window size.
        compress_ratio: must be ``>= 1`` even when topk==0.
        topk: configured topk; pass 0 for SWA-only layers.
        out_indices: optional preallocated ``(num_tokens, combined_topk)``
            int32 buffer. If provided, the kernel writes the per-query prefix
            ``[0, topk_len + swa_len)``; positions beyond are not touched.
            Caller must pre-fill with ``-1`` sentinels (and the chunk-invariant
            valid-prefix length must hold across reuses).
        out_lens: optional preallocated ``(num_tokens,)`` int32 buffer; the
            kernel fully overwrites it, so any dtype-correct buffer works.

    Returns:
        combined_indices: (num_tokens, padded_topk_swa) int32, padded to a
            multiple of 128 with -1 sentinels.
        combined_lens: (num_tokens,) int32, valid prefix length per token.
    """
    assert topk_indices.dtype == torch.int32
    assert query_start_loc.dtype == torch.int32
    assert seq_lens.dtype == torch.int32
    assert gather_lens.dtype == torch.int32
    assert compressed_base.dtype == torch.int32
    assert swa_base.dtype == torch.int32
    assert compress_ratio >= 1, "compress_ratio must be >= 1 (use topk=0 for SWA-only)"
    assert (
        topk_indices.shape[-1] >= topk
    ), f"topk_indices width {topk_indices.shape[-1]} must be >= topk {topk}"

    num_tokens = topk_indices.shape[0]
    num_reqs = seq_lens.shape[0]
    combined_topk = combined_topk_width(topk, window_size)
    if out_indices is None:
        combined_indices = torch.full(
            (num_tokens, combined_topk),
            -1,
            dtype=torch.int32,
            device=topk_indices.device,
        )
    else:
        assert out_indices.shape == (num_tokens, combined_topk)
        assert out_indices.dtype == torch.int32
        combined_indices = out_indices
    if out_lens is None:
        combined_lens = torch.zeros(
            num_tokens, dtype=torch.int32, device=topk_indices.device
        )
    else:
        assert out_lens.shape == (num_tokens,)
        assert out_lens.dtype == torch.int32
        combined_lens = out_lens

    NUM_WORKERS = 128
    _combine_topk_swa_indices_kernel[(num_reqs, NUM_WORKERS)](
        combined_indices,
        combined_indices.stride(0),
        combined_lens,
        topk_indices,
        topk_indices.stride(0),
        query_start_loc,
        seq_lens,
        gather_lens,
        compressed_base,
        swa_base,
        top_k=topk,
        COMPRESS_RATIO=compress_ratio,
        WINDOW_SIZE=window_size,
        PADDED_TOP_K=triton.next_power_of_2(topk_indices.shape[-1]),
    )
    return combined_indices, combined_lens


def build_swa_token_ids(
    seq_lens: torch.Tensor,
    extend_seq_lens: torch.Tensor,
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    full_to_swa: torch.Tensor,
    swa_window: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build a flat list of physical SWA-cache token IDs covering each
    request's positional union of every query's SWA window.

    Per request, the union spans seq positions
    ``[max(0, seq_len - extend - W + 1), seq_len)``, of length
    ``min(seq_len, extend + W - 1)``. Each position is translated through
    ``req_to_token`` (full kv-cache id) and then ``full_to_swa`` (SWA
    cache id) to land in the SWA-cache token-id space that
    ``dequantize_k_cache_paged`` consumes.

    Args:
        seq_lens: (num_reqs,) int32, per-request total sequence length.
        extend_seq_lens: (num_reqs,) int32, per-request query length.
        req_pool_indices: (num_reqs,) int32, per-request row in
            ``req_to_token``.
        req_to_token: (num_reqs_max, max_seq_len) int32. Full kv-cache id
            per (request, seq position).
        full_to_swa: (full_pool_size + extra,) int64. Maps full kv id to
            SWA-cache id.
        swa_window: int. SWA window size.

    Returns:
        swa_token_ids: (total_swa,) int32, flat physical SWA-cache token IDs.
        swa_first_pos: (num_reqs,) int32, first seq position covered per req.
        swa_gather_lens: (num_reqs,) int32, gather length per request.
        swa_offsets: (num_reqs+1,) int32, exclusive cumsum of swa_gather_lens.
    """
    assert seq_lens.dtype == torch.int32
    assert extend_seq_lens.dtype == torch.int32
    assert req_pool_indices.dtype == torch.int32
    assert req_to_token.dtype == torch.int32
    assert full_to_swa.dtype == torch.int64

    num_reqs = seq_lens.shape[0]
    device = seq_lens.device

    swa_gather_lens = torch.minimum(seq_lens, extend_seq_lens + (swa_window - 1)).to(
        torch.int32
    )
    swa_first_pos = (seq_lens - swa_gather_lens).to(torch.int32)
    swa_offsets = torch.zeros(num_reqs + 1, dtype=torch.int32, device=device)
    swa_offsets[1:] = torch.cumsum(swa_gather_lens, dim=0).to(torch.int32)
    total_swa = int(swa_offsets[-1].item())  # one CPU sync per chunk

    swa_token_ids = torch.empty(total_swa, dtype=torch.int32, device=device)
    if total_swa == 0:
        return swa_token_ids, swa_first_pos, swa_gather_lens, swa_offsets

    NUM_WORKERS = 128
    _build_swa_token_ids_kernel[(num_reqs, NUM_WORKERS)](
        swa_token_ids,
        swa_first_pos,
        swa_gather_lens,
        swa_offsets,
        req_pool_indices,
        req_to_token,
        req_to_token.stride(0),
        full_to_swa,
    )
    return swa_token_ids, swa_first_pos, swa_gather_lens, swa_offsets


@dataclass
class SparsePrefillChunkCache:
    """Chunk-invariant scaffolding for ``_forward_prefill_sparse``.

    The fields here depend only on the prefill chunk (forward_batch,
    req_to_token, full_to_swa_index_mapping, and the c4/c128 page tables)
    and not on the per-layer k_cache. Reused across every layer in the
    chunk to avoid rebuilding tiny tensors 61 times per forward pass.
    """

    # Geometry computed once per chunk.
    num_reqs: int
    num_qo_tokens: int
    # Actual maximum sequence length in this forward. CUDA-graph metadata may
    # have a much wider page table sized for the capture limit; gather only the
    # live sequence extent instead of materializing that padded capacity.
    max_seq_len: int
    # Model's SWA window — the per-query attention range. Used by
    # combine_topk_swa_indices' WINDOW_SIZE and by build_swa_token_ids's
    # gather_lens. Must match SWA_WINDOW from the backend (e.g. 128), NOT
    # the SWA pool's storage page size (often 256).
    swa_window_size: int
    # SWA cache pool's storage page size — used as the dequant kernel's
    # ``page_size`` so that ``slot // page_size`` recovers the right page.
    swa_page_size: int
    seq_lens: torch.Tensor  # (num_reqs,) int32
    query_start_loc: torch.Tensor  # (num_reqs+1,) int32

    # SWA-side (every layer needs these, all chunk-invariant).
    swa_token_ids: torch.Tensor  # (total_swa,) int32
    swa_first_pos: torch.Tensor  # (num_reqs,) int32
    swa_gather_lens: torch.Tensor  # (num_reqs,) int32
    swa_offsets: torch.Tensor  # (num_reqs+1,) int32

    # c0 pre-computed combine output (entire input set is chunk-invariant).
    c0_combined_indices: torch.Tensor = field(default=None)
    c0_combined_lens: torch.Tensor = field(default=None)
    # c128: positional layout of the c128 cache + pre-computed combine.
    c128_flat_token_ids: Optional[torch.Tensor] = None  # (num_reqs * c128_max,) int32
    c128_combined_indices: Optional[torch.Tensor] = None
    c128_combined_lens: Optional[torch.Tensor] = None

    # c4: positional layout of the c4 cache (combine output is per-layer).
    c4_flat_token_ids: Optional[torch.Tensor] = None  # (num_reqs * c4_max,) int32
    c4_page_size: Optional[int] = None
    c4_compressed_base: Optional[torch.Tensor] = None  # (num_reqs,) int32
    c4_swa_base: Optional[torch.Tensor] = None  # (num_reqs,) int32
    # Tail stays at the -1 sentinel because the valid prefix length is
    # chunk-invariant per request — subsequent layers only overwrite that prefix.
    c4_combined_indices: Optional[torch.Tensor] = None
    c4_combined_lens: Optional[torch.Tensor] = None

    @classmethod
    def build(
        cls,
        seq_lens: torch.Tensor,
        extend_seq_lens: torch.Tensor,
        req_pool_indices: torch.Tensor,
        req_to_token: torch.Tensor,
        full_to_swa: torch.Tensor,
        swa_window_size: int,
        swa_page_size: int,
        num_qo_tokens: int,
        max_seq_len: int,
    ) -> "SparsePrefillChunkCache":
        device = seq_lens.device
        num_reqs = seq_lens.shape[0]

        query_start_loc = torch.zeros(num_reqs + 1, dtype=torch.int32, device=device)
        query_start_loc[1:] = torch.cumsum(extend_seq_lens, dim=0).to(torch.int32)

        swa_token_ids, swa_first_pos, swa_gather_lens, swa_offsets = (
            build_swa_token_ids(
                seq_lens=seq_lens,
                extend_seq_lens=extend_seq_lens,
                req_pool_indices=req_pool_indices,
                req_to_token=req_to_token,
                full_to_swa=full_to_swa,
                swa_window=swa_window_size,
            )
        )

        cache = cls(
            num_reqs=num_reqs,
            num_qo_tokens=num_qo_tokens,
            max_seq_len=max_seq_len,
            swa_window_size=swa_window_size,
            swa_page_size=swa_page_size,
            seq_lens=seq_lens,
            query_start_loc=query_start_loc,
            swa_token_ids=swa_token_ids,
            swa_first_pos=swa_first_pos,
            swa_gather_lens=swa_gather_lens,
            swa_offsets=swa_offsets,
        )

        # Pre-compute the c0 combine output: TOPK=0, compressed_base=0,
        # swa_base = swa_offsets[:-1]. All inputs are chunk-invariant.
        zero_topk = torch.zeros((num_qo_tokens, 1), dtype=torch.int32, device=device)
        zero_compressed_base = torch.zeros(num_reqs, dtype=torch.int32, device=device)
        c0_swa_base = swa_offsets[:-1].to(torch.int32)
        cache.c0_combined_indices, cache.c0_combined_lens = combine_topk_swa_indices(
            topk_indices=zero_topk,
            query_start_loc=query_start_loc,
            seq_lens=seq_lens,
            gather_lens=swa_gather_lens,
            compressed_base=zero_compressed_base,
            swa_base=c0_swa_base,
            window_size=swa_window_size,
            compress_ratio=1,
            topk=0,
        )
        return cache

    def ensure_c128(self, c128_page_indices: torch.Tensor) -> None:
        """Populate c128-side fields from per-query c128 page indices.

        ``c128_page_indices[q, j]`` carries slot ids derived from
        ``page_table[q]`` (request-keyed; same across queries of a request)
        but masked per-token by ``j < seq_lens_casual[q] // 128`` — entries
        beyond that are -1. We need a row whose mask covers every j the
        combine kernel might reference, i.e. up to ``seq_lens[r] // 128``;
        that's the *last* query's mask. Pulling the first query in a fresh
        prefill (``seq_lens_casual = 1``) yields an all-`-1` row that
        clamp_min(0) collapses to slot 0, sending dequant to a polluted
        slot and producing garbage c128 entries.
        """
        if self.c128_flat_token_ids is not None:
            return
        device = self.seq_lens.device
        c128_max = max(self.max_seq_len // 128, 1)
        assert c128_max <= c128_page_indices.shape[-1], (
            f"live c128 extent {c128_max} exceeds metadata capacity "
            f"{c128_page_indices.shape[-1]}"
        )
        last_q_per_req = (self.query_start_loc[1:] - 1).long()
        per_req_c128 = c128_page_indices.narrow(1, 0, c128_max).index_select(
            0, last_q_per_req
        )
        # Clamp -1 -> 0 so dequant doesn't OOB; combine masks the invalid
        # tail via topk_len.
        flat_c128_ids = per_req_c128.reshape(-1).clamp_min(0).to(torch.int32)
        compressed_base = (
            torch.arange(self.num_reqs, dtype=torch.int32, device=device) * c128_max
        ).to(torch.int32)
        total_compressed = self.num_reqs * c128_max
        # Pre-compute the c128 combine output. topk_indices[q, j] = j is the
        # arange-broadcast pattern; we materialize it once here so the
        # combine kernel can read it like any other topk tensor.
        topk_indices = (
            torch.arange(c128_max, dtype=torch.int32, device=device)[None, :]
            .expand(self.num_qo_tokens, -1)
            .contiguous()
        )
        swa_base = (total_compressed + self.swa_offsets[:-1]).to(torch.int32)
        combined_indices, combined_lens = combine_topk_swa_indices(
            topk_indices=topk_indices,
            query_start_loc=self.query_start_loc,
            seq_lens=self.seq_lens,
            gather_lens=self.swa_gather_lens,
            compressed_base=compressed_base,
            swa_base=swa_base,
            window_size=self.swa_window_size,
            compress_ratio=128,
            topk=c128_max,
        )

        self.c128_flat_token_ids = flat_c128_ids
        self.c128_combined_indices = combined_indices
        self.c128_combined_lens = combined_lens

    def ensure_c4(
        self,
        page_table: torch.Tensor,
        c4_page_size: int,
    ) -> None:
        """Populate c4-side fields from the per-query page table.

        ``page_table`` is (num_qo_tokens, max_blocks); rows within a request
        are duplicates. The combine output is per-layer (depends on the
        layer's remapped topk_indices), so we only cache the gather-side
        scaffolding plus compressed/swa bases.
        """
        if self.c4_flat_token_ids is not None:
            return
        device = self.seq_lens.device
        c4_max = max(self.max_seq_len // 4, 1)
        c4_capacity = page_table.shape[-1] * c4_page_size
        assert (
            c4_max <= c4_capacity
        ), f"live c4 extent {c4_max} exceeds metadata capacity {c4_capacity}"
        first_q_per_req = self.query_start_loc[:-1].long()
        num_blocks = (c4_max + c4_page_size - 1) // c4_page_size
        assert num_blocks <= page_table.shape[1]
        per_req_page_table = page_table.narrow(1, 0, num_blocks).index_select(
            0, first_q_per_req
        )

        k_arange = torch.arange(c4_max, dtype=torch.int32, device=device)
        block_idx = (k_arange // c4_page_size).long()
        in_page = (k_arange % c4_page_size).to(torch.int32)
        c4_token_ids_2d = (
            per_req_page_table.index_select(1, block_idx) * c4_page_size + in_page
        ).to(torch.int32)
        flat_c4_ids = c4_token_ids_2d.reshape(-1).clamp_min(0)
        total_compressed = self.num_reqs * c4_max
        compressed_base = (
            torch.arange(self.num_reqs, dtype=torch.int32, device=device) * c4_max
        ).to(torch.int32)
        swa_base = (total_compressed + self.swa_offsets[:-1]).to(torch.int32)

        self.c4_flat_token_ids = flat_c4_ids
        self.c4_page_size = c4_page_size
        self.c4_compressed_base = compressed_base
        self.c4_swa_base = swa_base

    def combine_c4_layer(
        self,
        c4_sparse_raw_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-layer combine for c4. ``c4_sparse_raw_indices`` is the topk
        kernel's positional output (``block_in_seq * c_page_size + in_page``)
        — already in the request-local workspace coordinate that
        ``combine_topk_swa_indices`` expects, so no remap is needed.

        Reuses preallocated ``c4_combined_indices`` / ``c4_combined_lens``
        buffers across layers — the kernel only overwrites the valid prefix.
        """
        topk = c4_sparse_raw_indices.shape[-1]
        if self.c4_combined_indices is None:
            device = self.seq_lens.device
            self.c4_combined_indices = torch.full(
                (self.num_qo_tokens, combined_topk_width(topk, self.swa_window_size)),
                -1,
                dtype=torch.int32,
                device=device,
            )
            self.c4_combined_lens = torch.zeros(
                self.num_qo_tokens, dtype=torch.int32, device=device
            )
        return combine_topk_swa_indices(
            topk_indices=c4_sparse_raw_indices,
            query_start_loc=self.query_start_loc,
            seq_lens=self.seq_lens,
            gather_lens=self.swa_gather_lens,
            compressed_base=self.c4_compressed_base,
            swa_base=self.c4_swa_base,
            window_size=self.swa_window_size,
            compress_ratio=4,
            topk=topk,
            out_indices=self.c4_combined_indices,
            out_lens=self.c4_combined_lens,
        )
