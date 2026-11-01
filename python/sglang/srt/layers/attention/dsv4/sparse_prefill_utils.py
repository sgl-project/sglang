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

import os
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import torch

from sglang.kernels.ops.attention.dsv4.dequant_k_cache import DIM_NOPE, DIM_ROPE
from sglang.srt.utils.common import async_h2d

# Bf16 workspace per-token width, matching ``dequantize_k_cache_paged_dsv4``'s
# output: 448 fp8 nope (dequanted) + 64 bf16 rope = 512.
WORKSPACE_DIM = DIM_NOPE + DIM_ROPE
DSV4_Q8KV8_PREFILL_ENV = "SGLANG_DSV4_Q8KV8_PREFILL"
DSV4_Q8KV8_PREFILL_LOG_ENV = "SGLANG_DSV4_Q8KV8_PREFILL_LOG"

from sglang.kernels.ops.attention.dsv4.sparse_prefill_kernels import (
    build_swa_token_ids,
    combine_topk_swa_indices,
    combined_topk_width,
)


def use_dsv4_q8kv8_sparse_prefill(dsv4_prefill_backend: str = "auto") -> bool:
    """Return whether DeepSeek-V4 sparse prefill should use Q8KV8.

    ``dsv4_prefill_backend`` is the production configuration. The environment
    variable remains as a debug override while the runtime path is being
    hardened: truthy values force Q8 on, falsy values force it off.
    """
    env_value = os.getenv(DSV4_Q8KV8_PREFILL_ENV)
    if env_value is not None:
        return env_value.lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
    return dsv4_prefill_backend == "flashmla_sparse_q8"


class SparsePrefillKVBuffer:
    """Backend-owned scratch storage for sparse prefill KV dequantization.

    Callers overwrite the rows they read, except head rows ``get_with_head``
    reports as filled, so token buckets and compression ratios can safely share
    one buffer. Sparse prefill executes eagerly and serially on the supported
    paths, which makes it safe to replace the scratch allocation when a larger
    extent is needed.
    """

    def __init__(self, device: torch.device):
        self.device = device
        self._buffer: Optional[torch.Tensor] = None
        self._head_source: Optional[tuple] = None

    def get(
        self,
        num_tokens: int,
        dtype: torch.dtype = torch.bfloat16,
    ) -> torch.Tensor:
        self._head_source = None
        return self._reserve(num_tokens, dtype)

    def get_with_head(
        self, num_tokens: int, head_source: tuple
    ) -> Tuple[torch.Tensor, bool]:
        """Returns the rows and whether the head already holds ``head_source``."""
        buffer = self._reserve(num_tokens, torch.bfloat16)
        head_filled = head_source == self._head_source
        self._head_source = head_source
        return buffer, head_filled

    def forget_head(self) -> None:
        """Per forward: the next forward's tensors may reuse freed addresses."""
        self._head_source = None

    def _reserve(self, num_tokens: int, dtype: torch.dtype) -> torch.Tensor:
        assert num_tokens > 0
        current_capacity = self._buffer.shape[0] if self._buffer is not None else 0
        current_dtype = self._buffer.dtype if self._buffer is not None else None
        if num_tokens > current_capacity or dtype != current_dtype:
            self._buffer = torch.empty(
                (num_tokens, 1, WORKSPACE_DIM),
                dtype=dtype,
                device=self.device,
            )
            self._head_source = None
        return self._buffer[:num_tokens]


@dataclass
class CompressedGather:
    """Positional layout of one compressed cache inside the workspace."""

    flat_token_ids: torch.Tensor  # (total gathered rows,) int32
    compressed_base: torch.Tensor  # (num_reqs,) int32
    swa_base: torch.Tensor  # (num_reqs,) int32
    # Tail stays at the -1 sentinel because the valid prefix length is
    # chunk-invariant per request; subsequent layers only overwrite that prefix.
    combined_indices: Optional[torch.Tensor] = None
    combined_lens: Optional[torch.Tensor] = None


@dataclass
class SparsePrefillChunkCache:
    """Cache prefill-chunk metadata shared across layers.

    Fields depend on request/token mappings and compressed page tables, not
    per-layer k_cache; per-layer top-k combinations are recomputed into reused
    buffers.
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
    query_start_loc: torch.Tensor  # (num_reqs+1,) int32, query rows per request
    query_pos: torch.Tensor  # (num_qo_tokens,) int32, sequence position per row

    # SWA-side (every layer needs these, all chunk-invariant).
    swa_token_ids: torch.Tensor  # (total_swa,) int32
    swa_first_pos: torch.Tensor  # (num_reqs,) int32
    swa_gather_lens: torch.Tensor  # (num_reqs,) int32
    swa_offsets: torch.Tensor  # (num_reqs+1,) int32

    # c0 pre-computed combine output (entire input set is chunk-invariant).
    c0_combined_indices: torch.Tensor = field(default=None)
    c0_combined_lens: torch.Tensor = field(default=None)
    # Compressed caches keyed by compress ratio: c128 (every block, combined once
    # per chunk) and the top-k ratios (combined per layer).
    compressed: Dict[int, CompressedGather] = field(default_factory=dict)
    # Per-layer scratch only: combine overwrites all entries before attention.
    # c0/c128 are cached across layers and must retain separate storage.
    low_ratio_combined: dict = field(default_factory=dict)
    swa_indices: Optional[torch.Tensor] = None
    swa_lengths: Optional[torch.Tensor] = None
    # Host lengths avoid a device synchronization when sizing ragged gathers.
    seq_lens_cpu: Optional[list[int]] = None

    @classmethod
    def build(
        cls,
        seq_lens: torch.Tensor,
        extend_seq_lens: torch.Tensor,
        query_lens: torch.Tensor,
        query_pos: torch.Tensor,
        req_pool_indices: torch.Tensor,
        req_to_token: torch.Tensor,
        full_to_swa: torch.Tensor,
        swa_window_size: int,
        swa_page_size: int,
        num_qo_tokens: int,
        max_seq_len: int,
        total_swa: int,
        request_window_layout=None,
        seq_lens_cpu: Optional[list[int]] = None,
    ) -> "SparsePrefillChunkCache":
        """``query_lens`` / ``query_pos``: the rows this forward runs (the extend, or
        a CP rank's interleaved share of it); the SWA gather spans the whole extend."""
        device = seq_lens.device
        num_reqs = seq_lens.shape[0]

        query_start_loc = torch.zeros(num_reqs + 1, dtype=torch.int32, device=device)
        query_start_loc[1:] = torch.cumsum(query_lens, dim=0).to(torch.int32)

        if request_window_layout is None:
            swa_token_ids, swa_first_pos, swa_gather_lens, swa_offsets = (
                build_swa_token_ids(
                    seq_lens=seq_lens,
                    extend_seq_lens=extend_seq_lens,
                    req_pool_indices=req_pool_indices,
                    req_to_token=req_to_token,
                    full_to_swa=full_to_swa,
                    swa_window=swa_window_size,
                    total_swa=total_swa,
                )
            )
            swa_indices = swa_lengths = None
        else:
            swa_token_ids = torch.arange(
                request_window_layout.size, dtype=torch.int32, device=device
            )
            swa_first_pos = torch.zeros_like(seq_lens)
            swa_gather_lens = torch.zeros_like(seq_lens)
            # RequestWindow indices already address the shared workspace.
            swa_offsets = torch.zeros(num_reqs + 1, dtype=torch.int32, device=device)
            swa_indices = request_window_layout.indices
            swa_lengths = request_window_layout.lengths

        cache = cls(
            num_reqs=num_reqs,
            seq_lens_cpu=seq_lens_cpu,
            num_qo_tokens=num_qo_tokens,
            max_seq_len=max_seq_len,
            swa_window_size=swa_window_size,
            swa_page_size=swa_page_size,
            seq_lens=seq_lens,
            query_start_loc=query_start_loc,
            query_pos=query_pos,
            swa_token_ids=swa_token_ids,
            swa_first_pos=swa_first_pos,
            swa_gather_lens=swa_gather_lens,
            swa_offsets=swa_offsets,
            swa_indices=swa_indices,
            swa_lengths=swa_lengths,
        )

        # Pre-compute the c0 combine output: TOPK=0, compressed_base=0,
        # swa_base = swa_offsets[:-1]. All inputs are chunk-invariant.
        zero_topk = torch.zeros((num_qo_tokens, 1), dtype=torch.int32, device=device)
        zero_compressed_base = torch.zeros(num_reqs, dtype=torch.int32, device=device)
        c0_swa_base = swa_offsets[:-1].to(torch.int32)
        cache.c0_combined_indices, cache.c0_combined_lens = combine_topk_swa_indices(
            topk_indices=zero_topk,
            query_start_loc=query_start_loc,
            query_pos=query_pos,
            seq_lens=seq_lens,
            gather_lens=swa_gather_lens,
            compressed_base=zero_compressed_base,
            swa_base=c0_swa_base,
            window_size=swa_window_size,
            compress_ratio=1,
            topk=0,
            swa_indices=swa_indices,
            swa_lengths=swa_lengths,
        )
        return cache

    def _workspace_bases(self, c_max: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Flat workspace offsets of each request's compressed and SWA regions:
        ``c_max`` compressed slots per request, then the SWA gather."""
        device = self.seq_lens.device
        compressed_base = (
            torch.arange(self.num_reqs, dtype=torch.int32, device=device) * c_max
        ).to(torch.int32)
        swa_base = (self.num_reqs * c_max + self.swa_offsets[:-1]).to(torch.int32)
        return compressed_base, swa_base

    def layer_inputs(
        self,
        compress_ratio: int,
        core_attn_metadata,
        c_page_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``(flat_token_ids, combined_indices, combined_lens)`` for one compressed
        layer. c128 gathers every block and combines once per chunk; a top-k ratio
        gathers from the page table once and combines per layer from the indexer's
        raw top-k."""
        if compress_ratio == 128:
            page_indices = core_attn_metadata.sparse_page_indices(128)
            assert page_indices is not None
            gather = self.ensure_c128(page_indices)
            return gather.flat_token_ids, gather.combined_indices, gather.combined_lens
        raw_indices = core_attn_metadata.sparse_raw_indices(compress_ratio)
        assert raw_indices is not None, (
            f"sparse-prefill c{compress_ratio} path requires the raw top-k indices "
            "(allocated in init_flashmla_related when is_prefill=True)"
        )
        gather = self.ensure_compressed(
            compress_ratio, core_attn_metadata.page_table, c_page_size
        )
        combined_indices, combined_lens = self.combine_compressed(
            compress_ratio, raw_indices[: self.num_qo_tokens]
        )
        return gather.flat_token_ids, combined_indices, combined_lens

    def ensure_c128(self, c128_page_indices: torch.Tensor) -> CompressedGather:
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
        gather = self.compressed.get(128)
        if gather is not None:
            return gather
        device = self.seq_lens.device
        c128_max = max(self.max_seq_len // 128, 1)
        assert c128_max <= c128_page_indices.shape[-1], (
            f"live c128 extent {c128_max} exceeds metadata capacity "
            f"{c128_page_indices.shape[-1]}"
        )
        # A request without rows on this rank gathers into a region nothing reads.
        last_q_per_req = (self.query_start_loc[1:] - 1).clamp_min(0).long()
        per_req_c128 = c128_page_indices.narrow(1, 0, c128_max).index_select(
            0, last_q_per_req
        )
        # Clamp -1 -> 0 so dequant doesn't OOB; combine masks the invalid
        # tail via topk_len.
        flat_c128_ids = per_req_c128.reshape(-1).clamp_min(0).to(torch.int32)
        compressed_base, swa_base = self._workspace_bases(c128_max)
        # Pre-compute the c128 combine output. topk_indices[q, j] = j is the
        # arange-broadcast pattern; we materialize it once here so the
        # combine kernel can read it like any other topk tensor.
        topk_indices = (
            torch.arange(c128_max, dtype=torch.int32, device=device)[None, :]
            .expand(self.num_qo_tokens, -1)
            .contiguous()
        )
        combined_indices, combined_lens = combine_topk_swa_indices(
            topk_indices=topk_indices,
            query_start_loc=self.query_start_loc,
            query_pos=self.query_pos,
            seq_lens=self.seq_lens,
            gather_lens=self.swa_gather_lens,
            compressed_base=compressed_base,
            swa_base=swa_base,
            window_size=self.swa_window_size,
            compress_ratio=128,
            topk=c128_max,
            swa_indices=self.swa_indices,
            swa_lengths=self.swa_lengths,
        )

        gather = CompressedGather(
            flat_token_ids=flat_c128_ids,
            compressed_base=compressed_base,
            swa_base=swa_base,
            combined_indices=combined_indices,
            combined_lens=combined_lens,
        )
        self.compressed[128] = gather
        return gather

    def ensure_compressed(
        self,
        compress_ratio: int,
        page_table: torch.Tensor,
        c_page_size: int,
    ) -> CompressedGather:
        """``page_table`` is (num_qo_tokens, max_blocks); rows within a request
        are duplicates. The combine output is per-layer (depends on the
        layer's remapped topk_indices), so we only cache the gather-side
        scaffolding plus compressed/swa bases.
        """
        gather = self.compressed.get(compress_ratio)
        if gather is not None:
            return gather
        device = self.seq_lens.device
        c_max = max(self.max_seq_len // compress_ratio, 1)
        c_capacity = page_table.shape[-1] * c_page_size
        assert c_max <= c_capacity, (
            f"live c{compress_ratio} extent {c_max} exceeds metadata capacity {c_capacity}"
        )
        first_q_per_req = (
            self.query_start_loc[:-1].clamp_max(self.num_qo_tokens - 1).long()
        )
        # Each request needs only its own live compressed history. Keeping a
        # rectangular [num_reqs, c_max] gather duplicates the longest extent
        # for every short request in a mixed batch.
        seq_lens_cpu = self.seq_lens_cpu
        if seq_lens_cpu is None:
            seq_lens_cpu = self.seq_lens.tolist()
        lengths_cpu = [int(length) // compress_ratio for length in seq_lens_cpu]
        total_compressed = sum(lengths_cpu)
        lengths = async_h2d(lengths_cpu, dtype=torch.int32, device=device)
        ends = lengths.cumsum(0, dtype=torch.int32)
        compressed_base = ends - lengths
        swa_base = total_compressed + self.swa_offsets[:-1]

        # output_size is known on the host, so repeat_interleave does not need
        # to synchronize the device to determine its output allocation.
        request_ids = torch.repeat_interleave(
            torch.arange(self.num_reqs, device=device),
            lengths,
            output_size=total_compressed,
        )
        positions = (
            torch.arange(total_compressed, device=device) - compressed_base[request_ids]
        )
        pages = page_table[first_q_per_req[request_ids], positions // c_page_size]
        flat_ids = (pages * c_page_size + positions % c_page_size).to(torch.int32)
        flat_ids.clamp_min_(0)

        gather = CompressedGather(
            flat_token_ids=flat_ids,
            compressed_base=compressed_base,
            swa_base=swa_base,
        )
        self.compressed[compress_ratio] = gather
        return gather

    def combine_compressed(
        self,
        compress_ratio: int,
        sparse_raw_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``sparse_raw_indices`` is the top-k as request-local compressed
        positions, already the workspace coordinate ``combine_topk_swa_indices``
        expects, so no remap is needed.
        """
        gather = self.compressed[compress_ratio]
        topk = sparse_raw_indices.shape[-1]
        width = combined_topk_width(topk, self.swa_window_size)
        if compress_ratio in (1, 2) and width in self.low_ratio_combined:
            gather.combined_indices, gather.combined_lens = self.low_ratio_combined[
                width
            ]
        if gather.combined_indices is None:
            device = self.seq_lens.device
            gather.combined_indices = torch.full(
                (self.num_qo_tokens, combined_topk_width(topk, self.swa_window_size)),
                -1,
                dtype=torch.int32,
                device=device,
            )
            gather.combined_lens = torch.zeros(
                self.num_qo_tokens, dtype=torch.int32, device=device
            )
        if compress_ratio in (1, 2):
            self.low_ratio_combined[width] = (
                gather.combined_indices,
                gather.combined_lens,
            )
        return combine_topk_swa_indices(
            topk_indices=sparse_raw_indices,
            query_start_loc=self.query_start_loc,
            query_pos=self.query_pos,
            seq_lens=self.seq_lens,
            gather_lens=self.swa_gather_lens,
            compressed_base=gather.compressed_base,
            swa_base=gather.swa_base,
            window_size=self.swa_window_size,
            compress_ratio=compress_ratio,
            topk=topk,
            out_indices=gather.combined_indices,
            out_lens=gather.combined_lens,
            swa_indices=self.swa_indices,
            swa_lengths=self.swa_lengths,
        )
