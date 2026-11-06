"""Host offload of the QSA full-attention K/V with a per-request GPU hot cache.

The full K/V of every QSA layer lives in pinned host memory, indexed by the
logical KV slot (the allocator index stored in ``req_to_token``). The allocator,
radix tree, chunked prefill and the compressed indexer keep addressing logical
slots, so a radix prefix shared by several requests shares its host rows too.

The GPU K/V pool of the QSA layers holds only device rows:

* request slot ``r`` owns rows ``[r * region, (r + 1) * region)`` with
  ``region = (hot_blocks + 2) * compress_ratio``: a two-block tail ring (block
  ``b`` in ring slot ``b % 2``) for the newest tokens, then ``hot_blocks``
  least-recently-used slots of one compressed block (``compress_ratio`` tokens)
  each;
* an extend scratch above the request regions.

Every forward writes its new K/V to device rows and through to the host. A
decode or target-verify forward maps each selected logical position to a device
row, copying missing blocks from the host, and attention gathers through that
row table instead of ``req_to_token``. A block copy is valid only while
``req_to_token`` still maps its positions to the logical slots it was filled
from; the radix cache rewrites a running request's mapping when it deduplicates
an inserted prefix.
"""

from __future__ import annotations

import logging
import time
from typing import List, Optional, Tuple

import msgspec
import torch

from sglang.srt.environ import envs
from sglang.srt.mem_cache.pool_host.common import (
    HostTensorAllocator,
    _cuda_host_unregister,
    alloc_with_host_register,
)
from sglang.srt.utils import is_hip

logger = logging.getLogger(__name__)

# The block of a request's first new token and the next one: up to
# ``compress_ratio`` new tokens span at most two blocks.
TAIL_BLOCKS = 2
GB = 1e9


class QSAKVOffloadLayout(msgspec.Struct, frozen=True):
    """Device-row layout of an offloaded QSA K/V pool."""

    compress_ratio: int
    hot_blocks: int
    num_request_slots: int
    scratch_rows: int
    token_topk: int
    rows_per_request: int

    @property
    def region_rows(self) -> int:
        return (self.hot_blocks + TAIL_BLOCKS) * self.compress_ratio

    @property
    def tail_rows(self) -> int:
        return TAIL_BLOCKS * self.compress_ratio

    @property
    def scratch_base(self) -> int:
        return self.num_request_slots * self.region_rows

    @property
    def device_rows(self) -> int:
        return self.scratch_base + self.scratch_rows


def compute_offload_layout(
    *,
    compress_ratio: int,
    token_topk: int,
    hot_blocks: int,
    num_request_slots: int,
    rows_per_request: int,
    max_extend_tokens: int,
    page_size: int,
) -> QSAKVOffloadLayout:
    """Layout for ``num_request_slots`` request slots (``req_to_token`` rows).

    ``rows_per_request`` is the query rows of one request in a forward (the
    verify rows under speculative decoding, else 1); their selected blocks are
    resolved together and must fit in the hot cache at once.
    """
    if page_size % compress_ratio:
        raise ValueError(
            f"QSA K/V offload needs a page size ({page_size}) that is a multiple "
            f"of the compress ratio ({compress_ratio})"
        )
    if not 1 <= rows_per_request <= compress_ratio:
        raise ValueError(
            "QSA K/V offload supports at most compress_ratio "
            f"({compress_ratio}) query rows per request, got {rows_per_request}"
        )
    needed = rows_per_request * (token_topk // compress_ratio)
    if hot_blocks < needed:
        raise ValueError(
            f"--qsa-kv-offload-hot-blocks {hot_blocks} is below {needed}: "
            f"{rows_per_request} query row(s) per request x "
            f"{token_topk // compress_ratio} selected blocks must fit at once"
        )
    # Page-aligned so the device pool size is a whole number of pages.
    scratch = max_extend_tokens + 2 * page_size
    regions = num_request_slots * (hot_blocks + TAIL_BLOCKS) * compress_ratio
    scratch += -(regions + scratch) % page_size
    return QSAKVOffloadLayout(
        compress_ratio=compress_ratio,
        hot_blocks=hot_blocks,
        num_request_slots=num_request_slots,
        scratch_rows=scratch,
        token_topk=token_topk,
        rows_per_request=rows_per_request,
    )


def offload_layout_from_config(
    *, hf_config, num_request_slots: int, page_size: int
) -> QSAKVOffloadLayout:
    """Layout from the resolved server configuration and the model's QSA profile."""
    from sglang.srt.layers.attention.qsa.config import parse_qsa_profile
    from sglang.srt.runtime_context import (
        get_memory,
        get_schedule,
        max_speculative_num_draft_tokens,
    )

    profile = parse_qsa_profile(hf_config)
    schedule = get_schedule()
    return compute_offload_layout(
        compress_ratio=profile.compress_ratio,
        token_topk=profile.budget,
        hot_blocks=get_memory().qsa_kv_offload_hot_blocks,
        num_request_slots=num_request_slots,
        rows_per_request=max_speculative_num_draft_tokens() or 1,
        max_extend_tokens=max(
            schedule.chunked_prefill_size or 0, schedule.max_prefill_tokens or 0
        ),
        page_size=page_size,
    )


class QSAKVOffloadBudget(msgspec.Struct, frozen=True):
    """GPU and host cost of the offload, for the KV pool sizing."""

    # K/V bytes per logical token of the offloaded layers (all of them).
    offloaded_bytes_per_token: int
    # Device rows per request slot, and the scratch plus padding page.
    region_rows: int
    fixed_rows: int
    # Hot-cache tables per request slot, all layers.
    table_bytes_per_slot: int
    host_size_gb: float
    page_size: int
    # Verify mode keeps the full GPU pool: the whole cell stays on the GPU.
    verify: bool = False

    def slot_bytes(self) -> int:
        return self.region_rows * self.offloaded_bytes_per_token + (
            self.table_bytes_per_slot
        )

    def gpu_cell(self, cell_bytes: int) -> int:
        cell = cell_bytes - (0 if self.verify else self.offloaded_bytes_per_token)
        if cell <= 0:
            raise ValueError(f"QSA K/V offload: per-token GPU cost {cell} B")
        return cell

    def host_tokens(self) -> int:
        if self.host_size_gb <= 0:
            raise ValueError("--qsa-kv-offload-host-size must be positive")
        host_rows = int(self.host_size_gb * GB) // self.offloaded_bytes_per_token
        return host_rows - self.page_size

    def max_total_tokens(
        self, *, available_bytes: int, cell_bytes: int, num_request_slots: int
    ) -> int:
        """Logical KV pool size for ``num_request_slots`` request slots.

        Per logical token the GPU keeps the cell minus the offloaded K/V (the
        QSA index and any draft pool); each request slot costs its device rows
        and tables. The host budget caps the pool at its K/V rows.
        """
        fixed = num_request_slots * self.slot_bytes() + (
            self.fixed_rows * self.offloaded_bytes_per_token
        )
        if available_bytes <= fixed:
            raise ValueError(
                f"QSA K/V offload: the hot cache of {num_request_slots} request "
                f"slots ({fixed / 2**30:.2f} GiB) exceeds the "
                f"{available_bytes / 2**30:.2f} GiB KV budget; lower "
                "--qsa-kv-offload-hot-blocks or --max-running-requests"
            )
        gpu_tokens = (available_bytes - fixed) // self.gpu_cell(cell_bytes)
        tokens = min(gpu_tokens, self.host_tokens())
        logger.info(
            "QSA K/V offload sizing: %d request slots, hot cache %.2f GiB, GPU "
            "allows %d tokens, host %.1f GB allows %d tokens",
            num_request_slots,
            fixed / 2**30,
            gpu_tokens,
            self.host_size_gb,
            self.host_tokens(),
        )
        return tokens // self.page_size * self.page_size

    def tokens_with_unused_slots(
        self, *, max_total_tokens: int, cell_bytes: int, unused_slots: int
    ) -> int:
        """Pool size once ``unused_slots`` budgeted request slots are not built:
        their GPU bytes hold logical tokens instead, up to the host budget."""
        extra = unused_slots * self.slot_bytes() // self.gpu_cell(cell_bytes)
        tokens = min(max_total_tokens + extra, self.host_tokens())
        return tokens // self.page_size * self.page_size


def offload_table_bytes_per_slot(
    *, num_layers: int, hot_blocks: int, compress_ratio: int, context_len: int
) -> int:
    """Upper bound of the per-request-slot hot-cache tables of either resolve:
    block->slot (int32 per block of the context), per-slot block, age, LRU
    order, marks and source records, and the ring records."""
    per_layer = (
        (context_len // compress_ratio + 2 * TAIL_BLOCKS) * 4
        + hot_blocks * (8 + 8 + 2 * 4 + 4 + 4 * compress_ratio)
        + TAIL_BLOCKS * (8 + 4 * compress_ratio)
        + 8
    )
    return num_layers * per_layer


def _copy_rows(
    *,
    k_dst: torch.Tensor,
    v_dst: torch.Tensor,
    dst: torch.Tensor,
    k_src: torch.Tensor,
    v_src: torch.Tensor,
    src: torch.Tensor,
) -> None:
    """K/V rows ``src[i]`` -> ``dst[i]`` between device and pinned host buffers."""
    if dst.numel() == 0:
        return
    if dst.is_cuda:
        from sglang.kernels.ops.kvcache.hicache import transfer_hicache_one_layer

        transfer_hicache_one_layer(
            k_cache_dst=k_dst,
            v_cache_dst=v_dst,
            indices_dst=dst,
            k_cache_src=k_src,
            v_cache_src=v_src,
            indices_src=src,
            element_dim=k_dst[0].numel(),
        )
        return
    k_dst.index_copy_(0, dst, k_src.index_select(0, src))
    v_dst.index_copy_(0, dst, v_src.index_select(0, src))


class QSAKVHostOffload:
    """Host K/V store and GPU hot cache of one QSA K/V pool.

    On CUDA the resolve is the fused, CUDA-graph-safe one
    (``qsa_kv_offload_kernels``). The torch resolve here is the reference: eager,
    with host synchronizations; ``SGLANG_USE_QSA_KV_OFFLOAD_TORCH_RESOLVE``
    selects it on CUDA.
    """

    def __init__(
        self,
        *,
        layout: QSAKVOffloadLayout,
        k_buffers: List[torch.Tensor],
        v_buffers: List[torch.Tensor],
        host_rows: int,
        pin_memory: bool = True,
        verify: bool = False,
    ):
        self.layout = layout
        # Debug: the full K/V pool stays on the GPU; see check().
        self.verify = verify
        self._verify_logged = 0.0
        self.cr = layout.compress_ratio
        self.hot_blocks = layout.hot_blocks
        self.region = layout.region_rows
        self.tail_rows = layout.tail_rows
        self.dev_k = k_buffers
        self.dev_v = v_buffers
        self.num_layers = len(k_buffers)
        self.device = k_buffers[0].device
        row_shape = tuple(k_buffers[0].shape[1:])
        if tuple(v_buffers[0].shape[1:]) != row_shape:
            raise ValueError("QSA K/V offload needs equal K and V row shapes")
        if k_buffers[0].shape[0] < layout.device_rows:
            raise ValueError(
                f"device K/V buffers have {k_buffers[0].shape[0]} rows, the "
                f"offload layout needs {layout.device_rows}"
            )
        self.host_rows = host_rows
        dtype = k_buffers[0].dtype
        # Pageable memory of the exact size, page-locked with cudaHostRegister:
        # torch's pinned caching allocator rounds each block up to a power of two.
        dims = (self.num_layers, host_rows, *row_shape)
        self.host_k, self.host_v = (
            alloc_with_host_register(
                dims=dims,
                dtype=dtype,
                device="cpu",
                pin_memory=pin_memory,
                allocator=HostTensorAllocator(),
            )
            for _ in range(2)
        )
        self.pin_memory = pin_memory
        self.host_bytes = 2 * self.host_k.numel() * self.host_k.element_size()
        logger.info(
            "QSA K/V offload: %d layers, host K/V %.2f GB (%d rows), %d device "
            "rows (%d request slots x %d hot blocks + %d scratch rows)",
            self.num_layers,
            self.host_bytes / GB,
            host_rows,
            layout.device_rows,
            layout.num_request_slots,
            layout.hot_blocks,
            layout.scratch_rows,
        )
        self.req_to_token: Optional[torch.Tensor] = None
        # The fused resolve reads the host store through raw UVA pointers,
        # validated on CUDA only.
        self.use_fused = (
            self.device.type == "cuda"
            and not is_hip()
            and not envs.SGLANG_USE_QSA_KV_OFFLOAD_TORCH_RESOLVE.get()
        )
        self.fused = None
        self._resolve_errors_host = torch.zeros(
            1, dtype=torch.int64, pin_memory=self.use_fused
        )
        self._step = 0
        if verify:
            # [mismatching checks, checks]
            self._verify_counts = torch.zeros(2, dtype=torch.int64, device=self.device)
        self._arange = torch.arange(
            max(layout.scratch_rows, self.cr), dtype=torch.int64, device=self.device
        )

    def mirror(
        self,
        layer_idx: int,
        rows: torch.Tensor,
        loc: torch.Tensor,
        pool_k: torch.Tensor,
        pool_v: torch.Tensor,
    ) -> None:
        """Verify mode: the new K/V the full pool holds at ``loc`` -> device
        rows ``rows`` and host rows ``loc`` (no host sync: graph-capturable)."""
        loc, rows = loc.long(), rows.long()
        self.dev_k[layer_idx][rows] = pool_k[loc]
        self.dev_v[layer_idx][rows] = pool_v[loc]
        self.write_through(layer_idx=layer_idx, rows=rows, loc=loc)

    def check(self, what: str, layer_idx: int, actual, expected) -> None:
        """Verify mode: count offloaded tensors that differ from the full pool's,
        bit for bit, on the device (graph-capturable); ``poll_verify`` raises."""
        for got, ref in zip(actual, expected):
            self._verify_counts[0] += (
                got.view(torch.uint8) != ref.view(torch.uint8)
            ).any()
        self._verify_counts[1] += 1

    def poll_verify(self) -> None:
        """Verify mode: raise if any check so far differed (host sync)."""
        mismatches, checks = self._verify_counts.tolist()
        if mismatches:
            raise RuntimeError(
                f"QSA K/V offload verify: {mismatches} of {checks} checks differ "
                "from the full pool"
            )
        now = time.monotonic()
        if now - self._verify_logged >= 10:
            self._verify_logged = now
            logger.info("QSA K/V offload verify: %d checks, 0 mismatches", checks)

    def destroy(self) -> None:
        """Unpin the host K/V (see HostKVCache.destroy). Idempotent."""
        if self.verify:
            self._verify_logged = 0.0
            self.poll_verify()
        if self.fused is not None and int(self.fused.errors):
            logger.error(
                "QSA K/V offload: %d selected K/V columns had no device row",
                int(self.fused.errors),
            )
        if self.pin_memory:
            for buffer in (self.host_k, self.host_v):
                _cuda_host_unregister(buffer)
        self.pin_memory = False

    def bind(self, req_to_token: torch.Tensor) -> None:
        """Allocate the hot-cache tables for this ``req_to_token``."""
        if self.req_to_token is not None:
            return
        num_slots, width = req_to_token.shape
        if num_slots != self.layout.num_request_slots:
            raise ValueError(
                f"req_to_token has {num_slots} request slots, the QSA K/V offload "
                f"was built for {self.layout.num_request_slots}"
            )
        # Verify mode reads one snapshot of the mapping per forward
        # (snapshot_mapping); serving reads the live table, as upstream does.
        self._live_req_to_token = req_to_token
        self.req_to_token = req_to_token.clone() if self.verify else req_to_token
        self.width = width
        self.max_blocks = width // self.cr + TAIL_BLOCKS
        if self.use_fused:
            from sglang.srt.mem_cache.qsa_kv_offload_kernels import QSAKVFusedResolve

            self.fused = QSAKVFusedResolve(
                num_request_slots=num_slots,
                hot_blocks=self.hot_blocks,
                compress_ratio=self.cr,
                token_topk=self.layout.token_topk,
                width=width,
                region_rows=self.region,
                dev_k=self.dev_k,
                dev_v=self.dev_v,
                host_k=self.host_k,
                host_v=self.host_v,
                max_rows_per_request=self.layout.rows_per_request,
            )
            return
        layers, hot, cr, dev = self.num_layers, self.hot_blocks, self.cr, self.device

        def table(*shape, dtype=torch.int64):
            return torch.full(shape, -1, dtype=dtype, device=dev)

        # Block -> LRU slot, and per LRU slot its block, last use and the
        # logical slots its rows were filled from.
        self.blk_slot = table(layers, num_slots, self.max_blocks, dtype=torch.int32)
        self.slot_blk = table(layers, num_slots, hot)
        self.slot_age = table(layers, num_slots, hot)
        self.slot_src = table(layers, num_slots, hot, cr, dtype=torch.int32)
        # Tail ring: the block in each ring slot, the logical slots its rows
        # were filled from, and the end of the positions written into it.
        self.ring_blk = table(layers, num_slots, TAIL_BLOCKS)
        self.ring_src = table(layers, num_slots, TAIL_BLOCKS, cr, dtype=torch.int32)
        self.ring_end = table(layers, num_slots)

    def snapshot_mapping(self) -> None:
        """Verify mode: copy the live ``req_to_token`` once per forward. The
        scheduler rewrites a running request's mapping (radix dedup in
        ``checkpoint``) on its own stream while a forward may run, so the
        resolve and the full-pool reference must read the same copy
        (graph-capturable)."""
        if self.verify:
            self.req_to_token.copy_(self._live_req_to_token)

    def set_num_real(self, num_requests: int) -> None:
        """Requests of the next forward; CUDA-graph padding past them is skipped.
        A single device fill: graph-replay metadata prep may be captured."""
        if self.fused is not None:
            self.fused.num_real.fill_(num_requests)

    def real_rows(self, *, num_rows: int, rows_per_request: int) -> torch.Tensor:
        """Device mask of the forward's query rows that belong to real requests,
        not CUDA-graph padding (no host sync: graph-capturable)."""
        rows = torch.arange(num_rows, device=self.device)
        if self.fused is None:
            return rows >= 0
        return rows < self.fused.num_real * rows_per_request

    @property
    def graph_safe(self) -> bool:
        return self.use_fused

    def begin_forward(
        self,
        req_to_token: torch.Tensor,
        reset_req_pool_indices: Optional[torch.Tensor] = None,
    ) -> None:
        """Per-forward setup; an extend passes its requests to reset."""
        if (
            self.fused is None
            and self.device.type == "cuda"
            and torch.cuda.is_current_stream_capturing()
        ):
            raise RuntimeError(
                "The QSA K/V offload resolve runs eagerly; launch with "
                "--disable-cuda-graph"
            )
        self.bind(req_to_token)
        if reset_req_pool_indices is not None:
            self.reset(reset_req_pool_indices)
        if self.fused is not None:
            self._check_resolve_errors()
        if self.verify:
            self.poll_verify()

    def _check_resolve_errors(self) -> None:
        """Raise if the fused resolve left a selected column without a device row
        (it would not be attended). Reads the copy started at the previous eager
        forward, so it adds no host sync."""
        if int(self._resolve_errors_host[0]):
            raise RuntimeError(
                f"QSA K/V offload: {int(self._resolve_errors_host[0])} selected K/V "
                "columns had no device row"
            )
        self._resolve_errors_host.copy_(self.fused.errors, non_blocking=True)

    # ---- extend --------------------------------------------------------------

    def reset(self, req_pool_indices: torch.Tensor) -> None:
        """Empty the hot cache and tail ring of these request slots.

        Every extend calls this for its requests (new request, radix hit or
        the next prefill chunk); exact because the host holds every token.
        """
        if self.fused is not None:
            self.fused.reset(req_pool_indices)
            return
        idx = req_pool_indices.long()
        for table in (
            self.blk_slot,
            self.slot_blk,
            self.slot_age,
            self.slot_src,
            self.ring_blk,
            self.ring_src,
            self.ring_end,
        ):
            table.index_fill_(1, idx, -1)

    def extend_rows(self, num_tokens: int) -> torch.Tensor:
        """Device scratch rows that hold this extend's new K/V."""
        if num_tokens > self.layout.scratch_rows:
            raise RuntimeError(
                f"QSA K/V offload: extend of {num_tokens} tokens exceeds the "
                f"{self.layout.scratch_rows}-row scratch"
            )
        return self.layout.scratch_base + self._arange[:num_tokens]

    def write_through(
        self, layer_idx: int, rows: torch.Tensor, loc: torch.Tensor
    ) -> None:
        """Device rows ``rows`` -> host rows ``loc`` (their logical slots)."""
        if self.fused is not None:
            self.fused.write_through(layer_idx=layer_idx, rows=rows, loc=loc)
            return
        _copy_rows(
            k_dst=self.host_k[layer_idx],
            v_dst=self.host_v[layer_idx],
            dst=loc.long(),
            k_src=self.dev_k[layer_idx],
            v_src=self.dev_v[layer_idx],
            src=rows.long(),
        )

    def gather_context(
        self, layer_idx: int, slots: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Host K/V rows of these logical slots, in order, on the device."""
        rows = slots.numel()
        shape = (rows, *self.dev_k[layer_idx].shape[1:])
        k = torch.empty(shape, dtype=self.dev_k[layer_idx].dtype, device=self.device)
        v = torch.empty_like(k)
        dst = torch.arange(rows, dtype=torch.int64, device=self.device)
        _copy_rows(
            k_dst=k,
            v_dst=v,
            dst=dst,
            k_src=self.host_k[layer_idx],
            v_src=self.host_v[layer_idx],
            src=slots.long(),
        )
        return k, v

    # ---- decode / target verify ----------------------------------------------

    def new_token_rows(
        self, row_seq_lens: torch.Tensor, row_req: torch.Tensor
    ) -> torch.Tensor:
        """Ring row of each query row's new token (position ``seq_len - 1``)."""
        pos = row_seq_lens.long() - 1
        return (
            row_req.long() * self.region
            + (pos // self.cr) % TAIL_BLOCKS * self.cr
            + pos % self.cr
        )

    def resolve_rows(
        self,
        layer_idx: int,
        positions: torch.Tensor,
        row_seq_lens: torch.Tensor,
        row_req: torch.Tensor,
        rows_per_request: int,
    ) -> torch.Tensor:
        """Device row of every selected logical position.

        ``positions`` is ``[R, W]`` request-relative positions (``-1`` padding),
        the rows of one request contiguous and ``rows_per_request`` of them;
        ``row_seq_lens`` is each row's query position + 1. Returns int32
        ``[R, W]`` device rows, ``-1`` exactly where the position is not
        attended (``< 0`` or ``>= row_seq_lens``). Call after ``bind``; the
        forward's new K/V may already be stored at ``new_token_rows``.
        """
        if self.fused is not None:
            return self.fused.resolve(
                layer_idx=layer_idx,
                positions=positions,
                row_seq_lens=row_seq_lens.contiguous(),
                row_req=row_req,
                rows_per_request=rows_per_request,
                req_to_token=self.req_to_token,
            )
        cr, q = self.cr, rows_per_request
        num_rows, width = positions.shape
        if num_rows % q:
            raise ValueError(f"{num_rows} rows are not groups of {q}")
        groups = num_rows // q
        if layer_idx == 0:
            self._step += 1
        seq = row_seq_lens.long()
        req = row_req.long()
        group_req = req[::q]
        first_new = seq[::q] - 1  # committed length of each request
        self._load_ring(
            layer_idx=layer_idx,
            group_req=group_req,
            first_new=first_new,
            rows_per_request=q,
        )
        self._drop_remapped(layer_idx=layer_idx, group_req=group_req)

        pos = positions.long()
        valid = (pos >= 0) & (pos < seq[:, None])
        ring_start = torch.repeat_interleave(first_new // cr * cr, q)
        in_ring = valid & (pos >= ring_start[:, None])
        need = (valid & ~in_ring).reshape(groups, q * width)
        blk = torch.where(need, pos.reshape(groups, q * width) // cr, -1)
        blk_slot = self.blk_slot[layer_idx]
        req_x = group_req[:, None].expand_as(blk)
        slot = blk_slot[req_x, blk.clamp(min=0)].long()
        miss = need & (slot < 0)
        if bool(miss.any()):
            self._swap_in(
                layer_idx=layer_idx,
                group_req=group_req,
                blk=blk,
                slot=slot,
                need=need,
                miss=miss,
            )
            slot = blk_slot[req_x, blk.clamp(min=0)].long()
        hit_g, hit_c = need.nonzero(as_tuple=True)
        self.slot_age[layer_idx][group_req[hit_g], slot[hit_g, hit_c]] = self._step

        slot = slot.reshape(num_rows, width)
        offset = pos.remainder(cr)
        base = (req * self.region)[:, None]
        ring_rows = base + (pos // cr) % TAIL_BLOCKS * cr + offset
        lru_rows = base + self.tail_rows + slot.clamp(min=0) * cr + offset
        rows = torch.where(in_ring, ring_rows, lru_rows)
        return torch.where(valid, rows, -1).to(torch.int32)

    def _load_ring(
        self,
        layer_idx: int,
        group_req: torch.Tensor,
        first_new: torch.Tensor,
        rows_per_request: int,
    ) -> None:
        """Make each request's tail ring hold the committed rows of block
        ``first_new // cr`` and take the forward's new rows of that block and
        the next one."""
        cr = self.cr
        offsets = self._arange[:cr]
        block = first_new // cr
        ring_slot = block % TAIL_BLOCKS
        committed = offsets[None, :] < (first_new % cr)[:, None]
        cols = block[:, None] * cr + offsets
        src0 = self.req_to_token[group_req[:, None], cols.clamp(max=self.width - 1)]
        cols1 = cols + cr
        src1 = self.req_to_token[group_req[:, None], cols1.clamp(max=self.width - 1)]
        ring_blk = self.ring_blk[layer_idx]
        ring_src = self.ring_src[layer_idx]
        held = (
            (ring_blk[group_req, ring_slot] == block)
            & (first_new <= self.ring_end[layer_idx][group_req])
            & ((ring_src[group_req, ring_slot] == src0) | ~committed).all(1)
        )
        load = committed & ~held[:, None]
        if bool(load.any()):
            load_g, load_c = load.nonzero(as_tuple=True)
            dst = (
                group_req[load_g] * self.region
                + ring_slot[load_g] * cr
                + offsets[load_c]
            )
            _copy_rows(
                k_dst=self.dev_k[layer_idx],
                v_dst=self.dev_v[layer_idx],
                dst=dst,
                k_src=self.host_k[layer_idx],
                v_src=self.host_v[layer_idx],
                src=src0[load_g, load_c].long(),
            )
        # A block that re-enters the ring is rewritten there: drop its LRU copy.
        for b in (block, block + 1):
            slot = self.blk_slot[layer_idx][group_req, b.clamp(max=self.max_blocks - 1)]
            hit = (slot >= 0).nonzero(as_tuple=True)[0]
            if hit.numel():
                self.slot_src[layer_idx][group_req[hit], slot[hit].long()] = -1
        other = 1 - ring_slot
        ring_blk[group_req, ring_slot] = block
        ring_blk[group_req, other] = block + 1
        ring_src[group_req, ring_slot] = src0
        ring_src[group_req, other] = src1
        self.ring_end[layer_idx][group_req] = first_new + rows_per_request

    def _drop_remapped(self, layer_idx: int, group_req: torch.Tensor) -> None:
        """Invalidate LRU slots whose positions no longer map to the logical
        slots they were filled from."""
        slot_blk = self.slot_blk[layer_idx][group_req]
        live = slot_blk >= 0
        cols = slot_blk.clamp(min=0)[..., None] * self.cr + self._arange[: self.cr]
        current = self.req_to_token[
            group_req[:, None, None], cols.clamp(max=self.width - 1)
        ]
        stale = live & (current != self.slot_src[layer_idx][group_req]).any(-1)
        stale_g, stale_s = stale.nonzero(as_tuple=True)
        if stale_g.numel():
            req = group_req[stale_g]
            self.blk_slot[layer_idx][req, slot_blk[stale_g, stale_s]] = -1
            self.slot_blk[layer_idx][req, stale_s] = -1
            self.slot_age[layer_idx][req, stale_s] = -1
            self.slot_src[layer_idx][req, stale_s] = -1

    def _swap_in(
        self,
        layer_idx: int,
        group_req: torch.Tensor,
        blk: torch.Tensor,
        slot: torch.Tensor,
        need: torch.Tensor,
        miss: torch.Tensor,
    ) -> None:
        """Copy every missing block of each request into its least recently
        used slots that this forward does not read."""
        cr, groups = self.cr, blk.shape[0]
        group_ids = torch.arange(groups, device=blk.device)[:, None].expand_as(blk)
        key = torch.unique((group_ids * self.max_blocks + blk)[miss])
        miss_g = key // self.max_blocks
        miss_blk = key % self.max_blocks
        counts = torch.bincount(miss_g, minlength=groups)
        rank = (
            torch.arange(key.numel(), device=blk.device)
            - (torch.cumsum(counts, 0) - counts)[miss_g]
        )
        age = self.slot_age[layer_idx][group_req].clone()
        pinned = torch.iinfo(torch.int64).max
        hit_g, hit_c = (need & ~miss).nonzero(as_tuple=True)
        age[hit_g, slot[hit_g, hit_c]] = pinned
        if bool((counts > (age != pinned).sum(1)).any()):
            raise RuntimeError(
                f"QSA K/V offload: {self.hot_blocks} hot blocks cannot hold this "
                "forward's selection"
            )
        victim = torch.argsort(age, dim=1, stable=True)[miss_g, rank]
        req = group_req[miss_g]
        old = self.slot_blk[layer_idx][req, victim]
        evicted = old >= 0
        self.blk_slot[layer_idx][req[evicted], old[evicted]] = -1
        self.slot_blk[layer_idx][req, victim] = miss_blk
        self.blk_slot[layer_idx][req, miss_blk] = victim.to(torch.int32)
        offsets = self._arange[:cr]
        src = self.req_to_token[req[:, None], miss_blk[:, None] * cr + offsets]
        self.slot_src[layer_idx][req, victim] = src
        dst = req[:, None] * self.region + self.tail_rows + victim[:, None] * cr
        _copy_rows(
            k_dst=self.dev_k[layer_idx],
            v_dst=self.dev_v[layer_idx],
            dst=(dst + offsets).reshape(-1),
            k_src=self.host_k[layer_idx],
            v_src=self.host_v[layer_idx],
            src=src.long().reshape(-1),
        )
