"""SGLANG_ENABLE_HICACHE_RANK_SHARD: a rank-sharded HiCache host tier for replicated MLA/DSA KV.

Under TP without DP attention every TP rank holds the same MLA latent KV and DSA
index-K, so each rank backs up and loads back the same bytes. With the flag on,
each host layer has one owner rank (round-robin, below):

- backup (D2H): a rank copies only the layers it owns, into a host buffer that
  holds only those layers, so the same host budget holds ~tp_size x more tokens;
- load-back (H2D): a rank copies only the layers it owns from its host buffer,
  then every loaded layer is broadcast from its owner to the other TP ranks over
  a dedicated NCCL group, page by page into each rank's own device slots.

Tree, host-slot allocation and device allocation are untouched and stay
replicated: only the bytes are sharded. The exchange runs on its own stream in
layer order (the order every rank issues load-backs in), and each layer's load
event is recorded after that layer's exchange, so LayerDoneCounter keeps its
per-layer contract.

Forward gate: the shard group's broadcasts and the forward's collectives (TP
all-reduces, custom AR) are cross-rank blocking kernels on different
communicators; NCCL requires them to run at different times, in one order on
every rank. So before any forward kernel, the scheduler thread waits until the
whole exchange is enqueued (CPU gate), then makes the forward stream wait for
its end (GPU gate); see L2TransferEngine.gate_rank_shard_forward. Invariants:
  I1: on every rank, no command that waits on the exchange is pushed before the
      exchange's last command (a shared hardware queue could stall it);
  I2: no other cross-rank blocking kernel is runnable while a shard-group kernel
      is: the load fence (start_loading) holds the exchange behind every forward
      already enqueued, the gate holds every later forward behind the exchange.
"""

from __future__ import annotations

import logging
import threading
import time
import zlib
from collections import deque
from typing import Any, Callable, Optional

import msgspec
import torch

from sglang.srt.environ import envs
from sglang.srt.utils import get_device_module

logger = logging.getLogger(__name__)
device_module = get_device_module()

KV = "kv"
INDEXER = "indexer"

_logged_once: set = set()
_logged_once_lock = threading.Lock()


def log_once(key: str, msg: str, *args) -> None:
    with _logged_once_lock:
        if key in _logged_once:
            return
        _logged_once.add(key)
    logger.info(msg, *args)


class RankShardSpec(msgspec.Struct, frozen=True):
    """Which rank owns which host layer. Rank-independent apart from ``rank``."""

    rank: int
    size: int
    # Keep the unsharded host token capacity (host bytes shrink instead).
    keep_capacity: bool = False

    def owner(self, kind: str, host_layer: int) -> int:
        # Round-robin; index-K starts at the last rank, which is one KV layer
        # short when the KV layer count is not a multiple of size.
        offset = self.size - 1 if kind == INDEXER else 0
        return (host_layer + offset) % self.size

    def owned_layers(
        self, kind: str, num_layers: int, rank: Optional[int] = None
    ) -> list[int]:
        rank = self.rank if rank is None else rank
        if num_layers < self.size:
            # A rank with no layers would need a zero-size host buffer.
            raise ValueError(
                f"SGLANG_ENABLE_HICACHE_RANK_SHARD needs at least one {kind} host "
                f"layer per TP rank ({self.size}), got {num_layers}; unset it"
            )
        return [h for h in range(num_layers) if self.owner(kind, h) == rank]

    def max_owned(self, kind: str, num_layers: int) -> int:
        return max(
            len(self.owned_layers(kind, num_layers, r)) for r in range(self.size)
        )


def _ineligible_reasons(kv_pool) -> list[str]:
    from sglang.srt.layers.dp_attention import is_dp_attention_enabled
    from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool, MLATokenToKVPoolFP4
    from sglang.srt.runtime_context import get_disagg, get_memory, get_parallel
    from sglang.srt.utils import is_cuda

    parallel = get_parallel()
    memory = get_memory()
    reasons = []
    if not is_cuda():
        reasons.append("not CUDA")
    if parallel.tp_size <= 1:
        reasons.append("tp_size=1")
    if is_dp_attention_enabled() or parallel.attn_cp_size != 1:
        reasons.append("DP attention or attention CP (KV not replicated over TP)")
    if parallel.pp_size != 1:
        reasons.append(f"pp_size={parallel.pp_size}")
    if parallel.dcp_enabled:
        reasons.append("DCP")
    if parallel.nnodes != 1:
        reasons.append(f"nnodes={parallel.nnodes}")
    if memory.hicache_io_backend != "direct":
        reasons.append(f"io_backend={memory.hicache_io_backend!r} (direct only)")
    if memory.hicache_mem_layout != "page_first_direct":
        reasons.append(
            f"mem_layout={memory.hicache_mem_layout!r} (page_first_direct only)"
        )
    if memory.hicache_storage_backend is not None:
        reasons.append("storage backend attached (L2 only)")
    if memory.enable_hisparse:
        reasons.append("hisparse")
    if get_disagg().disaggregation_mode == "decode":
        reasons.append("disaggregation decode")
    if not isinstance(kv_pool, MLATokenToKVPool) or isinstance(
        kv_pool, MLATokenToKVPoolFP4
    ):
        reasons.append(f"{type(kv_pool).__name__} (MLA/DSA non-FP4 only)")
    elif kv_pool.layer_shard_enabled:
        reasons.append("device layer shard")
    return reasons


def resolve_rank_shard_spec(kv_pool) -> Optional[RankShardSpec]:
    """The spec for this rank, or None (then every path is unchanged).

    Every condition is configuration only, so all ranks decide alike.
    """
    if not envs.SGLANG_ENABLE_HICACHE_RANK_SHARD.get():
        return None
    reasons = _ineligible_reasons(kv_pool)
    if reasons:
        logger.info(
            "HiCache rank shard requested (SGLANG_ENABLE_HICACHE_RANK_SHARD=1) "
            "but inactive: %s",
            ", ".join(reasons),
        )
        return None
    from sglang.srt.runtime_context import get_parallel

    parallel = get_parallel()
    return RankShardSpec(
        rank=parallel.tp_rank,
        size=parallel.tp_size,
        keep_capacity=envs.SGLANG_HICACHE_RANK_SHARD_KEEP_CAPACITY.get(),
    )


def is_rank_sharded(host_pool) -> bool:
    from sglang.srt.mem_cache.pool_host.base import HostKVCache

    return isinstance(host_pool, HostKVCache) and host_pool.rank_shard is not None


def is_rank_shard_load(transfers: list) -> bool:
    return any(is_rank_sharded(t.host_pool) for t in transfers)


def rank_sharded_host_pools(mem_pool_host) -> list:
    from sglang.srt.mem_cache.pool_host.group import HostPoolGroup

    if isinstance(mem_pool_host, HostPoolGroup):
        pools = [entry.host_pool for entry in mem_pool_host.entries]
    else:
        pools = [mem_pool_host]
    return [p for p in pools if is_rank_sharded(p)]


def _to_device_async(cpu_tensor: torch.Tensor, device) -> torch.Tensor:
    if torch.device(device).type == "cuda":
        return cpu_tensor.pin_memory().to(device, non_blocking=True)
    return cpu_tensor.to(device)


# Device-side steps of one exchange chunk (module level so a CPU simulation can
# order them on its fake streams).
def gather_rows(rows: torch.Tensor, index: torch.Tensor, out: torch.Tensor) -> None:
    torch.index_select(rows, 0, index, out=out)


def scatter_rows(rows: torch.Tensor, index: torch.Tensor, src: torch.Tensor) -> None:
    rows.index_copy_(0, index, src)


class _Stats:
    def __init__(self, log_every: int):
        self.log_every = max(0, log_every)
        self.total_loads = 0
        self._reset()

    def _reset(self):
        self.loads = 0
        self.items = 0
        self.items_owned = 0
        self.pages = 0
        self.broadcasts = 0
        self.bytes_sent = 0
        self.bytes_received = 0
        self.issue_s = 0.0

    def line(self) -> str:
        loads = max(1, self.loads)
        return (
            f"HiCache rank shard: {self.loads} loads; {self.items} layer exchanges "
            f"({self.items_owned} loaded from this rank's host shard), "
            f"{self.pages / loads:.0f} pages per load, {self.broadcasts} broadcasts; "
            f"sent {self.bytes_sent / 1e9:.2f} GB, received "
            f"{self.bytes_received / 1e9:.2f} GB via broadcast; exchange issue "
            f"{1e3 * self.issue_s / loads:.1f} ms per load"
        )


class _GateStats:
    """Forward-gate counters (scheduler thread), logged once per log_every gates."""

    def __init__(self, log_every: int):
        self.log_every = max(0, log_every)
        self.total = 0
        self._pending = deque()  # (mark, done) timing pairs not yet complete
        self._reset()

    def _reset(self):
        self.gates = 0
        self.loads = 0
        self.host_s = 0.0
        self.host_max_s = 0.0
        self.gpu_n = 0
        self.gpu_idle = 0
        self.gpu_ms = 0.0
        self.gpu_max_ms = 0.0

    def note(self, loads: int, host_s: float, mark, done) -> None:
        self.total += 1
        self.gates += 1
        self.loads += loads
        self.host_s += host_s
        self.host_max_s = max(self.host_max_s, host_s)
        self._pending.append((mark, done))
        self._harvest()
        if self.total == 1:
            logger.info(
                "HiCache rank shard: forward gated on the whole exchange: the "
                "scheduler waited %.1f ms for the exchange to be enqueued, then the "
                "forward stream waits for its last broadcast, so no shard-group op "
                "runs beside a forward collective",
                1e3 * host_s,
            )
        if self.log_every and self.gates >= self.log_every:
            logger.info(self.line())
            self._reset()

    def _harvest(self) -> None:
        # GPU wait: how long the forward stream sat at the gate (<= 0: none).
        while self._pending:
            mark, done = self._pending[0]
            if not (mark.query() and done.query()):
                return
            self._pending.popleft()
            gap_ms = mark.elapsed_time(done)
            self.gpu_n += 1
            if gap_ms > 0:
                self.gpu_idle += 1
                self.gpu_ms += gap_ms
                self.gpu_max_ms = max(self.gpu_max_ms, gap_ms)

    def line(self) -> str:
        return (
            f"HiCache rank shard gate: {self.gates} forwards gated on {self.loads} "
            f"loads; host wait mean {1e3 * self.host_s / max(1, self.gates):.1f} ms "
            f"max {1e3 * self.host_max_s:.1f} ms; GPU wait mean "
            f"{self.gpu_ms / max(1, self.gpu_n):.1f} ms max {self.gpu_max_ms:.1f} ms "
            f"over {self.gpu_n} measured ({self.gpu_idle} waited)"
        )


class RankShardExchange:
    """Broadcasts each loaded host layer from its owner over a dedicated group."""

    def __init__(
        self,
        spec: RankShardSpec,
        group,
        group_ranks: list[int],
        device,
        staging_bytes: int,
        verify_group=None,
    ):
        self.spec = spec
        self.group = group
        self.group_ranks = list(group_ranks)
        self.device = device
        self.staging = torch.empty(staging_bytes, dtype=torch.uint8, device=device)
        self.stream = device_module.Stream()
        self.stats = _Stats(envs.SGLANG_HICACHE_RANK_SHARD_LOG_EVERY.get())
        self.verify_every = max(0, envs.SGLANG_HICACHE_RANK_SHARD_VERIFY_EVERY.get())
        self.verify_group = verify_group
        self._verify_calls = 0
        # Forward gate. submitted/gated: scheduler thread; finished/done_event:
        # whichever thread runs finish() (read by the scheduler after the CPU gate).
        self.submitted = 0
        self.finished = 0
        self.gated = 0
        self.done_event = None
        self.gate_stats = _GateStats(envs.SGLANG_HICACHE_RANK_SHARD_LOG_EVERY.get())

    def gate_forward(self, stream, host_wait_s: float) -> None:
        """GPU gate, after the caller's CPU gate: `stream` waits for every exchange."""
        if self.finished != self.submitted or self.done_event is None:
            raise RuntimeError(
                f"HiCache rank shard: forward gate with {self.submitted} sharded "
                f"loads submitted but {self.finished} exchanges enqueued"
            )
        mark = device_module.Event(enable_timing=True)
        mark.record(stream)
        stream.wait_event(self.done_event)
        loads = self.submitted - self.gated
        self.gated = self.submitted
        self.gate_stats.note(loads, host_wait_s, mark, self.done_event)

    @classmethod
    def build(cls, pools: list, tp_group, device) -> RankShardExchange:
        """Collective on every TP rank: create the group and warm every root."""
        from sglang.srt.distributed.parallel_state import create_custom_parallel_group

        spec = pools[0].rank_shard
        assert all(p.rank_shard == spec for p in pools), "rank shard specs differ"
        group_ranks = list(torch.distributed.get_process_group_ranks(tp_group))
        if (
            len(group_ranks) != spec.size
            or group_ranks.index(torch.distributed.get_rank()) != spec.rank
        ):
            raise RuntimeError(
                f"HiCache rank shard: TP group ranks {group_ranks} do not match "
                f"rank {spec.rank} of {spec.size}"
            )
        group = create_custom_parallel_group(group_ranks=group_ranks, backend="nccl")
        verify_group = None
        if envs.SGLANG_HICACHE_RANK_SHARD_VERIFY_EVERY.get() > 0:
            verify_group = create_custom_parallel_group(
                group_ranks=group_ranks, backend="gloo"
            )
        staging_mb = max(1, envs.SGLANG_HICACHE_RANK_SHARD_STAGING_MB.get())
        exchange = cls(spec, group, group_ranks, device, staging_mb << 20, verify_group)
        exchange.warmup()
        logger.info(
            "HiCache rank shard enabled (SGLANG_ENABLE_HICACHE_RANK_SHARD=1): rank %d "
            "of %d; %s; load-back exchange over a dedicated NCCL group (warm), "
            "staging %d MB",
            spec.rank,
            spec.size,
            "; ".join(p.rank_shard_summary() for p in pools),
            staging_mb,
        )
        return exchange

    def warmup(self) -> None:
        buf = self.staging[:1]
        for src in self.group_ranks:
            torch.distributed.broadcast(buf, src=src, group=self.group)
        if torch.device(self.device).type == "cuda":
            device_module.synchronize()

    def maybe_verify(self, host_indices: torch.Tensor) -> None:
        """Every N load-backs (scheduler thread, all ranks): same host pages?"""
        if self.verify_group is None:
            return
        self._verify_calls += 1
        if self._verify_calls % self.verify_every:
            return
        cpu = host_indices.cpu().contiguous()
        mine = torch.tensor(
            [cpu.numel(), zlib.crc32(cpu.numpy().tobytes())], dtype=torch.int64
        )
        every = [torch.empty_like(mine) for _ in range(self.spec.size)]
        torch.distributed.all_gather(every, mine, group=self.verify_group)
        if any(not torch.equal(x, mine) for x in every):
            raise RuntimeError(
                "HiCache rank shard: load-back host pages differ across TP ranks "
                f"(load {self._verify_calls}: {[x.tolist() for x in every]})"
            )
        log_once(
            "verify",
            "HiCache rank shard verify passed: load-back host pages match on all "
            "%d ranks (checked every %d loads)",
            self.spec.size,
            self.verify_every,
        )

    def begin_load(self, transfers: list, h2d_stream) -> RankShardLoad:
        return RankShardLoad(self, transfers, h2d_stream)

    def exchange(self, rows: torch.Tensor, page_index: torch.Tensor, owner: int):
        """Enqueue (on the current stream) owner -> all copies of rows[page_index]."""
        row_bytes = rows.shape[1]
        per_chunk = self.staging.numel() // row_bytes
        if per_chunk <= 0:
            raise RuntimeError(
                f"HiCache rank shard staging ({self.staging.numel()} B) is smaller "
                f"than one page ({row_bytes} B)"
            )
        is_owner = owner == self.spec.rank
        src = self.group_ranks[owner]
        num = page_index.numel()
        stats = self.stats
        for start in range(0, num, per_chunk):
            index = page_index[start : start + per_chunk]
            count = index.numel()
            buf = self.staging[: count * row_bytes].view(count, row_bytes)
            if is_owner:
                gather_rows(rows, index, buf)
            torch.distributed.broadcast(buf, src=src, group=self.group)
            if not is_owner:
                scatter_rows(rows, index, buf)
            stats.broadcasts += 1
            if is_owner:
                stats.bytes_sent += count * row_bytes
            else:
                stats.bytes_received += count * row_bytes


class RankShardLoad:
    """One load-back: per layer, broadcast each sharded transfer's layer rows."""

    def __init__(self, exchange: RankShardExchange, transfers: list, h2d_stream):
        self.ex = exchange
        self.transfers = transfers
        self.h2d = h2d_stream
        self.primary = transfers[0] if transfers else None
        self.issue_s = 0.0
        self.items = 0
        self.pages = 0
        stream = exchange.stream
        self._sharded = [is_rank_sharded(t.host_pool) for t in transfers]
        self._page_index: dict[int, torch.Tensor] = {}
        with device_module.stream(stream):
            # Covers the start event and load fence already on h2d.
            stream.wait_stream(h2d_stream)
            for t, sharded in zip(transfers, self._sharded):
                if not sharded:
                    continue
                key = id(t.device_indices)
                if key in self._page_index:
                    continue
                page = t.host_pool.page_size
                di = t.device_indices
                if di.numel() % page:
                    raise ValueError(
                        f"HiCache rank shard needs page-aligned device indices, got "
                        f"{di.numel()} slots for page_size={page}"
                    )
                pages = di.cpu()[::page] // page
                self._page_index[key] = _to_device_async(pages, exchange.device)
                self.pages = max(self.pages, pages.numel())

    def _mapped(self, t, layer_id: int) -> Optional[int]:
        # Same skip rule as L2TransferEngine._load_layer.
        local = t.layer_mapper(layer_id) if t.layer_mapper is not None else layer_id
        if local is None or (
            t is not self.primary
            and t.layer_mapper is None
            and layer_id >= t.host_pool.layer_num
        ):
            return None
        return local

    def exchange_layer(
        self, layer_id: int, on_layer_done: Optional[Callable[[int], None]]
    ) -> None:
        t0 = time.perf_counter()
        ex = self.ex
        rank = ex.spec.rank
        items = []
        wait_h2d = False
        for t, sharded in zip(self.transfers, self._sharded):
            local = self._mapped(t, layer_id)
            if local is None:
                continue
            if not sharded:
                wait_h2d = True  # an unsharded pool copied this layer on h2d
                continue
            item = t.host_pool.rank_shard_exchange_rows(
                t.device_pool, local, is_draft=t.is_draft
            )
            if item is None:
                continue
            owner, rows = item
            wait_h2d = wait_h2d or owner == rank
            items.append((owner, rows, self._page_index[id(t.device_indices)]))
        h2d_done = None
        if wait_h2d:
            h2d_done = device_module.Event()
            h2d_done.record(self.h2d)
        stream = ex.stream
        with device_module.stream(stream):
            if h2d_done is not None:
                stream.wait_event(h2d_done)
            for owner, rows, page_index in items:
                ex.exchange(rows, page_index, owner)
                self.items += 1
                if owner == rank:
                    ex.stats.items_owned += 1
            if on_layer_done is not None:
                on_layer_done(layer_id)
        self.issue_s += time.perf_counter() - t0

    def finish(self) -> None:
        # The exchange's last command: the forward gate waits for this event.
        done = device_module.Event(enable_timing=True)
        done.record(self.ex.stream)
        # The caller records its ack on h2d next: order it after every exchange.
        self.h2d.wait_stream(self.ex.stream)
        self.ex.done_event = done
        self.ex.finished += 1
        stats = self.ex.stats
        stats.loads += 1
        stats.total_loads += 1
        stats.items += self.items
        stats.pages += self.pages
        stats.issue_s += self.issue_s
        if stats.total_loads == 1:
            logger.info(
                "HiCache rank shard engaged: the first load-back exchanged %d layers "
                "(%d pages each) from their owner ranks over the NCCL group",
                self.items,
                self.pages,
            )
        if stats.log_every and stats.loads >= stats.log_every:
            logger.info(stats.line())
            stats._reset()


def host_layers_desc(layers: list[int]) -> str:
    return "[" + ",".join(str(x) for x in layers) + "]"


def shard_summary(kind: str, pool: Any) -> str:
    return (
        f"{kind} host layers {host_layers_desc(pool.shard_host_layer_ids)} "
        f"({pool.shard_layer_num} of {pool.layer_num})"
    )
