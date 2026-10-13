"""LayerSplit transfers: two fixed buffers, one shared PG scheduler.

The native controller's workers enter prefetch/backup here. This engine owns
plans, window execution, L2 copies, L3 I/O and operation completion. Only buffer
layout and cross-rank communication scheduling live in helper classes.

The optional HybridCacheController attachment implements StorageCoordination;
prefetch admission, request-release gates and backup ACK retirement remain
scheduler-owned.
Backup execution starts directly from the native worker FIFO.
"""

from __future__ import annotations

import hashlib
import logging
import os
import signal
import time
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import astuple, dataclass, field, replace
from typing import Any, Callable, Optional, Sequence

import psutil
import torch
import torch.distributed as dist

from sglang.srt.managers.cache_controller import PrefetchAck
from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.layer_split.layer_split_config import StagingBufferConfig
from sglang.srt.mem_cache.layer_split.layer_split_host_view import LayerSplitHostView
from sglang.srt.mem_cache.layer_split.layer_split_plan import (
    EXCHANGE_COMPONENTS,
    PageTransferPlan,
    build_exchange_rounds,
    build_transfer_plan,
)
from sglang.srt.mem_cache.layer_split.layer_split_staging_pool import FixedStageBuffer
from sglang.srt.mem_cache.layer_split.layer_split_utils import (
    BACKUP,
    LOGICAL_COMPONENT_POOLS,
    PREFETCH,
    STAGE_NAMES,
    SharedWindowPGPool,
    WindowJob,
    complete_page_mask,
    owned_layer_range,
)

logger = logging.getLogger(__name__)


@dataclass
class StagingPrefetchOp:
    """Local input/result of a serial prefetch; the native ACK publishes progress."""

    plan: PageTransferPlan
    host_indices: Any = field(default=None, repr=False)
    published_pages: int = 0
    truncation_reason: Optional[str] = None


@dataclass
class StagingBackupOp:
    plan: PageTransferPlan
    written_pages: int = 0
    skipped_pages: int = 0


@dataclass
class _Stage:
    """Resources owned by one direction; both stages use the engine's PG pool."""

    buffer: FixedStageBuffer
    executor: ThreadPoolExecutor
    active: bool = False
    io: Future | None = None
    # Work plus tensor references must survive any exception/timeout.
    works: list = field(default_factory=list)


class LayerSplitTransferEngine:
    """Own both transfer paths, with separate memory and one communication pool.

    Native workers enter prefetch()/backup(); their run methods walk windows.
    prefetch_window()/backup_window() order the same I/O, copy and communication
    primitives differently. Scheduler coordination admits prefetch allocations
    and decides when L2 resources can be released; backup has no extra admission.
    """

    def __init__(
        self,
        rank: int,
        *,
        staging_buffer_config: StagingBufferConfig,
        controller=None,
        storage_backend=None,
        layer_count=None,
        exchange_ranks=None,
        l2_pools=None,
        pin_memory=None,
        component_layer_counts=None,
    ):
        self._controller = controller
        self.host_view = None
        self._outstanding_backups = {}
        self._backup_drops = 0
        self._write_ack_stall_since = None
        self.rank = rank
        self.staging_buffer_config = staging_buffer_config
        # Validate the resolved, immutable layout once before any transfer runs.
        staging_buffer_config.require_host_layout()
        self.storage_backend = storage_backend
        self.layer_count = layer_count
        self.component_layer_counts = component_layer_counts
        self.l2_pools = dict(l2_pools or {})
        self.pin_memory = (
            torch.cuda.is_available() if pin_memory is None else pin_memory
        )
        self.pg_pool = (
            SharedWindowPGPool(staging_buffer_config, exchange_ranks)
            if exchange_ranks is not None
            else None
        )
        self.stages: dict[int, _Stage] = {}
        self.backup_ops_run = 0
        self.backup_pages_written = 0
        self.backup_pages_skipped = 0

    @classmethod
    def for_controller(cls, controller, config):
        """Create the optional engine without starting workers or touching L3."""
        host = LayerSplitHostView(controller.mem_pool_host)
        config = config.with_host_layout(
            page_size=host.page_size, shard_size=host.shard_size
        )
        engine = cls(
            host.shard_rank,
            staging_buffer_config=config,
            controller=controller,
        )
        engine.host_view = host
        return engine

    @staticmethod
    def configure_storage(config):
        # Must happen before backend construction/host registration.
        return replace(
            config, external_buffer_pools=(str(PoolName.KV), str(PoolName.INDEXER))
        )

    def attach(self):
        if self._controller is not None:
            cc = self._controller
            cfg = self.staging_buffer_config
            digest = self._identity(astuple(cfg))
            agreed = torch.tensor([digest, -digest], dtype=torch.int64)
            self._reduce_control(agreed, torch.distributed.ReduceOp.MIN)
            if agreed[0].item() != -agreed[1].item():
                raise ValueError(
                    "LayerSplit storage configuration differs across ranks"
                )
            ranks = list(torch.distributed.get_process_group_ranks(cc.attn_cp_group))
            if len(ranks) != self.host_view.shard_size:
                raise ValueError(
                    "LayerSplit storage requires the device pool's CP group"
                )
            logger.info(
                "LayerSplit shared-PG staging: %s; one fixed window per direction", cfg
            )
            self.storage_backend = cc.storage_backend
            self.layer_count = self.host_view.layer_count
            self.component_layer_counts = self.host_view.component_layer_counts
            self.l2_pools = dict(self.host_view.shard_views)
            self.pg_pool = SharedWindowPGPool(cfg, ranks)
        self._initialize_stages()
        self.pg_pool.attach()

    def _initialize_stages(self):
        cfg = self.staging_buffer_config
        counts = tuple(
            hi - lo
            for lo, hi in (
                owned_layer_range(r, cfg.shard_size, self.layer_count)
                for r in range(cfg.shard_size)
            )
        )
        if self.component_layer_counts is None:
            self.component_layer_counts = {c: counts for c in EXCHANGE_COMPONENTS}
        if set(self.l2_pools) != set(EXCHANGE_COMPONENTS):
            raise ValueError("Both DSA host shard views are required")
        for name, view in self.l2_pools.items():
            if (
                view.component != name
                or view.owned_layers != self.component_layer_counts[name][self.rank]
                or view.page_size != cfg.page_size
            ):
                raise ValueError("Host shard view differs from shared staging geometry")
        row_bytes = {c: self.l2_pools[c].row_bytes for c in EXCHANGE_COMPONENTS}
        self.pg_pool.geometry = tuple(
            (c, self.component_layer_counts[c], row_bytes[c])
            for c in EXCHANGE_COMPONENTS
        )
        for lane in (PREFETCH, BACKUP):
            buffer = FixedStageBuffer(
                lane,
                cfg,
                self.component_layer_counts,
                row_bytes,
                self.rank,
                self.pin_memory,
            )
            self.stages[lane] = _Stage(
                buffer,
                ThreadPoolExecutor(
                    max_workers=1, thread_name_prefix=f"l3-{STAGE_NAMES[lane]}"
                ),
            )
            for component in EXCHANGE_COMPONENTS:
                self.storage_backend.register_mem_host_pool_v2(
                    buffer.components[component], buffer.names[component]
                )

    @staticmethod
    def prepare_prefetch(operation):
        """Attach staging-only request metadata before the native query queue."""
        if operation.pool_transfers is None:
            operation.pool_transfers = [
                PoolTransfer(PoolName.INDEXER, indices_from_pool=PoolName.KV)
            ]
        operation.pool_transfers_done = False

    def build_plan(self, *, op_id, request_id, page_hashes: Sequence[str]):
        """Both directions use the same full-page/window geometry.

        The controller supplies the already-admitted page list. Prefetch query
        and L2 allocation have already truncated their common hit prefix.
        """
        return build_transfer_plan(
            op_id=op_id,
            request_id=request_id,
            page_hashes=page_hashes,
            staging_buffer_config=self.staging_buffer_config,
        )

    def prefetch_run(self, op, *, stop_requested: Callable[[], bool] = lambda: False):
        with self._transfer_errors("prefetch"):
            with self._operation(PREFETCH, op.plan):
                windows = op.plan.windows()
                for window in windows:
                    copied, stop_agreed = self.prefetch_window(
                        op.plan,
                        window,
                        op.host_indices,
                        stop_requested=stop_requested,
                    )
                    op.published_pages = window.page_start + copied
                    if copied < window.page_count:
                        op.truncation_reason = "get"
                        break
                    if stop_agreed and window.index != len(windows) - 1:
                        op.truncation_reason = "aborted"
                        break
            return op

    def backup_run(self, op, host_indices):
        # BACKUP_ORDER_CONTRACT: the single native backup worker preserves the
        # upstream operation order. A common plan then fixes every window/round.
        with self._transfer_errors("backup"):
            with self._operation(BACKUP, op.plan):
                for window in op.plan.windows():
                    written, skipped = self.backup_window(op.plan, window, host_indices)
                    op.written_pages += written
                    op.skipped_pages += skipped
            self.backup_ops_run += 1
            self.backup_pages_written += op.written_pages
            self.backup_pages_skipped += op.skipped_pages
            return op

    def assert_idle(self):
        if self._outstanding_backups:
            raise RuntimeError(
                "Cannot reset/detach before backup completions are consumed"
            )
        if any(stage.active for stage in self.stages.values()):
            raise RuntimeError("Cannot clean up shared staging with active operations")
        self.pg_pool.assert_idle()

    def reset(self):
        self.assert_idle()
        # Idle fixed buffers and live process groups are reused.
        self._write_ack_stall_since = None

    def detach(self):
        self.assert_idle()
        self.pg_pool.close()
        for stage in self.stages.values():
            stage.executor.shutdown(wait=True)
        self.stages.clear()
        self.pg_pool = None
        self.storage_backend = None
        self.host_view = None
        self.l2_pools.clear()
        self._controller = None

    @contextmanager
    def _transfer_errors(self, direction):
        try:
            yield
        except Exception:
            # Do not retire buffers on failure: native I/O may still be using them.
            self.pg_pool.abort()
            logger.exception(
                "LayerSplit %s failed; requesting service shutdown", direction
            )
            try:
                psutil.Process(os.getppid()).send_signal(signal.SIGQUIT)
            except Exception:
                logger.exception("Could not signal the parent process")
            raise

    # Shared window mechanics and the two directional execution sequences.
    @contextmanager
    def _operation(self, lane, plan):
        """One operation per direction; retire only on successful return.

        There is deliberately no finally-cleanup: an exception does not prove
        native I/O has stopped using the fixed buffers. The run entrypoint handles
        fail-stop and leaves this operation active until process exit.
        """
        stage = self.stages[lane]
        if plan.pages_per_window != self.staging_buffer_config.pages_per_window:
            raise ValueError("Plan window size differs from fixed buffer capacity")
        stage.active = True
        yield
        stage.io = None
        stage.works.clear()
        stage.active = False

    def _transfers(self, stage, plan, window, ordinals):
        # A page's owner-local ordinal equals its round index under round-robin ownership.
        positions = [(o - window.page_start) // plan.shard_size for o in ordinals]
        indices = torch.tensor(positions, dtype=torch.int64)
        return [
            PoolTransfer(
                name=LOGICAL_COMPONENT_POOLS[c],
                keys=[plan.page_hashes[o] for o in ordinals],
                host_indices=indices,
                buffer_pool_name=stage.buffer.names[c],
                indices_from_pool=PoolName.KV if c == "indexer" else None,
            )
            for c in EXCHANGE_COMPONENTS
        ]

    def _start_io(self, lane, plan, window, ordinals):
        stage = self.stages[lane]
        if not ordinals:
            # A short window may assign this rank no L3 pages. Its local I/O
            # is already done, but it must still join the window's exchange.
            future = Future()
            future.set_result({name: [] for name in LOGICAL_COMPONENT_POOLS.values()})
        else:
            transfers = self._transfers(stage, plan, window, ordinals)
            call = (
                self.storage_backend.batch_get_v2
                if lane == PREFETCH
                else self.storage_backend.batch_set_v2
            )
            future = stage.executor.submit(call, transfers)
        stage.io = future
        timeout = (
            self.staging_buffer_config.get_timeout_s
            if lane == PREFETCH
            else self.staging_buffer_config.set_timeout_s
        )
        return future, time.monotonic() + timeout

    def _finish_io(self, pending, page_count):
        future, deadline = pending
        while not future.done():
            if time.monotonic() > deadline:
                raise TimeoutError(
                    "Shared staging native I/O did not settle; retain buffers"
                )
            time.sleep(0)
        try:
            return complete_page_mask(future.result(), page_count)
        except OSError as exc:
            logger.warning("Shared staging settled I/O failed: %s", exc)
            return [False] * page_count

    def _communicate(self, lane, plan, window, rounds):
        """Only the communication thread calls this, for either direction."""
        stage = self.stages[lane]
        buffer = stage.buffer
        for component in EXCHANGE_COMPONENTS:
            sizes = buffer.shard_bytes[component]
            for exchange_round in rounds:
                contributions = exchange_round.contributions
                active = sum(page is not None for page in contributions)
                shard_splits = [
                    sizes[self.rank] if page is not None else 0
                    for page in contributions
                ]
                shard = buffer.scratch[component][
                    exchange_round.index, :active
                ].reshape(-1)
                owns_page = contributions[self.rank] is not None
                full = (
                    buffer.components[component]
                    .buffer[exchange_round.index]
                    .reshape(-1)
                )
                if not owns_page:
                    full = full[:0]
                full_splits = list(sizes) if owns_page else [0] * plan.shard_size
                if lane == PREFETCH:
                    send, recv, inputs, outputs = full, shard, full_splits, shard_splits
                else:
                    send, recv, inputs, outputs = shard, full, shard_splits, full_splits
                group = self.pg_pool.data_groups[
                    exchange_round.index % len(self.pg_pool.data_groups)
                ]
                work = dist.all_to_all_single(
                    recv,
                    send,
                    output_split_sizes=outputs,
                    input_split_sizes=inputs,
                    group=group,
                    async_op=True,
                )
                stage.works.append((work, recv, send, time.monotonic()))
        for work, recv, send, issued in stage.works:
            while not work.is_completed():
                if (
                    time.monotonic() - issued
                    > self.staging_buffer_config.exchange_timeout_s
                ):
                    raise TimeoutError("Shared staging all-to-all did not finish")
                time.sleep(0)
            work.wait()

    def _copy_shards(self, lane, window, rounds, host_indices):
        buffer = self.stages[lane].buffer
        readable = [True] * window.page_count
        for component in EXCHANGE_COMPONENTS:
            view = self.l2_pools[component]
            for exchange_round in rounds:
                active = [o for o in exchange_round.contributions if o is not None]
                for compact, ordinal in enumerate(active):
                    payload = buffer.scratch[component][exchange_round.index, compact]
                    token_base = int(
                        host_indices[ordinal * self.staging_buffer_config.page_size]
                    )
                    if lane == PREFETCH:
                        view.write_page(token_base, payload)
                    else:
                        try:
                            view.read_page(token_base, payload)
                        except Exception as exc:
                            logger.warning(
                                "Shared backup cannot read page %d: %s", ordinal, exc
                            )
                            payload.zero_()
                            readable[ordinal - window.page_start] = False
        return readable

    def prefetch_window(self, plan, window, host_indices, *, stop_requested):
        """GET -> shared communication -> L2; return copied pages and agreed stop."""
        owned = plan.owned_ordinals(self.rank, window)
        pending = self._start_io(PREFETCH, plan, window, owned)
        mask = self._finish_io(pending, len(owned))
        successful_owned_prefix = mask.index(False) if False in mask else len(mask)
        rounds = build_exchange_rounds(plan, window)
        # A rank with fewer (or zero) owned pages still receives shards in the
        # remaining rounds. Only a failed local GET limits the shared prefix.
        supported_rounds = (
            len(rounds)
            if successful_owned_prefix == len(owned)
            else successful_owned_prefix
        )

        def exchange(agreed):
            usable, keep_going = agreed
            self._communicate(PREFETCH, plan, window, rounds[:usable])
            return usable, keep_going

        usable, keep_going = self.pg_pool.submit(
            PREFETCH,
            WindowJob(
                plan.window_fingerprint(window),
                lambda: [supported_rounds, int(not stop_requested())],
                exchange,
            ),
        )
        self._copy_shards(PREFETCH, window, rounds[:usable], host_indices)
        self.stages[PREFETCH].works.clear()
        return min(usable * plan.shard_size, window.page_count), not keep_going

    def backup_window(self, plan, window, host_indices):
        """L2 -> shared communication -> SET; return locally written/skipped pages."""
        rounds = build_exchange_rounds(plan, window)
        readable = self._copy_shards(BACKUP, window, rounds, host_indices)

        def exchange(agreed):
            writable = [o for o, ok in zip(window.ordinals(), agreed) if ok]
            if writable:
                self._communicate(BACKUP, plan, window, rounds)
            return writable

        writable = self.pg_pool.submit(
            BACKUP,
            WindowJob(
                plan.window_fingerprint(window),
                lambda: [int(ok) for ok in readable],
                exchange,
            ),
        )
        self.stages[BACKUP].works.clear()
        owned = [o for o in plan.owned_ordinals(self.rank, window) if o in writable]
        written = 0
        if owned:
            pending = self._start_io(BACKUP, plan, window, owned)
            written = sum(self._finish_io(pending, len(owned)))
        return written, window.page_count - len(writable)

    # Native-controller adapter and scheduler-owned StorageCoordination hooks.
    def _reduce_control(self, tensor, op, timeout_s=None):
        if timeout_s is None:
            timeout_s = self.staging_buffer_config.control_timeout_s
        deadline = time.monotonic() + timeout_s
        groups = [
            g
            for g in (self._controller.attn_cp_group, self._controller.attn_tp_group)
            if g is not None
        ]
        if not groups:
            groups = [self._controller.tp_group]
        for group in groups:
            if torch.distributed.get_world_size(group) <= 1:
                continue
            work = torch.distributed.all_reduce(
                tensor, op=op, group=group, async_op=True
            )
            while not work.is_completed():
                if time.monotonic() >= deadline:
                    raise TimeoutError("LayerSplit control collective timed out")
                time.sleep(0.0005)
            work.wait()  # completed Work can still contain a transport error

    @staticmethod
    def _identity(values):
        payload = "\x1f".join(str(v) for v in values).encode()
        return int.from_bytes(
            hashlib.blake2b(payload, digest_size=8).digest(), "little"
        ) & ((1 << 63) - 1)

    def align_prefetch_allocation(self, operation, length):
        digest = self._identity(
            [operation.request_id, len(operation.hash_value), *operation.hash_value]
        )
        states = torch.tensor([length, digest, -digest], dtype=torch.int64)
        self._reduce_control(states, torch.distributed.ReduceOp.MIN)
        agreed_length, minimum_digest, negative_maximum_digest = states.tolist()
        if minimum_digest != -negative_maximum_digest:
            raise RuntimeError("Prefetch allocation identities differ across ranks")
        return agreed_length

    def prefetch(self, operation):
        """Publish through the community ACK pipeline, never mutate scheduler progress.

        One terminal progress ACK per operation keeps the ACK schedule identical
        even after a short GET or an agreed early stop. The native auxiliary
        worker supplies completed_req; its ACK consumer owns tail release.
        """
        plan = self.build_plan(
            op_id=operation.id,
            request_id=operation.request_id,
            page_hashes=operation.hash_value,
        )
        handle = StagingPrefetchOp(plan, host_indices=operation.host_indices)
        self.prefetch_run(handle, stop_requested=operation.is_terminated)
        self._controller.prefetch_sync_queue.put(
            PrefetchAck(
                rid=operation.request_id,
                operation=operation,
                completed_tokens=handle.published_pages
                * self.staging_buffer_config.page_size,
                pool_hits={},
            )
        )

    def enqueue_backup(self, operation):
        """Bound outstanding work and publish directly to the native backup worker.

        BACKUP_ORDER_CONTRACT: upstream ranks must publish the same page lists in
        the same order after D2H completion. Local operation IDs may differ.
        This is an input contract, not something ACK-count MIN proves. There is
        no scheduler admission, background reordering or missing-task recovery.
        The shared PG rejects mismatched windows or times out waiting for a
        missing peer; it does not repair the upstream sequence.

        Both this method and retire_backup run on the scheduler thread. Under
        that contract, the same enqueue stream and count-MIN ACK retirement keep
        the outstanding counts (and cap decisions) equal across ranks. A worker
        must NOT retire work itself: I/O completion timing is rank-local.
        """
        if (
            len(self._outstanding_backups)
            >= self.staging_buffer_config.backup_max_outstanding_operations
        ):
            self._record_backup_drop("outstanding cap")
            return None
        self._outstanding_backups[operation.id] = operation
        self._controller.backup_queue.put(operation)
        return operation.id

    def retire_backup(self, operation_id):
        # Retire only after synchronized ACK consumption and local host unlock.
        self._outstanding_backups.pop(operation_id, None)

    def _record_backup_drop(self, reason):
        self._backup_drops += 1
        if (
            self._backup_drops == 1
            or self._backup_drops % self.staging_buffer_config.backup_drop_log_interval
            == 0
        ):
            logger.warning(
                "LayerSplit dropped %d backups: %s", self._backup_drops, reason
            )

    def check_write_ack_progress(self, ready_count, finish_count):
        # This detects, but does not repair, upstream count-MIN divergence.
        # Non-ready GPU events require a separate transfer deadline.
        if ready_count <= finish_count:
            self._write_ack_stall_since = None
        elif self._write_ack_stall_since is None:
            self._write_ack_stall_since = time.monotonic()
        elif (
            time.monotonic() - self._write_ack_stall_since
            > self.staging_buffer_config.write_ack_stall_timeout_s
        ):
            raise TimeoutError("D-to-H write ACKs remain undrainable across ranks")

    def _make_backup_op(self, operation):
        hashes = list(operation.hash_value or [])
        plan = self.build_plan(
            op_id=operation.id,
            request_id=f"backup:{len(hashes)}:{hashes[-1] if hashes else ''}",
            page_hashes=hashes,
        )
        return StagingBackupOp(plan=plan)

    def backup(self, operation):
        handle = self._make_backup_op(operation)
        self.backup_run(handle, operation.host_indices)
        operation.completed_tokens = (
            handle.written_pages * self.staging_buffer_config.page_size
        )
