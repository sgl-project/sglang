"""Communication helpers for the FlexKV connector.

FlexKV runs a single KVManager per DP group (typically the TP/CP/PP
sync leader's process). Every other rank in the same KV-cache-sharing
fan-out must be told the leader's decisions: which prefix matched in
FlexKV, which task id the leader allocated, which slot mappings to
send, etc.

This file provides:

* ``FlexKVComm`` — a 3-axis (PP × CP × TP) hierarchical sync context
  built on torch.distributed (gloo CPU groups). Exposes ``scatter``,
  ``scatter_pp``, ``barrier`` and ``all_reduce_min`` plus role flags
  (``is_sync_leader`` etc.) that the connector branches on.
* libc / ``eventfd`` shims used by the layerwise transfer worker
  socket handshake.
* ``FlexKVLayerLoadingEvent`` and ``FlexKVLayerDoneCounter`` — the
  eventfd-backed per-layer completion structures that the FlexKV
  layerwise transfer worker signals into. Hooked into sglang's
  ``register_layer_transfer_counter`` so each layer's forward waits
  for its own host→device copy.
"""

from __future__ import annotations

import ctypes
import errno
import logging
import os
import pickle
import socket
import struct
from datetime import timedelta
from typing import Any, Dict, List

import torch
import torch.distributed as dist

from sglang.srt.distributed.parallel_state import get_world_group

logger = logging.getLogger(__name__)

# PP-channel command tags (used by ``scatter_pp`` payloads). Sender and
# receiver assert on these to catch protocol drift early.
CMD_PUT_META = 2
CMD_LAYERWISE = 3
CMD_STORE_COMPLETE = 5


class FlexKVComm:
    """3-axis (PP × CP × TP) hierarchical sync for the FlexKV connector.

    Notation:
      * "sync leader" is the unique rank that talks to the FlexKV
        KVManager: pp_rank=0, attn_cp_rank=0, attn_tp_rank=0.
      * "PP stage leader" is the (cp=0, tp=0) rank within a PP stage —
        it does cross-PP P2P (``scatter_pp``).
      * Every rank participates in collective layers it belongs to.

    Communication strategy:
      * P2P (send/recv/isend/irecv) on CPU tensors → ``world_cpu_group``
        (the global gloo group). Sub-group cpu_groups have unreliable
        TCP pairs for direct P2P.
      * Collectives (all_reduce / barrier) → sglang's sub-group
        cpu_groups (fine for collectives).
    """

    # P2P tags. World group is shared with sglang's own P2P, so we pick
    # 4-byte tags that won't collide.
    _TAG_SCATTER = int.from_bytes(b"FxSc", byteorder="big")
    _TAG_PP = int.from_bytes(b"FxPP", byteorder="big")
    _TAG_CP = int.from_bytes(b"FxCP", byteorder="big")
    _TAG_TP = int.from_bytes(b"FxTP", byteorder="big")
    _TAG_PP_AR_MIN = int.from_bytes(b"FxA2", byteorder="big")
    _TAG_PP_BARRIER = int.from_bytes(b"FxB2", byteorder="big")
    _TAG_PP_BARRIER_BCAST = int.from_bytes(b"FxB3", byteorder="big")
    _TAG_AR_BCAST = int.from_bytes(b"FxAR", byteorder="big")

    # Adaptive async-work reaper. gloo's isend Work objects do not auto-
    # advance their "completed" state on poll, so a pure poll-based reaper
    # leaks. We actively wait() the oldest works with a tiny timeout;
    # the watermark grows on stuck reaps (slow / asymmetric peer) and
    # shrinks back on clean reaps.
    _REAP_HIGH_BASE = 1024
    _REAP_HIGH_MAX = 32768
    _REAP_MAX_DRAIN = 512
    _REAP_PROBE = timedelta(milliseconds=1)
    _REAP_LOG_EVERY = 64

    def __init__(
        self,
        rank_info,
        world_rank: int,
        pp_group=None,
        attn_tp_group=None,
        attn_cp_group=None,
    ):
        model_config = rank_info.model_config
        self.world_rank = world_rank
        self._async_works: List = []
        self._reap_high: int = self._REAP_HIGH_BASE
        self._reap_calls: int = 0
        self._reap_stuck_total: int = 0
        self._reap_drained_total: int = 0

        # Accept either GroupCoordinator wrappers (has ``.cpu_group``) or
        # raw ProcessGroups.
        self.pp_cpu_group = (
            getattr(pp_group, "cpu_group", pp_group) if pp_group is not None else None
        )
        self.attn_tp_cpu_group = (
            getattr(attn_tp_group, "cpu_group", attn_tp_group)
            if attn_tp_group is not None
            else None
        )
        self.attn_cp_cpu_group = (
            getattr(attn_cp_group, "cpu_group", attn_cp_group)
            if attn_cp_group is not None
            else None
        )

        self.pp_size = model_config.pp_size
        self.attn_tp_size = model_config.attn_tp_size
        self.attn_cp_size = model_config.attn_cp_size

        self.pp_rank = rank_info.pp_rank
        self.attn_tp_rank = rank_info.attn_tp_rank
        self.attn_cp_rank = rank_info.attn_cp_rank

        self.is_pp_stage_leader = self.attn_tp_rank == 0 and self.attn_cp_rank == 0
        self.is_sync_leader = self.pp_rank == 0 and self.is_pp_stage_leader
        self.is_pp_leader = self.pp_rank == 0 and self.is_pp_stage_leader
        self.is_cp_leader = self.attn_cp_rank == 0
        self.is_tp_leader = self.attn_tp_rank == 0

        # P2P routing tables (computed once).
        stride = self.attn_tp_size * self.attn_cp_size
        self._pp_stage_leader_ranks = [s * stride for s in range(self.pp_size)]
        pp_stage_offset = self.pp_rank * stride
        self._cp_leader_ranks = (
            [
                pp_stage_offset + cp * self.attn_tp_size
                for cp in range(self.attn_cp_size)
            ]
            if self.attn_cp_size > 1
            else []
        )
        if self.attn_tp_size > 1:
            if self.attn_tp_cpu_group is None:
                raise RuntimeError(
                    f"[FlexKV] attn_tp_size={self.attn_tp_size} > 1 but "
                    f"attn_tp_cpu_group is None — TP CPU group is required "
                    f"for scatter/collectives."
                )
            self._tp_group_ranks = [
                dist.get_global_rank(self.attn_tp_cpu_group, i)
                for i in range(self.attn_tp_cpu_group.size())
            ]
        else:
            self._tp_group_ranks = []
        self._pp_group_global_ranks = (
            [
                dist.get_global_rank(self.pp_cpu_group, i)
                for i in range(self.pp_cpu_group.size())
            ]
            if self.pp_size > 1 and self.pp_cpu_group is not None
            else []
        )
        self._pp_stage_member_ranks = list(
            range(pp_stage_offset, pp_stage_offset + stride)
        )

        self.needs_sync = (
            self.pp_size > 1 or self.attn_tp_size > 1 or self.attn_cp_size > 1
        )

        self._world_cpu_group = get_world_group().cpu_group

        self.pp_group = (
            self.pp_cpu_group
            if (self.pp_size > 1 and self.is_pp_stage_leader)
            else None
        )
        self.is_pp_active = self.pp_size > 1
        self.is_pp_sender = self.is_pp_leader
        self.is_pp_receiver = self.is_pp_stage_leader and not self.is_pp_leader

        self.is_cross_node_pp = self.pp_size > rank_info.pp_size_per_node
        self.should_send_slot_mapping_to_remote = (
            self.is_pp_receiver and self.is_cross_node_pp
        )

        logger.info(
            "[FlexKV] Comm init: rank=%d, pp=%d/%d, tp=%d/%d, cp=%d/%d, "
            "sync_leader=%s, stage_leader=%s, cross_node_pp=%s",
            world_rank,
            self.pp_rank,
            self.pp_size,
            self.attn_tp_rank,
            self.attn_tp_size,
            self.attn_cp_rank,
            self.attn_cp_size,
            self.is_sync_leader,
            self.is_pp_stage_leader,
            self.is_cross_node_pp,
        )

    # ------------------------------------------------------------------
    # Public collectives
    # ------------------------------------------------------------------

    def scatter(self, data: Any, blocking: bool = False) -> Any:
        """Hierarchical fan-out: sync_leader → PP stage leaders →
        CP leaders → TP ranks. Returns the leader's payload on every rank.

        ``blocking=False`` queues isends and reaps later — fine for the
        hot path; ``True`` blocks until the leader's sends drain (used
        on shutdown / barriers).
        """
        if self.pp_size > 1 and self.is_pp_stage_leader:
            data = self._scatter_group(
                data,
                self._pp_stage_leader_ranks,
                self.is_pp_leader,
                self._TAG_PP,
                blocking,
            )
        if self._cp_leader_ranks:
            data = self._scatter_group(
                data,
                self._cp_leader_ranks,
                self.is_cp_leader,
                self._TAG_CP,
                blocking,
            )
        if self._tp_group_ranks:
            data = self._scatter_group(
                data,
                self._tp_group_ranks,
                self.is_tp_leader,
                self._TAG_TP,
                blocking,
            )
        return data

    def scatter_pp(self, data: Any) -> Any:
        """PP-only fan-out across PP stages (only stage leaders participate)."""
        if not self._pp_group_global_ranks:
            return data
        is_leader = self._pp_group_global_ranks[0] == self.world_rank
        return self._scatter_group(
            data,
            self._pp_group_global_ranks,
            is_leader,
            self._TAG_SCATTER,
            blocking=False,
        )

    def all_reduce_min(self, value: int) -> int:
        """Hierarchical all_reduce(MIN) across TP, CP, PP.

        Used to align FlexKV block-count limits across all ranks that
        will register GPU buffers (each rank computes the maximum it can
        support, and we take the MIN to land on a value everyone can
        honor).
        """
        tensor = torch.tensor(value, dtype=torch.int64)
        if self.attn_tp_size > 1 and self.attn_tp_cpu_group is not None:
            dist.all_reduce(tensor, op=dist.ReduceOp.MIN, group=self.attn_tp_cpu_group)
        if self.attn_cp_size > 1 and self.attn_cp_cpu_group is not None:
            dist.all_reduce(tensor, op=dist.ReduceOp.MIN, group=self.attn_cp_cpu_group)
        if self.pp_size > 1 and self.is_pp_stage_leader:
            self._pp_all_reduce_min_p2p(tensor)
        if self.pp_size > 1:
            self._bcast_to_stage_members(tensor, self._TAG_AR_BCAST)
        return int(tensor.item())

    def barrier(self) -> None:
        if self.attn_tp_size > 1 and self.attn_tp_cpu_group is not None:
            dist.barrier(group=self.attn_tp_cpu_group)
        if self.attn_cp_size > 1 and self.attn_cp_cpu_group is not None:
            dist.barrier(group=self.attn_cp_cpu_group)
        if self.pp_size > 1 and self.is_pp_stage_leader:
            self._pp_barrier_p2p()
        if self.pp_size > 1:
            dummy = torch.tensor([0], dtype=torch.int64)
            self._bcast_to_stage_members(dummy, self._TAG_PP_BARRIER_BCAST)

    # ------------------------------------------------------------------
    # Internal scatter helper
    # ------------------------------------------------------------------

    def _scatter_group(
        self,
        data: Any,
        group_ranks: List[int],
        is_leader: bool,
        tag: int,
        blocking: bool = False,
    ) -> Any:
        if not group_ranks or self.world_rank not in group_ranks:
            return data
        if is_leader:
            dsts = [r for r in group_ranks if r != self.world_rank]
            works = []
            for dst in dsts:
                works.extend(self._isend(dst, data, tag, self._world_cpu_group))
            if blocking:
                for w in works:
                    w.wait()
            else:
                self._reap_completed_async_works()
                self._async_works.extend(works)
            return data
        return self._recv(group_ranks[0], tag, self._world_cpu_group)

    def _reap_completed_async_works(self) -> None:
        n = len(self._async_works)
        if n <= self._reap_high:
            return

        drained = 0
        stuck = False
        for _ in range(self._REAP_MAX_DRAIN):
            if not self._async_works:
                break
            w = self._async_works[0]
            try:
                w.wait(self._REAP_PROBE)
            except RuntimeError:
                stuck = True
                break
            self._async_works.pop(0)
            drained += 1

        self._reap_calls += 1
        self._reap_drained_total += drained
        if stuck:
            self._reap_stuck_total += 1

        prev_high = self._reap_high
        if stuck:
            self._reap_high = min(self._REAP_HIGH_MAX, self._reap_high * 2)
        else:
            self._reap_high = max(self._REAP_HIGH_BASE, self._reap_high // 2)
        if self._reap_high != prev_high:
            logger.debug(
                "[FlexKV] reap watermark rank=%d %d->%d "
                "(stuck=%s drained=%d backlog=%d)",
                self.world_rank,
                prev_high,
                self._reap_high,
                stuck,
                drained,
                n,
            )
        if self._reap_calls % self._REAP_LOG_EVERY == 0:
            logger.debug(
                "[FlexKV] reap stats rank=%d calls=%d drained=%d stuck=%d "
                "backlog=%d high=%d",
                self.world_rank,
                self._reap_calls,
                self._reap_drained_total,
                self._reap_stuck_total,
                len(self._async_works),
                self._reap_high,
            )

    # ------------------------------------------------------------------
    # Low-level send / recv on the world cpu group
    # ------------------------------------------------------------------

    def _isend(self, dst: int, data: Any, tag: int = 0, group=None) -> list:
        serialized = bytearray(pickle.dumps(data))
        t_size = torch.tensor([len(serialized)], dtype=torch.long)
        t_data = torch.frombuffer(serialized, dtype=torch.uint8)
        return [
            dist.isend(t_size, dst=dst, tag=tag, group=group),
            dist.isend(t_data, dst=dst, tag=tag, group=group),
        ]

    def _recv(self, src: int, tag: int = 0, group=None) -> Any:
        t_size = torch.tensor([0], dtype=torch.long)
        dist.irecv(t_size, src=src, tag=tag, group=group).wait()
        size = int(t_size.item())
        if size == 0:
            return []
        t_data = torch.empty(size, dtype=torch.uint8)
        dist.irecv(t_data, src=src, tag=tag, group=group).wait()
        return safe_pickle_loads(t_data.numpy().tobytes())

    def _send_tensor(
        self, tensor: torch.Tensor, dst: int, tag: int = 0, group=None
    ) -> None:
        dist.send(tensor, dst=dst, tag=tag, group=group)

    def _recv_tensor(
        self, tensor: torch.Tensor, src: int, tag: int = 0, group=None
    ) -> None:
        dist.recv(tensor, src=src, tag=tag, group=group)

    def _bcast_to_stage_members(self, tensor: torch.Tensor, tag: int) -> None:
        if not self.is_pp_stage_leader:
            self._recv_tensor(
                tensor,
                src=self._pp_stage_leader_ranks[self.pp_rank],
                tag=tag,
                group=self._world_cpu_group,
            )
            return
        for rank in self._pp_stage_member_ranks:
            if rank != self.world_rank:
                self._send_tensor(
                    tensor, dst=rank, tag=tag, group=self._world_cpu_group
                )

    def _pp_all_reduce_min_p2p(self, tensor: torch.Tensor) -> None:
        leader_rank = self._pp_stage_leader_ranks[0]
        other_leaders = self._pp_stage_leader_ranks[1:]
        tag = self._TAG_PP_AR_MIN
        if self.world_rank == leader_rank:
            result = int(tensor.item())
            for src in other_leaders:
                other = torch.tensor(0, dtype=torch.int64)
                self._recv_tensor(other, src=src, tag=tag, group=self._world_cpu_group)
                result = min(result, int(other.item()))
            tensor.fill_(result)
            for dst in other_leaders:
                self._send_tensor(tensor, dst=dst, tag=tag, group=self._world_cpu_group)
        else:
            self._send_tensor(
                tensor, dst=leader_rank, tag=tag, group=self._world_cpu_group
            )
            self._recv_tensor(
                tensor, src=leader_rank, tag=tag, group=self._world_cpu_group
            )

    def _pp_barrier_p2p(self) -> None:
        leader_rank = self._pp_stage_leader_ranks[0]
        other_leaders = self._pp_stage_leader_ranks[1:]
        tag = self._TAG_PP_BARRIER
        dummy = torch.tensor([1], dtype=torch.int64)
        if self.world_rank == leader_rank:
            for src in other_leaders:
                self._recv_tensor(dummy, src=src, tag=tag, group=self._world_cpu_group)
            for dst in other_leaders:
                self._send_tensor(dummy, dst=dst, tag=tag, group=self._world_cpu_group)
        else:
            self._send_tensor(
                dummy, dst=leader_rank, tag=tag, group=self._world_cpu_group
            )
            self._recv_tensor(
                dummy, src=leader_rank, tag=tag, group=self._world_cpu_group
            )


# ----------------------------------------------------------------------
# libc / eventfd / SCM_RIGHTS shims for the layerwise UDS handshake
# ----------------------------------------------------------------------

_libc = ctypes.CDLL("libc.so.6", use_errno=True)
_libc.eventfd.argtypes = [ctypes.c_uint, ctypes.c_int]
_libc.eventfd.restype = ctypes.c_int
_libc.read.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t]
_libc.read.restype = ctypes.c_ssize_t
_libc.write.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t]
_libc.write.restype = ctypes.c_ssize_t

EFD_SEMAPHORE = 0x1
EFD_NONBLOCK = 0x800


def eventfd(initval: int = 0, flags: int = 0) -> int:
    fd = _libc.eventfd(ctypes.c_uint(initval), ctypes.c_int(flags))
    if fd == -1:
        err = ctypes.get_errno()
        raise OSError(err, os.strerror(err))
    return fd


def eventfd_write(fd: int, val: int) -> None:
    v = ctypes.c_uint64(val)
    n = _libc.write(fd, ctypes.byref(v), ctypes.sizeof(v))
    if n != ctypes.sizeof(v):
        err = ctypes.get_errno()
        raise OSError(err, f"eventfd write failed: {os.strerror(err)}")


def eventfd_read(fd: int) -> int:
    v = ctypes.c_uint64()
    n = _libc.read(fd, ctypes.byref(v), ctypes.sizeof(v))
    if n != ctypes.sizeof(v):
        err = ctypes.get_errno()
        if err == errno.EAGAIN:
            return 0
        raise OSError(err, f"eventfd read failed: {os.strerror(err)}")
    return v.value


def send_fds(sock: socket.socket, fds: list, extra_data: bytes = b"x") -> None:
    """SCM_RIGHTS-send a list of file descriptors over a UDS socket."""
    fds_packed = struct.pack(f"{len(fds)}i", *fds)
    ancdata = [(socket.SOL_SOCKET, socket.SCM_RIGHTS, fds_packed)]
    sock.sendmsg([extra_data], ancdata)


# ----------------------------------------------------------------------
# Layerwise transfer signaling (eventfd-backed)
# ----------------------------------------------------------------------


class FlexKVLayerLoadingEvent:
    """One per producer slot. Holds ``num_layers`` semaphore eventfds —
    the FlexKV layerwise worker writes 1 to each as the corresponding
    layer's H2D copy completes; the consumer (sglang's
    ``register_layer_transfer_counter`` hook) reads to wait for them."""

    def __init__(self, num_layers: int):
        self._num_layers = num_layers
        # Semaphore mode so each read consumes exactly one signal. NONBLOCK
        # lets ``reset_for_new_transfer`` drain leftover counter values
        # without blocking; ``wait`` re-arms the fd to blocking before
        # reading so consumers still get the desired blocking semantics.
        self.load_event_fds: List[int] = [
            eventfd(0, EFD_SEMAPHORE | EFD_NONBLOCK) for _ in range(num_layers)
        ]
        self._finished = True
        self.wait_remaining: List[int] = [1] * num_layers

    def reset_for_new_transfer(self) -> None:
        """Drain any leftover signals from prior transfers, then arm.

        Without this drain, a previous transfer that wrote N eventfd
        signals but only had N-K reads (e.g. because the attention
        backend skipped a layer's ``get_key_buffer`` call) leaves K
        pending. The next transfer's first ``wait(layer)`` returns
        immediately reading one of those stale signals, even though
        the FlexKV worker hasn't actually finished that layer's H2D
        yet — and forward proceeds with wrong KV data.
        """
        import os

        for fd in self.load_event_fds:
            # The fd is NONBLOCK: read until EAGAIN. Each read is 8 bytes.
            while True:
                try:
                    if not os.read(fd, 8):
                        break
                except BlockingIOError:
                    break
                except OSError:
                    break
        self._finished = False
        self.wait_remaining = [1] * self._num_layers

    def wait(self, layer_index: int) -> None:
        """Block until the FlexKV worker signals layer ``layer_index``.

        The fd was created with EFD_NONBLOCK so reset can drain it. We
        re-introduce the blocking semantics with ``select.select`` on a
        NONBLOCK fd: the read after select is guaranteed to consume one
        signal.
        """
        import os
        import select

        assert 0 <= layer_index < self._num_layers
        fd = self.load_event_fds[layer_index]
        while True:
            select.select([fd], [], [])
            try:
                buf = os.read(fd, 8)
                if buf:
                    break
            except BlockingIOError:
                # Spurious wakeup; loop and re-select.
                continue
        if layer_index == self._num_layers - 1:
            self._finished = True

    def close(self) -> None:
        for fd in self.load_event_fds:
            try:
                os.close(fd)
            except Exception:
                pass
        self.load_event_fds.clear()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


class FlexKVLayerDoneCounter:
    """Triple-buffered slot-based layerwise counter.

    The KV pool calls ``wait_until(layer_id)`` once per layer during
    forward. We track which producer slot the current task is using and
    block on that slot's ``layer_id``-th eventfd. Producer rotation lets
    the next prefetch start before the current one finishes consuming.
    """

    def __init__(self, num_layers: int, num_counters: int = 3):
        self.num_layers = num_layers
        self.num_counters = num_counters
        self.events: List[FlexKVLayerLoadingEvent] = [
            FlexKVLayerLoadingEvent(num_layers) for _ in range(num_counters)
        ]
        self.producer_index = -1
        self.consumer_index = -1
        self._task_to_producer: Dict[int, int] = {}

    def register_task(self, task_id: int, producer_id: int) -> None:
        self._task_to_producer[task_id] = producer_id

    def register_task_with_explicit_counter_id(
        self, task_id: int, counter_id: int
    ) -> None:
        if not 0 <= counter_id < self.num_counters:
            raise ValueError(
                f"Invalid counter_id={counter_id}, must be in [0, {self.num_counters})"
            )
        self._task_to_producer[task_id] = counter_id
        self.events[counter_id].reset_for_new_transfer()

    def update_producer(self) -> int:
        self.producer_index = (self.producer_index + 1) % self.num_counters
        assert self.events[
            self.producer_index
        ]._finished, "Producer event should be finished before reuse"
        return self.producer_index

    def set_consumer(self, task_id: int) -> None:
        if task_id < 0:
            self.consumer_index = -1
            return
        producer_id = self._task_to_producer.pop(task_id, None)
        self.consumer_index = producer_id if producer_id is not None else -1

    def wait_until(self, threshold: int) -> None:
        if self.consumer_index < 0:
            return
        event = self.events[self.consumer_index]
        if event.wait_remaining[threshold] <= 0:
            return
        event.wait_remaining[threshold] -= 1
        event.wait(threshold)

    def reset(self) -> None:
        self.producer_index = -1
        self.consumer_index = -1
        self._task_to_producer.clear()

    def __del__(self) -> None:
        try:
            for event in self.events:
                event.close()
            self.events.clear()
        except Exception:
            pass
