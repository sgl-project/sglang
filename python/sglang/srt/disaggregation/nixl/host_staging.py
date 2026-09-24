"""NIXL WRITEs through pinned host memory, beneath native NIXL's transfer planner.

The planners in nixl/conn.py reduce every KV and state transfer to (addr, len,
device) rows and post them through NixlKVManager's transport seam. HOST staging
replaces that seam: prefill packs a WRITE's source rows into a pinned slot behind a
header and the WRITE's destination row table, then WRITEs the slot into the
decode's pinned ring. Decode copies the header and table out of the ring, checks
every row against the room's own registered memory, scatters the rows, and only
then delivers the WRITE's native notification, so native completion tracking is
unchanged. Aux stays a native DRAM WRITE. No recovery of uncertain remote writes:
a posted WRITE that cannot be settled fail-stops the process.
"""

from __future__ import annotations

import logging
import threading
import time
import traceback
from collections import defaultdict, deque
from dataclasses import dataclass

import numpy as np
import torch
import zmq

from sglang.srt.disaggregation.base.conn import KVPoll, StateType
from sglang.srt.disaggregation.common.staging_buffer import (
    StagingAllocator,
    StagingBuffer,
)
from sglang.srt.disaggregation.common.staging_handler import DecodeStagingHandler
from sglang.srt.disaggregation.utils import DisaggregationMode, fail_stop

SLOT_COUNT = 2
VERSION = 3
MAGIC = 0x32545348  # b"HST2"
HEADER = np.dtype(
    [
        ("magic", "<u4"),
        ("part", "<u4"),
        ("parts", "<u4"),
        ("notif_len", "<u4"),
        ("room", "<u8"),
        ("seq", "<u8"),
        ("group", "<u8"),
        ("rows", "<u8"),
        ("payload", "<u8"),
        ("notif", "S192"),
    ]
)
ROW = np.dtype([("addr", "<u8"), ("len", "<u8")])
PAYLOAD_ALIGN = 256
# From STAGING_REQ, which precedes the grant: inside the decode's default 30 s
# drain-ack wait, so an unacked room's ring regions are free to reuse on release.
POST_DEADLINE_S, WRITE_DEADLINE_S = 15.0, 10.0
logger = logging.getLogger(__name__)


def payload_offset(n_rows: int) -> int:
    end = HEADER.itemsize + n_rows * ROW.itemsize
    return -(-end // PAYLOAD_ALIGN) * PAYLOAD_ALIGN


def as_rows(rows):
    """(addr, len) columns of planner rows; the device column is implied."""
    rows = np.asarray(rows, dtype=np.uint64).reshape(-1, 3)
    return rows[:, 0].copy(), rows[:, 1].copy()


def coalesce(src, dst, lens):
    """Merge rows that continue their predecessor on both sides."""
    if len(lens) < 2:
        return src, dst, lens
    joins = (src[1:] == src[:-1] + lens[:-1]) & (dst[1:] == dst[:-1] + lens[:-1])
    starts = np.flatnonzero(np.r_[True, ~joins])
    return src[starts], dst[starts], np.add.reduceat(lens, starts)


def copy_rows(dst, src, sizes, device, waits=()):
    """Copy sizes[i] bytes src[i] -> dst[i] on the current stream after `waits`."""
    from sglang.kernels.ops.kvcache.pd_dcp_gather import copy_byte_ranges

    # Pinned and async: a pageable upload would block this thread until the
    # stream's queued work (earlier scatters, producer waits) finished.
    table = (
        torch.from_numpy(np.stack([dst, src, sizes]).astype(np.int64))
        .pin_memory()
        .to(device, non_blocking=True)
    )
    stream = torch.cuda.current_stream(device)
    for event in waits:
        if event is not None:
            stream.wait_event(event)
    copy_byte_ranges(table, int(sizes.max()))


@dataclass
class HostWrite:
    """Pseudo xfer handle the native transfer worker polls."""

    parts: int
    state: str = "PROC"  # PROC -> DONE | ERR
    settled: int = 0
    failed: bool = False

    def settle(self, state="DONE"):
        self.settled += 1
        self.failed |= state == "ERR"
        # Native treats ERR as settled and acks the room's drain, so report
        # only once no part of this WRITE can still land in the peer's ring.
        if self.settled == self.parts:
            self.state = "ERR" if self.failed else "DONE"


@dataclass
class HostPart:
    write: HostWrite
    peer: str
    room: int
    seq: int
    group: int
    part: int
    parts: int
    src: np.ndarray
    dst: np.ndarray
    lens: np.ndarray
    notif: bytes
    requested: float = 0.0

    @property
    def size(self) -> int:
        return payload_offset(len(self.lens)) + int(self.lens.sum())


@dataclass
class HostDlist:
    rows: np.ndarray
    mem_kind: str


@dataclass
class HostSlot:
    buffer: StagingBuffer
    stream: object
    part: HostPart | None = None
    copy_done: object = None
    handle: object = None


class HostStaging:
    def __init__(self, manager):
        self.manager = manager
        args = manager.kv_args
        # Capacity is the TOTAL pinned budget, not a per-slot multiplier.
        self.capacity = manager.host_staging_bytes
        self.slot_bytes = self.capacity // SLOT_COUNT
        if self.slot_bytes < payload_offset(1) + PAYLOAD_ALIGN:
            raise ValueError("Host staging capacity is too small")
        if set(args.kv_data_mem_kinds) - {"VRAM"}:
            raise ValueError("Host staging requires CUDA KV pools")
        # FP4 pools keep scales outside the exported KV buffers, so no NIXL path
        # (native included) moves them yet: refuse rather than send bare data.
        if args.kv_cache_dtype_str == "nvfp4":
            raise ValueError("NIXL PD does not transfer FP4 KV scales yet")
        self.lock = threading.Lock()
        self.device = torch.device(f"cuda:{args.gpu_id}")
        self.config = {"version": VERSION, "capacity": self.capacity}
        self.slots, self.allocator = [], None
        # Prefill: queued parts FIFO, per-room seqs and producer events.
        self.queue = deque()
        self.seqs = defaultdict(int)
        self.ready = {}
        if manager.disaggregation_mode == DisaggregationMode.PREFILL:
            for _ in range(SLOT_COUNT):
                buffer = StagingBuffer(
                    self.slot_bytes, "cpu", args.gpu_id, pin_memory=True
                )
                manager._register_staging_memory(buffer.get_ptr(), self.slot_bytes)
                self.slots.append(HostSlot(buffer, torch.cuda.Stream(self.device)))
            threading.Thread(target=self.run, args=(self.worker,), daemon=True).start()
        else:
            # One stream serializes the scatters; distinct ring regions are written
            # concurrently and retired only by their own scatter events.
            self.stream = torch.cuda.Stream(device=self.device)
            self.allocator = StagingAllocator(
                self.capacity, "cpu", args.gpu_id, pin_memory=True
            )
            self.allocator._scatter_stream = self.stream
            manager._register_staging_memory(
                self.allocator.get_base_ptr(), self.capacity
            )

    def config_mismatch(self, peer) -> bool:
        return (
            peer.host_staging_config != self.config
            or peer.staging_base_ptr <= 0
            or peer.staging_total_size != self.capacity
        )

    def run(self, operation, *args):
        try:
            return operation(*args)
        except Exception as exc:
            # Fail-stop is the contract for every HOST transport error, but keep
            # the cause: a local bug must not read like an unfenced RDMA fault.
            fail_stop(f"{operation.__name__}: {exc!r}\n{traceback.format_exc()}")

    # -- Seam: NixlKVManager binds these over its native transport methods. ---

    def prep_dlist(self, peer_name, rows, mem_kind):
        # Keeps every prepared row in host RAM (slots x layers x 24 B per
        # peer); store per-entry (base, stride, count) if that ever matters.
        return HostDlist(np.asarray(rows, dtype=np.uint64).reshape(-1, 3), mem_kind)

    def post_prepped(self, peer_name, src, src_indices, dst, dst_indices, notif, what):
        return self.post_write(
            peer_name,
            src.rows[src_indices],
            src.mem_kind,
            dst.rows[dst_indices],
            dst.mem_kind,
            notif,
            what,
        )

    def post_write(
        self, peer_name, src_rows, src_mem_kind, dst_rows, dst_mem_kind, notif, what
    ):
        manager = self.manager
        if src_mem_kind == dst_mem_kind == "DRAM":
            # Aux: registered host memory on both sides, a native WRITE.
            return type(manager)._post_write(
                manager,
                peer_name,
                src_rows,
                src_mem_kind,
                dst_rows,
                dst_mem_kind,
                notif,
                what,
            )
        if src_mem_kind != "VRAM" or dst_mem_kind != "VRAM":
            raise NotImplementedError(
                f"Host staging does not stage {src_mem_kind}->{dst_mem_kind} {what}"
            )
        room = int(notif.split("_", 1)[0])
        src, src_lens = as_rows(src_rows)
        dst, dst_lens = as_rows(dst_rows)
        if not np.array_equal(src_lens, dst_lens):
            raise ValueError(f"Host staging {what} row lengths differ")
        # Zero-length rows are placeholder entries (e.g. DSA skip-topk layers).
        live = src_lens > 0
        src, dst, lens = coalesce(src[live], dst[live], src_lens[live])
        if not len(lens):
            manager._post_notif(peer_name, notif)  # No bytes: nothing to order.
            return HostWrite(1, "DONE")
        encoded = notif.encode("ascii")
        if len(encoded) > HEADER["notif"].itemsize:
            raise ValueError("Host staging notification too long")
        # Split on row boundaries so each part fits one slot.
        budget = self.slot_bytes - payload_offset(0) - PAYLOAD_ALIGN
        cuts, used = [0], 0
        for i, n in enumerate(lens.tolist()):
            if n + ROW.itemsize > budget:
                raise ValueError(f"Host staging row of {n} bytes exceeds the slot")
            if used + n + ROW.itemsize > budget:
                cuts.append(i)
                used = 0
            used += n + ROW.itemsize
        cuts.append(len(lens))
        write = HostWrite(len(cuts) - 1)
        with self.lock:
            if self._failed(room):
                write.state = "DONE"  # Nothing posts for a failed room.
                return write
            seqs = self.seqs
            group = seqs[room, peer_name]
            seqs[room, peer_name] += write.parts
            for part, (a, b) in enumerate(zip(cuts, cuts[1:])):
                self.queue.append(
                    HostPart(
                        write,
                        peer_name,
                        room,
                        group + part,
                        group,
                        part,
                        write.parts,
                        src[a:b],
                        dst[a:b],
                        lens[a:b],
                        encoded,
                    )
                )
        return write

    def xfer_state(self, handle):
        if isinstance(handle, HostWrite):
            return handle.state
        return self.manager.agent.check_xfer_state(handle)

    # -- Prefill pipeline -----------------------------------------------------

    def _failed(self, room) -> bool:
        # A cleared room is gone, not pending: treat it as failed.
        return self.manager.request_status.get(room) in (None, KVPoll.Failed)

    def fail_room(self, room):
        """Drop a failed room's unposted parts. Caller holds self.lock."""
        for part in [p for p in self.queue if p.room == room]:
            self.queue.remove(part)
            part.write.settle()

    def gathering(self, room) -> bool:
        """Whether a part of room may still read its source pages."""
        with self.lock:
            return any(p.room == room for p in self.queue) or any(
                s.part is not None
                and s.part.room == room
                and (s.copy_done is None or not s.copy_done.query())
                for s in self.slots
            )

    def forget_room(self, room):
        with self.lock:
            self.ready.pop(room, None)
            for key in [k for k in self.seqs if k[0] == room]:
                del self.seqs[key]

    def worker(self):
        torch.cuda.set_device(self.device)
        while True:
            self.progress()
            time.sleep(0.0005)

    def progress(self):
        for slot in self.slots:
            if slot.part is not None:
                self._advance(slot)
        for slot in self.slots:
            if slot.part is None:
                with self.lock:
                    if not self.queue:
                        return
                    part = self.queue.popleft()
                    req = self.manager.transfer_infos.get(part.room, {}).get(part.peer)
                    if req is None:  # Room cleared since; nothing to post.
                        part.write.settle()
                        continue
                    # Claimed under the lock: gathering() never misses this part.
                    slot.part, slot.handle, slot.copy_done = part, None, None
                self._start(slot, part, req)

    def _start(self, slot, part, req):
        from sglang.srt.utils.network import NetworkAddress

        manager = self.manager
        part.requested = time.monotonic()
        view = slot.buffer.buffer.numpy()
        header = np.zeros(1, HEADER)
        header[0] = (
            MAGIC,
            part.part,
            part.parts,
            len(part.notif),
            part.room,
            part.seq,
            part.group,
            len(part.lens),
            int(part.lens.sum()),
            part.notif,
        )
        view[: HEADER.itemsize] = header.view(np.uint8)
        table = np.empty(len(part.lens), ROW)
        table["addr"], table["len"] = part.dst, part.lens
        table_end = HEADER.itemsize + table.nbytes
        view[HEADER.itemsize : table_end] = table.view(np.uint8)
        copy_done = torch.cuda.Event()
        with torch.cuda.stream(slot.stream):
            start = slot.buffer.get_ptr() + payload_offset(len(part.lens))
            offsets = np.r_[0, np.cumsum(part.lens)[:-1]].astype(np.uint64)
            copy_rows(
                offsets + np.uint64(start),
                part.src,
                part.lens,
                self.device,
                waits=self.ready.get(part.room, ()),
            )
            copy_done.record(slot.stream)
        slot.copy_done = copy_done  # Published once recorded.
        address = NetworkAddress(req.endpoint, req.dst_port)
        manager._send_multipart_locked(
            address.to_tcp(),
            [
                b"STAGING_REQ",
                str(part.room).encode(),
                str(part.seq).encode(),
                str(part.size).encode(),
                part.peer.encode(),
                manager.agent.name.encode(),
                # Our bootstrap endpoint: the response goes to this writer only.
                str(manager.local_ip).encode(),
                str(manager.rank_port).encode(),
            ],
            is_ipv6=address.is_ipv6,
        )

    def _advance(self, slot):
        part, manager = slot.part, self.manager
        if slot.handle is None:
            # Nothing leaves the slot, not even a dropped part, while its gather
            # (queued behind the forward) may still write it.
            if not slot.copy_done.query():
                return
            with self.lock:
                req = manager.transfer_infos.get(part.room, {}).get(part.peer)
                if req is None or self._failed(part.room):
                    # Unposted: the peer never sees this part.
                    self._finish(slot, "DONE")
                    return
                if time.monotonic() - part.requested >= POST_DEADLINE_S:
                    logger.error(
                        "Host staging part timed out before posting room=%s seq=%s",
                        part.room,
                        part.seq,
                    )
                    self._finish(slot, "ERR")
                    return
                staging = req.staging
                # Native set_chunk publishes ends last, after offsets/rounds.
                if (
                    staging is None
                    or part.seq >= len(staging.ends)
                    or staging.ends[part.seq] < 0
                ):
                    return
                offset, rnd, end = (
                    staging.offsets[part.seq],
                    staging.rounds[part.seq],
                    staging.ends[part.seq],
                )
                if offset < 0 or end != offset + part.size or end > self.capacity:
                    raise ValueError("Invalid host staging allocation response")
                if not manager._is_watermark_ready(part.peer, rnd, end):
                    return
                peer = manager.decode_kv_args_table[part.peer]
                # Posted under the lock fail_room() takes: a failure either sees
                # this part queued (dropped) or posted (fenced by the drain ack).
                slot.handle = type(manager)._post_write(
                    manager,
                    part.peer,
                    [(slot.buffer.get_ptr(), part.size, 0)],
                    "DRAM",
                    [(peer.staging_base_ptr + offset, part.size, 0)],
                    "DRAM",
                    f"{part.room}_hst_{part.seq}",
                    "host staging WRITE",
                )
            return
        state = manager.agent.check_xfer_state(slot.handle)
        if state == "PROC":
            if time.monotonic() - part.requested >= POST_DEADLINE_S + WRITE_DEADLINE_S:
                fail_stop(f"Host staging WRITE timeout room={part.room} seq={part.seq}")
            return
        # ERR (e.g. the peer died) has settled: it lands nothing later, and the
        # decode holds the region until the drain ack. Fail the room like native.
        manager.agent.release_xfer_handle(slot.handle)
        with self.lock:
            self._finish(slot, "DONE" if state == "DONE" else "ERR")

    def _finish(self, slot, state):
        # Caller holds self.lock.
        slot.part.write.settle(state)
        slot.part = slot.handle = None


class HostDecodeStagingHandler(DecodeStagingHandler):
    """Ring allocation, row validation and scatter for HOST WRITEs."""

    def __init__(self, manager, scheduler, tp_rank):
        host = manager.host_staging
        super().__init__(
            manager,
            host.allocator,
            {"page_size": manager.kv_args.page_size},
            manager.attn_tp_size,
            1,
            tp_rank,
            scheduler,
        )
        self.host = host
        self.lock = threading.RLock()
        # Registered destinations: (base, length, item bytes, owner), by base.
        args = manager.kv_args
        regions = [
            (ptr, length, item, "kv")
            for ptr, length, item in zip(
                args.kv_data_ptrs, args.kv_data_lens, args.kv_item_lens
            )
        ]
        for comp, (ptrs, lengths, items) in enumerate(
            zip(args.state_data_ptrs, args.state_data_lens, args.state_item_lens)
        ):
            regions += [
                (ptr, length, item, comp)
                for ptr, length, item in zip(ptrs, lengths, items)
            ]
        regions = sorted(r for r in regions if r[0] and r[1] and r[2])
        self.region_base = np.array([r[0] for r in regions], dtype=np.uint64)
        self.region_end = np.array([r[0] + r[1] for r in regions], dtype=np.uint64)
        self.region_item = np.array([r[2] for r in regions], dtype=np.uint64)
        self.region_owner = [r[3] for r in regions]

    def allowed_items(self, kv_indices, state_indices):
        """Per region owner, the item indices this room may be written at."""
        allowed = {"kv": np.unique(np.asarray(kv_indices, dtype=np.uint64))}
        for comp, st in enumerate(self.kv_manager.kv_args.state_types):
            indices = state_indices[comp] if state_indices else None
            if indices is None:
                indices = []  # A component without indices takes no rows.
            # DSA_TAIL indices are ring descriptors, not rows: region checks only.
            allowed[comp] = (
                None
                if st == StateType.DSA_TAIL
                else np.unique(np.asarray(indices, dtype=np.uint64))
            )
        return allowed

    def check_rows(self, receiver, addr, lens):
        region = np.searchsorted(self.region_base, addr, side="right") - 1
        ok = (region >= 0) & (lens > 0)
        region = np.maximum(region, 0)
        ok &= addr + lens <= self.region_end[region]
        for r in np.unique(region[ok]):
            allowed = receiver.host_allowed.get(self.region_owner[r])
            if allowed is None:
                continue
            rows = ok & (region == r)
            base, item = self.region_base[r], self.region_item[r]
            lo = (addr[rows] - base) // item
            hi = (addr[rows] + lens[rows] - np.uint64(1) - base) // item
            covered = np.searchsorted(allowed, hi, side="right") - np.searchsorted(
                allowed, lo
            )
            ok[rows] = covered == (hi - lo + np.uint64(1))
        if not ok.all():
            raise ValueError(
                f"Host staging row outside room {receiver.bootstrap_room} memory"
            )

    def register_wm_subscriber(self, receiver, session_id):
        # One watermark per prefill endpoint set, held while any of its rooms is
        # live: not per room (fan-out), nor for a historical peer that may be gone.
        key = tuple(str(i) for i in receiver.bootstrap_infos)
        rooms = self._wm_subscribers.setdefault(key, {})
        rooms[receiver.bootstrap_room] = (receiver, session_id)

    def _free_and_send_watermark(self, alloc_id, _decode_req):
        self.staging_allocator.free(alloc_id)
        watermark = self.staging_allocator.get_watermark()
        for rooms in list(self._wm_subscribers.values()):
            self._send_watermark(*next(iter(rooms.values())), watermark)

    @staticmethod
    def _send_watermark(receiver, session_id, watermark):
        # Never block (under self.lock, on the scheduler thread) on a dead or
        # stuck prefill: a dropped watermark is superseded by the next free.
        wm_round, wm_tail = watermark
        parts = [b"WATERMARK", b"%d" % wm_round, b"%d" % wm_tail, session_id.encode()]
        for info in receiver.bootstrap_infos:
            sock, lock = receiver._connect_to_bootstrap_server(info)
            with lock:
                try:
                    sock.send_multipart(parts, flags=zmq.NOBLOCK)
                except zmq.Again:
                    logger.debug("Dropped host staging watermark to %s", info)

    def is_done(self, decode_req):
        return True  # Completion is native: inner notifs follow their scatter.

    def is_failed(self, decode_req):
        return False

    def advance_scatter(self, decode_req):
        pass

    def allocate(self, msg):
        with self.lock:
            room, seq, size = map(int, msg[1:4])
            session, writer = msg[4].decode(), msg[5].decode()
            endpoint = (msg[6].decode(), int(msg[7]))
            receiver = self._room_to_receiver.get(room)
            if session != self.kv_manager.agent.name:
                raise ValueError("Host staging allocation for unknown session")
            if (
                receiver is None
                or receiver.host_allowed is None
                or receiver.conclude_state is not None
                or self.kv_manager.request_status.get(room) in (None, KVPoll.Failed)
            ):
                # A concluded, failed or cleared room stays fenced: never give its
                # late peer a ring region. The peer drops the part on our ABORT.
                logger.debug("Dropping host staging allocation for room=%s", room)
                return
            if not any(
                (str(i["rank_ip"]), int(i["rank_port"])) == endpoint
                for i in receiver.bootstrap_infos
            ):
                raise ValueError("Host staging allocation from outside the room")
            key = (writer, seq)
            if key not in receiver.host_allocs:
                if not payload_offset(1) < size <= self.host.slot_bytes:
                    raise ValueError("Invalid host staging allocation size")
                receiver.host_allocs[key] = (*self.staging_allocator.assign(size), size)
            _, offset, rnd, known = receiver.host_allocs[key]
            if known != size:
                raise ValueError("Conflicting host staging allocation")
            self.register_wm_subscriber(receiver, session)
            parts = [
                b"STAGING_RSP",
                str(room).encode(),
                str(seq).encode(),
                str(offset).encode(),
                str(rnd).encode(),
                str(offset + size).encode(),
                session.encode(),
            ]
            # Only the writer may learn its region: another prefill rank of this
            # room would take it for its own allocation of the same seq.
            info = next(
                i
                for i in receiver.bootstrap_infos
                if (str(i["rank_ip"]), int(i["rank_port"])) == endpoint
            )
            # Blocking like native control sends: a lost STAGING_RSP on an
            # allocated room cannot be fenced, so its error fail-stops.
            sock, lock = receiver._connect_to_bootstrap_server(info)
            with lock:
                sock.send_multipart(parts)
            # A new session has no watermark until the first free; publish now.
            self._send_watermark(
                receiver, session, self.staging_allocator.get_watermark()
            )

    def on_write(self, room, seq, writer):
        """A HOST WRITE landed in the ring: validate its table, start its scatter."""
        with self.lock:
            receiver = self._room_to_receiver.get(room)
            if receiver is None:
                # Released on its drain ack, which can overtake this notif: the
                # ack proves the WRITE had landed, so there is nothing to do.
                return
            alloc = receiver.host_allocs.get((writer, seq))
            if alloc is None or any(
                s[1:3] == (writer, seq) for s in receiver.host_scatters
            ):
                raise ValueError(f"Unknown host staging WRITE room={room} seq={seq}")
            _, offset, _, size = alloc
            ring = self.staging_allocator.buffer.buffer.numpy()
            # Copy the header and table out first: the scatter must use what
            # was validated, whatever lands in the ring afterwards.
            header = ring[offset : offset + HEADER.itemsize].copy().view(HEADER)[0]
            n = int(header["rows"])
            start = payload_offset(n)
            if (
                header["magic"] != MAGIC
                or header["room"] != room
                or header["seq"] != seq
                or not header["part"] < header["parts"]
                or header["group"] + header["part"] != seq
                or start + int(header["payload"]) != size
            ):
                raise ValueError(f"Invalid host staging header room={room}")
            table = (
                ring[
                    offset + HEADER.itemsize : offset
                    + HEADER.itemsize
                    + n * ROW.itemsize
                ]
                .copy()
                .view(ROW)
            )
            lens = table["len"]
            if int(lens.sum()) != int(header["payload"]):
                raise ValueError(f"Host staging table/payload mismatch room={room}")
            notif = bytes(header["notif"][: header["notif_len"]]).decode("ascii")
            tag = notif.split("_", 2)
            if tag[0] != str(room) or tag[1] not in ("kv", "state"):
                raise ValueError(f"Invalid host staging inner notification {notif!r}")
            self.check_rows(receiver, table["addr"], lens)
            src = np.r_[0, np.cumsum(lens)[:-1]].astype(np.uint64) + np.uint64(
                self.staging_allocator.get_base_ptr() + offset + start
            )
            event = torch.cuda.Event()
            with torch.cuda.stream(self.host.stream):
                copy_rows(
                    table["addr"],
                    src,
                    lens,
                    self.host.device,
                    waits=(receiver.host_mapping_ready,),
                )
                event.record(self.host.stream)
            receiver.host_scatters.append((event, writer, seq, header))

    def poll(self, room):
        """Retire finished scatters; deliver each WRITE's notif once all parts land."""
        with self.lock:
            receiver = self._room_to_receiver.get(room)
            if receiver is None or not receiver.host_scatters:
                return
            pending, notifs = [], []
            for scatter in receiver.host_scatters:
                event, writer, seq, header = scatter
                if not event.query():
                    pending.append(scatter)
                    continue
                alloc_id = receiver.host_allocs.pop((writer, seq))[0]
                self._free_and_send_watermark(alloc_id, None)
                key = (writer, int(header["group"]))
                receiver.host_groups[key] += 1
                if receiver.host_groups[key] == header["parts"]:
                    del receiver.host_groups[key]
                    notifs.append(
                        (writer, bytes(header["notif"][: header["notif_len"]]))
                    )
            receiver.host_scatters = pending
        for writer, notif in notifs:
            self.kv_manager._dispatch_notif(writer, notif)

    def release_room(self, room, decode_req, receiver):
        with self.lock:
            if receiver is None:
                return
            if receiver.host_scatters:
                self.host.stream.synchronize()  # No scatter reads a freed region.
                receiver.host_scatters.clear()
            # Unsubscribe first: this room's prefill may be the unreachable one.
            key = tuple(str(i) for i in receiver.bootstrap_infos or ())
            rooms = self._wm_subscribers.get(key, {})
            rooms.pop(room, None)
            if not rooms:
                self._wm_subscribers.pop(key, None)
            # Even without a drain ack: the writer's deadlines have passed by now.
            for alloc_id, *_ in receiver.host_allocs.values():
                self._free_and_send_watermark(alloc_id, decode_req)
            receiver.host_allocs.clear()
