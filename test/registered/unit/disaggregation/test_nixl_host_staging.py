"""CPU tests for NIXL HOST staging under the native transfer planner.

Production code runs end to end: the native prefill transfer worker and planners,
HOST's seam backend and slot pipeline, the decode ring allocator, row validation,
scatter, native completion tracking and decode admission. Only NIXL itself and the
CUDA runtime are simulated: "device" memory is CPU tensors, the fake agent copies
bytes on completion, and the row copy is a memmove. Not live NIXL validation.
"""

import ctypes
import threading
import time
import unittest
from collections import defaultdict, deque
from contextlib import ExitStack, nullcontext
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import numpy as np
import torch
import zmq

from sglang.srt.disaggregation.base.conn import KVPoll, StateType
from sglang.srt.disaggregation.common.conn import KVTransferError
from sglang.srt.disaggregation.common.staging_buffer import (
    StagingAllocator,
    StagingBuffer,
)
from sglang.srt.disaggregation.common.staging_handler import (
    DecodeStagingContext,
    PrefillStagingContext,
    handle_staging_rsp,
    handle_watermark_msg,
)
from sglang.srt.disaggregation.common.utils import FastQueue
from sglang.srt.disaggregation.decode import DecodeTransferQueue
from sglang.srt.disaggregation.nixl import host_staging as H
from sglang.srt.disaggregation.nixl.conn import (
    GUARD,
    KVArgsRegisterInfo,
    NixlKVManager,
    NixlKVReceiver,
    NixlKVSender,
    TransferInfo,
    TransferStatus,
)
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

ROOM = 37
MODES = {"prefill": DisaggregationMode.PREFILL, "decode": DisaggregationMode.DECODE}
PARALLEL = lambda: NS(num_dp_ranks=1, tp_size=1, enable_dsa_cache_layer_split=False)


class FailStop(BaseException):
    pass


class Event:
    auto_complete = True

    def __init__(self):
        self.done = False

    def record(self, stream=None):
        self.done = self.auto_complete

    def query(self):
        return self.done


class Stream:
    def wait_event(self, event):
        pass

    def synchronize(self):
        pass


def memmove_rows(dst, src, sizes, device, waits=()):
    for d, s, n in zip(dst.tolist(), src.tolist(), sizes.tolist()):
        ctypes.memmove(d, s, n)


class Agent:
    """NIXL stand-in. A WRITE lands (memmove) when completed; then its notif.

    Under HOST it accepts only DRAM descriptors: KV/state memory is never
    registered, so any GPU descriptor or prepared dlist is a bug.
    """

    def __init__(self, name, registry, host):
        self.name, self.registry, self.host = name, registry, host
        self.pending, self.handles = defaultdict(list), []
        self.lock = threading.Lock()
        self.manual = False
        registry[name] = self

    def get_agent_metadata(self):
        return b"fake-agent"

    def add_remote_agent(self, metadata):
        pass

    def prep_xfer_dlist(self, peer, rows, kind):
        assert not self.host, "HOST prepared a NIXL dlist"
        rows = np.asarray(rows, dtype=np.uint64)
        if rows.shape[1] == 5:
            rows = NixlKVManager._expand_stride_descs(rows)
        return NS(rows=rows, peer=peer, kind=kind)

    def make_prepped_xfer(self, op, src, src_idx, dst, dst_idx, notif):
        assert not self.host, "HOST posted a prepped NIXL xfer"
        return self.initialize_xfer(
            op, src.rows[src_idx].tolist(), dst.rows[dst_idx].tolist(), dst.peer, notif
        )

    def get_xfer_descs(self, rows, kind):
        rows = [tuple(int(v) for v in row) for row in rows]
        if self.host and (kind != "DRAM" or any(row[2] != 0 for row in rows)):
            raise AssertionError("HOST issued a GPU descriptor")
        return rows

    def initialize_xfer(self, op, src, dst, peer, notif):
        handle = NS(src=src, dst=dst, peer=peer, notif=notif, done=False)
        handle.released = False
        self.handles.append(handle)
        return handle

    def transfer(self, handle):
        return "PROC"

    def check_xfer_state(self, handle):
        if not handle.done and not self.manual:
            self.complete(handle)
        return "DONE" if handle.done else "PROC"

    def complete(self, handle):
        if handle.done:
            return
        for src, dst in zip(handle.src, handle.dst, strict=True):
            assert src[1] == dst[1], "Descriptor length mismatch"
            ctypes.memmove(int(dst[0]), int(src[0]), int(src[1]))
        handle.done = True
        self.send_notif(handle.peer, handle.notif)

    def release_xfer_handle(self, handle):
        assert handle.done or getattr(handle, "err", False), (
            "Released an uncertain write"
        )
        handle.released = True

    def send_notif(self, peer, notif):
        target = self.registry[peer]
        with target.lock:
            target.pending[self.name].append(notif)

    def get_new_notifs(self):
        with self.lock:
            pending, self.pending = self.pending, defaultdict(list)
        return dict(pending)


class BusyLock:
    """A socket send lock another sender thread holds."""

    def acquire(self, blocking=True):
        assert not blocking, "HOST blocked on a busy socket lock"
        return False


def endpoint_name(endpoint):
    return endpoint.split("//")[-1].rsplit(":", 1)[0]


class Rank:
    """One decode rank: manager, HOST handler, receiver, request and queue."""


class Rig:
    """One prefill rank -> decode_tp decode ranks, over HOST or native NIXL.

    MLA rigs carry KV (target + draft) plus state components; MHA rigs carry
    per-layer K and V with kv heads split across decode ranks (hetero TP).
    """

    def __init__(
        self,
        count=4,
        slot_bytes=4096,
        state=("dsa",),
        pages=16,
        ring=1,
        host=True,
        decode_tp=1,
        mla=True,
        pages_src=(11, 3, 14, 1, 9, 5, 12, 7, 2),
        pages_dst=(4, 13, 2, 10, 1, 15, 8, 3, 6),
        dcp=1,
        prefill_tp=1,
        prefill_pp=1,
        draft=0,
    ):
        self.source = np.array(pages_src[:count], dtype=np.int32)
        self.destination = np.array(pages_dst[:count], dtype=np.int32)
        self.pages, self.ring, self.host, self.mla = pages, ring, host, mla
        self.registry, self.dcp, self.draft = {}, dcp, draft
        self.control_full = False  # Prefill control sockets refuse sends.
        if dcp > 1:
            # DCP relayout moves single tokens: 4-token pages, 4/8 bytes per token.
            # A draft entry (the last) stays unsharded: dcp tokens per decode row.
            self.page_size, self.src_kv = 4, [16, 32]
            self.dst_kv = [16, 32 * dcp if draft else 32]
        elif mla:
            # Bytes per page per entry: two latent layers (target + draft); each
            # prefill PP stage owns its own layers of the decode's full model.
            self.page_size, self.src_kv = 64, [12, 20]
            self.dst_kv = [12, 20] * prefill_pp
        else:
            # K and V of one layer: 4 tokens x 2 heads x 4 bytes per page.
            self.page_size = 4
            self.src_kv = [32 // prefill_tp] * 2
            self.dst_kv = [32 // decode_tp] * 2
        # Per state component: type, bytes per row per entry, src/dst row indices.
        # DSA rows are pages (placeholder entry 1 is zero-row); Mamba rows are
        # per-request state slots.
        kinds = {
            "dsa": (StateType.DSA, [8, 0, 16], self.source, self.destination),
            "mamba": (StateType.MAMBA, [24, 40], np.array([5]), np.array([9])),
        }
        self.components = [kinds[k] for k in state]
        self.state_items = [n for _, items, _, _ in self.components for n in items]
        self.src_aux = [torch.tensor([[37]]), torch.tensor([[123]])]
        self.src_aux.append(torch.tensor([[64, 64, 0, 0, 0, 0, 0]]))
        # Each prefill rank owns distinct source bytes (its own kv heads).
        self.prefill_pp = prefill_pp
        writers = prefill_tp * prefill_pp
        self.srcs = [
            [t + p for t in self.buffers(self.src_kv, fill=None)]
            for p in range(writers)
        ]
        self.prefills = [
            self.manager(
                "prefill" if p == 0 else f"prefill{p}",
                self.srcs[p],
                self.src_aux,
                slot_bytes,
                prefill_tp,
                p,
            )
            for p in range(writers)
        ]
        self.prefill, self.src = self.prefills[0], self.srcs[0]
        tp = max(decode_tp, dcp)
        self.ranks = [self.make_rank(r, tp, slot_bytes) for r in range(tp)]
        rank0 = self.ranks[0]
        self.decode, self.handler, self.receiver = (
            rank0.m,
            rank0.handler,
            rank0.receiver,
        )
        self.req, self.queue, self.dst = rank0.req, rank0.queue, rank0.dst
        self.senders = []
        for m in self.prefills:
            sender = NixlKVSender(m, "prefill", ROOM, [0], 0)
            sender.init(count, 0)
            sender._host_ready_event = Event()
            m.update_status(ROOM, KVPoll.WaitingForInput)
            self.senders.append(sender)
            threading.Thread(
                target=m.transfer_worker,
                args=(m.transfer_queues[0], None, 0),
                daemon=True,
            ).start()
        self.sender = self.senders[0]

    def buffers(self, kv_items, fill):
        out = []
        for n in kv_items + self.state_items:
            if fill is None:
                t = torch.arange(self.pages * n).remainder(251).to(torch.uint8)
            else:
                t = torch.full((self.pages * n,), fill, dtype=torch.uint8)
            out.append(t.reshape(self.pages, n))
        return out

    def manager(self, name, buffers, aux, slot_bytes, tp, rank):
        m = NixlKVManager.__new__(NixlKVManager)
        m.agent = Agent(name, self.registry, self.host)
        m.host_staging_bytes = slot_bytes * H.SLOT_COUNT * self.ring
        prefill = name.startswith("prefill")
        m.disaggregation_mode = MODES["prefill" if prefill else "decode"]
        m.request_status, m.transfer_infos, m.decode_kv_args_table = {}, {}, {}
        m.transfer_statuses = defaultdict(TransferStatus)
        m.required_prefill_response_num_table, m.prefill_response_tracker = {}, {}
        m._deferred_abort_ack_tracker, m._deferred_ack_targets = {}, {}
        m._deferred_ack_fanout_snapshots = {}
        m._staging_outstanding = defaultdict(int)
        m._abandoned, m._abandoned_lock, m._aux_in_flight = {}, threading.Lock(), set()
        m.transfer_queues = [FastQueue()]
        m.exceptions, m.failure_records = {}, {}
        m.failure_lock = threading.Lock()
        m.req_to_decode_prefix_len = {}
        m.addr_to_rooms_tracker = defaultdict(set)
        m.attn_tp_rank, m.attn_tp_size = rank, tp
        m.pp_rank = m.attn_cp_rank = 0
        m.pp_size = m.attn_cp_size = 1
        if prefill and self.prefill_pp > 1:
            m.pp_size, m.pp_rank, m.attn_tp_rank, rank = self.prefill_pp, rank, 0, 0
        dcp = not prefill and self.dcp > 1
        m.dcp_size, m.dcp_rank = (self.dcp, rank) if dcp else (1, 0)
        m.transfer_source_rank = m.pp_rank * tp + rank if prefill else 0
        m.enable_all_cp_ranks_for_transfer = m.is_dummy_cp_rank = False
        m.is_mla_backend, m.is_hybrid_mla_backend = self.mla, False
        m.enable_staging = self.host
        m.enable_deferred_decode_kv_release = True
        m.local_ip, m.rank_port, m.waiting_timeout = name, 1, 10
        m.src_mem_kind = "VRAM"
        m.prep_handles, m.prep_handles_slice_dst = {}, {}
        m.prep_handles_segment_src, m.prep_handle_slice_src = {}, None
        m._num_slots_src = self.pages
        m.kv_buffer_tensors = m._staging_handler = None
        m._dcp_pack_buffers = []  # Per-token rows in both modes; no pack kernel.
        nkv = len(self.src_kv) if prefill else len(self.dst_kv)
        state, comps = buffers[nkv:], []
        for _, items, _, _ in self.components:
            comps.append(state[: len(items)])
            state = state[len(items) :]
        m.kv_args = NS(
            page_size=self.page_size,
            engine_rank=rank,
            gpu_id=0,
            prefill_start_layer=m.pp_rank * nkv,
            prefill_end_layer=(m.pp_rank + 1) * nkv,
            mla_compression_ratios=None,
            num_draft_entries=self.draft,
            total_kv_head_num=2,
            kv_head_num=2 // tp,
            kv_data_ptrs=[b.data_ptr() for b in buffers[:nkv]],
            kv_data_lens=[b.nbytes for b in buffers[:nkv]],
            kv_item_lens=[b.shape[1] for b in buffers[:nkv]],
            kv_data_mem_kinds=["VRAM"] * nkv,
            kv_layer_ids=[],
            state_types=[c[0] for c in self.components],
            state_data_ptrs=[
                [b.data_ptr() if b.nbytes else 0 for b in c] for c in comps
            ],
            state_data_lens=[[b.nbytes for b in c] for c in comps],
            state_item_lens=[list(c[1]) for c in self.components],
            state_layer_ids=[[] for _ in self.components],
            state_dim_per_tensor=[],
            aux_data_ptrs=[t.data_ptr() for t in aux],
            aux_data_lens=[t.nbytes for t in aux],
            aux_item_lens=[t.nbytes for t in aux],
        )
        m._socket_send_locks = defaultdict(threading.Lock)
        m._connect = lambda endpoint, is_ipv6=False: NS(
            send_multipart=lambda parts, flags=0: self.control(endpoint, parts, flags)
        )
        if not self.host:
            m.host_staging = None
            m._staging_ctx = None
            return m
        h = m.host_staging = H.HostStaging.__new__(H.HostStaging)
        h.manager, h.capacity, h.slot_bytes = m, m.host_staging_bytes, slot_bytes
        h.lock, h.device = threading.Lock(), torch.device("cpu")
        h.config = {
            "version": H.VERSION,
            "capacity": h.capacity,
            "post_deadline_sec": H.POST_DEADLINE_S,
            "write_deadline_sec": H.WRITE_DEADLINE_S,
        }
        h.queue, h.seqs, h.ready, h.failed_rooms = deque(), defaultdict(int), {}, set()
        h.slots, h.allocator, h.stream = [], None, Stream()
        if prefill:
            m._start_host_abort_ack_worker()
            h.slots = [
                H.HostSlot(StagingBuffer(slot_bytes, "cpu", 0), Stream())
                for _ in range(H.SLOT_COUNT)
            ]
            m._staging_ctx = PrefillStagingContext()
        else:
            h.allocator = StagingAllocator(h.capacity, "cpu", 0)
            h.allocator._scatter_stream = h.stream
            m._staging_ctx = DecodeStagingContext(allocator=h.allocator)
        m._prep_dlist, m._post_write = h.prep_dlist, h.post_write
        m._post_prepped, m._xfer_state = h.post_prepped, h.xfer_state
        return m

    def make_rank(self, r, tp, slot_bytes):
        rank = Rank()
        name = "decode" if r == 0 else f"decode{r}"
        rank.dst = self.buffers(self.dst_kv, fill=255)
        rank.aux = [torch.zeros_like(t) for t in self.src_aux]
        rank.m = m = self.manager(name, rank.dst, rank.aux, slot_bytes, tp, r)
        rank.handler = H.HostDecodeStagingHandler(m, NS(), r) if self.host else None
        m._staging_handler = rank.handler
        rank.receiver = rcv = NixlKVReceiver(m, "prefill", ROOM)
        rcv.bootstrap_infos = [
            {
                "rank_ip": p.agent.name,
                "rank_port": 1,
                "pp_rank": p.pp_rank,
                "is_dummy": False,
            }
            for p in self.prefills
        ]
        prefill_tp = len(self.prefills) // self.prefill_pp
        rcv.prefill_info = NS(
            attn_tp_size=prefill_tp, pp_size=self.prefill_pp, attn_cp_size=1
        )
        # Each prefill rank sees every decode rank it feeds.
        rcv.require_staging, rcv.required_dst_info_num = (
            self.host,
            max(tp // prefill_tp, 1),
        )
        m.required_prefill_response_num_table[ROOM] = len(self.prefills)
        rcv._connect_to_bootstrap_server = lambda info: (
            NS(
                send_multipart=lambda parts, flags=0: self.to_prefill(
                    info["rank_ip"], parts
                )
            ),
            threading.Lock(),
        )
        rank.req = NS(
            req=NS(
                bootstrap_room=ROOM,
                rid=f"rid-{r}",
                kv=NS(cache_protected_len=0),
                finished_reason=None,
                bootstrap_host="prefill",
                output_ids=[],
                return_logprob=False,
                return_sampling_mask=False,
                time_stats=NS(set_wait_queue_entry_time=Mock()),
            ),
            kv_receiver=rcv,
            metadata_buffer_index=0,
            hicache_restore_status="done",
            hicache_load_consumer_index=-1,
            host_staged=False,
            is_rebootstrap=False,
        )
        if self.host:
            rank.handler.register_decode_req(ROOM, rank.req)
        rcv._register_kv_args()
        rcv.send_metadata(self.destination, 0, [c[3] for c in self.components], 0)
        rank.queue = q = DecodeTransferQueue.__new__(DecodeTransferQueue)
        q.queue, q.enable_staging, q.staging_handler = (
            [rank.req],
            self.host,
            rank.handler,
        )
        q.gloo_group, q.spec_algorithm = None, NS(is_none=lambda: True)
        q.metadata_buffers = NS(
            bootstrap_room=rank.aux[0],
            get_buf=lambda idx: (
                rank.aux[1][idx],
                rank.aux[2][idx],
                *([None] * 11),
                rank.aux[0][idx],
            ),
        )
        q.scheduler = NS(
            enable_decode_hicache=False,
            kv_checksum_computer=None,
            enable_hisparse=False,
            server_args=NS(disaggregation_transfer_backend="nixl"),
            metrics_reporter=NS(enable_metrics=False),
            batch_result_processor=NS(_maybe_update_reasoning_tokens=Mock()),
        )
        q.req_to_metadata_buffer_idx_allocator = NS(free=Mock())
        q._commit_hicache_local_restore_to_req = Mock()
        rank.name = name
        return rank

    def control(self, endpoint, parts, flags):
        if self.control_full:
            assert flags & zmq.NOBLOCK, "HOST blocked on a full control socket"
            raise zmq.Again()
        self.to_decode(endpoint_name(endpoint), parts)

    def to_decode(self, name, msg):
        m = next(rank.m for rank in self.ranks if rank.name == name)
        if msg[0] == b"ABORT_ACK":
            m.note_abort_ack(int(msg[1]), int(msg[2]))
        elif msg[0] == b"STAGING_REQ":
            m._handle_staging_req(msg)  # KV_STATUS etc. are not modeled.

    def to_prefill(self, name, msg):
        prefill = next(p for p in self.prefills if p.agent.name == name)
        if msg[0] == b"STAGING_RSP":
            handle_staging_rsp(msg, prefill.transfer_infos)
        elif msg[0] == b"WATERMARK":
            handle_watermark_msg(prefill._staging_ctx, msg)
        elif msg[0] == b"ABORT":
            prefill._handle_abort_notification(msg)
        elif msg[0] == GUARD and msg[1] == b"None":
            prefill._add_remote_peer(KVArgsRegisterInfo.from_zmq(msg[1:]))
        elif msg[0] == GUARD:
            info = TransferInfo.from_zmq(msg[1:])
            prefill.transfer_infos.setdefault(info.room, {})[info.agent_name] = info
        else:
            raise AssertionError(msg)

    def submit(self, chunks):
        offset = 0
        for n in chunks:
            end = offset + n
            last = end == len(self.source)
            state = [c[2] for c in self.components] if last else None
            for sender in self.senders:
                sender.send(
                    self.source[offset:end],
                    state_indices=state or None,
                    num_kv_tokens=n * self.page_size,
                )
            offset = end
        assert offset == len(self.source), "Incomplete test input"

    def tick(self):
        if self.host:
            for m in self.prefills:
                m.host_staging.progress()
        return [req for rank in self.ranks for req in rank.queue.pop_transferred()]

    def drive(self, chunks=None, timeout=5):
        self.submit(chunks or [len(self.source)])
        admitted, deadline = [], time.monotonic() + timeout
        while any(r.queue.queue for r in self.ranks) and time.monotonic() < deadline:
            admitted += self.tick()
            time.sleep(0.001)
        return admitted

    def decoded(self):
        """Every decode destination byte, per rank."""
        return [[bytes(t.numpy()) for t in rank.dst] for rank in self.ranks]

    def assert_bytes(self, case):
        rows = [(self.source, self.destination)] * len(self.src_kv)
        for _, items, src_idx, dst_idx in self.components:
            rows += [(src_idx, dst_idx)] * len(items)
        for src, dst, (src_idx, dst_idx) in zip(self.src, self.dst, rows, strict=True):
            untouched = sorted(set(range(self.pages)) - set(dst_idx.tolist()))
            case.assertTrue(torch.equal(src[src_idx], dst[dst_idx]))
            case.assertTrue(torch.all(dst[untouched] == 255))


class HostStagingTest(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)

        def fail_stop(reason):
            raise FailStop(reason)

        Event.auto_complete = True
        for context in (
            patch.object(H, "fail_stop", side_effect=fail_stop),
            patch(
                "sglang.srt.disaggregation.nixl.conn.fail_stop", side_effect=fail_stop
            ),
            patch.object(H, "copy_rows", memmove_rows),
            patch.object(torch.cuda, "Event", Event),
            patch.object(torch.cuda, "current_stream", lambda *a: Stream()),
            patch.object(torch.cuda, "stream", lambda s: nullcontext()),
            patch.object(torch.cuda, "set_device"),
            patch("torch.distributed.get_world_size", lambda group=None: 1),
            *(
                patch(f"sglang.srt.disaggregation.{mod}.get_parallel", PARALLEL)
                for mod in ("common.conn", "nixl.conn")
            ),
        ):
            self.stack.enter_context(context)

    def test_mla_dsa_bytes_land_through_native_planner(self):
        rig = Rig()
        self.assertEqual(rig.drive(), [rig.req.req])
        rig.assert_bytes(self)
        # Admission consumed the natively written aux (and cleared its room tag).
        self.assertEqual(rig.req.req.output_ids, [123])
        self.assertEqual(rig.req.req.cached_tokens, 64)
        # Only pinned slot -> ring WRITEs and aux ever reached NIXL.
        notifs = {h.notif.split(b"_")[1] for h in rig.prefill.agent.handles}
        self.assertEqual(notifs, {b"hst", b"aux"})
        self.assertFalse(rig.handler.staging_allocator.allocations)
        self.assertEqual(rig.prefill.check_status(ROOM), KVPoll.Success)

    def test_host_bytes_equal_native_nixl(self):
        """Native NIXL is the oracle: HOST must leave identical decode bytes."""
        cases = {
            "mla+dsa": {},
            "mla+mamba+dsa": {"state": ("mamba", "dsa")},
            "mla fan-out tp1->2": {"decode_tp": 2},
            "mha equal tp": {"mla": False, "state": ()},
            "mha hetero tp1->2 head slices": {
                "mla": False,
                "state": (),
                "decode_tp": 2,
            },
            "mla dcp1->2 token relayout": {"dcp": 2, "state": ()},
            "mha prefill tp2 -> decode tp1 (two writers)": {
                "mla": False,
                "state": (),
                "prefill_tp": 2,
            },
            "mla prefill pp2 -> decode pp1 (stage layer offsets)": {
                "state": (),
                "prefill_pp": 2,
            },
            "K3 shape: dcp1->2 + draft KV entry + mamba": {
                "dcp": 2,
                "draft": 1,
                "state": ("mamba",),
            },
            "contiguous source pages": {"pages_src": (3, 4, 5, 6)},
            "contiguous both sides (rows merge)": {
                "pages_src": (3, 4, 5, 6),
                "pages_dst": (8, 9, 10, 11),
            },
            "rows coalesce past a slot (row split)": {
                "count": 5,  # 5 x 20-byte pages: a 100-byte row, 88-byte payload.
                "pages_src": (3, 4, 5, 6, 7),
                "pages_dst": (8, 9, 10, 11, 12),
                "slot_bytes": 600,
            },
            "chunked, split parts, ring wrap": {
                "count": 9,
                "slot_bytes": 600,
                "chunks": [2, 4, 3],
            },
        }
        for name, kwargs in cases.items():
            with self.subTest(name):
                chunks = kwargs.pop("chunks", None)
                decoded = {}
                for host in (False, True):
                    rig = Rig(host=host, **kwargs)
                    fresh = rig.decoded()
                    self.assertEqual(len(rig.drive(chunks)), len(rig.ranks))
                    decoded[host] = rig.decoded()
                    if host:
                        self.assertFalse(rig.handler.staging_allocator.allocations)
                self.assertNotEqual(decoded[False], fresh)  # Native wrote bytes.
                self.assertEqual(decoded[True], decoded[False])

    def test_mamba_and_dsa_components_land_and_all_count(self):
        # Two state components: Mamba (whole-slot rows) and DSA (page rows) take
        # different native planner paths; decode waits for both notifs.
        rig = Rig(state=("mamba", "dsa"))
        status = rig.decode.transfer_statuses[ROOM]  # Admission pops it.
        self.assertEqual(rig.drive(), [rig.req.req])
        rig.assert_bytes(self)
        self.assertEqual(status.expected_state_per_pp, {0: 2})
        self.assertEqual(status.received_state_per_pp[0], {0, 1})

    def test_split_parts_ring_wrap_and_out_of_order_completion(self):
        # Tiny slots split every WRITE into parts; a 2-slot ring wraps constantly.
        rig = Rig(count=9, slot_bytes=600)
        rig.prefill.agent.manual = True
        rig.submit([2, 4, 3])
        deadline = time.monotonic() + 5
        parts = 0
        while rig.queue.queue and time.monotonic() < deadline:
            rig.tick()
            posted = [h for h in rig.prefill.agent.handles if not h.done]
            # Land WRITEs newest first, so parts of one WRITE arrive out of order.
            for handle in reversed(posted):
                parts += handle.notif.split(b"_")[1] == b"hst"
                rig.prefill.agent.complete(handle)
            time.sleep(0.001)
        self.assertFalse(rig.queue.queue)
        self.assertGreater(parts, 6)  # More ring WRITEs than native WRITEs.
        rig.assert_bytes(self)
        self.assertFalse(rig.handler.staging_allocator.allocations)

    def test_inner_notif_waits_for_every_part(self):
        # The KV WRITE splits into 4 parts; the ring holds all of them at once.
        rig = Rig(count=4, slot_bytes=600, ring=2)
        rig.prefill.agent.manual = True
        rig.submit([4])
        status = rig.decode.transfer_statuses[ROOM]
        ring = lambda seq: next(  # noqa: E731
            (
                h
                for h in rig.prefill.agent.handles
                if h.notif == f"{ROOM}_hst_{seq}".encode()
            ),
            None,
        )
        # Land parts 1..3 while part 0 stays in flight in its slot.
        for _ in range(500):
            rig.tick()
            for seq in (1, 2, 3):
                if ring(seq) is not None:
                    rig.prefill.agent.complete(ring(seq))
            if ring(3) is not None and ring(3).done:
                break
            time.sleep(0.001)
        rig.tick()
        self.assertFalse(status.received_kvs_per_pp[0])
        rig.prefill.agent.complete(ring(0))
        rig.tick()
        self.assertEqual(status.received_kvs_per_pp[0], {0})

    def test_rows_outside_the_room_fail_stop(self):
        rig = Rig()
        rig.prefill.agent.manual = True
        rig.submit([4])
        for _ in range(200):
            rig.tick()
            ring = [h for h in rig.prefill.agent.handles if b"_hst_" in h.notif]
            if ring:
                break
            time.sleep(0.001)
        (src, _, _), (dst, _, _) = ring[0].src[0], ring[0].dst[0]
        rig.prefill.agent.complete(ring[0])
        # Retarget the first row at page 0, which this room does not own.
        table = (ctypes.c_uint64 * 2).from_address(dst + H.HEADER.itemsize)
        table[0] = rig.dst[0].data_ptr()
        with self.assertRaisesRegex(FailStop, "outside room"):
            rig.tick()

    def test_metadata_only_room_completes_on_native_aux(self):
        rig = Rig(count=0)
        self.assertEqual(rig.drive(), [rig.req.req])
        self.assertFalse([h for h in rig.prefill.agent.handles if b"_hst_" in h.notif])
        # A peer that sends no component count: done on its first component.
        rig.decode._dispatch_notif("prefill", b"38_state_0_1")
        status = rig.decode.transfer_statuses[38]
        self.assertEqual(status.expected_state_per_pp, {0: 1})
        self.assertEqual(status.received_state_per_pp, {0: {1}})

    def test_incompatible_peer_is_refused_not_fatal(self):
        rig = Rig()
        peer = NS(
            agent_name="other",
            host_staging_config=None,
            staging_base_ptr=0,
            staging_total_size=0,
        )
        rig.prefill._add_remote_peer(peer)  # HOST prefill, non-HOST decode.
        self.assertNotIn("other", rig.prefill.decode_kv_args_table)

    def test_failed_room_drops_unposted_parts(self):
        rig = Rig(count=4, slot_bytes=600)
        rig.submit([4])
        host = rig.prefill.host_staging
        for _ in range(200):
            if len(host.queue) > H.SLOT_COUNT:
                break
            time.sleep(0.001)
        rig.decode._handle_staging_req = Mock()  # Allocation granted below, later.
        host.progress()  # Both slots gathered and asked for ring space.
        self.assertTrue(all(slot.part for slot in host.slots))
        rig.prefill.update_status(ROOM, KVPoll.Failed)
        self.assertFalse(host.queue)  # Queued parts dropped at once.
        for call in rig.decode._handle_staging_req.call_args_list:
            rig.handler.allocate(call.args[0])  # Decode had not heard yet.
        for _ in range(50):
            rig.prefill.host_staging.progress()
            time.sleep(0.001)
        # No ring WRITE was posted (native aux may land; deferred release fences it).
        self.assertFalse([h for h in rig.prefill.agent.handles if b"_hst_" in h.notif])
        self.assertFalse(rig.sender.is_source_pending())

    def test_failed_room_stays_released_after_clear(self):
        # A room fails (client AbortReq, or a transfer error) with a non-last
        # chunk still outstanding (no aux posted). The scheduler's failure path
        # calls failure_exception(), whose clear() pops the room's status, then
        # re-checks is_source_pending() before releasing the KV pages. The room
        # was drained when it failed; clearing it must not make it pending.
        for fail in ("abort", "transfer"):
            with self.subTest(fail=fail):
                rig = Rig(count=4, slot_bytes=600)
                rig.decode._handle_staging_req = Mock()  # Parts wait for ring space.
                rig.submit([2, 2])
                host = rig.prefill.host_staging
                for _ in range(200):
                    if host.queue:
                        break
                    time.sleep(0.001)
                host.progress()  # Slots claimed: the first chunk stays outstanding.
                self.assertFalse(rig.sender.is_aux_in_flight())
                if fail == "abort":
                    rig.sender.abort()
                else:
                    rig.prefill.record_failure(ROOM, "transfer failed")
                    rig.prefill.update_status(ROOM, KVPoll.Failed)
                self.assertGreater(rig.prefill._staging_outstanding.get(ROOM, 0), 0)
                self.assertEqual(rig.sender.poll(), KVPoll.Failed)
                with self.assertRaises(KVTransferError):
                    rig.sender.failure_exception()
                self.assertNotIn(ROOM, rig.prefill.request_status)
                # The release guards (release_kv_cache, metadata buffer, clear).
                self.assertFalse(rig.sender.is_source_pending())
                rig.sender.clear()
                # The claimed parts retire unposted and the chunk settles.
                for _ in range(200):
                    host.progress()
                    if not rig.prefill._staging_outstanding.get(ROOM, 0):
                        break
                    time.sleep(0.001)
                self.assertFalse(rig.prefill._staging_outstanding.get(ROOM, 0))
                self.assertFalse(
                    [h for h in rig.prefill.agent.handles if b"_hst_" in h.notif]
                )

    def test_dropped_part_keeps_its_slot_until_its_gather_finished(self):
        Event.auto_complete = False  # Gathers wait on a forward still running.
        rig = Rig(count=4, slot_bytes=600)
        rig.submit([4])
        host = rig.prefill.host_staging
        for _ in range(200):
            if len(host.queue) > H.SLOT_COUNT:
                break
            time.sleep(0.001)
        host.progress()
        worker = threading.Thread(
            target=rig.prefill.update_status, args=(ROOM, KVPoll.Failed)
        )
        worker.start()
        try:
            for _ in range(200):
                if ROOM in host.failed_rooms:
                    break
                time.sleep(0.001)
            self.assertIn(ROOM, host.failed_rooms)
            self.assertNotEqual(rig.prefill.check_status(ROOM), KVPoll.Failed)
            host.progress()
            self.assertTrue(all(slot.part for slot in host.slots))
        finally:
            for slot in host.slots:
                slot.copy_done.done = True
            host.progress()
            worker.join(timeout=2)
        self.assertFalse(worker.is_alive())
        self.assertEqual(rig.prefill.check_status(ROOM), KVPoll.Failed)
        self.assertFalse(any(slot.part for slot in host.slots))

    def test_timeouts_before_and_after_posting(self):
        rig = Rig(count=4)
        rig.decode._handle_staging_req = lambda msg: None  # Never allocates.
        rig.submit([4])
        deadline = time.monotonic() + 5
        with patch.object(H, "POST_DEADLINE_S", 0.05):
            while rig.prefill.check_status(ROOM) != KVPoll.Failed:
                self.assertLess(time.monotonic(), deadline)
                rig.prefill.host_staging.progress()
                time.sleep(0.01)
        # No ring WRITE was posted (native aux may land; deferred release fences it).
        self.assertFalse([h for h in rig.prefill.agent.handles if b"_hst_" in h.notif])

        rig = Rig(count=4)
        rig.prefill.agent.manual = True
        rig.submit([4])
        for _ in range(200):
            rig.tick()
            if any(b"_hst_" in h.notif for h in rig.prefill.agent.handles):
                break
            time.sleep(0.001)
        with (
            patch.object(H, "POST_DEADLINE_S", 0),
            patch.object(H, "WRITE_DEADLINE_S", 0),
            self.assertRaisesRegex(FailStop, "WRITE timeout"),
        ):
            rig.prefill.host_staging.progress()
        for handle in rig.prefill.agent.handles:
            rig.prefill.agent.complete(handle)
        rig.prefill.host_staging.progress()  # Settled past the deadline: no fail-stop.

    def test_post_deadline_runs_from_the_staging_request(self):
        # Time queued behind busy slots is not transport time: only the step the
        # decode's quarantine must outlast (request, grant, WRITE) is bounded.
        rig = Rig(count=4)
        rig.submit([4])
        for _ in range(200):
            if rig.prefill.host_staging.queue:
                break
            time.sleep(0.001)
        time.sleep(0.3)
        admitted, deadline = [], time.monotonic() + 5
        with patch.object(H, "POST_DEADLINE_S", 0.25):
            while not admitted and time.monotonic() < deadline:
                admitted = rig.tick()
                time.sleep(0.001)
        self.assertTrue(admitted)
        self.assertNotEqual(rig.prefill.check_status(ROOM), KVPoll.Failed)

    def test_source_pending_tracks_host_gathers_only(self):
        # Optimistic prefill sends before decode bootstraps: the native worker
        # skips the chunk, nothing gathers, so releasing the pages is safe.
        rig = Rig()
        del rig.prefill.transfer_infos[ROOM]
        rig.submit([4])
        for _ in range(200):
            if not rig.prefill._staging_outstanding.get(ROOM):
                break
            time.sleep(0.001)
        time.sleep(0.05)
        self.assertFalse(rig.sender.is_source_pending())
        # With metadata, queued parts pin the source until gathered.
        rig = Rig()
        rig.submit([4])
        for _ in range(200):
            if rig.prefill.host_staging.queue:
                break
            time.sleep(0.001)
        self.assertTrue(rig.sender.is_source_pending())

    def test_host_peer_must_use_same_writer_deadlines(self):
        rig = Rig()
        host = rig.prefill.host_staging
        peer = NS(
            host_staging_config=host.config.copy(),
            staging_base_ptr=1,
            staging_total_size=host.capacity,
        )
        self.assertFalse(host.config_mismatch(peer))
        peer.host_staging_config["write_deadline_sec"] += 1
        self.assertTrue(host.config_mismatch(peer))

    def test_ring_write_err_fails_the_room_not_the_worker(self):
        rig = Rig(count=4)
        rig.prefill.agent.manual = True
        rig.submit([4])
        for _ in range(500):
            rig.tick()
            ring = [x for x in rig.prefill.agent.handles if b"_hst_" in x.notif]
            if ring:
                break
            time.sleep(0.001)
        for x in ring:
            x.err = True  # The peer died: NIXL settles the WRITE as ERR.
        rig.prefill.agent.check_xfer_state = lambda handle: "ERR"
        deadline = time.monotonic() + 5
        while rig.prefill.check_status(ROOM) != KVPoll.Failed:
            self.assertLess(time.monotonic(), deadline)
            rig.prefill.host_staging.progress()  # No fail-stop.
            time.sleep(0.01)

    def test_stride_descriptors_expand_only_for_host(self):
        runs = np.array([[100, 4, 0, 8, 2], [200, 2, 0, 4, 3]], dtype=np.uint64)
        expanded = np.array(
            [[100, 4, 0], [108, 4, 0], [200, 2, 0], [204, 2, 0], [208, 2, 0]],
            dtype=np.uint64,
        )
        for host in (None, object()):
            with self.subTest(host=host is not None):
                manager = NixlKVManager.__new__(NixlKVManager)
                manager.host_staging = host
                manager.agent = NS(prep_xfer_dlist=Mock(return_value="native"))
                manager._prep_dlist = Mock(return_value="host")
                result = manager._prep_xfer_dlist("peer", runs, "VRAM")
                prepare = (
                    manager._prep_dlist
                    if host is not None
                    else manager.agent.prep_xfer_dlist
                )
                args = prepare.call_args.args
                self.assertEqual((args[0], args[2]), ("peer", "VRAM"))
                np.testing.assert_array_equal(
                    args[1], expanded if host is not None else runs
                )
                self.assertEqual(result, "host" if host is not None else "native")
                unused = (
                    manager.agent.prep_xfer_dlist
                    if host is not None
                    else manager._prep_dlist
                )
                unused.assert_not_called()

    def test_control_send_never_blocks_the_progress_thread(self):
        # This thread alone enforces WRITE deadlines: a socket lock another sender
        # holds, or a full socket, defers the STAGING_REQ to a later tick.
        rig = Rig(count=4)
        host = rig.prefill.host_staging
        rig.submit([4])
        for _ in range(200):
            if host.queue:
                break
            time.sleep(0.001)
        locks = rig.prefill._socket_send_locks
        rig.prefill._socket_send_locks = defaultdict(BusyLock)
        host.progress()
        rig.prefill._socket_send_locks, rig.control_full = locks, True
        host.progress()
        self.assertTrue(any(slot.request for slot in host.slots))
        self.assertFalse(rig.handler.staging_allocator.allocations)
        rig.control_full = False
        admitted, deadline = [], time.monotonic() + 5
        while not admitted and time.monotonic() < deadline:
            admitted = rig.tick()
            time.sleep(0.001)
        self.assertEqual(admitted, [rig.req.req])
        rig.assert_bytes(self)

    def test_unfinished_chunk_insert_skip_is_rank_invariant(self):
        # Gather state differs per TP rank; the insert decision must not.
        from sglang.srt.mem_cache.common import checkpoint_kv_cache

        tree = Mock()
        tree.req_to_token_pool.req_to_token = torch.arange(8).reshape(1, 8)
        for host, pending in ((None, False), (object(), False), (object(), True)):
            sender = NixlKVSender.__new__(NixlKVSender)
            sender.kv_mgr = NS(host_staging=host)
            sender.is_source_pending = lambda pending=pending: pending
            req = NS(
                disagg_kv_sender=sender,
                kv=NS(req_pool_idx=0),
                extend_range=NS(end=4),
                finished=lambda: False,
                skip_radix_cache_insert=False,
                prefix_indices=None,
            )
            tree.insert_req.reset_mock()
            checkpoint_kv_cache(req, tree)
            self.assertEqual(tree.insert_req.called, host is None)
            if host is not None:
                self.assertEqual(req.prefix_indices.tolist(), [0, 1, 2, 3])

    def test_write_errs_only_after_every_part_settled(self):
        write = H.HostWrite(2)
        write.settle("ERR")  # One part timed out before posting...
        self.assertEqual(write.state, "PROC")  # ...while its sibling may land.
        write.settle()
        self.assertEqual(write.state, "ERR")

    def test_allocation_for_failed_or_cleared_room_is_dropped(self):
        rig = Rig()
        msg = [
            b"STAGING_REQ",
            b"37",
            b"0",
            b"600",
            b"decode",
            b"prefill",
            b"prefill",
            b"1",
        ]
        rig.decode.update_status(ROOM, KVPoll.Failed)
        rig.handler.allocate(msg)
        self.assertFalse(rig.handler.staging_allocator.allocations)
        self.assertFalse(rig.receiver.host_allocs)


class HostRowValidationTest(unittest.TestCase):
    def setUp(self):
        self.handler = H.HostDecodeStagingHandler.__new__(H.HostDecodeStagingHandler)
        self.handler.region_base = np.array([1000, 2000, 4000, 8000], dtype=np.uint64)
        self.handler.region_end = np.array([1128, 2256, 4256, 8256], dtype=np.uint64)
        self.handler.region_item = np.array([16, 32, 64, 8], dtype=np.uint64)
        self.handler.region_owner = np.array([-1, -1, 0, 1], dtype=np.int64)
        self.receiver = NS(
            bootstrap_room=ROOM,
            host_allowed={
                "kv": np.array([0, 1, 3, 4, 5, 7], dtype=np.uint64),
                0: np.array([1, 2], dtype=np.uint64),
                1: None,  # DSA tail: region bounds only, as before.
            },
        )

    def check(self, rows):
        table = np.asarray(rows, dtype=np.uint64).reshape(-1, 2)
        self.handler.check_rows(self.receiver, table[:, 0], table[:, 1])

    def test_mixed_owners_partial_items_and_shared_layers(self):
        rows = [(1049, 46), (2097, 94), (4065, 126), (8001, 255), (1112, 16)]
        self.check(rows)
        self.check(rows[::-1] + rows)  # Unsorted and duplicate rows are valid.
        self.check([])

    def test_rejects_bounds_holes_and_wrong_owner(self):
        for row in (
            (999, 1),  # Before the first region.
            (1500, 1),  # In a gap.
            (1127, 2),  # Crosses a region end.
            (1000, 0),
            (1016, 48),  # Allowed endpoints, but item 2 is missing.
            (4000, 64),  # KV owns item 0; this state component does not.
            (8255, 2),  # Even region-only owners cannot cross bounds.
            (1000, 2**64 - 1),  # Unsigned end-address overflow.
            (8000, 2**64 - 1),  # Region-only owners must reject overflow too.
        ):
            with (
                self.subTest(row=row),
                self.assertRaisesRegex(ValueError, "outside room"),
            ):
                self.check([row])

    def test_empty_allowed_owner_rejects_rows(self):
        self.receiver.host_allowed["kv"] = np.array([], dtype=np.uint64)
        with self.assertRaisesRegex(ValueError, "outside room"):
            self.check([(1000, 1)])
        self.check([(4064, 64)])

    def test_large_unsigned_addresses(self):
        self.handler.region_base += np.uint64(2**63)
        self.handler.region_end += np.uint64(2**63)
        self.check([(2**63 + 1048, 48), (2**63 + 4064, 128)])
        with self.assertRaisesRegex(ValueError, "outside room"):
            self.check([(2**63 + 1048, 2**63)])

    def test_matches_independent_scalar_validation(self):
        rng = np.random.default_rng(9321)
        regions = list(
            zip(
                self.handler.region_base.tolist(),
                self.handler.region_end.tolist(),
                self.handler.region_item.tolist(),
                ("kv", "kv", 0, 1),
                strict=True,
            )
        )
        for _ in range(1000):
            base, end, _, _ = regions[int(rng.integers(len(regions)))]
            addr = int(rng.integers(base - 5, end + 5))
            length = int(rng.integers(0, 300))
            valid = False
            for start, stop, item, owner in regions:
                if start <= addr < stop and 0 < length <= stop - addr:
                    allowed = self.receiver.host_allowed[owner]
                    valid = allowed is None or all(
                        index in allowed
                        for index in range(
                            (addr - start) // item,
                            (addr + length - 1 - start) // item + 1,
                        )
                    )
                    break
            with self.subTest(addr=addr, length=length):
                if valid:
                    self.check([(addr, length)])
                else:
                    with self.assertRaisesRegex(ValueError, "outside room"):
                        self.check([(addr, length)])


class CudaByteRanges(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
    def test_device_and_pinned_host_ranges(self):
        from sglang.kernels.ops.kvcache.pd_dcp_gather import copy_byte_ranges

        src = torch.randint(0, 255, (1 << 20,), dtype=torch.uint8, device="cuda")
        host = torch.zeros(1 << 20, dtype=torch.uint8).pin_memory()
        back = torch.zeros_like(src)
        sizes = np.array([1, 4095, 4096, 70001, 300000], dtype=np.int64)
        offs = np.r_[0, np.cumsum(sizes)[:-1]]
        rev = np.r_[0, np.cumsum(sizes[::-1])[:-1]][::-1].copy()
        for dst, s, o_dst, o_src in (
            (host, src, rev, offs),  # gather: device rows -> pinned slot
            (back, host, offs, rev),  # scatter: pinned ring -> device rows
        ):
            table = torch.as_tensor(
                np.stack([o_dst + dst.data_ptr(), o_src + s.data_ptr(), sizes]),
                device="cuda",
            )
            copy_byte_ranges(table, int(sizes.max()))
        torch.cuda.synchronize()
        n = int(sizes.sum())
        self.assertTrue(torch.equal(back[:n], src[:n]))


if __name__ == "__main__":
    unittest.main()
