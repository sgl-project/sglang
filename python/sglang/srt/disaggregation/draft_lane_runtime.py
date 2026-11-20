"""Synchronous cold-PD adapter for the versioned DFlash context lane.

The scheduler owns leases and GPU state. ZMQ threads only enqueue control
messages. A source slab is reused only after successful synchronous DMA;
uncertain transport or cancellation quarantines the destination until restart.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
import uuid
from dataclasses import asdict, replace
from pathlib import Path
from queue import Empty, SimpleQueue

from sglang.srt.disaggregation.draft_lane import (
    CONTROL_TAG,
    DraftLease,
    DraftLeaseState,
    DraftPeer,
    DraftRegion,
    decode_control,
    destination_head_shards,
    encode_control,
    peer_from_dict,
    peer_to_dict,
    plan_region_head_copy,
    validate_cohort,
)

logger = logging.getLogger(__name__)


def validate_typed_target_pool(pool):
    import torch

    if (
        not getattr(pool, "use_dsa", False)
        or pool.dtype != torch.float8_e4m3fn
        or not getattr(pool, "dsa_kv_cache_store_fp8", False)
        or getattr(pool, "kv_cache_dim", None) != 656
        or getattr(pool, "kv_scale_ue8m0", False)
    ):
        raise ValueError("Typed DFLASH target pool must use scaled656 FP8 DSA storage")


class DraftPDLane:
    def __init__(self, worker, scheduler, *, prefill):

        from sglang.srt.runtime_context import get_parallel

        self.worker, self.scheduler, self.prefill = worker, scheduler, prefill
        validate_typed_target_pool(scheduler.tp_worker.model_runner.token_to_kv_pool)
        self.manager = None
        self.inbox = SimpleQueue()
        self.owners = {}
        self.destinations = {}
        self.slot_leases = {}
        self.sources = {}
        self.retired_sources = set()
        self.receivers = {}
        self.peer = None
        self.peers = ()
        self.quarantined = set()
        parallel = get_parallel()
        self.authority = not prefill or parallel.attn_cp_rank < 2
        self.regions, self.tensors = [], {}
        pool = worker.draft_model_runner.token_to_kv_pool
        self.dtype = str(worker.draft_model_runner.kv_cache_dtype).removeprefix(
            "torch."
        )
        if self.dtype not in ("bfloat16", "float16"):
            raise ValueError("The typed draft lane requires unscaled floating-point KV")
        layout = worker.draft_parallel_layout
        self.head_begin = (
            layout.tp_group.rank_in_group
            * worker.draft_model.layers[0].self_attn.num_kv_heads
        )
        self.head_end = (
            self.head_begin + worker.draft_model.layers[0].self_attn.num_kv_heads
        )
        for layer in worker.draft_model.layers:
            attention = layer.self_attn
            index = attention.attn.layer_id
            kind = "WINDOW" if attention.sliding_window_size >= 0 else "FULL"
            for label, tensor in [
                ("K", pool.get_key_buffer(index)),
                ("V", pool.get_value_buffer(index)),
            ]:
                if (
                    tensor.ndim != 3
                    or not tensor.is_contiguous()
                    or tensor.shape[1:] != (attention.num_kv_heads, attention.head_dim)
                ):
                    raise ValueError(
                        "Typed draft PD requires physically packed NHD regions"
                    )
                region = DraftRegion(
                    index,
                    label,
                    kind,
                    tensor.data_ptr(),
                    tensor.numel() * tensor.element_size(),
                    tensor.stride(0) * tensor.element_size(),
                    tensor.shape[0],
                )
                self.regions.append(region)
                self.tensors[region.key] = tensor
        manifest_path = os.environ.get("SGLANG_DFLASH_PD_CHECKPOINT_MANIFEST")
        if not manifest_path:
            raise ValueError("Typed draft PD requires a SHA256 checkpoint manifest")
        manifest = json.loads(Path(manifest_path).read_text())
        model_root = Path(worker.server_args.speculative_draft_model_path)
        for item in manifest["files"]:
            info = (model_root / item["name"]).stat()
            if (info.st_size, info.st_mtime_ns) != (item["size"], item["mtime_ns"]):
                raise ValueError(
                    "Draft checkpoint changed after SHA256 manifest creation"
                )
        self.fingerprint = manifest["fingerprint"]
        if not prefill:
            worker._independent_cache.release_callback = self.release_owner_slot

    def registerable_regions(self):
        return tuple((r.base, r.registered_bytes) for r in self.regions)

    def bind_engine(self, manager):
        self.manager = manager
        from sglang.srt.utils.network import NetworkAddress

        self.endpoint = NetworkAddress(manager.local_ip, manager.rank_port).to_tcp()

    def start(self):
        if not self.prefill:
            group = self.worker._cohort.group
            local = DraftPeer(
                uuid.uuid4().hex,
                "pending",
                "pending",
                group.rank_in_group,
                self.manager.get_session_id(),
                self.endpoint,
                self.fingerprint,
                self.dtype,
                self.head_begin,
                self.head_end,
                16,
                128,
                self.worker.page_size,
                self.worker.block_size,
                tuple(self.regions),
            )
            members = group.all_gather_object(local)
            epoch = hashlib.sha256(
                json.dumps(
                    [(p.registration_id, p.engine_session) for p in members]
                ).encode()
            ).hexdigest()
            self.peers = validate_cohort(
                [replace(p, cohort_id=epoch[:16], cohort_epoch=epoch) for p in members]
            )
            self.peer = self.peers[group.rank_in_group]
        self.worker._draft_pd_lane = self
        logger.info(
            "Typed DFLASH lane registered: role=%s heads=[%s,%s) regions=%s authority=%s",
            "P" if self.prefill else "D",
            self.head_begin,
            self.head_end,
            len(self.regions),
            self.authority,
        )

    def handle_control(self, frames):
        if not frames or frames[0] != CONTROL_TAG:
            return False
        try:
            record = decode_control(frames)
        except (ValueError, TypeError, KeyError):
            # An invalid envelope never authorizes DMA or acknowledges a lease.
            # Keep receiving unrelated target/control messages; its sender's
            # deadline will fail closed rather than losing the ZMQ thread.
            logger.exception("Rejected malformed typed draft control envelope")
            return True
        self.inbox.put(record)
        return True

    def _send(self, endpoint, kind, payload):
        self.manager._send_multipart_locked(endpoint, encode_control(kind, payload))

    def _error(self, payload, error):
        room = payload.get("room")
        if room is not None:
            self.quarantined.add(room)
            self.manager.record_failure(room, "Typed draft lane: " + str(error))
            from sglang.srt.disaggregation.base.conn import KVPoll

            self.manager.update_status(room, KVPoll.Failed)
        if payload.get("reply_to"):
            self._send(
                payload["reply_to"], "ERROR", {"room": room, "error": str(error)}
            )
        logger.error("Typed draft lane failed closed: room=%s: %s", room, error)

    def begin_lease(self, req, receiver, logical_end):
        from sglang.srt.disaggregation.mode import FAKE_BOOTSTRAP_HOST

        diagnostic = getattr(req, "bootstrap_host", None) == FAKE_BOOTSTRAP_HOST
        if diagnostic and logical_end > 16:
            raise ValueError("Native diagnostic draft context is limited to16 tokens")
        room = getattr(req, "bootstrap_room", None)
        if room is None:
            room = receiver.bootstrap_room
        cache = self.worker._independent_cache
        owner = self.worker._cohort.owner_rank
        target_slot = req.kv.req_pool_idx
        slot = target_slot + owner * self.worker._cohort.owner_slots
        generation = int(cache.target_requests.req_generation[target_slot])
        layout = cache.layout
        lease = DraftLease(
            uuid.uuid4().hex,
            int(room),
            req.rid,
            generation,
            owner,
            slot,
            logical_end,
            layout.context_length,
            layout.full_loc(slot, 0),
            layout.stride,
            layout.window_loc(slot, 0),
            layout.window_stride,
            self.peer.cohort_epoch,
        ).validate()
        prior = self.owners.get(lease.room)
        if prior is not None and (
            not prior["aborted"]
            or lease.room in self.quarantined
            or len(prior["ready"]) != len(self.peers)
        ):
            raise RuntimeError("Duplicate or quarantined typed draft bootstrap room")
        self.owners[lease.room] = dict(
            lease=lease,
            reserved=set(),
            prepared=set(),
            ready=set(),
            published=diagnostic,
            writers=[] if diagnostic else None,
            diagnostic=diagnostic,
            pending=None,
            receiver=receiver,
            aborted=False,
            admitted=False,
            deadline=time.monotonic() + self.manager.waiting_timeout,
        )
        for peer in self.peers:
            self._send(
                peer.control_endpoint,
                "RESERVE",
                dict(
                    room=lease.room,
                    lease=asdict(lease),
                    registration_id=peer.registration_id,
                    diagnostic=diagnostic,
                    reply_to=self.endpoint,
                ),
            )
        return lease

    def defer_metadata(self, receiver, kwargs):
        state = self.owners.get(int(receiver.bootstrap_room))
        if state is None:
            raise RuntimeError("Target metadata has no typed draft lease")
        if state["published"]:
            return False
        if state["pending"] is not None:
            raise RuntimeError("Target metadata published twice")
        state["pending"] = kwargs
        return True

    def ready(self, room):
        state = self.owners.get(int(room))
        return (
            state is not None
            and not state["aborted"]
            and len(state["ready"]) == len(self.peers)
        )

    def mark_admitted(self, room):
        self.owners[int(room)]["admitted"] = True

    def _prepare_writers(self, state):
        receiver = state["receiver"]
        lease = state["lease"]
        info = receiver.prefill_info
        if info.attn_tp_size != 1 or info.attn_cp_size < 2 or info.pp_size != 1:
            raise ValueError("Typed draft context writer topology mismatch")
        writers = []
        for cp in (0, 1):
            record = receiver._get_bootstrap_info_from_server(
                receiver.prefill_dp_rank, cp, 0, 0
            )
            if record is None:
                raise RuntimeError("Missing typed draft writer endpoint")
            from sglang.srt.utils.network import NetworkAddress

            writers.append(
                NetworkAddress(record["rank_ip"], record["rank_port"]).to_tcp()
            )
        state["writers"] = writers
        for writer in writers:
            self._send(
                writer,
                "PREPARE",
                dict(
                    room=lease.room,
                    lease=asdict(lease),
                    peers=[peer_to_dict(p) for p in self.peers],
                    reply_to=self.endpoint,
                ),
            )

    def service(self):

        while True:
            try:
                record = self.inbox.get_nowait()
            except Empty:
                break
            payload = record["payload"]
            try:
                self._service_message(record["kind"], payload)
            except Exception as error:
                self._error(payload, error)
        if self.prefill:
            return
        for room, state in list(self.owners.items()):
            if state["aborted"] or room in self.quarantined:
                continue
            if not state["admitted"] and time.monotonic() > state["deadline"]:
                self._error(
                    dict(room=room),
                    RuntimeError(
                        "Draft PD preparation/READY deadline exceeded; slot quarantined"
                    ),
                )
                continue
            try:
                if (
                    len(state["reserved"]) == len(self.peers)
                    and state["writers"] is None
                ):
                    self._prepare_writers(state)
                if (
                    state["writers"] is not None
                    and len(state["prepared"]) == len(state["writers"])
                    and state["pending"] is not None
                    and not state["published"]
                ):
                    state["published"] = True
                    kwargs, state["pending"] = state["pending"], None
                    state["receiver"].send_metadata(**kwargs)
            except Exception as error:
                self._error(dict(room=room), error)

    def _publish_ready(self, state, reply_to, *, diagnostic=False):
        lease = state.lease
        cache = self.worker._independent_cache
        cache.generations[lease.slot], cache.allocated_lens[lease.slot] = (
            lease.generation,
            lease.logical_end,
        )
        self.worker._cohort.committed[lease.slot] = (
            lease.generation,
            lease.request_id,
            lease.logical_end,
        )
        self._send(
            reply_to,
            "READY",
            dict(room=lease.room, lease_id=lease.lease_id, peer=self.peer.peer),
        )
        logger.info(
            "Typed draft context READY: room=%s peer=%s tokens=%s diagnostic=%s",
            lease.room,
            self.peer.peer,
            lease.logical_end,
            diagnostic,
        )

    def _service_message(self, kind, payload):
        import torch

        room = int(payload["room"])
        if kind == "ERROR":
            raise RuntimeError(payload["error"])
        if kind == "RESERVE":
            if self.prefill:
                raise ValueError("Reserve sent to a source encoder")
            lease = DraftLease(**payload["lease"]).validate()
            if payload["registration_id"] != self.peer.registration_id:
                raise ValueError("Stale destination registration")
            layout = self.worker._independent_cache.layout
            owner_slots = self.worker._cohort.owner_slots
            if not (
                lease.owner * owner_slots
                < lease.slot
                <= (lease.owner + 1) * owner_slots
                and lease.full_row0 == layout.full_loc(lease.slot, 0)
                and lease.window_row0 == layout.window_loc(lease.slot, 0)
                and lease.full_stride == layout.stride
                and lease.window_stride == layout.window_stride
            ):
                raise ValueError("Lease does not match private destination storage")
            prior = self.slot_leases.get(lease.slot)
            if prior is not None and prior != lease.lease_id:
                raise RuntimeError("Destination slot is still leased or quarantined")
            state = self.destinations.get(lease.lease_id)
            if state is None:
                torch.cuda.synchronize()
                self.destinations[lease.lease_id] = DraftLeaseState(lease, self.peer)
                self.slot_leases[lease.slot] = lease.lease_id
            self._send(
                payload["reply_to"],
                "RESERVE_ACK",
                dict(room=room, lease_id=lease.lease_id, peer=self.peer.peer),
            )
            if payload.get("diagnostic", False):
                if lease.logical_end > 16:
                    raise ValueError("Oversized native diagnostic context")
                state = self.destinations[lease.lease_id]
                if state.phase == "RESERVED":
                    ranges = []
                    for key, tensor in self.tensors.items():
                        start = lease.window_floor if key[2] == "WINDOW" else 0
                        rows = [
                            lease.row(p, key[2])
                            for p in range(start, lease.logical_end)
                        ]
                        tensor.index_fill_(
                            0,
                            torch.tensor(
                                rows, device=self.worker.device, dtype=torch.int64
                            ),
                            0,
                        )
                        ranges.append((key, start, lease.logical_end))
                    state.transferred("native-diagnostic-zero-context", ranges)
                    state.mark_ready()
                self._publish_ready(state, payload["reply_to"], diagnostic=True)
        elif kind == "RESERVE_ACK":
            state = self.owners[room]
            if payload["lease_id"] != state["lease"].lease_id:
                raise ValueError("Stale reserve ACK")
            state["reserved"].add(int(payload["peer"]))
        elif kind == "PREPARE":
            if not self.prefill or not self.authority:
                raise ValueError("Invalid context writer")
            lease = DraftLease(**payload["lease"]).validate()
            peers = validate_cohort([peer_from_dict(p) for p in payload["peers"]])
            destinations = destination_head_shards(
                peers, self.head_begin, self.head_end
            )
            for peer in destinations:
                if peer.fingerprint != self.fingerprint or peer.dtype != self.dtype:
                    raise ValueError(
                        "Draft source/destination checkpoint or dtype mismatch"
                    )
                if {r.key for r in peer.regions} != {r.key for r in self.regions}:
                    raise ValueError("Draft source/destination layer types mismatch")
            if lease.lease_id in self.retired_sources:
                raise ValueError("Retired source lease cannot be prepared again")
            prior = self.sources.get(room)
            if prior is not None:
                if (
                    prior["lease"] != lease
                    or prior["peers"] != destinations
                    or prior["owner"] != payload["reply_to"]
                ):
                    raise ValueError(
                        "Source room already leased with a different registration"
                    )
            else:
                self.sources[room] = dict(
                    lease=lease,
                    peers=destinations,
                    owner=payload["reply_to"],
                    sent=set(),
                )
            self._send(
                payload["reply_to"],
                "PREPARE_ACK",
                dict(room=room, lease_id=lease.lease_id, writer=self.endpoint),
            )
        elif kind == "PREPARE_ACK":
            state = self.owners[room]
            if (
                payload["lease_id"] != state["lease"].lease_id
                or payload["writer"] not in state["writers"]
            ):
                raise ValueError("Unrecognized context writer ACK")
            state["prepared"].add(payload["writer"])
        elif kind == "DONE":
            if self.prefill:
                raise ValueError("Transfer DONE sent to source")
            state = self.destinations[payload["lease_id"]]
            if state.lease.room != room:
                raise ValueError("Transfer room/lease mismatch")
            if payload["chunk_id"] in state.chunks:
                if state.phase == "READY":
                    self._send(
                        payload["reply_to"],
                        "READY",
                        dict(
                            room=room,
                            lease_id=state.lease.lease_id,
                            peer=self.peer.peer,
                        ),
                    )
                return
            # Check samples before authorizing first decode, independently of
            # the target MLA checksum and acceptance/answer statistics.
            for sample in payload["samples"]:
                key = tuple(sample["key"])
                positions = sample["positions"]
                rows = [state.lease.row(p, key[2]) for p in positions]
                actual = self.tensors[key][
                    torch.tensor(rows, device=self.worker.device)
                ].contiguous()
                digest = hashlib.sha256(
                    actual.view(torch.uint8).cpu().numpy().tobytes()
                ).hexdigest()
                if digest != sample["sha256"]:
                    raise RuntimeError("Draft PD K/V sample checksum mismatch")
            ranges = [(tuple(key), start, end) for key, start, end in payload["ranges"]]
            state.transferred(payload["chunk_id"], ranges)
            if payload["last"]:
                state.last_seen = True
            if getattr(state, "last_seen", False):
                try:
                    state.mark_ready()
                except RuntimeError as error:
                    if "incomplete" not in str(error) and "missing" not in str(error):
                        raise
                else:
                    self._publish_ready(state, payload["reply_to"])
        elif kind == "READY":
            state = self.owners[room]
            if payload["lease_id"] != state["lease"].lease_id:
                raise ValueError("Stale draft READY")
            state["ready"].add(int(payload["peer"]))
        elif kind == "RELEASE":
            if self.prefill:
                source = self.sources.get(room)
                if (
                    source is not None
                    and source["lease"].lease_id == payload["lease_id"]
                ):
                    self.sources.pop(room)
                    self.retired_sources.add(payload["lease_id"])
                return
            state = self.destinations.get(payload["lease_id"])
            if state is None:
                return
            # Only a successful complete transfer can be reclaimed in v1.
            # Cancellation/uncertain transport deliberately keeps its slot.
            drained = bool(payload["transport_drained"]) and state.phase == "READY"
            torch.cuda.synchronize()
            if state.retire(readers_drained=True, transport_drained=drained):
                slot = state.lease.slot
                self.worker._cohort.committed.pop(slot, None)
                cache = self.worker._independent_cache
                cache.generations[slot], cache.allocated_lens[slot] = -1, 0
                self.slot_leases.pop(slot, None)
                self.destinations.pop(payload["lease_id"])
            else:
                self.quarantined.add(room)
        else:
            raise ValueError("Unknown typed draft control message " + kind)

    def send_context_chunk(self, batch, positions, source_rows):
        import torch

        if not self.prefill or not self.authority:
            return
        # No other source slab write can occur before this synchronous batch
        # completes. Failed sync DMA never permits slab reuse.
        torch.cuda.synchronize()
        positions, source_rows = positions.cpu().tolist(), source_rows.cpu().tolist()
        offset = 0
        from sglang.srt.disaggregation.mode import FAKE_BOOTSTRAP_HOST

        for req, length in zip(batch.reqs, batch.extend_lens):
            if getattr(req, "bootstrap_host", None) == FAKE_BOOTSTRAP_HOST:
                # Native health/loader warmups deliberately have no remote D.
                offset += length
                continue
            room = int(req.disagg_kv_sender.bootstrap_room)
            state = self.sources.get(room)
            if state is None:
                raise RuntimeError("Prefill started before typed draft PREPARE")
            lease = state["lease"]
            logical = positions[offset : offset + length]
            physical = source_rows[offset : offset + length]
            offset += length
            if not logical:
                continue
            for peer in state["peers"]:
                chunk_id = (
                    f"{logical[0]}:{logical[-1] + 1}:"
                    f"{peer.head_begin}:{peer.head_end}:{peer.registration_id}"
                )
                if chunk_id in state["sent"]:
                    raise RuntimeError("Attempted to rewrite a completed draft chunk")
                blocks, ranges, samples = [], [], []
                destinations = {r.key: r for r in peer.regions}
                for region in self.regions:
                    blocks += plan_region_head_copy(
                        region,
                        destinations[region.key],
                        physical,
                        logical,
                        lease,
                        source_head_begin=self.head_begin,
                        source_head_end=self.head_end,
                        destination_head_begin=peer.head_begin,
                        destination_head_end=peer.head_end,
                        head_dim=peer.head_dim,
                    )
                    first = (
                        max(logical[0], lease.window_floor)
                        if region.pool == "WINDOW"
                        else logical[0]
                    )
                    end = logical[-1] + 1
                    if first >= end:
                        continue
                    ranges.append((region.key, first, end))
                    sample_pos = sorted({first, end - 1})
                    rows = [physical[p - logical[0]] for p in sample_pos]
                    values = self.tensors[region.key][
                        torch.tensor(rows, device=self.worker.device),
                        peer.head_begin - self.head_begin : peer.head_end
                        - self.head_begin,
                        :,
                    ].contiguous()
                    samples.append(
                        dict(
                            key=region.key,
                            positions=sample_pos,
                            sha256=hashlib.sha256(
                                values.view(torch.uint8).cpu().numpy().tobytes()
                            ).hexdigest(),
                        )
                    )
                # Bound each descriptor batch: TP2 -> TP4 has strided source rows.
                # Source reuse and DONE remain gated on every synchronous batch.
                for block_offset in range(0, len(blocks), 4096):
                    code = self.manager._transfer_data(
                        peer.engine_session, blocks[block_offset : block_offset + 4096]
                    )
                    if code != 0:
                        self.quarantined.add(room)
                        raise RuntimeError(
                            "Unsettled typed draft RDMA; restart required before source/destination reuse"
                        )
                state["sent"].add(chunk_id)
                self._send(
                    peer.control_endpoint,
                    "DONE",
                    dict(
                        room=room,
                        lease_id=lease.lease_id,
                        chunk_id=chunk_id,
                        ranges=ranges,
                        samples=samples,
                        last=logical[-1] + 1 == lease.logical_end,
                        reply_to=state["owner"],
                    ),
                )

    def release_owner_slot(self, target_slot):
        for room, state in self.owners.items():
            lease = state["lease"]
            if (
                lease.slot
                != target_slot
                + self.worker._cohort.owner_rank * self.worker._cohort.owner_slots
                or state["aborted"]
            ):
                continue
            state["aborted"] = True
            drained = len(state["ready"]) == len(self.peers)
            for peer in self.peers:
                self._send(
                    peer.control_endpoint,
                    "RELEASE",
                    dict(room=room, lease_id=lease.lease_id, transport_drained=drained),
                )
            for writer in state["writers"] or ():
                self._send(
                    writer,
                    "RELEASE",
                    dict(room=room, lease_id=lease.lease_id, transport_drained=drained),
                )


class DraftReadyPoller:
    """Gate every receiver, including native fake-transfer diagnostics."""

    def __init__(self, receiver, lane, room):
        self.receiver, self.lane, self.room = receiver, lane, int(room)

    def poll(self):
        from sglang.srt.disaggregation.base.conn import KVPoll

        status = self.receiver.poll()
        if status == KVPoll.Success:
            if not self.lane.ready(self.room):
                return KVPoll.Transferring
            self.lane.mark_admitted(self.room)
        return status
