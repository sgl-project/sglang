"""KVServe adapter for native Mooncake TP1 full-attention PD.

The scheduler retains page ownership. Mooncake writes only bounded private codec
buffers; the existing worker may conclude Success only after a restore ACK.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import os
import threading
import time
import uuid
from dataclasses import asdict
from pathlib import Path

import msgspec
import torch

from sglang.srt.disaggregation.base.conn import KVPoll, StateType
from sglang.srt.disaggregation.mooncake.kvserve_pipeline import InlineNotifications
from sglang.srt.disaggregation.mooncake.kvserve_transport import ReceiveCredits
from sglang.srt.utils.network import NetworkAddress

READY = b"KVSERVE_READY_V3"
ACK = b"KVSERVE_ACK_V3"


def endpoint(host, port):
    return "tcp://" + NetworkAddress(host, port).to_host_port_str()


class KVServeAdapter:
    """Registration, protocol and lifecycle shared by the pipelined adapter."""

    def __init__(self, manager, k_buffers, v_buffers, page_size, layer_ids):
        # Import the optional dependency only on explicitly enabled deployments.
        from kvserve_v1.compression.runtime import CacheSpec

        self.manager = manager
        self.device = torch.device("cuda", manager.kv_args.gpu_id)
        if (
            manager.is_mla_backend
            or manager.is_hybrid_mla_backend
            or manager.attn_tp_size != 1
            or manager.attn_cp_size != 1
            or manager.dcp_size != 1
            or manager.pp_size != 1
            or len(manager.kv_args.state_types) > 1
            or any(st != StateType.MAMBA for st in manager.kv_args.state_types)
            or manager.kv_args.num_draft_entries
            or manager.enable_staging
            or getattr(manager.server_args, "enable_pd_role_switch", False)
        ):
            raise ValueError(
                "KVServe requires BF16 full attention, TP/PP/CP/DCP1, no "
                "staging/draft/role switching or non-Mamba state"
            )
        self.has_mamba = StateType.MAMBA in manager.kv_args.state_types
        if not manager.enable_deferred_decode_kv_release:
            raise ValueError(
                "KVServe requires SGLANG_DISAGGREGATION_DEFERRED_DECODE_KV_RELEASE=1"
            )
        values = json.loads(Path(os.environ["SGLANG_KVSERVE_CONFIG"]).read_text())
        from sglang.srt.disaggregation.kvserve_runtime import parse_compression_config

        self.config = parse_compression_config(values)
        if not self.config.enabled or self.config.uses_aggregate:
            raise ValueError(
                "KVServe must enable a compression pipeline; raw aggregation is not compression"
            )
        if self.config.slot_count not in (1, 2, 4, 8):
            raise ValueError("KVServe adapter supports 1, 2, 4 or 8 bounded slots")
        self.chunk_size = self.config.logical_blocks_per_chunk
        self.page_size = page_size
        if page_size < 1 or self.chunk_size % page_size:
            raise ValueError(
                "KVServe chunk tokens must be divisible by native page size"
            )
        if not layer_ids:
            layer_ids = list(range(len(k_buffers)))
        elif len(layer_ids) == 2 * len(k_buffers):
            if list(layer_ids[: len(k_buffers)]) != list(layer_ids[len(k_buffers) :]):
                raise ValueError("K/V staging metadata must reference identical layers")
            layer_ids = layer_ids[: len(k_buffers)]
        if not (len(k_buffers) == len(v_buffers) == len(layer_ids)):
            raise ValueError("Incomplete full-attention KV buffer list")
        caches, specs = {}, {}
        for layer, k, v in zip(layer_ids, k_buffers, v_buffers):
            # NHD token pools become HND page views without a copy, page size 1.
            k = k.unsqueeze(2) if k.ndim == 3 else k
            v = v.unsqueeze(2) if v.ndim == 3 else v
            if k.ndim != 4 or v.ndim != 4 or k.shape[2] != 1 or v.shape[2] != 1:
                raise ValueError("KVServe requires NHD token or HND page-size-1 pools")
            name = f"layer.{layer}"
            caches[name] = (k, v)
            specs[name] = CacheSpec(
                layer_index=layer,
                block_size=1,
                num_kv_heads=k.shape[1],
                head_size=k.shape[-1],
                head_size_v=v.shape[-1],
                dtype=k.dtype,
            )
        self.cache_blocks = min(x[0].shape[0] for x in caches.values())
        from sglang.srt.disaggregation.kvserve_runtime import take_runtime

        self.runtime = take_runtime(self.device)
        if self.runtime is None:
            raise RuntimeError(
                "KVServe runtime was not reserved before KV cache sizing"
            )
        plan = self.runtime.plan
        if self.runtime.config != self.config:
            raise ValueError("KVServe reserved plan does not match bound caches")
        owned_bytes = plan.reserved_bytes - plan.retained_driver_bytes
        self.runtime.bind_kv_caches(caches, {name: 0 for name in caches})
        self.runtime.stream.synchronize()
        ptrs, sizes = self.runtime.registered_memory()
        rc = manager.engine.batch_register(ptrs, sizes)
        if rc not in (None, 0):
            raise RuntimeError(f"Failed to register KVServe buffers: {rc}")
        self.registered_ptrs = ptrs
        self.epoch = uuid.uuid4().hex
        self.slot_locks = [threading.Lock() for _ in self.runtime.slots]
        self.lock = threading.Lock()
        self.pending_lock = threading.Lock()
        self.pending = {}
        self.credit_pools = {}
        self.submit_lock = threading.Lock()
        self.notifier = InlineNotifications(
            manager._send_multipart_locked, self.control_failed, self.record
        )
        self.sequence = itertools.count()
        self.poisoned = False
        self.restored_rooms = {}
        self.expected_rooms = {}
        self._source_unsettled_rooms = set()
        self._remote_restore_unsettled = set()
        self._closing = False
        self._prefill_owner = None
        self.verify = os.environ.get("SGLANG_KVSERVE_VERIFY", "0") == "1"
        self.log_path = os.environ.get("SGLANG_KVSERVE_STATS")
        self.log_lock = threading.Lock()
        self.raw_per_token = sum(
            k.shape[1] * k.shape[-1] * 2 + v.shape[1] * v.shape[-1] * 2
            for k, v in caches.values()
        )
        self.record(
            {
                "event": "init",
                "profile": self.runtime.wire_profile,
                "reserved_bytes": plan.reserved_bytes,
                "owned_plan_bytes": owned_bytes,
                "driver_retention_estimate_bytes": plan.retained_driver_bytes,
                "bulk_bytes": self.runtime.bulk.numel(),
                "cache_blocks": self.cache_blocks,
                "raw_per_token": self.raw_per_token,
                "config": values,
                "effective_config": asdict(self.config),
                "quant_sections": {
                    label: {
                        str(level): sum(
                            1
                            for part in self.runtime.layout.parts(kv_part)
                            for section in part.sections
                            if section.num_levels == level
                        )
                        for level in sorted(
                            {
                                section.num_levels
                                for part in self.runtime.layout.parts(kv_part)
                                for section in part.sections
                            }
                        )
                    }
                    for kv_part, label in ((0, "K"), (1, "V"))
                },
            }
        )

    def record(self, value):
        if self.log_path:
            with self.log_lock, open(self.log_path, "a") as f:
                f.write(json.dumps({"time": time.time(), **value}) + "\n")

    def registration(self):
        return {
            "version": 3,
            "profile": self.runtime.wire_profile,
            "page_size": self.page_size,
            "has_mamba": self.has_mamba,
            "slots": [
                {
                    "ptrs": [slot.arena_a.data_ptr(), slot.aux.data_ptr()],
                    "capacities": [slot.arena_a.numel(), slot.aux.numel()],
                }
                for slot in self.runtime.slots
            ],
        }

    def expect_room(self, room, indices, is_dummy):
        if not is_dummy:
            with self.lock:
                self.expected_rooms[room] = {int(x) for x in indices}
                self.restored_rooms[room] = {0: set(), 1: set()}

    def drop_room(self, room):
        self.cancel_room(room)
        with self.lock:
            self.expected_rooms.pop(room, None)
            self.restored_rooms.pop(room, None)

    def cancel_room(self, room):
        pipeline = getattr(self, "_receive_pipeline_v2", None)
        if pipeline is not None:
            pipeline.cancel_room(room)

    def local_restore_release_safe(self, room):
        pipeline = getattr(self, "_receive_pipeline_v2", None)
        return pipeline is None or pipeline.room_is_idle(room)

    def source_encode_release_safe(self, room):
        with self.lock:
            return room not in getattr(self, "_source_unsettled_rooms", set())

    def quarantine_remote_restore(self, room):
        # A remote timeout only permits releasing native pages after local
        # restores drain. It does not permit deregistering scratch still exposed
        # to a possibly outstanding remote write.
        with self.lock:
            if not hasattr(self, "_remote_restore_unsettled"):
                self._remote_restore_unsettled = set()
            self._remote_restore_unsettled.add(room)

    def claim_prefill_peer(self, bootstrap_info):
        """One producer owns this receive bank until the Decode manager exits.

        Sender-local credits are not a distributed multi-producer grant. Reject
        a second endpoint before publishing any codec/native addresses to it.
        """
        if bootstrap_info.get("is_dummy", False):
            return
        peer = (str(bootstrap_info["rank_ip"]), int(bootstrap_info["rank_port"]))
        with self.lock:
            owner = getattr(self, "_prefill_owner", None)
            if owner is not None and owner != peer:
                raise ValueError(
                    "KVServe receive banks support one Prefill endpoint per Decode manager"
                )
            self._prefill_owner = peer

    def assert_close_owner(self):
        current = threading.current_thread()
        workers = list(getattr(self.manager, "_worker_threads", []))
        source = getattr(self, "_source_pool", None)
        workers.extend(getattr(source, "_threads", ()))
        tx = getattr(self, "_tx_v2", None)
        workers.extend(getattr(tx, "threads", ()))
        rx = getattr(self, "_receive_pipeline_v2", None)
        workers.extend(getattr(rx, "waiters", ()))
        workers.append(getattr(rx, "submit_thread", None))
        if current in workers:
            raise RuntimeError(
                "KVServe close must run on its lifecycle owner, not a control/codec worker"
            )

    def validate_peer(self, peer):
        local = self.registration()["slots"]
        remote = peer.get("slots", []) if isinstance(peer, dict) else []
        if (
            not isinstance(peer, dict)
            or peer.get("version") != 3
            or peer.get("page_size") != self.page_size
            or peer.get("profile") != self.runtime.wire_profile
            or type(peer.get("has_mamba")) is not bool
            or peer["has_mamba"] != self.has_mamba
            or len(local) not in (1, 2, 4, 8)
            or len(remote) not in (1, 2, 4, 8)
            or any(own["capacities"] != local[0]["capacities"] for own in local)
            or any(
                not isinstance(slot, dict)
                or len(slot.get("ptrs", [])) != 2
                or slot.get("capacities") != local[0]["capacities"]
                for slot in remote
            )
        ):
            raise ValueError(
                "KVServe peer is disabled or uses an incompatible compression profile"
            )

    def digest(self, sizes, slot_id):
        slot = self.runtime.slots[slot_id]
        return self._buffers_digest((slot.arena_a, slot.aux), sizes)

    @staticmethod
    def _buffers_digest(buffers, sizes):
        digest = hashlib.sha256()
        for tensor, size in zip(buffers, sizes):
            digest.update(tensor[:size].cpu().numpy().tobytes())
        return digest.hexdigest()

    @staticmethod
    def token_ids(pages, page_size, valid_tokens):
        if (
            len(pages) == 0
            or not (len(pages) - 1) * page_size < valid_tokens <= len(pages) * page_size
        ):
            raise ValueError("Invalid valid-token count for KVServe native pages")
        return [
            int(page) * page_size + offset
            for page in pages
            for offset in range(page_size)
        ][:valid_tokens]

    def control_failed(self, error):
        self.poisoned = True
        with self.pending_lock:
            for event, result, _, _ in self.pending.values():
                result["error"] = error
                event.set()

    def _prepare_send(
        self, request, registration, source_ids, destination_ids, num_tokens
    ):
        self.validate_peer(registration.kvserve)
        if len(source_ids) != len(destination_ids):
            raise ValueError("KVServe source/destination page counts differ")
        num_tokens = (
            num_tokens if num_tokens is not None else len(source_ids) * self.page_size
        )
        self.token_ids(source_ids, self.page_size, num_tokens)
        with self.pending_lock:
            pool = self.credit_pools.setdefault(
                request.mooncake_session_id,
                ReceiveCredits(len(registration.kvserve["slots"])),
            )
        return num_tokens, pool

    def _frame_chunks(self, source_ids, destination_ids, num_tokens):
        page_chunk = self.chunk_size // self.page_size
        for start in range(0, len(source_ids), page_chunk):
            src_pages = [int(x) for x in source_ids[start : start + page_chunk]]
            dst_pages = [int(x) for x in destination_ids[start : start + page_chunk]]
            tokens = min(self.chunk_size, num_tokens - start * self.page_size)
            yield self.token_ids(src_pages, self.page_size, tokens), dst_pages

    @staticmethod
    def _validate_lengths(lengths, capacities, error):
        if any(
            not 0 <= size <= capacity for size, capacity in zip(lengths, capacities)
        ):
            raise ValueError(error)

    def _begin_frame(
        self, room, part, dst_pages, tokens, lengths, u8_bytes, pool, slot_id
    ):
        sequence = str(next(self.sequence))
        event, result = threading.Event(), {}
        with self.pending_lock:
            self.pending[sequence] = (event, result, pool, slot_id)
        meta = {
            "version": 3,
            "epoch": self.epoch,
            "slot_id": slot_id,
            "profile": self.runtime.wire_profile,
            "room": room,
            "sequence": sequence,
            "part": part,
            "indices": dst_pages,
            "tokens": tokens,
            "codec_bytes": lengths[0],
            "u8_bytes": u8_bytes,
            "aux_bytes": lengths[1],
            "reply_host": self.manager.local_ip,
            "reply_port": self.manager.rank_port,
        }
        return sequence, event, result, meta

    @staticmethod
    def _frame_blocks(buffers, peer, lengths):
        return [
            (tensor.data_ptr(), address, size)
            for tensor, address, size in zip(buffers, peer["ptrs"], lengths)
            if size
        ]

    def finish_frame(self, frame):
        stats, result = frame["stats"], frame["result"]
        try:
            if not frame["event"].wait(60):
                raise TimeoutError("KVServe restoration ACK timed out")
            if result.get("cancelled"):
                self.record({"event": "send_cancelled", **stats})
                return
            if result.get("error"):
                raise RuntimeError(result["error"])
            self.record(
                {
                    "event": "send",
                    **stats,
                    "restore_ms": result["restore_ms"],
                    "restore_gpu_ms": result["restore_gpu_ms"],
                    "total_ms": (result["ack_perf"] - frame["begin"]) * 1000,
                    "time": result["ack_time"],
                }
            )
        finally:
            with self.pending_lock:
                self.pending.pop(stats["sequence"], None)

    def handle_ack(self, msg):
        if msg[0] != ACK:
            return False
        for value in msgspec.msgpack.decode(msg[1]):
            if value.get("epoch") != self.epoch:
                continue
            with self.pending_lock:
                item = self.pending.get(value["sequence"])
                if item:
                    event, result, pool, slot = item
                    if not event.is_set():
                        result.update(value)
                        result["ack_perf"] = time.perf_counter()
                        result["ack_time"] = time.time()
                        if not value.get("error"):
                            pool.release(slot)
                        event.set()
        return True

    def _validate_restore(self, meta, slot_id):
        if (
            self.poisoned
            or meta.get("version") != 3
            or meta["profile"] != self.runtime.wire_profile
        ):
            raise ValueError("Invalid or poisoned KVServe receive session")
        pages = meta["indices"]
        ids = self.token_ids(pages, self.page_size, meta["tokens"])
        part = meta["part"]
        if (
            part not in (0, 1)
            or not 0 < len(ids) <= self.chunk_size
            or len(set(ids)) != len(ids)
            or any(type(x) is not int or not 0 <= x < self.cache_blocks for x in ids)
        ):
            raise ValueError("Invalid KVServe destination pages")
        if meta["aux_bytes"] != self.runtime.aux_bytes(part, {0: len(ids)}):
            raise ValueError("Invalid KVServe auxiliary length")
        if not 0 <= meta["codec_bytes"] <= self.runtime.slots[slot_id].arena_a.numel():
            raise ValueError("Invalid KVServe encoded length")
        return pages, ids, part

    def reply(self, meta, result):
        self.notifier.put(endpoint(meta["reply_host"], meta["reply_port"]), ACK, result)

    def check_success(self, room, status):
        expected = self.expected_rooms.get(room)
        restored = self.restored_rooms.get(room)
        if status == KVPoll.Success and (
            expected is None
            or restored is None
            or restored[0] != expected
            or restored[1] != expected
        ):
            # Reject an uncompressed sender, even if it ignored our registration.
            self.manager.record_failure(
                room,
                "KVServe completion preceded restoration of all registered K/V pages",
            )
            return KVPoll.Failed
        return status

    def close(self):
        self.assert_close_owner()
        with self.submit_lock:
            self._closing = True
        if self.poisoned:
            raise RuntimeError(
                "Poisoned KVServe arena must remain registered until process exit"
            )
        # ACK/READY listeners must still be alive while this owner drains.
        with self.pending_lock:
            pending = list(self.pending.values())
        deadline = time.monotonic() + 60
        for event, _, _, _ in pending:
            if not event.wait(max(0, deadline - time.monotonic())):
                self.poisoned = True
                raise TimeoutError(
                    "KVServe close retains registered arenas after an unsettled ACK"
                )
        with self.lock:
            if getattr(self, "_source_unsettled_rooms", set()) or getattr(
                self, "_remote_restore_unsettled", set()
            ):
                raise RuntimeError(
                    "Unsettled KVServe GPU/remote writes retain registered arenas"
                )
            if any(
                self.restored_rooms.get(room) != {0: expected, 1: expected}
                for room, expected in self.expected_rooms.items()
            ):
                raise RuntimeError(
                    "KVServe receive rooms still accept writes; registrations retained"
                )
        self.notifier.close()
        self.manager.engine.batch_deregister(self.registered_ptrs)
        self.runtime.shutdown()
