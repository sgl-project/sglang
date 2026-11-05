"""Explicitly composed KVServe codec/TX/RX pipeline for native Mooncake P/D.

No import-time class replacement or alternate transport protocol is used. The
Runtime owns all codec/output memory; this adapter owns its worker lifetimes and
retains native request/page ownership until restored-KV ACK completion.
"""

from __future__ import annotations

import time

import msgspec
import torch

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.mooncake.kvserve import READY, KVServeAdapter, endpoint
from sglang.srt.disaggregation.mooncake.kvserve_pipeline import (
    BoundedSourcePool,
    EncodedTransferPipeline,
    RestorePipeline,
)


class OptimizedKVServeAdapter(KVServeAdapter):
    """Pipelined default codec using explicit Runtime ``encode_into`` leases."""

    def __init__(self, manager, k_buffers, v_buffers, page_size, layer_ids):
        mode = getattr(manager, "disaggregation_mode", None)
        role = str(getattr(mode, "value", mode)).lower()
        if role not in ("prefill", "decode"):
            raise ValueError(
                "Optimized KVServe requires an explicit Prefill/Decode role"
            )
        super().__init__(manager, k_buffers, v_buffers, page_size, layer_ids)
        self.role = role
        self._source_pool = None
        self._tx_v2 = None
        self._receive_pipeline_v2 = None
        self._optimized_closed = False
        self.raw_per_part = {
            part: sum(
                row.num_heads * row.head_size * 2
                for row in self.runtime.layout.parts(part)
            )
            for part in (0, 1)
        }
        if self.role == "prefill":
            banks = self.runtime.output_slots
            if len(banks) != 4:
                raise ValueError(
                    "Optimized KVServe needs four pre-reserved output slots"
                )
            self._tx_v2 = EncodedTransferPipeline(
                banks, self._write_encoded, self.control_failed, workers=2
            )
            self._source_pool = BoundedSourcePool(len(self.runtime.slots))
        else:
            # Base cancel_room/local_restore_release_safe delegate to this object.
            self._receive_pipeline_v2 = RestorePipeline(self, torch, KVPoll.Failed)
        self.record(
            dict(
                event="optimized_init",
                role=self.role,
                codec_slots=len(self.runtime.slots),
                output_slots=len(self.runtime.output_slots),
                tx_workers=2,
            )
        )

    def send(
        self, room, request, registration, source_ids, destination_ids, num_tokens=None
    ):
        if self._source_pool is None:
            raise RuntimeError("Decode adapter cannot submit Prefill codec frames")
        tokens_total, pool = self._prepare_send(
            request, registration, source_ids, destination_ids, num_tokens
        )
        futures, transfers, frames = [], [], []
        error = None
        try:
            for src, dst_pages in self._frame_chunks(
                source_ids, destination_ids, tokens_total
            ):
                for part in (0, 1):
                    with self.submit_lock:
                        if getattr(self, "_closing", False):
                            raise RuntimeError("KVServe is closing")
                    # Backpressure waits on a native transfer worker, outside
                    # submit_lock; scheduler/control threads do not enter here.
                    futures.append(
                        self._source_pool.submit(
                            self.send_frame,
                            room,
                            request,
                            registration.kvserve,
                            src,
                            dst_pages,
                            part,
                            pool,
                        )
                    )
        except Exception as exc:
            error = exc
        # Submission failure must not abandon previously submitted GPU/TX work.
        for future in futures:
            try:
                transfers.append(future.result())
            except Exception as exc:
                error = error or exc
        for future in transfers:
            try:
                frames.append(future.result())
            except Exception as exc:
                error = error or exc
        for frame in frames:
            try:
                self.finish_frame(frame)
            except Exception as exc:
                error = error or exc
        if error is not None:
            self._tx_v2.fail(error)
            self.manager._poison_unsettled_send(room)
            raise error
        return 0

    def send_frame(
        self, slot_id, room, request, registration, src, dst_pages, part, pool
    ):
        lease = None
        source_drained = True
        try:
            if self.poisoned:
                raise RuntimeError("KVServe source is quarantined")
            bank_begin = time.perf_counter()
            lease = self._tx_v2.acquire()
            bank_wait_ms = (time.perf_counter() - bank_begin) * 1000
            with torch.cuda.device(self.device):
                begin = time.perf_counter()
                # encode_into may enqueue work before raising. Only a completed
                # event (or a successful error-path stream drain) releases pages.
                source_drained = False
                encoded = self.runtime.encode_into(
                    slot_id,
                    [src],
                    part,
                    valid_tokens_by_group={0: len(src)},
                    output=lease.buffers[0],
                    auxiliary=lease.buffers[1],
                )
                # This event covers payload, aux and size/status D2H. Reading
                # all metadata before returning is part of source-slot ownership.
                encoded.event.synchronize()
                source_drained = True
                lengths = [
                    encoded.codec_bytes,
                    self.runtime.aux_bytes(part, {0: len(src)}),
                ]
                if encoded.raw_tail_bytes:
                    raise RuntimeError("Token codec unexpectedly produced raw tails")
                self._validate_lengths(
                    lengths,
                    [tensor.numel() for tensor in lease.buffers],
                    "Encoded payload exceeds its output-slot capacity",
                )
                digest = (
                    self._buffers_digest(lease.buffers, lengths)
                    if self.verify
                    else None
                )
                u8_bytes = encoded.u8_bytes
                encode_ms = (time.perf_counter() - begin) * 1000
            stats = dict(
                room=room,
                source_slot_id=slot_id,
                part=part,
                tokens=len(src),
                raw_bytes=len(src) * self.raw_per_part[part],
                wire_bytes=sum(lengths),
                codec_bytes=lengths[0],
                aux_bytes=lengths[1],
                encode_ms=encode_ms,
                encode_gpu_ms=None,
                staging_bank_wait_ms=bank_wait_ms,
                source_lease_ms=(time.perf_counter() - begin) * 1000,
                source_done_time=time.time(),
                completion_host_waits=1,
            )
            job = dict(
                room=room,
                request=request,
                registration=registration,
                pool=pool,
                part=part,
                dst_pages=dst_pages,
                tokens=len(src),
                lengths=lengths,
                u8_bytes=u8_bytes,
                digest=digest,
                begin=begin,
                stats=stats,
            )
            future = self._tx_v2.submit(lease, job)
            lease = None
            return future
        except BaseException as exc:
            if not source_drained:
                try:
                    with torch.cuda.device(self.device):
                        self.runtime.streams[slot_id].synchronize()
                except BaseException:
                    with self.lock:
                        self._source_unsettled_rooms.add(room)
            if lease is not None and lease.state == "encoding_copy":
                # A submission/wait exception does not establish GPU quiescence.
                with self._tx_v2.condition:
                    lease.state = "quarantined"
            self._tx_v2.fail(exc)
            self.manager._poison_unsettled_send(room)
            raise

    def _write_encoded(self, lease, job):
        if self.poisoned:
            raise RuntimeError("KVServe transfer pipeline is quarantined")
        receive_slot = job["pool"].acquire()[0]
        peer = job["registration"]["slots"][receive_slot]
        lengths = job["lengths"]
        self._validate_lengths(
            lengths,
            peer["capacities"],
            "Encoded frame exceeds negotiated receive capacity",
        )
        sequence, event, result, meta = self._begin_frame(
            job["room"],
            job["part"],
            job["dst_pages"],
            job["tokens"],
            lengths,
            job["u8_bytes"],
            job["pool"],
            receive_slot,
        )
        if job["digest"] is not None:
            meta["digest"] = job["digest"]
        blocks = self._frame_blocks(lease.buffers, peer, lengths)
        start = time.perf_counter()
        with torch.cuda.device(self.device):
            status = self.manager._transfer_data(
                job["request"].mooncake_session_id, blocks
            )
        if status != 0:
            raise RuntimeError(f"Mooncake compressed payload write failed: {status}")
        elapsed_ms = (time.perf_counter() - start) * 1000
        self.notifier.put(
            endpoint(job["request"].endpoint, job["request"].dst_port), READY, meta
        )
        stats = dict(
            job["stats"],
            sequence=sequence,
            slot_id=receive_slot,
            transfer_ms=elapsed_ms,
            tx_lease_ms=(time.perf_counter() - job["begin"]) * 1000,
            transport_done_time=time.time(),
            batched_write=False,
        )
        # EncodedTransferPipeline releases this output bank after return: sync
        # RDMA is finished. Destination credit remains owned until restore ACK.
        return dict(event=event, result=result, begin=job["begin"], stats=stats)

    def handle_ready(self, msg):
        if msg[0] != READY:
            return False
        if self._receive_pipeline_v2 is None:
            raise ValueError("Prefill adapter cannot accept restore READY frames")
        self._receive_pipeline_v2.accept(msgspec.msgpack.decode(msg[1]))
        return True

    def close(self):
        self.assert_close_owner()
        if self._optimized_closed:
            return
        with self.submit_lock:
            self._closing = True
        if self._source_pool is not None:
            self._source_pool.shutdown(wait=True)
        if self._tx_v2 is not None:
            self._tx_v2.close()
        if self._receive_pipeline_v2 is not None:
            self._receive_pipeline_v2.close()
        # Base close drains ACKs while control listeners remain live, then
        # deregisters the Runtime's authoritative memory set before freeing it.
        super().close()
        self._optimized_closed = True
