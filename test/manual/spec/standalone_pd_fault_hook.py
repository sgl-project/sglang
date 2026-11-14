"""Test-only producer corruption and read-only Decode resource observations.

Loaded explicitly through --forward-hooks by the manual recovery test. Only the
selected RID's CPU seed row is changed, after the real producer has written it
and before its caller sends the final chunk. Decode methods, transport polling,
validation, allocation and release retain their original calls and return values.
The bounded allocator census synchronizes small GPU index tensors to the host;
this is a functional test, not performance or timing/race coverage.
"""

import functools
import json
import os
import threading
from pathlib import Path

_config = None
_lock = threading.RLock()
_sequence = 0
_injected = False
_success = set()
_pending_idle = True
_completed = []
LIMIT = 256


def emit(phase, **fields):
    global _sequence
    with _lock:
        if _sequence >= LIMIT:
            raise RuntimeError("STANDALONE recovery observation limit exceeded")
        _sequence += 1
        line = (
            json.dumps(
                dict(phase=phase, sequence=_sequence, pid=os.getpid(), **fields),
                allow_nan=False,
            ).encode()
            + b"\n"
        )
        flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW
        fd = os.open(_config["log"], flags, 0o600)
        try:
            if os.write(fd, line) != len(line):
                raise RuntimeError("Incomplete recovery observation")
        finally:
            os.close(fd)


def cpu_values(tensor):
    return tensor.detach().cpu().reshape(-1).tolist()


def resources(queue):
    scheduler = queue.scheduler
    pool = scheduler.req_to_token_pool
    tokens = scheduler.token_to_kv_pool_allocator
    metadata = queue.req_to_metadata_buffer_idx_allocator
    return dict(
        allocator=type(tokens).__name__,
        page_size=tokens.page_size,
        token_capacity=tokens.size,
        token_available=tokens.available_size(),
        token_free_ids=cpu_values(tokens.get_all_free_pages()),
        free_group_pending=(
            None if tokens.free_group is None else len(tokens.free_group)
        ),
        request_pool=type(pool).__name__,
        request_capacity=pool.size + pool.pre_alloc_size,
        request_available=pool.available_size(),
        request_free_slots=list(pool.free_slots),
        metadata_capacity=metadata.size,
        metadata_available=metadata.available_size(),
        metadata_free_slots=list(metadata.free_slots),
    )


def request(queue, entry):
    req = entry.req
    pool = queue.scheduler.req_to_token_pool
    index = req.kv.req_pool_idx
    reason = req.finished_reason
    return dict(
        rid=req.rid,
        room=req.bootstrap_room,
        metadata_slot=entry.metadata_buffer_index,
        request_slot=index,
        generation=None if index is None else int(pool.req_generation[index]),
        allocated_length=req.kv.kv_allocated_len,
        kv_ids=(
            []
            if index is None
            else cpu_values(pool.req_to_token[index, : req.kv.kv_allocated_len])
        ),
        output_ids=list(req.output_ids),
        holds_kv=req.kv.holds_kv,
        kv_released=req.kv.is_kv_released,
        receiver_cleared=entry.kv_receiver is None,
        finish=None if reason is None else type(reason).__name__,
        status_code=getattr(reason, "status_code", None),
        message=getattr(reason, "message", None),
    )


def install_prefill():
    from sglang.srt.disaggregation.utils import MetadataBuffers

    original = MetadataBuffers.set_buf

    @functools.wraps(original)
    def set_buf(self, req):
        global _injected
        result = original(self, req)
        slot = req.metadata_buffer_index
        if self.output_topk_p.device.type != "cpu":
            raise RuntimeError("Recovery test requires CPU metadata transport")
        before = self.output_topk_p[slot, :1].tolist()
        room = int(self.bootstrap_room[slot, 0])
        if req.rid == _config["bad_rid"]:
            if _injected or room != req.bootstrap_room or not before[0] > 0:
                raise RuntimeError(
                    "Bad-RID injection ownership/publication gate failed"
                )
            # The real _send_kv_chunk invokes set_buf before final sender.send.
            # Corrupt its actual registered row, not a clone or a Decode input.
            self.output_topk_p[slot].zero_()
            _injected = True
        emit(
            "producer",
            rid=req.rid,
            room=room,
            slot=slot,
            probability_before=before,
            probability_sent=self.output_topk_p[slot, :1].tolist(),
            injected=req.rid == _config["bad_rid"],
        )
        return result

    MetadataBuffers.set_buf = set_buf


def install_decode():
    import sglang.srt.speculative.standalone_disaggregation as standalone
    from sglang.srt.disaggregation.base import KVPoll
    from sglang.srt.disaggregation.decode import (
        DecodePreallocQueue,
        DecodeTransferQueue,
    )
    from sglang.srt.managers.scheduler import Scheduler
    from sglang.srt.speculative.eagle_worker_v2 import EagleDraftWorker, EAGLEWorkerV2

    original_send = DecodePreallocQueue._send_kv_metadata

    @functools.wraps(original_send)
    def send(self, entry, *args, **kwargs):
        global _pending_idle
        result = original_send(self, entry, *args, **kwargs)
        _pending_idle = True
        queue = self.transfer_queue
        emit("allocated", request=request(queue, entry), resources=resources(queue))
        return result

    DecodePreallocQueue._send_kv_metadata = send
    original_poll = DecodeTransferQueue._poll_with_metadata_gate

    @functools.wraps(original_poll)
    def poll(self):
        result = original_poll(self)
        for entry, status in zip(self.queue.items(), result):
            if status == KVPoll.Success and entry.req.rid not in _success:
                _success.add(entry.req.rid)
                slot = entry.metadata_buffer_index
                emit(
                    "transfer_success",
                    rid=entry.req.rid,
                    room=int(self.metadata_buffers.bootstrap_room[slot, 0]),
                    probability_received=self.metadata_buffers.output_topk_p[
                        slot, :1
                    ].tolist(),
                )
        return result

    DecodeTransferQueue._poll_with_metadata_gate = poll
    original_commit = DecodeTransferQueue._commit_transfer_to_req

    @functools.wraps(original_commit)
    def commit(self, entry):
        if entry.req.rid not in _success:
            raise RuntimeError("Commit was not preceded by real transport success")
        emit("commit_before", request=request(self, entry))
        result = original_commit(self, entry)
        emit("commit_after", request=request(self, entry))
        return result

    DecodeTransferQueue._commit_transfer_to_req = commit
    original_release = DecodeTransferQueue._release_request

    @functools.wraps(original_release)
    def release(self, entry):
        emit("release_before", request=request(self, entry), resources=resources(self))
        result = original_release(self, entry)
        emit("release_after", request=request(self, entry), resources=resources(self))
        return result

    DecodeTransferQueue._release_request = release
    original_pop = DecodeTransferQueue.pop_transferred_with_outcomes

    @functools.wraps(original_pop)
    def pop(self, *args, **kwargs):
        entries = {entry.req.rid: entry for entry in self.queue.items()}
        result = original_pop(self, *args, **kwargs)
        admitted, outcomes = result
        if outcomes:
            _completed.extend(outcomes)
            emit(
                "pop_after",
                admitted=[req.rid for req in admitted],
                outcomes=outcomes,
                requests=[request(self, entries[rid]) for rid in outcomes],
                cleared_rooms={
                    rid: int(
                        self.metadata_buffers.bootstrap_room[
                            entries[rid].metadata_buffer_index, 0
                        ]
                    )
                    for rid in outcomes
                },
                resources=resources(self),
            )
        return result

    DecodeTransferQueue.pop_transferred_with_outcomes = pop
    original_idle = Scheduler.on_idle

    @functools.wraps(original_idle)
    def idle(self):
        global _pending_idle
        result = original_idle(self)
        if _pending_idle and self.is_fully_idle():
            queue = self.disagg_decode_transfer_queue
            emit("idle", completed=list(_completed), resources=resources(queue))
            _pending_idle = False
        return result

    Scheduler.on_idle = idle

    def observe_batch(owner, name):
        original = getattr(owner, name)

        @functools.wraps(original)
        def observed(self, batch, *args, **kwargs):
            emit(name, rids=[req.rid for req in batch.reqs])
            return original(self, batch, *args, **kwargs)

        setattr(owner, name, observed)

    observe_batch(EagleDraftWorker, "draft")
    observe_batch(EAGLEWorkerV2, "verify")
    original_factory = standalone.build_standalone_disagg_draft_input

    @functools.wraps(original_factory)
    def factory(batch, *args, **kwargs):
        emit("factory", rids=[req.rid for req in batch.reqs])
        return original_factory(batch, *args, **kwargs)

    standalone.build_standalone_disagg_draft_input = factory


def make_hook(config):
    global _config
    with _lock:
        if _config is not None:
            if config != _config:
                raise RuntimeError("Conflicting recovery hooks in one process")
        else:
            if set(config) != {"role", "log", "bad_rid"}:
                raise ValueError("Unexpected recovery hook configuration")
            if not Path(config["log"]).is_absolute() or not config["bad_rid"]:
                raise ValueError("An absolute test log and exact bad RID are required")
            _config = dict(config)
            if config["role"] == "prefill":
                install_prefill()
            elif config["role"] == "decode":
                install_decode()
            else:
                raise ValueError("Only the two explicit PD roles are supported")
            emit("installed", role=config["role"])
    return lambda *_args, **_kwargs: None
