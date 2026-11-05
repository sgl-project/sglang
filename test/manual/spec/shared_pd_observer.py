"""Bounded, read-only PD metadata observations for the manual shared-path test.

Loaded only with the official forward-hook option. Tensor fingerprints use
bounded CPU copies (including D2H at the EAGLE factory); this is functional
coverage, not a performance or race test. Original calls and returns are kept.
"""

import contextvars
import functools
import hashlib
import json
import os
import threading
from pathlib import Path

_config = None
_lock = threading.RLock()
_sequence = 0
_commit = contextvars.ContextVar("shared_pd_commit", default=None)
_draft = contextvars.ContextVar("shared_pd_draft", default=())
LIMIT = 1024


def tensor_record(tensor):
    if tensor is None:
        return None
    import torch

    if tensor.numel() > 65536:
        raise RuntimeError("Shared PD tensor observation exceeds bound")
    value = tensor.detach().cpu().contiguous()
    raw = value.reshape(-1).view(torch.uint8).numpy().tobytes()
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "nonzero": int(value.count_nonzero()),
    }


def emit(phase, **fields):
    global _sequence
    with _lock:
        if _sequence >= LIMIT:
            raise RuntimeError("Shared PD observation limit exceeded")
        _sequence += 1
        raw = (
            json.dumps(
                dict(phase=phase, pid=os.getpid(), sequence=_sequence, **fields),
                allow_nan=False,
            ).encode()
            + b"\n"
        )
        fd = os.open(
            _config["log"],
            os.O_CREAT | os.O_APPEND | os.O_WRONLY | os.O_NOFOLLOW,
            0o600,
        )
        try:
            if os.write(fd, raw) != len(raw):
                raise RuntimeError("Incomplete shared PD observation")
        finally:
            os.close(fd)


def install_prefill():
    from sglang.srt.disaggregation.prefill import SchedulerDisaggregationPrefillMixin
    from sglang.srt.disaggregation.utils import MetadataBuffers

    original_set = MetadataBuffers.set_buf

    @functools.wraps(original_set)
    def set_buf(self, req):
        result = original_set(self, req)
        idx = req.metadata_buffer_index
        if self.output_hidden_states.device.type != "cpu":
            raise RuntimeError("Shared PD observer requires CPU metadata transport")
        emit(
            "producer",
            rid=req.rid,
            room=int(self.bootstrap_room[idx, 0]),
            token=int(self.output_ids[idx, 0]),
            topk=tensor_record(self.output_topk_p[idx]),
            indices=tensor_record(self.output_topk_index[idx]),
            hidden=tensor_record(self.output_hidden_states[idx]),
            active_topk=tensor_record(req.output_topk_p),
            active_indices=tensor_record(req.output_topk_index),
            active_hidden=tensor_record(req.hidden_states_tensor),
        )
        return result

    MetadataBuffers.set_buf = set_buf
    original_send = SchedulerDisaggregationPrefillMixin._send_kv_chunk

    @functools.wraps(original_send)
    def send(self, req, last_chunk=False, end_idx=None):
        start = req.start_send_idx
        result = original_send(self, req, last_chunk=last_chunk, end_idx=end_idx)
        emit(
            "chunk_sent",
            rid=req.rid,
            last_chunk=last_chunk,
            start=start,
            end=req.start_send_idx,
        )
        return result

    SchedulerDisaggregationPrefillMixin._send_kv_chunk = send


def install_decode():
    import sglang.srt.speculative.eagle_disaggregation as eagle
    from sglang.srt.disaggregation.decode import DecodeTransferQueue
    from sglang.srt.disaggregation.utils import MetadataBuffers
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.speculative.eagle_worker_v2 import EagleDraftWorker, EAGLEWorkerV2

    original_get = MetadataBuffers.get_buf

    @functools.wraps(original_get)
    def get_buf(self, idx):
        result = original_get(self, idx)
        entry = _commit.get()
        if entry is not None:
            if len(result) != 14 or idx != entry.metadata_buffer_index:
                raise RuntimeError("Unexpected real Decode metadata read")
            emit(
                "received",
                rid=entry.req.rid,
                room=int(result[13][0]),
                token=int(result[0][0]),
                topk=tensor_record(result[9]),
                indices=tensor_record(result[10]),
                hidden=tensor_record(result[11]),
                active_topk=tensor_record(result[9][:1]),
                active_indices=tensor_record(result[10][:1]),
            )
        return result

    MetadataBuffers.get_buf = get_buf
    original_commit = DecodeTransferQueue._commit_transfer_to_req

    @functools.wraps(original_commit)
    def commit(self, decode_req):
        entry = decode_req
        token = _commit.set(entry)
        try:
            result = original_commit(self, entry)
            req = entry.req
            emit(
                "committed",
                rid=req.rid,
                room=req.bootstrap_room,
                output_ids=list(req.output_ids),
                receiver_cleared=entry.kv_receiver is None,
                finished=req.finished_reason is not None,
                hidden=tensor_record(req.hidden_states_tensor),
            )
            return result
        finally:
            _commit.reset(token)

    DecodeTransferQueue._commit_transfer_to_req = commit
    original_factory = eagle.build_eagle_disagg_draft_input

    @functools.wraps(original_factory)
    def factory(batch, *args, **kwargs):
        rids = [req.rid for req in batch.reqs]
        result = original_factory(batch, *args, **kwargs)
        for i, rid in enumerate(rids):
            emit(
                "eagle_factory",
                rid=rid,
                hidden=tensor_record(result.hidden_states[i]),
                topk=tensor_record(result.topk_p[i]),
                indices=tensor_record(result.topk_index[i]),
                capture_hidden_mode=result.capture_hidden_mode.name,
            )
        return result

    eagle.build_eagle_disagg_draft_input = factory
    for cls, name in ((EagleDraftWorker, "draft"), (EAGLEWorkerV2, "verify")):
        original = getattr(cls, name)

        def wrap(original, phase):
            @functools.wraps(original)
            def observed(self, batch, *args, **kwargs):
                rids = [req.rid for req in batch.reqs]
                token = _draft.set(tuple(rids) if phase == "draft" else ())
                try:
                    result = original(self, batch, *args, **kwargs)
                    emit(phase, rids=rids)
                    return result
                finally:
                    _draft.reset(token)

            return observed

        setattr(cls, name, wrap(original, name))

    original_forward = ModelRunner.forward

    @functools.wraps(original_forward)
    def forward(self, forward_batch, *args, **kwargs):
        rids = _draft.get()
        observe = (
            rids and self.is_draft_worker and forward_batch.forward_mode.is_decode()
        )
        result = original_forward(self, forward_batch, *args, **kwargs)
        if observe:
            emit(
                "inner_draft_decode",
                rids=list(rids),
                model_class=type(self.model).__name__,
                forward_mode=forward_batch.forward_mode.name,
            )
        return result

    ModelRunner.forward = forward


def make_hook(config):
    global _config
    with _lock:
        if _config is not None:
            if config != _config:
                raise RuntimeError("Conflicting shared PD observer configuration")
        else:
            if set(config) != {"role", "log"} or not Path(config["log"]).is_absolute():
                raise ValueError("Explicit PD role and absolute log path required")
            _config = dict(config)
            if config["role"] == "prefill":
                install_prefill()
            elif config["role"] == "decode":
                install_decode()
            else:
                raise ValueError("Only explicit PD peers can use this observer")
            emit("installed", role=config["role"])
    return lambda *_args, **_kwargs: None
