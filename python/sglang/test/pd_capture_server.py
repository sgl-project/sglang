"""Test-only P source observations and handoff faults in spawned workers."""

import hashlib
import itertools
import os
import sys
from pathlib import Path

import msgspec
import torch

from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.training_capture.coordinator import CaptureCoordinator
from sglang.srt.training_capture.pd_capture import PrefillCaptureCoordinator
from sglang.srt.training_capture.pd_protocol import (
    PrefillTeacherHandoff,
    decode_handoff,
    encode_handoff,
)
from sglang.test.dspark_capture_observer import observe

_pools = {}
_sequence = itertools.count()
_init = TpModelWorker.init_training_capture
_prefill_forward = PrefillCaptureCoordinator.after_forward
_decode_forward = CaptureCoordinator.after_forward
_handoff = PrefillCaptureCoordinator.finish_handoff


def observed_init(self, **kwargs):
    _init(self, **kwargs)
    if isinstance(self.training_capture, PrefillCaptureCoordinator):
        _pools[id(self.training_capture)] = (
            self.model_runner.token_to_kv_pool,
            self.model_runner.req_to_token_pool,
        )


def observed_prefill(self, batch, forward_batch, logits_output, **kwargs):
    pool, req_pool = _pools[id(self)]
    root = Path(self.config.journal_directory).parent / "capture-reference"
    root.mkdir(parents=True, exist_ok=True)
    for row, req in enumerate(batch.reqs):
        if req.training_capture_pd is None:
            continue
        end = int(batch.seq_lens_cpu[row])
        slots = req_pool.req_to_token[req.req_pool_idx, :end].long()
        last = end == len(req.origin_input_ids)
        torch.save(
            {
                "trace_id": hashlib.sha256(req.rid.encode()).hexdigest(),
                "tokens": list(req.origin_input_ids[:end]),
                "kv_start": 0,
                "kv": {
                    f"target_{component}.{layer}": buffer[slots].cpu()
                    for layer in self.kv.selected_layer_ids
                    for component, buffer in (
                        ("k", pool.get_key_buffer(layer)),
                        ("v", pool.get_value_buffer(layer)),
                    )
                },
                "predictions": [end] if last else [],
                "logits": logits_output.next_token_logits[
                    row : row + int(last), : self.teacher.vocab_size
                ]
                .float()
                .cpu(),
                "batch_size": len(batch.reqs),
                "cuda_graph": kwargs.get("can_run_cuda_graph", False),
            },
            root / f"prefill-{next(_sequence):06d}.pt",
        )
    return _prefill_forward(self, batch, forward_batch, logits_output, **kwargs)


def observed_decode(self, batch, forward_batch, logits_output, **kwargs):
    observe(self, batch, forward_batch, logits_output, **kwargs)
    return _decode_forward(self, batch, forward_batch, logits_output, **kwargs)


def handoff_faults(self, req):
    payload = _handoff(self, req)
    if req.rid.startswith("missing-"):
        return None
    if payload is not None and req.rid.startswith("stale-"):
        value = decode_handoff(payload, PrefillTeacherHandoff)
        context = msgspec.structs.replace(
            value.context, fencing_token=value.context.fencing_token + 1
        )
        return encode_handoff(msgspec.structs.replace(value, context=context))
    return payload


TpModelWorker.init_training_capture = observed_init
PrefillCaptureCoordinator.after_forward = observed_prefill
PrefillCaptureCoordinator.finish_handoff = handoff_faults
CaptureCoordinator.after_forward = observed_decode


if __name__ == "__main__":
    from sglang.launch_server import run_server
    from sglang.srt.server_args import prepare_server_args
    from sglang.srt.utils import kill_process_tree

    try:
        run_server(prepare_server_args(sys.argv[1:]))
    finally:
        kill_process_tree(os.getpid(), include_parent=False)
