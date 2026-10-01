"""Test-only P source observations and handoff faults in spawned workers."""

import hashlib
import importlib
import itertools
import json
import os
import sys
import threading
from pathlib import Path

import msgspec
import torch

from sglang.srt.disaggregation.decode import SchedulerDisaggregationDecodeMixin
from sglang.srt.disaggregation.prefill import SchedulerDisaggregationPrefillMixin
from sglang.srt.distributed import (
    get_pipeline_model_parallel_rank,
    get_pipeline_model_parallel_world_size,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.training_capture.pd_capture import PrefillCaptureCoordinator
from sglang.srt.training_capture.pd_protocol import (
    PrefillTeacherHandoff,
    decode_handoff,
    encode_handoff,
)
from sglang.test.dspark_capture_observer import install_capture_observer

_pools = {}
_sequence = itertools.count()
_init = TpModelWorker.init_training_capture
_prefill_forward = PrefillCaptureCoordinator.after_forward
_handoff = PrefillCaptureCoordinator.finish_handoff
_prebuilt = SchedulerDisaggregationDecodeMixin.get_new_prebuilt_batch
_send_kv_chunk = SchedulerDisaggregationPrefillMixin.send_kv_chunk


def observe_stage_state(capture, stage):
    previous = None
    while not capture.stop.wait(0.5):
        state = capture.stats()
        writer = state.get("cohort_writer", {})
        current = {
            "counters": state["counters"],
            "states": state["states"],
            "writer_states": writer.get("states"),
            "writer_error": writer.get("error"),
            "disabled_reason": state["disabled_reason"],
        }
        if current != previous:
            print(json.dumps({"pd_capture_stage": stage, "state": current}), flush=True)
            previous = current


def observed_init(self, **kwargs):
    _init(self, **kwargs)
    if isinstance(self.training_capture, PrefillCaptureCoordinator):
        _pools[id(self.training_capture)] = (
            self.model_runner.token_to_kv_pool,
            self.model_runner.req_to_token_pool,
        )
    elif (
        self.training_capture is not None
        and get_pipeline_model_parallel_world_size() > 1
    ):
        threading.Thread(
            target=observe_stage_state,
            args=(
                self.training_capture,
                {
                    "pp": get_pipeline_model_parallel_rank(),
                    "tp": get_tensor_model_parallel_rank(),
                },
            ),
            daemon=True,
        ).start()


def observed_prefill(self, batch, forward_batch, logits_output, **kwargs):
    root = Path(self.config.journal_directory).parent / "capture-reference"
    rank = get_tensor_model_parallel_rank()
    root = root / f"pp{get_pipeline_model_parallel_rank()}" / f"tp{rank}"
    root.mkdir(parents=True, exist_ok=True)
    for row, req in enumerate(batch.reqs):
        if req.training_capture_pd is None:
            continue
        end = int(batch.seq_lens_cpu[row])
        last = end == len(req.origin_input_ids) and logits_output is not None
        torch.save(
            {
                "trace_id": hashlib.sha256(req.rid.encode()).hexdigest(),
                "tp_rank": rank,
                "tp_size": get_tensor_model_parallel_world_size(),
                "pp_rank": get_pipeline_model_parallel_rank(),
                "tokens": list(req.origin_input_ids[:end]),
                "kv_start": 0,
                "kv": {},
                "predictions": [end] if last else [],
                "logits": (
                    logits_output.next_token_logits[
                        row : row + int(last), : self.teacher.vocab_size
                    ]
                    .float()
                    .cpu()
                    if logits_output is not None
                    else torch.empty(0, self.teacher.vocab_size)
                ),
                "batch_size": len(batch.reqs),
                "cuda_graph": kwargs.get("can_run_cuda_graph", False),
            },
            root / f"prefill-{next(_sequence):06d}.pt",
        )
    return _prefill_forward(self, batch, forward_batch, logits_output, **kwargs)


def observed_send(self, req, last_chunk=False, end_idx=None):
    capture = self.tp_worker.training_capture
    # Cached-prefix sends can precede capture admission on P. Observe every
    # send; the reader selects the references by the published trace ID.
    if capture is not None:
        pool, req_pool = _pools[id(capture)]
        start = req.start_send_idx
        end = (
            end_idx
            if end_idx is not None
            else min(req.extend_range.end, len(req.origin_input_ids))
        )
        if not last_chunk:
            end -= end % self.token_to_kv_pool_allocator.page_size
        if start < end:
            # Radix insertion can replace newly computed rows before transfer.
            # Observe the canonical source slots that the sender will read.
            slots = req_pool.req_to_token[req.req_pool_idx, start:end].long()
            root = Path(capture.config.journal_directory).parent / "capture-reference"
            rank = get_tensor_model_parallel_rank()
            stage = get_pipeline_model_parallel_rank()
            root = root / f"pp{stage}" / f"tp{rank}"
            root.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "trace_id": hashlib.sha256(req.rid.encode()).hexdigest(),
                    "tp_rank": rank,
                    "tp_size": get_tensor_model_parallel_world_size(),
                    "pp_rank": stage,
                    "tokens": list(req.origin_input_ids[:end]),
                    "kv_start": start,
                    "kv": {
                        f"target_{component}.{layer}": buffer[slots].cpu()
                        for layer in capture.kv.selected_layer_ids
                        if pool.start_layer <= layer < pool.start_layer + pool.layer_num
                        for component, buffer in (
                            ("k", pool.get_key_buffer(layer)),
                            ("v", pool.get_value_buffer(layer)),
                        )
                    },
                    "predictions": [],
                    "logits": torch.empty(0, capture.teacher.vocab_size),
                    "batch_size": 1,
                    "cuda_graph": False,
                },
                root / f"transfer-{next(_sequence):06d}.pt",
            )
    return _send_kv_chunk(self, req, last_chunk=last_chunk, end_idx=end_idx)


def handoff_faults(self, req):
    payload = _handoff(self, req)
    if req.rid.startswith("missing-"):
        return None
    if (
        payload is not None
        and req.rid.startswith("stale-")
        and get_tensor_model_parallel_rank() == 0
        and get_pipeline_model_parallel_rank() == 0
    ):
        value = decode_handoff(payload, PrefillTeacherHandoff)
        context = msgspec.structs.replace(
            value.context, fencing_token=value.context.fencing_token + 1
        )
        return encode_handoff(msgspec.structs.replace(value, context=context))
    return payload


def grouped_prebuilt(self, running_batch):
    # PD transfers may complete separately even for one batched HTTP request.
    # Hold the tagged pair in the ready queue to exercise an actual decode batch.
    if (
        running_batch.is_empty()
        and sum(req.rid.startswith("batch-") for req in self.waiting_queue) == 1
    ):
        return None
    return _prebuilt(self, running_batch)


TpModelWorker.init_training_capture = observed_init
PrefillCaptureCoordinator.after_forward = observed_prefill
PrefillCaptureCoordinator.finish_handoff = handoff_faults
install_capture_observer()
if "--speculative-draft-model-path" in sys.argv:
    draft_path = Path(sys.argv[sys.argv.index("--speculative-draft-model-path") + 1])
    if (
        json.loads((draft_path / "config.json").read_text()).get("input_mode")
        == "target_kv"
    ):
        importlib.import_module("sglang.test.dspark_target_kv_server")
    else:
        importlib.import_module("sglang.test.dspark_capture_server")
SchedulerDisaggregationDecodeMixin.get_new_prebuilt_batch = grouped_prebuilt
SchedulerDisaggregationPrefillMixin.send_kv_chunk = observed_send


if __name__ == "__main__":
    from sglang.launch_server import run_server
    from sglang.srt.server_args import prepare_server_args
    from sglang.srt.utils import kill_process_tree

    try:
        run_server(prepare_server_args(sys.argv[1:]))
    finally:
        kill_process_tree(os.getpid(), include_parent=False)
