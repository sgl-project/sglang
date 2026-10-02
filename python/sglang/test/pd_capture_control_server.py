"""Test-only chunk gate for real HTTP control while a P/D capture is in flight."""

import json
import os
import sys
import time
from pathlib import Path

from sglang.srt.distributed import (
    get_pipeline_model_parallel_rank,
    get_tensor_model_parallel_rank,
)
from sglang.srt.managers.io_struct import SetInternalStateReqOutput
from sglang.srt.managers.schedule_batch import NextBatchPlan
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.training_capture.pd_capture import (
    DecodeCaptureMixin,
    PrefillCaptureCoordinator,
)
from sglang.test import pd_capture_server  # noqa: F401 - install source observers

_prefill = Scheduler.get_new_batch_prefill
_handoff = PrefillCaptureCoordinator.finish_handoff
_internal_state = Scheduler.get_internal_state
_set_internal_state = Scheduler.set_internal_state
_begin = DecodeCaptureMixin.begin_pd_transfer
_entered = {}
_released = set()
_release_prefix = "training_capture_test_release:"


def rank_name():
    return (
        f"pp{get_pipeline_model_parallel_rank()}"
        f"-tp{get_tensor_model_parallel_rank()}"
    )


def write_observation(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value))
    temporary.replace(path)


def release_gate(self, req):
    if len(req.server_args) == 1:
        name, value = next(iter(req.server_args.items()))
        if name.startswith(_release_prefix) and value == 1:
            # PP ranks must resume in the same request-message iteration.
            _released.add(name.removeprefix(_release_prefix))
            return SetInternalStateReqOutput(updated=True)
    return _set_internal_state(self, req)


def observed_admission(self, req):
    payload = _begin(self, req)
    if req.rid.startswith("control-"):
        record = req.training_capture_context
        root = Path(self.config.journal_directory).parent / "control-gates"
        write_observation(
            root / rank_name() / (req.rid + ".admission"),
            {"capture_id": record.lease.capture_id if record is not None else None},
        )
    return payload


def observed_internal_state(self, req):
    result = _internal_state(self, req)
    capture = self.tp_worker.training_capture
    if capture is not None:
        root = Path(capture.config.journal_directory).parent / "control-gates"
        request = root / "state-request"
        if request.exists():
            state = dict(result.internal_state["training_capture"])
            state["test_released_gates"] = sorted(_released)
            service = getattr(capture, "service", None)
            if service is not None:
                with service.lock:
                    state["test_service_admission_ready"] = service.admission_ready
                    state["test_invalid_captures"] = [
                        key
                        for key, handle in service.records.items()
                        if handle.invalid_reason is not None
                    ]
            write_observation(
                root / rank_name() / "state.json",
                {
                    "nonce": request.read_text(),
                    "state": state,
                },
            )
    return result


def gated_prefill(self, running_batch):
    req = self.chunked_req
    capture = self.tp_worker.training_capture
    if (
        isinstance(capture, PrefillCaptureCoordinator)
        and req is not None
        and req.rid.startswith("control-gated-")
    ):
        root = Path(capture.config.journal_directory).parent / "control-gates"
        hold = root / (req.rid + ".hold")
        if req.rid not in _released and hold.exists():
            if req.rid not in _entered:
                _entered[req.rid] = time.monotonic()
                state = req.training_capture_pd
                payload = {
                    "rid": req.rid,
                    "prompt_length": len(req.origin_input_ids),
                    "chunk_end": req.extend_range.end,
                    "attempted": req.training_capture_attempted,
                    "selected": state is not None,
                    "epoch": state.epoch if state is not None else None,
                }
                write_observation(root / rank_name() / (req.rid + ".entered"), payload)
            if time.monotonic() - _entered[req.rid] > 60:
                raise TimeoutError("P/D capture control test did not release its gate")
            # Return to the event loop so real HTTP management requests run.
            time.sleep(0.001)
            return NextBatchPlan(batch_to_run=None, running_batch=running_batch)
    return _prefill(self, running_batch)


def observed_handoff(self, req):
    state = req.training_capture_pd
    epoch = state.epoch if state is not None else None
    teacher_present = state is not None and state.teacher is not None
    payload = _handoff(self, req)
    if req.rid.startswith("control-"):
        root = Path(self.config.journal_directory).parent / "control-gates"
        write_observation(
            root / rank_name() / (req.rid + ".handoff"),
            {
                "capture_epoch": epoch,
                "current_epoch": self.abort_epoch,
                "teacher_present": teacher_present,
                "payload_present": payload is not None,
            },
        )
    return payload


Scheduler.get_new_batch_prefill = gated_prefill
PrefillCaptureCoordinator.finish_handoff = observed_handoff
Scheduler.get_internal_state = observed_internal_state
Scheduler.set_internal_state = release_gate
DecodeCaptureMixin.begin_pd_transfer = observed_admission


if __name__ == "__main__":
    from sglang.launch_server import run_server
    from sglang.srt.server_args import prepare_server_args
    from sglang.srt.utils import kill_process_tree

    try:
        args = prepare_server_args(sys.argv[1:])
        run_server(args)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)
