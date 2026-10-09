"""Observe real allocator-driven retraction without forcing its scheduling."""

import json
import os
import sys
from pathlib import Path

from sglang.srt.distributed import (
    get_pipeline_model_parallel_rank,
    get_pipeline_model_parallel_world_size,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.training_capture.cohort_coordinator import CohortCaptureCoordinator
from sglang.srt.training_capture.coordinator import CaptureCoordinator
from sglang.test.dspark_capture_observer import install_capture_observer

_retract_decode = ScheduleBatch.retract_decode
_internal_state = Scheduler.get_internal_state


def rank_name():
    return (
        f"pp{get_pipeline_model_parallel_rank()}-tp{get_tensor_model_parallel_rank()}"
    )


def reference_root(root):
    root = root / "capture-reference"
    if get_pipeline_model_parallel_world_size() > 1:
        root /= f"pp{get_pipeline_model_parallel_rank()}"
    if get_tensor_model_parallel_world_size() > 1:
        root /= f"tp{get_tensor_model_parallel_rank()}"
    return root


def observe_admission(admit):
    def observed(self, req):
        result = admit(self, req)
        record = req.training_capture_context
        if record is not None:
            root = Path(os.environ["TRAINING_CAPTURE_TEST_OUTPUT"])
            with (root / f"admissions-{rank_name()}.jsonl").open("a") as stream:
                stream.write(
                    json.dumps(
                        {
                            "rid": req.rid,
                            "capture_id": record.lease.capture_id,
                            "active_capture_rids": sorted(
                                r.rid for r in self.requests.values()
                            ),
                        }
                    )
                    + "\n"
                )
        return result

    return observed


def observed_internal_state(self, req):
    result = _internal_state(self, req)
    capture = self.tp_worker.training_capture
    root = Path(os.environ["TRAINING_CAPTURE_TEST_OUTPUT"]) / "rank-states"
    request = root / "request"
    if capture is not None and request.exists():
        state = dict(result.internal_state["training_capture"])
        service = getattr(capture, "service", None)
        if service is not None:
            with service.lock:
                state["test_service_admission_ready"] = service.admission_ready
        path = root / (rank_name() + ".json")
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps({"nonce": request.read_text(), "state": state}))
        temporary.replace(path)
    return result


def retract_decode(self, server_args):
    root = Path(os.environ["TRAINING_CAPTURE_TEST_OUTPUT"])
    event = {
        "available_before": self.token_to_kv_pool_allocator.available_size(),
        "required_next_decode": self.new_tokens_required_next_decode(),
        "debug_retract": envs.SGLANG_TEST_RETRACT.get(),
        "reference_boundary": max(
            (int(path.stem) for path in reference_root(root).glob("*.pt")),
            default=-1,
        ),
    }
    before = {
        req.rid: {
            "rid": req.rid,
            "capture_id": (
                req.training_capture_context.lease.capture_id
                if req.training_capture_context is not None
                else None
            ),
            "committed_tokens": req.kv_committed_len,
            "slots": self.req_to_token_pool.req_to_token[
                req.req_pool_idx, : req.kv_committed_len
            ].tolist(),
        }
        for req in self.reqs
    }
    result = _retract_decode(self, server_args)
    retracted, _, aborted = result
    event["available_after"] = self.token_to_kv_pool_allocator.available_size()
    event["aborted"] = [req.rid for req in aborted]
    event["retracted"] = [
        before[req.rid]
        | {
            "context_detached": req.training_capture_context is None,
            "finalizer_detached": req.training_capture_finalize is None,
            "capture_attempted": req.training_capture_attempted,
            "is_retracted": req.is_retracted,
        }
        for req in retracted
    ]
    with (root / f"retractions-{rank_name()}.jsonl").open("a") as stream:
        stream.write(json.dumps(event) + "\n")
    return result


ScheduleBatch.retract_decode = retract_decode
Scheduler.get_internal_state = observed_internal_state
CaptureCoordinator._admit = observe_admission(CaptureCoordinator._admit)
CohortCaptureCoordinator._admit = observe_admission(CohortCaptureCoordinator._admit)
install_capture_observer()


if __name__ == "__main__":
    from sglang.launch_server import run_server
    from sglang.srt.server_args import prepare_server_args
    from sglang.srt.utils import kill_process_tree

    try:
        run_server(prepare_server_args(sys.argv[1:]))
    finally:
        kill_process_tree(os.getpid(), include_parent=False)
