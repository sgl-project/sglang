"""Observe P/D pressure admission and rank-local state without changing allocation."""

import json
import os
import sys
from pathlib import Path

from sglang.srt.distributed import (
    get_pipeline_model_parallel_rank,
    get_tensor_model_parallel_rank,
)
from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.training_capture.pd_capture import DecodeCaptureMixin
from sglang.test import (
    pd_capture_server,  # noqa: F401 - install source/restore observers
)

_internal_state = Scheduler.get_internal_state
_begin = DecodeCaptureMixin.begin_pd_transfer
_retract = ScheduleBatch.retract_decode


def rank_name():
    return (
        f"pp{get_pipeline_model_parallel_rank()}-tp{get_tensor_model_parallel_rank()}"
    )


def append(root, name, value):
    with (root / f"{name}-{rank_name()}.jsonl").open("a") as stream:
        stream.write(json.dumps(value) + "\n")


def observed_begin(self, req):
    result = _begin(self, req)
    record = req.training_capture_context
    if record is not None:
        append(
            Path(self.config.journal_directory).parent,
            "pressure-admissions",
            {
                "rid": req.rid,
                "capture_id": record.lease.capture_id,
                "active_capture_rids": sorted(r.rid for r in self.requests.values()),
            },
        )
    return result


def observed_internal_state(self, req):
    result = _internal_state(self, req)
    capture = self.tp_worker.training_capture
    if capture is not None:
        root = Path(capture.config.journal_directory).parent / "pressure-states"
        request = root / "request"
        if request.exists():
            path = root / (rank_name() + ".json")
            temporary = path.with_suffix(".tmp")
            temporary.write_text(
                json.dumps(
                    {
                        "nonce": request.read_text(),
                        "state": result.internal_state["training_capture"],
                    }
                )
            )
            temporary.replace(path)
    return result


def observed_retract(self, server_args):
    before = self.token_to_kv_pool_allocator.available_size()
    required = self.new_tokens_required_next_decode()
    result = _retract(self, server_args)
    retracted, _, aborted = result
    root = Path(server_args.training_capture_config).parent
    append(
        root,
        "pressure-retractions",
        {
            "available_before": before,
            "required_next_decode": required,
            "available_after": self.token_to_kv_pool_allocator.available_size(),
            "debug_retract": envs.SGLANG_TEST_RETRACT.get(),
            "retracted": [req.rid for req in retracted],
            "aborted": [req.rid for req in aborted],
        },
    )
    return result


DecodeCaptureMixin.begin_pd_transfer = observed_begin
Scheduler.get_internal_state = observed_internal_state
ScheduleBatch.retract_decode = observed_retract


if __name__ == "__main__":
    from sglang.launch_server import run_server
    from sglang.srt.server_args import prepare_server_args
    from sglang.srt.utils import kill_process_tree

    try:
        run_server(prepare_server_args(sys.argv[1:]))
    finally:
        kill_process_tree(os.getpid(), include_parent=False)
