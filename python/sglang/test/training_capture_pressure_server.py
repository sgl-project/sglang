"""Observe real allocator-driven retraction without forcing its scheduling."""

import json
import os
import sys
from pathlib import Path

from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.test.dspark_capture_observer import install_capture_observer

_retract_decode = ScheduleBatch.retract_decode


def retract_decode(self, server_args):
    root = Path(os.environ["TRAINING_CAPTURE_TEST_OUTPUT"])
    event = {
        "available_before": self.token_to_kv_pool_allocator.available_size(),
        "required_next_decode": self.new_tokens_required_next_decode(),
        "debug_retract": envs.SGLANG_TEST_RETRACT.get(),
        "reference_boundary": max(
            (int(path.stem) for path in (root / "capture-reference").glob("*.pt")),
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
    with (root / "retractions.jsonl").open("a") as stream:
        stream.write(json.dumps(event) + "\n")
    return result


ScheduleBatch.retract_decode = retract_decode
install_capture_observer()


if __name__ == "__main__":
    from sglang.launch_server import run_server
    from sglang.srt.server_args import prepare_server_args
    from sglang.srt.utils import kill_process_tree

    try:
        run_server(prepare_server_args(sys.argv[1:]))
    finally:
        kill_process_tree(os.getpid(), include_parent=False)
