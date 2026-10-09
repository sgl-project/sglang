"""Pause only manifest writes while observing source tensors and every rank."""

import json
import os
import sys
import time
from pathlib import Path

from sglang.srt.distributed import (
    get_pipeline_model_parallel_rank,
    get_tensor_model_parallel_rank,
)
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.training_capture.mooncake_store import MooncakeSnapshotStore
from sglang.test.dspark_capture_observer import install_capture_observer

_put = MooncakeSnapshotStore.put_registered
_internal_state = Scheduler.get_internal_state


def rank_name():
    return (
        f"pp{get_pipeline_model_parallel_rank()}-tp{get_tensor_model_parallel_rank()}"
    )


def stalled_manifest(self, key, tensor, expected_digest):
    root = Path(os.environ["TRAINING_CAPTURE_TEST_OUTPUT"])
    gate = root / "manifest.pause"
    if key.endswith("/manifest") and gate.exists():
        (root / "manifest.started").write_text(
            json.dumps({"rank": rank_name(), "key": key})
        )
        deadline = time.monotonic() + 90
        while gate.exists():
            if time.monotonic() >= deadline:
                raise TimeoutError("manifest pause was not released")
            time.sleep(0.01)
    return _put(self, key, tensor, expected_digest)


def observed_internal_state(self, req):
    result = _internal_state(self, req)
    root = Path(os.environ["TRAINING_CAPTURE_TEST_OUTPUT"]) / "rank-states"
    request = root / "request"
    if self.tp_worker.training_capture is not None and request.exists():
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


MooncakeSnapshotStore.put_registered = stalled_manifest
Scheduler.get_internal_state = observed_internal_state
install_capture_observer()


if __name__ == "__main__":
    from sglang.launch_server import run_server
    from sglang.srt.server_args import prepare_server_args
    from sglang.srt.utils import kill_process_tree

    try:
        run_server(prepare_server_args(sys.argv[1:]))
    finally:
        kill_process_tree(os.getpid(), include_parent=False)
