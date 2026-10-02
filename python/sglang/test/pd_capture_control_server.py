"""Test-only chunk gate for real HTTP control while a P/D capture is in flight."""

import json
import os
import sys
import time
from pathlib import Path

from sglang.srt.managers.schedule_batch import NextBatchPlan
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.training_capture.pd_capture import PrefillCaptureCoordinator
from sglang.test import pd_capture_server  # noqa: F401 - install source observers

_prefill = Scheduler.get_new_batch_prefill
_handoff = PrefillCaptureCoordinator.finish_handoff
_entered = {}


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
        if hold.exists():
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
                temporary = root / (req.rid + ".tmp")
                temporary.write_text(json.dumps(payload))
                temporary.replace(root / (req.rid + ".entered"))
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
        root.mkdir(exist_ok=True)
        (root / (req.rid + ".handoff")).write_text(
            json.dumps(
                {
                    "capture_epoch": epoch,
                    "current_epoch": self.abort_epoch,
                    "teacher_present": teacher_present,
                    "payload_present": payload is not None,
                }
            )
        )
    return payload


Scheduler.get_new_batch_prefill = gated_prefill
PrefillCaptureCoordinator.finish_handoff = observed_handoff


if __name__ == "__main__":
    from sglang.launch_server import run_server
    from sglang.srt.server_args import prepare_server_args
    from sglang.srt.utils import kill_process_tree

    try:
        args = prepare_server_args(sys.argv[1:])
        if args.tp_size != 1 or args.pp_size != 1:
            raise ValueError("the chunk-gate test requires TP1/PP1")
        run_server(args)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)
