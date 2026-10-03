"""Hold request arrival at a decode boundary, then observe native mixed batches."""

import json
import os
import sys
import time
from pathlib import Path

from sglang.srt.managers.io_struct import BatchTokenizedGenerateReqInput
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler_components.request_receiver import (
    SchedulerRequestReceiver,
)
from sglang.test import training_capture_prefill_server  # noqa: F401

_pull = SchedulerRequestReceiver._pull_raw_reqs
_mix = ScheduleBatch.mix_with_running
_gated = set()


def pull(self):
    batch = self.get_last_batch()
    decode = next(
        (
            req
            for req in (batch.reqs if batch is not None else [])
            if req.rid.endswith("-decode") and len(req.output_ids) >= 2
        ),
        None,
    )
    if (
        self.ps.pp_rank
        or self.ps.attn_tp_rank
        or decode is None
        or decode.rid in _gated
    ):
        return _pull(self)
    _gated.add(decode.rid)
    root = Path(os.environ["TRAINING_CAPTURE_TEST_OUTPUT"])
    (root / "decode-ready").touch()
    deadline = time.monotonic() + 30
    received = []
    expected = {"partial", "one", "excluded"}
    while expected:
        fresh = _pull(self)
        received.extend(fresh)
        for message in fresh:
            for req in (
                message.batch
                if isinstance(message, BatchTokenizedGenerateReqInput)
                else [message]
            ):
                expected.discard(getattr(req, "rid", "").rsplit("-", 1)[-1])
        if time.monotonic() >= deadline:
            raise TimeoutError(f"mixed request arrival gate: {expected}")
        if expected:
            time.sleep(0.005)
    return received


def mix(self, running):
    result = _mix(self, running)
    root = Path(os.environ["TRAINING_CAPTURE_TEST_OUTPUT"])
    with (root / "mixed-batches.jsonl").open("a") as stream:
        stream.write(
            json.dumps(
                {
                    "rids": [r.rid for r in self.reqs],
                    "decode_rids": [r.rid for r in running.reqs],
                    "extend_lens": self.extend_lens,
                    "prefix_lens": self.prefix_lens,
                }
            )
            + "\n"
        )
    return result


SchedulerRequestReceiver._pull_raw_reqs = pull
ScheduleBatch.mix_with_running = mix


if __name__ == "__main__":
    from sglang.launch_server import run_server
    from sglang.srt.server_args import prepare_server_args
    from sglang.srt.utils import kill_process_tree

    try:
        run_server(prepare_server_args(sys.argv[1:]))
    finally:
        kill_process_tree(os.getpid(), include_parent=False)
