"""Test-only file-controlled writer stall, inherited by spawned server workers."""

import os
import sys
import time

from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.training_capture.snapshot_writer import SnapshotWriter

_write = SnapshotWriter.write


def controlled_write(self, *args, **kwargs):
    pause = self.journal.root / "writer.pause"
    deadline = time.monotonic() + 60
    if pause.exists():
        (self.journal.root / "writer.started").touch()
    while pause.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError("test writer pause was not released")
        time.sleep(0.02)
    return _write(self, *args, **kwargs)


SnapshotWriter.write = controlled_write

_process_batch_result = Scheduler.process_batch_result


def delayed_result(self, batch, result):
    capture = self.tp_worker.training_capture
    if capture is not None and (capture.journal.root / "latency.pause").exists():
        time.sleep(0.75)
    return _process_batch_result(self, batch, result)


Scheduler.process_batch_result = delayed_result


if __name__ == "__main__":
    from sglang.launch_server import run_server
    from sglang.srt.server_args import prepare_server_args
    from sglang.srt.utils import kill_process_tree

    try:
        run_server(prepare_server_args(sys.argv[1:]))
    finally:
        kill_process_tree(os.getpid(), include_parent=False)
