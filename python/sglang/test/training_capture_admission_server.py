"""Test-only file-controlled writer stall, inherited by spawned server workers."""

import os
import sys
import time

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


if __name__ == "__main__":
    from sglang.launch_server import run_server
    from sglang.srt.server_args import prepare_server_args
    from sglang.srt.utils import kill_process_tree

    try:
        run_server(prepare_server_args(sys.argv[1:]))
    finally:
        kill_process_tree(os.getpid(), include_parent=False)
