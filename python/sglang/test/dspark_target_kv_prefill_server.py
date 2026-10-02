"""Combine real prefill graph and independent target-KV projection observers."""

import os
import sys

from sglang.test import (  # noqa: F401
    dspark_target_kv_server,
    training_capture_prefill_server,
)

if __name__ == "__main__":
    from sglang.launch_server import run_server
    from sglang.srt.server_args import prepare_server_args
    from sglang.srt.utils import kill_process_tree

    try:
        run_server(prepare_server_args(sys.argv[1:]))
    finally:
        kill_process_tree(os.getpid(), include_parent=False)
