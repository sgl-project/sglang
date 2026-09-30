"""Overlap capture test entrypoint; observers also install in spawned workers."""

import os
import sys

from sglang.test.dspark_capture_observer import install_capture_observer

install_capture_observer()


if __name__ == "__main__":
    from sglang.launch_server import run_server
    from sglang.srt.server_args import prepare_server_args
    from sglang.srt.utils import kill_process_tree

    try:
        run_server(prepare_server_args(sys.argv[1:]))
    finally:
        kill_process_tree(os.getpid(), include_parent=False)
