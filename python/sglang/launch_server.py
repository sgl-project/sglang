"""Launch the inference server."""

import sys

from sglang.srt.server import launch_server
from sglang.srt.server_args import prepare_server_args
from sglang.srt.utils import kill_child_process

if __name__ == "__main__":
    server_args = prepare_server_args(sys.argv[1:])

    try:
        launch_server(server_args)
    finally:
        kill_child_process()
