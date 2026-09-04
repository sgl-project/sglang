"""Launch SGLang with RACER registered as an out-of-tree spec algorithm."""

import os
import sys

from sglang.srt.plugins import load_plugins
from sglang.srt.server_args import prepare_server_args
from sglang.srt.utils import kill_process_tree

from sglang.srt.speculative.racer import registration as _registration  # noqa: F401


def main() -> None:
    load_plugins()
    server_args = prepare_server_args(sys.argv[1:])

    from sglang.launch_server import run_server

    try:
        run_server(server_args)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)


if __name__ == "__main__":
    main()
