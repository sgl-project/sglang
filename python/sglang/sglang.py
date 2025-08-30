"""SGLang CLI entry point for the 'sglang' console command. Currently launches the inference server."""

import os
import sys

from sglang.srt.entrypoints.http_server import launch_server
from sglang.srt.server_args import prepare_server_args
from sglang.srt.utils import kill_process_tree


def main(argv=None):
    """Entrypoint for the 'sglang' command."""
    if argv is None:
        argv = sys.argv[1:]
    server_args = prepare_server_args(argv)

    try:
        launch_server(server_args)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)


if __name__ == "__main__":
    main()
