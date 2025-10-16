"""Launch the inference server."""

import asyncio
import os
import sys

from sglang.srt.server_args import prepare_server_args
from sglang.srt.utils import kill_process_tree

if __name__ == "__main__":
    server_args = prepare_server_args(sys.argv[1:])

    try:
        if server_args.grpc_mode:
            from sglang.srt.entrypoints.grpc_server import serve_grpc

            asyncio.run(serve_grpc(server_args))
        else:
            from sglang.srt.entrypoints.http_server import launch_server

            launch_server(server_args)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)
