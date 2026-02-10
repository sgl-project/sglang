"""Launch the inference server."""

import asyncio
import os
import sys
import warnings

# Suppress FutureWarning from deprecated cuda.cudart and cuda.nvrtc modules
warnings.filterwarnings(
    "ignore", message="The cuda.cudart module is deprecated", category=FutureWarning
)
warnings.filterwarnings(
    "ignore", message="The cuda.nvrtc module is deprecated", category=FutureWarning
)

from sglang.srt.server_args import prepare_server_args
from sglang.srt.utils import kill_process_tree


def run_server(server_args):
    """Run the server based on server_args.grpc_mode and server_args.encoder_only."""
    if server_args.grpc_mode:
        from sglang.srt.entrypoints.grpc_server import serve_grpc

        asyncio.run(serve_grpc(server_args))
    elif server_args.encoder_only:
        from sglang.srt.disaggregation.encode_server import launch_server

        launch_server(server_args)
    else:
        # Default mode: HTTP mode.
        from sglang.srt.entrypoints.http_server import launch_server

        launch_server(server_args)


if __name__ == "__main__":
    server_args = prepare_server_args(sys.argv[1:])

    try:
        run_server(server_args)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)
