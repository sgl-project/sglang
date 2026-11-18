"""Launch the inference server."""

import asyncio
import os
import sys
import warnings

from sglang.srt.arg_groups.model_override_base import resolving_view
from sglang.srt.plugins import load_plugins
from sglang.srt.server_args import prepare_server_args
from sglang.srt.utils import kill_process_tree
from sglang.srt.utils.log_utils import suppress_noisy_warnings

suppress_noisy_warnings()


def run_server(server_args):
    """Start the server server_args selects: encoder (HTTP or gRPC), Ray, or HTTP."""
    # `prepare_server_args` returns raw input. Resolve it before any server
    # starts, so that its checks, among them that `--grpc-mode` requires
    # `--encoder-only`, run first.
    server_args.resolve_once()
    cfg = resolving_view(server_args)

    if cfg.encoder_only:
        # For encoder disaggregation
        if cfg.grpc_mode:
            from sglang.srt.disaggregation.encoder.grpc_server import (
                serve_grpc_encoder,
            )

            asyncio.run(serve_grpc_encoder(server_args))
        else:
            from sglang.srt.disaggregation.encoder.http_server import launch_server

            launch_server(server_args)
    elif cfg.use_ray:
        # Ray mode: HTTP mode with Ray backend.
        try:
            from sglang.srt.ray.http_server import launch_server
        except ImportError:
            raise ImportError(
                "Ray is required for --use-ray mode. "
                "Install it with: pip install 'sglang[ray]'"
            )

        launch_server(server_args)
    else:
        # Default mode: HTTP mode.
        from sglang.srt.entrypoints.http_server import launch_server

        launch_server(server_args)


if __name__ == "__main__":
    warnings.warn(
        "'python -m sglang.launch_server' is still supported, but "
        "'sglang serve' is the recommended entrypoint.\n"
        "  Example: sglang serve --model-path <model> [options]",
        UserWarning,
        stacklevel=1,
    )

    load_plugins()

    server_args = prepare_server_args(sys.argv[1:])

    try:
        run_server(server_args)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)
