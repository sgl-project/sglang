# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
import os

import typer

logger = logging.getLogger(__name__)


def is_diffusion_model(model_path: str) -> bool:
    """
    Check if the model path is for a diffusion model.

    NOTE: This is a simple heuristic and might need to be improved.
    """
    model_path = model_path.lower()
    return "diffusion" in model_path or "sd" in model_path or "stable-diffusion" in model_path


def serve_command(model_path: str, ctx: typer.Context):
    """
    Serve a model.
    """
    # Reconstruct the argument list
    args = ["--model-path", model_path] + ctx.args

    if is_diffusion_model(model_path):
        from sglang.multimodal_gen.runtime.launch_server import (
            launch_server as launch_diffusion_server,
        )
        from sglang.multimodal_gen.runtime.server_args import (
            prepare_server_args as prepare_diffusion_server_args,
        )

        server_args = prepare_diffusion_server_args(args)
        launch_diffusion_server(server_args)
    else:
        from sglang.srt.entrypoints.grpc_server import serve_grpc
        from sglang.srt.entrypoints.http_server import launch_server as launch_http_server
        from sglang.srt.server_args import prepare_server_args
        from sglang.srt.utils import kill_process_tree

        server_args = prepare_server_args(args)

        try:
            if server_args.grpc_mode:
                asyncio.run(serve_grpc(server_args))
            else:
                launch_http_server(server_args)
        finally:
            kill_process_tree(os.getpid(), include_parent=False)
