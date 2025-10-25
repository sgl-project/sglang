# SPDX-License-Identifier: Apache-2.0

import argparse
import asyncio
import logging
import os

from sglang.srt.utils import kill_process_tree

logger = logging.getLogger(__name__)


def get_is_diffusion_model(model_path: str):
    lowered_path = model_path.lower()
    return (
        "diffusion" in lowered_path
        or "wan" in lowered_path
        or "video" in lowered_path
        or "image" in lowered_path
        or "hunyuan" in lowered_path
        or "flux" in lowered_path
    )


def serve(args, extra_argv):
    # Find the model_path argument
    model_path = None
    if "--model-path" in extra_argv:
        try:
            model_path_index = extra_argv.index("--model-path") + 1
            if model_path_index < len(extra_argv):
                model_path = extra_argv[model_path_index]
        except (ValueError, IndexError):
            pass

    if model_path is None:
        # Fallback for --help or other cases where model-path is not provided
        if any(h in extra_argv for h in ["-h", "--help"]):
            print(
                "Usage: sglang serve --model-path <model-name-or-path> [additional-arguments]\n\n"
                "This command can launch either a standard language model server or a diffusion model server.\n"
                "The server type is determined by the model path.\n"
                "For specific arguments, please provide a model_path."
            )
            return
        else:
            print(
                "Error: --model-path is required. "
                "Please provide the path to the model."
            )
            return

    try:
        is_diffusion_model = get_is_diffusion_model(model_path)
        if is_diffusion_model:
            logger.info("Diffusion model detected")

        if is_diffusion_model:
            # Logic for Diffusion Models
            from sglang.multimodal_gen.runtime.entrypoints.cli.serve import (
                add_image_serve_args,
            )
            from sglang.multimodal_gen.runtime.entrypoints.cli.serve import (
                serve as image_serve,
            )

            parser = argparse.ArgumentParser(
                description="SGLang Diffusion Model Serving"
            )
            add_image_serve_args(parser)
            parsed_args = parser.parse_args(extra_argv)
            image_serve(parsed_args)
        else:
            # Logic for Standard Language Models
            from sglang.srt.entrypoints.grpc_server import serve_grpc
            from sglang.srt.entrypoints.http_server import launch_server
            from sglang.srt.server_args import prepare_server_args

            # Add a dummy argument for the program name, expected by prepare_server_args
            # as it typically processes sys.argv
            server_args = prepare_server_args(extra_argv)

            if server_args.grpc_mode:
                asyncio.run(serve_grpc(server_args))
            else:
                launch_server(server_args)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)
