# SPDX-License-Identifier: Apache-2.0

import argparse
import logging
import os

from sglang.cli.utils import get_is_diffusion_model, get_model_path
from sglang.srt.utils import kill_process_tree

logger = logging.getLogger(__name__)


def serve(args, extra_argv):
    if any(h in extra_argv for h in ("-h", "--help")):
        # Since the server type is determined by the model, and we don't have a model path,
        # we can't show the exact help. Instead, we show a general help message and then
        # the help for both possible server types.
        print(
            "Usage: sglang serve --model-path <model-name-or-path> [additional-arguments]\n"
        )
        print(
            "This command can launch either a standard language model server or a diffusion model server."
        )
        print("The server type is determined by the model path.\n")
        print("For specific arguments, please provide a model_path.")
        print("\n--- Help for Standard Language Model Server ---")
        from sglang.srt.server_args import prepare_server_args

        try:
            prepare_server_args(["--help"])
        except SystemExit:
            pass  # argparse --help calls sys.exit

        print("\n--- Help for Diffusion Model Server ---")
        from sglang.multimodal_gen.runtime.entrypoints.cli.serve import (
            add_multimodal_gen_serve_args,
        )

        parser = argparse.ArgumentParser(description="SGLang Diffusion Model Serving")
        add_multimodal_gen_serve_args(parser)
        parser.print_help()
        return

    model_path = get_model_path(extra_argv)
    try:
        is_diffusion_model = get_is_diffusion_model(model_path)
        if is_diffusion_model:
            logger.info("Diffusion model detected")

        if is_diffusion_model:
            # Logic for Diffusion Models
            from sglang.multimodal_gen.runtime.entrypoints.cli.serve import (
                add_multimodal_gen_serve_args,
                execute_serve_cmd,
            )

            parser = argparse.ArgumentParser(
                description="SGLang Diffusion Model Serving"
            )
            add_multimodal_gen_serve_args(parser)
            parsed_args, remaining_argv = parser.parse_known_args(extra_argv)

            execute_serve_cmd(parsed_args, remaining_argv)
        else:
            # Logic for Standard Language Models
            from sglang.launch_server import run_server
            from sglang.srt.server_args import prepare_server_args

            # Add a dummy argument for the program name, expected by prepare_server_args
            # as it typically processes sys.argv
            server_args = prepare_server_args(extra_argv)

            run_server(server_args)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)
