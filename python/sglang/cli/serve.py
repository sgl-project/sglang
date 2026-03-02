# SPDX-License-Identifier: Apache-2.0

import argparse
import logging
import os

from sglang.cli.utils import get_is_diffusion_model, get_model_path
from sglang.srt.utils import kill_process_tree

logger = logging.getLogger(__name__)


def _extract_model_type_override(extra_argv):
    """Extract and remove --model-type override from argv."""
    model_type = "auto"
    filtered_argv = []
    i = 0
    while i < len(extra_argv):
        arg = extra_argv[i]
        if arg == "--model-type":
            if i + 1 >= len(extra_argv):
                raise Exception(
                    "Error: --model-type requires a value. "
                    "Valid values are: auto, llm, diffusion."
                )
            model_type = extra_argv[i + 1]
            i += 2
            continue

        if arg.startswith("--model-type="):
            model_type = arg.split("=", 1)[1]
            i += 1
            continue

        filtered_argv.append(arg)
        i += 1

    if model_type not in ("auto", "llm", "diffusion"):
        raise Exception(
            f"Error: invalid --model-type '{model_type}'. "
            "Valid values are: auto, llm, diffusion."
        )
    return model_type, filtered_argv


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
        print(
            "Optional override: --model-type {auto,llm,diffusion} "
            "(default: auto, fallback to LLM on detection failure).\n"
        )
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

    model_type, dispatch_argv = _extract_model_type_override(extra_argv)
    model_path = get_model_path(dispatch_argv)
    try:
        if model_type == "auto":
            is_diffusion_model = get_is_diffusion_model(model_path)
            if is_diffusion_model:
                logger.info("Diffusion model detected")
        else:
            is_diffusion_model = model_type == "diffusion"
            logger.info(
                "Dispatch override enabled: --model-type=%s " "(skip auto detection)",
                model_type,
            )

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
            parsed_args, remaining_argv = parser.parse_known_args(dispatch_argv)

            execute_serve_cmd(parsed_args, remaining_argv)
        else:
            # Logic for Standard Language Models
            from sglang.launch_server import run_server
            from sglang.srt.server_args import prepare_server_args

            # Add a dummy argument for the program name, expected by prepare_server_args
            # as it typically processes sys.argv
            server_args = prepare_server_args(dispatch_argv)

            run_server(server_args)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)
