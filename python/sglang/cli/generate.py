import argparse

from sglang.cli.utils import get_is_diffusion_model, get_model_path
from sglang.multimodal_gen.runtime.entrypoints.cli.generate import (
    add_multimodal_gen_generate_args,
    generate_cmd,
)


def generate(args, extra_argv):
    # If help is requested, show generate subcommand help without requiring --model-path
    if any(h in extra_argv for h in ("-h", "--help")):
        parser = argparse.ArgumentParser(description="SGLang Multimodal Generation")
        add_multimodal_gen_generate_args(parser)
        parser.parse_args(extra_argv)
        return

    model_path = get_model_path(extra_argv)
    is_diffusion_model = get_is_diffusion_model(model_path)
    if is_diffusion_model:
        parser = argparse.ArgumentParser(description="SGLang Multimodal Generation")
        add_multimodal_gen_generate_args(parser)
        parsed_args = parser.parse_args(extra_argv)
        generate_cmd(parsed_args)
    else:
        raise Exception(
            f"Generate subcommand is not yet supported for model: {model_path}"
        )
