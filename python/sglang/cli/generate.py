import argparse

from sglang.cli.main import get_is_diffusion_model, get_model_path
from sglang.multimodal_gen.runtime.entrypoints.cli.generate import (
    add_multimodal_gen_generate_args,
    generate_cmd,
)


def generate(args, extra_argv):
    model_path = get_model_path(extra_argv)
    is_diffusion_model = get_is_diffusion_model(model_path)
    if is_diffusion_model:
        parser = argparse.ArgumentParser(description="SGLang Multimodal Generation")
        add_multimodal_gen_generate_args(parser)
        parsed_args = parser.parse_args(extra_argv)
        generate_cmd(parsed_args)
    else:
        raise Exception(
            f"Generate subcommand is not supported for model: {model_path} for now"
        )
