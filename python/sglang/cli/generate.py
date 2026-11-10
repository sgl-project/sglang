import argparse

from sglang.cli.main import COOKBOOK_EPILOG
from sglang.cli.utils import (
    get_is_diffusion_model,
    get_model_path,
    get_pipeline_class_name,
    is_registered_native_pipeline,
)


def generate(args, extra_argv):
    # If help is requested, show generate subcommand help without requiring --model-path
    if any(h in extra_argv for h in ("-h", "--help")):
        from sglang.multimodal_gen.runtime.entrypoints.cli.generate import (
            add_multimodal_gen_generate_args,
        )

        parser = argparse.ArgumentParser(
            description="SGLang Multimodal Generation",
            epilog=COOKBOOK_EPILOG,
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        add_multimodal_gen_generate_args(parser)
        parser.parse_args(extra_argv)
        return

    model_path = get_model_path(extra_argv)
    # An explicit --pipeline-class-name naming a registered native diffusion
    # pipeline (e.g. FLUX.3's Flux3ImagePipeline) is treated as a diffusion signal even
    # when model_path itself is not auto-detectable (bare .safetensors, no
    # model_index.json). All downstream resolution already keys off this flag.
    is_diffusion_model = get_is_diffusion_model(
        model_path
    ) or is_registered_native_pipeline(get_pipeline_class_name(extra_argv))
    if is_diffusion_model:
        from sglang.multimodal_gen.runtime.entrypoints.cli.generate import (
            add_multimodal_gen_generate_args,
            generate_cmd,
        )

        parser = argparse.ArgumentParser(description="SGLang Multimodal Generation")
        add_multimodal_gen_generate_args(parser)
        parsed_args, unknown_args = parser.parse_known_args(extra_argv)
        generate_cmd(parsed_args, unknown_args)
    else:
        raise Exception(
            f"Generate subcommand is not yet supported for model: {model_path}"
        )
