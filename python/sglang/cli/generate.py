import argparse

from sglang.cli.utils import get_is_diffusion_model, get_model_path


def _get_pipeline_class_name(extra_argv):
    for i, arg in enumerate(extra_argv):
        if arg == "--pipeline-class-name" and i + 1 < len(extra_argv):
            return extra_argv[i + 1]
        if arg.startswith("--pipeline-class-name="):
            return arg.split("=", 1)[1]
    return None


def _has_registered_pipeline_class(extra_argv):
    pipeline_class_name = _get_pipeline_class_name(extra_argv)
    if not pipeline_class_name:
        return False
    try:
        from sglang.multimodal_gen.registry import (
            _PIPELINE_REGISTRY,
            _discover_and_register_pipelines,
        )
    except ImportError:
        return False
    _discover_and_register_pipelines()
    return pipeline_class_name in _PIPELINE_REGISTRY


def generate(args, extra_argv):
    # If help is requested, show generate subcommand help without requiring --model-path
    if any(h in extra_argv for h in ("-h", "--help")):
        from sglang.multimodal_gen.runtime.entrypoints.cli.generate import (
            add_multimodal_gen_generate_args,
        )

        parser = argparse.ArgumentParser(description="SGLang Multimodal Generation")
        add_multimodal_gen_generate_args(parser)
        parser.parse_args(extra_argv)
        return

    model_path = get_model_path(extra_argv)
    is_diffusion_model = get_is_diffusion_model(
        model_path
    ) or _has_registered_pipeline_class(extra_argv)
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
