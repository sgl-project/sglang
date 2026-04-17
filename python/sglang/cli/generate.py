import argparse
import json

from sglang.cli.utils import get_is_diffusion_model, get_model_path


def _get_generate_config_path(extra_argv):
    for i, arg in enumerate(extra_argv):
        if arg == "--config" and i + 1 < len(extra_argv):
            return extra_argv[i + 1]
        if arg.startswith("--config="):
            return arg.split("=", 1)[1]
    return None


def _get_model_path_for_generate(extra_argv):
    try:
        return get_model_path(extra_argv)
    except Exception as original_error:
        config_path = _get_generate_config_path(extra_argv)
        if config_path is None:
            raise

        config_args = _load_generate_config_file(config_path) or {}
        model_path = config_args.get("model_path")
        if not model_path:
            raise original_error
        return model_path


def _load_generate_config_file(config_path):
    if config_path.endswith(".json"):
        with open(config_path, "r") as f:
            return json.load(f)
    if config_path.endswith((".yaml", ".yml")):
        try:
            import yaml
        except ImportError as e:
            raise ImportError(
                "Please install PyYAML to use YAML config files. "
                "`pip install pyyaml`"
            ) from e
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    raise ValueError(f"Unsupported config file format: {config_path}")


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

    model_path = _get_model_path_for_generate(extra_argv)
    is_diffusion_model = get_is_diffusion_model(model_path)
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
