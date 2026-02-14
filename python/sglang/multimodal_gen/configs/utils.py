# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

import argparse
from typing import Any


def update_config_from_args(
    config: Any, args_dict: dict[str, Any], prefix: str = "", pop_args: bool = False
) -> bool:
    """
    Update configuration object from arguments dictionary.

    Args:
        config: The configuration object to update
        args_dict: Dictionary containing arguments
        prefix: Prefix for the configuration parameters in the args_dict.
               If None, assumes direct attribute mapping without prefix.
    """
    # Handle top-level attributes (no prefix)
    args_not_to_remove = [
        "model_path",
    ]
    args_to_remove = []
    if prefix.strip() == "":
        for key, value in args_dict.items():
            if hasattr(config, key) and value is not None:
                if key == "text_encoder_precisions" and isinstance(value, list):
                    setattr(config, key, tuple(value))
                else:
                    setattr(config, key, value)
                if pop_args:
                    args_to_remove.append(key)
    else:
        # Handle nested attributes with prefix
        prefix_with_dot = f"{prefix}."
        for key, value in args_dict.items():
            if key.startswith(prefix_with_dot) and value is not None:
                attr_name = key[len(prefix_with_dot) :]
                if hasattr(config, attr_name):
                    setattr(config, attr_name, value)
                if pop_args:
                    args_to_remove.append(key)

    if pop_args:
        for key in args_to_remove:
            if key not in args_not_to_remove:
                args_dict.pop(key)

    return len(args_to_remove) > 0


def clean_cli_args(args: argparse.Namespace) -> dict[str, Any]:
    """
    Clean the arguments by removing the ones that not explicitly provided by the user.
    """
    provided_args = {}
    for k, v in vars(args).items():
        if v is not None and hasattr(args, "_provided") and k in args._provided:
            provided_args[k] = v

    return provided_args
