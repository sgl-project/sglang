# SPDX-License-Identifier: Apache-2.0
# Adapted from https://github.com/vllm-project/vllm/blob/v0.7.3/vllm/utils.py

import argparse
import sys
from typing import Any

import yaml

from sglang.multimodal_gen.runtime.utils.logging_utils import (
    SortedHelpFormatter,
    init_logger,
)

logger = init_logger(__name__)


class StoreBoolean(argparse.Action):
    def __init__(self, option_strings, dest, default=False, required=False, help=None):
        super().__init__(
            option_strings=option_strings,
            dest=dest,
            nargs="?",
            const=True,
            default=default,
            required=required,
            help=help,
        )

    def __call__(self, parser, namespace, values, option_string=None):
        if values is None:
            setattr(namespace, self.dest, True)
        elif isinstance(values, str):
            if values.lower() == "true":
                setattr(namespace, self.dest, True)
            elif values.lower() == "false":
                setattr(namespace, self.dest, False)
            else:
                raise ValueError(
                    f"Invalid boolean value: {values}. Expected 'true' or 'false'."
                )
        else:
            setattr(namespace, self.dest, bool(values))


class FlexibleArgumentParser(argparse.ArgumentParser):
    """ArgumentParser that allows both underscore and dash in names."""

    def __init__(self, *args, **kwargs) -> None:
        # Set the default 'formatter_class' to SortedHelpFormatter
        if "formatter_class" not in kwargs:
            kwargs["formatter_class"] = SortedHelpFormatter
        super().__init__(*args, **kwargs)

    def parse_args(  # type: ignore[override]
        self, args=None, namespace=None
    ) -> argparse.Namespace:
        if args is None:
            args = sys.argv[1:]

        if any(arg.startswith("--config") for arg in args):
            args = self._pull_args_from_config(args)

        # Convert underscores to dashes and vice versa in argument names
        processed_args = []
        for arg in args:
            if arg.startswith("--"):
                if "=" in arg:
                    key, value = arg.split("=", 1)
                    key = "--" + key[len("--") :].replace("_", "-")
                    processed_args.append(f"{key}={value}")
                else:
                    processed_args.append("--" + arg[len("--") :].replace("_", "-"))
            elif arg.startswith("-O") and arg != "-O" and len(arg) == 2:
                # allow -O flag to be used without space, e.g. -O3
                processed_args.append("-O")
                processed_args.append(arg[2:])
            else:
                processed_args.append(arg)

        namespace = super().parse_args(processed_args, namespace)

        # Track which arguments were explicitly provided
        namespace._provided = set()

        i = 0
        while i < len(args):
            arg = args[i]
            if arg.startswith("--"):
                # Handle --key=value format
                if "=" in arg:
                    key = arg.split("=")[0][2:].replace("-", "_")
                    namespace._provided.add(key)
                    i += 1
                # Handle --key value format
                else:
                    key = arg[2:].replace("-", "_")
                    namespace._provided.add(key)
                    # Skip the value if there is one
                    if i + 1 < len(args) and not args[i + 1].startswith("-"):
                        i += 2
                    else:
                        i += 1
            else:
                i += 1

        return namespace  # type: ignore[no-any-return]

    def _pull_args_from_config(self, args: list[str]) -> list[str]:
        """Method to pull arguments specified in the config file
        into the command-line args variable.

        The arguments in config file will be inserted between
        the argument list.

        example:
        ```yaml
            port: 12323
            tensor-parallel-size: 4
        ```
        ```python
        $: vllm {serve,chat,complete} "facebook/opt-12B" \
            --config config.yaml -tp 2
        $: args = [
            "serve,chat,complete",
            "facebook/opt-12B",
            '--config', 'config.yaml',
            '-tp', '2'
        ]
        $: args = [
            "serve,chat,complete",
            "facebook/opt-12B",
            '--port', '12323',
            '--tp-size', '4',
            '-tp', '2'
            ]
        ```

        Please note how the config args are inserted after the sub command.
        this way the order of priorities is maintained when these are args
        parsed by super().
        """
        index = -1
        config_arg = None
        for i, arg in enumerate(args):
            if arg.startswith("--config"):
                if index != -1:
                    raise ValueError("More than one config file specified!")
                index = i
                config_arg = arg

        if config_arg is None:
            return args
        args_before_config = args[:index]
        if "=" in config_arg:
            file_path = config_arg.split("=", 1)[1]
            args_after_config = args[index + 1 :]
        else:
            if index == len(args) - 1:
                raise ValueError(
                    "No config file specified! "
                    "Please check your command-line arguments."
                )
            file_path = args[index + 1]
            args_after_config = args[index + 2 :]

        config_args = self._load_config_file(file_path)

        # 0th index is for {serve,chat,complete}
        # followed by model_tag (only for serve)
        # followed by config args
        # followed by rest of cli args.
        # maintaining this order will enforce the precedence
        # of cli > config > defaults
        if args[0] == "serve":
            if index == 1:
                raise ValueError(
                    "No model_tag specified! Please check your command-line arguments."
                )
            command = args_before_config[0]
            model_tag = args_before_config[1]
            other_args_before = args_before_config[2:]
            args = (
                [command, model_tag]
                + config_args
                + other_args_before
                + args_after_config
            )
        else:
            command = args_before_config[0]
            other_args_before = args_before_config[1:]
            args = [command] + config_args + other_args_before + args_after_config

        return args

    def _load_config_file(self, file_path: str) -> list[str]:
        """Loads a yaml file and returns the key value pairs as a
        flattened list with argparse like pattern
        ```yaml
            port: 12323
            tensor-parallel-size: 4
            vae_config:
                load_encoder: false
                load_decoder: true
        ```
        returns:
            processed_args: list[str] = [
                '--port': '12323',
                '--tp-size': '4',
                '--vae-config.load-encoder': 'false',
                '--vae-config.load-decoder': 'true'
            ]
        """

        extension: str = file_path.split(".")[-1]
        if extension not in ("yaml", "yml", "json"):
            raise ValueError(
                "Config file must be of a yaml/yml/json type.\
                              %s supplied",
                extension,
            )

        processed_args: list[str] = []

        config: dict[str, Any] = {}
        try:
            with open(file_path) as config_file:
                config = yaml.safe_load(config_file)
        except Exception as ex:
            logger.error(
                "Unable to read the config file at %s. \
                Make sure path is correct",
                file_path,
            )
            raise ex

        store_boolean_arguments = [
            action.dest for action in self._actions if isinstance(action, StoreBoolean)
        ]

        def process_dict(prefix: str, d: dict[str, Any]):
            for key, value in d.items():
                full_key = f"{prefix}.{key}" if prefix else key

                if isinstance(value, bool) and full_key not in store_boolean_arguments:
                    if value:
                        processed_args.append("--" + full_key)
                    else:
                        processed_args.append("--" + full_key)
                        processed_args.append("false")
                elif isinstance(value, list):
                    processed_args.append("--" + full_key)
                    for item in value:
                        processed_args.append(str(item))
                elif isinstance(value, dict):
                    process_dict(full_key, value)
                else:
                    processed_args.append("--" + full_key)
                    processed_args.append(str(value))

        process_dict("", config)

        return processed_args
