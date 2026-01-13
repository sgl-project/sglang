"""
Configuration argument parser for command-line applications.
Handles merging of YAML configuration files with command-line arguments.
"""

import argparse
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List

import yaml

logger = logging.getLogger(__name__)


class ConfigArgumentMerger:
    """Handles merging of configuration file arguments with command-line arguments."""

    def __init__(self, parser: argparse.ArgumentParser):
        """Initialize with list of store_true action names."""
        # NOTE: The current code does not support actions other than "store_true" and "store".
        self.parser = parser
        self.store_true_actions = [
            action.dest
            for action in parser._actions
            if isinstance(action, argparse._StoreTrueAction)
        ]
        self.unsupported_actions = {
            a.dest: a
            for a in parser._actions
            if a.option_strings
            and not isinstance(a, argparse._StoreTrueAction)
            and not isinstance(a, argparse._StoreAction)
            and "--config" not in a.option_strings
            and "--help" not in a.option_strings
            and "-h" not in a.option_strings
        }

    def parse_config(self, cli_args: List[str]) -> Dict[str, Any]:
        config_file_path = self._extract_config_file_path(cli_args)
        if not config_file_path:
            return {}

        config_data = self._parse_yaml_config(config_file_path)
        dest_map = {action.dest: action for action in self.parser._actions}

        parsed_config = {}
        for key, value in config_data.items():
            key_norm = key.replace("-", "_")

            if key_norm in self.unsupported_actions:
                action = self.unsupported_actions[key_norm]
                msg = f"Unsupported config option '{key_norm}' with action '{action.__class__.__name__}'"
                raise ValueError(msg)

            if key_norm not in dest_map:
                logger.warning(
                    f"Unknown config option '{key}' will be ignored. "
                    f"Check if the option name is correct."
                )
                continue

            action = dest_map[key_norm]

            if action.choices and value not in action.choices:
                raise ValueError(
                    f"Invalid value for '{key}': {value}. Allowed choices: {action.choices}"
                )
            # Validate store_true actions accept only boolean values
            if key_norm in self.store_true_actions and not isinstance(value, bool):
                raise ValueError(
                    f"Invalid value for '{key}': {value}. Expected boolean (true/false), got {type(value).__name__}"
                )
            # Apply type conversion if action.type is defined
            elif action.type is not None and callable(action.type):
                try:
                    # Handle nargs: apply type conversion to each list element
                    if action.nargs is not None and isinstance(value, list):
                        value = [self._convert_value(v, action.type) for v in value]
                    else:
                        value = self._convert_value(value, action.type)
                except (ValueError, TypeError, argparse.ArgumentTypeError) as e:
                    raise ValueError(
                        f"Invalid value for '{key}': {value}. Type conversion failed: {e}"
                    ) from e

            parsed_config[action.dest] = value

        return parsed_config

    def remove_config_from_argv(self, argv: List[str]) -> List[str]:
        result = []
        skip_next = False
        for arg in argv:
            if skip_next:
                skip_next = False
                continue
            if arg == "--config":
                skip_next = True
                continue
            result.append(arg)
        return result

    def _extract_config_file_path(self, args: List[str]) -> str:
        """Extract the config file path from arguments."""
        config_indices = [i for i, arg in enumerate(args) if arg == "--config"]

        if len(config_indices) > 1:
            raise ValueError("Multiple config files specified! Only one allowed.")

        if not config_indices:
            return None

        config_index = config_indices[0]
        if config_index == len(args) - 1:
            raise ValueError("No config file specified after --config flag!")

        return args[config_index + 1]

    def _parse_yaml_config(self, file_path: str) -> Dict[str, Any]:
        """
        Parse YAML configuration file and convert to argument list.

        Args:
            file_path: Path to the YAML configuration file

        Returns:
            List of arguments in format ['--key', 'value', ...]

        Raises:
            ValueError: If file is not YAML or cannot be read
        """
        self._validate_yaml_file(file_path)

        try:
            with open(file_path, "r") as file:
                config_data = yaml.safe_load(file)
        except Exception as e:
            logger.error(f"Failed to read config file {file_path}: {e}")
            raise

        # Handle empty files or None content
        if config_data is None:
            config_data = {}

        if not isinstance(config_data, dict):
            raise ValueError("Config file must contain a dictionary at root level")

        return config_data

    def _validate_yaml_file(self, file_path: str) -> None:
        """Validate that the file is a YAML file."""
        path = Path(file_path)
        if path.suffix.lower() not in [".yaml", ".yml"]:
            raise ValueError(f"Config file must be YAML format, got: {path.suffix}")

        if not path.exists():
            raise ValueError(f"Config file not found: {file_path}")

    def _convert_value(self, value: Any, type_func: Callable) -> Any:
        """
        Convert a value using the specified type function.

        Handles special cases:
        - None: pass through without conversion
        - Already correct type: skip redundant conversion for basic types
        - JSON type functions: skip if value is already dict/list from YAML
        """
        # Skip conversion for None values (YAML null)
        if value is None:
            return None

        # Skip conversion if value already has the expected type (for basic types)
        if type_func in (int, float, str) and isinstance(value, type_func):
            return value

        # Handle JSON type functions (json.loads, json_list_type, etc.)
        # YAML already parses the value to dict/list, no need to re-parse
        type_func_name = getattr(type_func, "__name__", "")
        if type_func_name in ("loads", "json_list_type") and isinstance(
            value, (dict, list)
        ):
            return value

        # Apply type conversion
        return type_func(value)
