"""
Configuration argument parser for command-line applications.
Handles merging of YAML configuration files with command-line arguments.
"""

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, List

import yaml

logger = logging.getLogger(__name__)


class ConfigArgumentMerger:
    """Handles merging of configuration file arguments with command-line arguments."""

    def __init__(
        self,
        parser: argparse.ArgumentParser = None,
        boolean_actions: List[str] = None,
    ):
        """Initialize with list of store_true action names."""
        # NOTE: The current code does not support actions other than "store_true" and "store".
        if parser is not None:
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
        elif boolean_actions is not None:
            # Legacy interface for compatibility
            self.store_true_actions = boolean_actions
            self.unsupported_actions = {}
        else:
            self.store_true_actions = []
            self.unsupported_actions = {}

    def merge_config_with_args(self, cli_args: List[str]) -> List[str]:
        """
        Merge configuration file arguments with command-line arguments.

        Configuration arguments are inserted after the subcommand to maintain
        proper precedence: CLI > Config > Defaults

        Args:
            cli_args: List of command-line arguments

        Returns:
            Merged argument list with config values inserted

        Raises:
            ValueError: If multiple config files specified or no config file provided
        """
        config_file_path = self._extract_config_file_path(cli_args)
        if not config_file_path:
            return cli_args

        config_data = self._parse_yaml_config(config_file_path)
        config_args = self._convert_config_to_args(config_data)

        # Merge config args into CLI args
        config_index = cli_args.index("--config")

        # Split arguments around config file
        before_config = cli_args[:config_index]
        after_config = cli_args[config_index + 2 :]  # Skip --config and file path

        # Simple merge: config args + CLI args
        return config_args + before_config + after_config

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

    def _convert_config_to_args(self, config: Dict[str, Any]) -> List[str]:
        """Convert configuration dictionary to argument list."""
        args = []

        for key, value in config.items():
            key_norm = key.replace("-", "_")
            if key_norm in self.unsupported_actions:
                action = self.unsupported_actions[key_norm]
                msg = f"Unsupported config option '{key_norm}' with action '{action.__class__.__name__}'"
                raise ValueError(msg)
            if isinstance(value, bool):
                self._add_boolean_arg(args, key, value)
            elif isinstance(value, list):
                self._add_list_arg(args, key, value)
            else:
                self._add_scalar_arg(args, key, value)

        return args

    def _add_boolean_arg(self, args: List[str], key: str, value: bool) -> None:
        """
        Add boolean argument to the list.

        Only store_true flags:
            - value True -> add flag
            - value False -> skip
        Regular booleans:
            - always add --key true/false
        """
        key_norm = key.replace("-", "_")
        if key_norm in self.store_true_actions:
            if value:
                args.append(f"--{key}")
        else:
            args.extend([f"--{key}", str(value).lower()])

    def _add_list_arg(self, args: List[str], key: str, value: List[Any]) -> None:
        """Add list argument to the list."""
        if value:  # Only add if list is not empty
            args.append(f"--{key}")
            args.extend(str(item) for item in value)

    def _add_scalar_arg(self, args: List[str], key: str, value: Any) -> None:
        """Add scalar argument to the list."""
        args.extend([f"--{key}", str(value)])
