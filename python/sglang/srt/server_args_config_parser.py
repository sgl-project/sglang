"""
Configuration argument parser for command-line applications.
Handles merging of YAML configuration files with command-line arguments.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List

import yaml

logger = logging.getLogger(__name__)


class ConfigArgumentMerger:
    """Handles merging of configuration file arguments with command-line arguments."""

    def __init__(self, boolean_actions: List[str] = None):
        """Initialize with list of boolean action destinations."""
        self.boolean_actions = boolean_actions or []

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

        config_args = self._parse_yaml_config(config_file_path)
        return self._insert_config_args(cli_args, config_args, config_file_path)

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

    def _insert_config_args(
        self, cli_args: List[str], config_args: List[str], config_file_path: str
    ) -> List[str]:
        """Insert configuration arguments into the CLI argument list."""
        config_index = cli_args.index("--config")

        # Split arguments around config file
        before_config = cli_args[:config_index]
        after_config = cli_args[config_index + 2 :]  # Skip --config and file path

        # Simple merge: config args + CLI args
        return config_args + before_config + after_config

    def _parse_yaml_config(self, file_path: str) -> List[str]:
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

        return self._convert_config_to_args(config_data)

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
            if isinstance(value, bool):
                self._add_boolean_arg(args, key, value)
            elif isinstance(value, list):
                self._add_list_arg(args, key, value)
            else:
                self._add_scalar_arg(args, key, value)

        return args

    def _add_boolean_arg(self, args: List[str], key: str, value: bool) -> None:
        """Add boolean argument to the list."""
        if key in self.boolean_actions:
            # For boolean actions, always add the flag and value
            args.extend([f"--{key}", str(value).lower()])
        else:
            # For regular booleans, only add flag if True
            if value:
                args.append(f"--{key}")

    def _add_list_arg(self, args: List[str], key: str, value: List[Any]) -> None:
        """Add list argument to the list."""
        if value:  # Only add if list is not empty
            args.append(f"--{key}")
            args.extend(str(item) for item in value)

    def _add_scalar_arg(self, args: List[str], key: str, value: Any) -> None:
        """Add scalar argument to the list."""
        args.extend([f"--{key}", str(value)])
