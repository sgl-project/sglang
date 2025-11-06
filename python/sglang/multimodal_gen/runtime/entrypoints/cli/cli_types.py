# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
# adapted from vllm: https://github.com/vllm-project/vllm/blob/v0.7.3/vllm/entrypoints/cli/types.py

import argparse

from sglang.multimodal_gen.utils import FlexibleArgumentParser


class CLISubcommand:
    """Base class for CLI subcommands"""

    name: str

    def cmd(self, args: argparse.Namespace) -> None:
        """Execute the command with the given arguments"""
        raise NotImplementedError

    def validate(self, args: argparse.Namespace) -> None:
        """Validate the arguments for this command"""
        pass

    def subparser_init(
        self, subparsers: argparse._SubParsersAction
    ) -> FlexibleArgumentParser:
        """Initialize the subparser for this command"""
        raise NotImplementedError
