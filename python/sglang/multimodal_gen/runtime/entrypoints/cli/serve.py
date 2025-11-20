# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0

import argparse
import os
from typing import cast

from sglang.multimodal_gen.runtime.entrypoints.cli.cli_types import CLISubcommand
from sglang.multimodal_gen.runtime.launch_server import launch_server
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.utils import FlexibleArgumentParser

logger = init_logger(__name__)


def add_multimodal_gen_serve_args(parser: argparse.ArgumentParser):
    """Add the arguments for the serve command."""
    parser.add_argument(
        "--config",
        type=str,
        default="",
        required=False,
        help="Read CLI options from a config JSON or YAML file.",
    )
    return ServerArgs.add_cli_args(parser)


def execute_serve_cmd(args: argparse.Namespace, unknown_args: list[str] | None = None):
    """The entry point for the serve command."""
    server_args = ServerArgs.from_cli_args(args, unknown_args)
    server_args.post_init_serve()
    launch_server(server_args)


class ServeSubcommand(CLISubcommand):
    """The `serve` subcommand for the sgl-diffusion CLI"""

    def __init__(self) -> None:
        self.name = "serve"
        super().__init__()

    def cmd(
        self, args: argparse.Namespace, unknown_args: list[str] | None = None
    ) -> None:
        execute_serve_cmd(args, unknown_args)

    def validate(self, args: argparse.Namespace) -> None:
        """Validate the arguments for this command"""
        if args.config and not os.path.exists(args.config):
            raise ValueError(f"Config file not found: {args.config}")

    def subparser_init(
        self, subparsers: argparse._SubParsersAction
    ) -> FlexibleArgumentParser:
        serve_parser = subparsers.add_parser(
            "serve",
            help="Launch the server and start FastAPI listener.",
            usage="sgl_diffusion serve --model-path MODEL_PATH_OR_ID [OPTIONS]",
        )

        serve_parser = add_multimodal_gen_serve_args(serve_parser)

        return cast(FlexibleArgumentParser, serve_parser)


def cmd_init() -> list[CLISubcommand]:
    return [ServeSubcommand()]
