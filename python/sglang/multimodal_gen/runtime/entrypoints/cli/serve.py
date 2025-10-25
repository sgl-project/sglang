# SPDX-License-Identifier: Apache-2.0

import argparse
import os
from typing import cast

from sgl_diffusion.runtime.entrypoints.cli.cli_types import CLISubcommand
from sgl_diffusion.runtime.launch_server import launch_server
from sgl_diffusion.runtime.server_args import ServerArgs
from sgl_diffusion.runtime.utils.logging_utils import init_logger
from sgl_diffusion.utils import FlexibleArgumentParser

logger = init_logger(__name__)


class ServeSubcommand(CLISubcommand):
    """The `serve` subcommand for the sgl-diffusion CLI"""

    def __init__(self) -> None:
        self.name = "serve"
        super().__init__()

    def cmd(self, args: argparse.Namespace) -> None:
        server_args = ServerArgs.from_cli_args(args)
        server_args.post_init_serve()

        launch_server(server_args)

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

        serve_parser.add_argument(
            "--config",
            type=str,
            default="",
            required=False,
            help="Read CLI options from a config JSON or YAML file.",
        )

        serve_parser = ServerArgs.add_cli_args(serve_parser)

        return cast(FlexibleArgumentParser, serve_parser)


def cmd_init() -> list[CLISubcommand]:
    return [ServeSubcommand()]
