# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
# adapted from vllm: https://github.com/vllm-project/vllm/blob/v0.7.3/vllm/entrypoints/cli/main.py

from sglang.multimodal_gen.runtime.entrypoints.cli.cli_types import CLISubcommand
from sglang.multimodal_gen.runtime.entrypoints.cli.generate import GenerateSubcommand
from sglang.multimodal_gen.runtime.entrypoints.cli.serve import ServeSubcommand
from sglang.multimodal_gen.utils import FlexibleArgumentParser


def generate_cmd_init() -> list[CLISubcommand]:
    return [GenerateSubcommand(), ServeSubcommand()]


def cmd_init() -> list[CLISubcommand]:
    """Initialize all commands from separate modules"""
    commands = []
    commands.extend(generate_cmd_init())
    return commands


def main() -> None:
    parser = FlexibleArgumentParser(description="sgl-diffusion CLI")
    parser.add_argument("-v", "--version", action="version", version="0.1.0")

    subparsers = parser.add_subparsers(required=False, dest="subparser")

    cmds = {}
    for cmd in cmd_init():
        cmd.subparser_init(subparsers).set_defaults(dispatch_function=cmd.cmd)
        cmds[cmd.name] = cmd
    args = parser.parse_args()
    if args.subparser in cmds:
        cmds[args.subparser].validate(args)

    if hasattr(args, "dispatch_function"):
        args.dispatch_function(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
