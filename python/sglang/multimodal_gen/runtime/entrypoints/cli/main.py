# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
# adapted from vllm: https://github.com/vllm-project/vllm/blob/v0.7.3/vllm/entrypoints/cli/main.py

from sglang.multimodal_gen.plugins import apply_plugin_hooks
from sglang.multimodal_gen.runtime.entrypoints.cli.cli_types import CLISubcommand
from sglang.multimodal_gen.utils import FlexibleArgumentParser


def generate_cmd_init() -> list[CLISubcommand]:
    # Command modules import the runtime graph. Activate plugins first so OOT
    # platforms can prepare that graph before its modules are evaluated.
    apply_plugin_hooks()

    from sglang.multimodal_gen.runtime.entrypoints.cli.generate import (
        GenerateSubcommand,
    )
    from sglang.multimodal_gen.runtime.entrypoints.cli.serve import ServeSubcommand

    return [GenerateSubcommand(), ServeSubcommand()]


def cmd_init() -> list[CLISubcommand]:
    """Initialize all commands from separate modules"""
    commands = []
    commands.extend(generate_cmd_init())
    return commands


def main() -> None:
    apply_plugin_hooks()

    parser = FlexibleArgumentParser(description="sglang-diffusion CLI")
    parser.add_argument("-v", "--version", action="version", version="0.1.0")

    subparsers = parser.add_subparsers(required=False, dest="subparser")

    cmds = {}
    for cmd in cmd_init():
        cmd.subparser_init(subparsers).set_defaults(dispatch_function=cmd.cmd)
        cmds[cmd.name] = cmd
    args, unknown_args = parser.parse_known_args()
    if args.subparser in cmds:
        cmds[args.subparser].validate(args)

    if hasattr(args, "dispatch_function"):
        args.dispatch_function(args, unknown_args=unknown_args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
