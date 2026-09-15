# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
"""The `sglang snapshot` command surface: argument declarations and dispatch."""

import argparse

from sglang.srt.engine_snapshot.errors import SnapshotUsageError


def _timeout_seconds(value):
    try:
        seconds = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid timeout: {value!r}") from None
    if seconds <= 0:
        raise argparse.ArgumentTypeError("timeout must be positive")
    return seconds


def _port(value):
    try:
        port = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid port: {value!r}") from None
    if not 0 < port < 65536:
        raise argparse.ArgumentTypeError("port must be between 1 and 65535")
    return port


def add_cli_args(parser):
    """Declare the `create` and `restore` subcommands on `parser`."""
    commands = parser.add_subparsers(dest="action", required=True)

    create = commands.add_parser(
        "create", help="Initialize an engine and checkpoint it into an artifact."
    )
    create.add_argument(
        "--output",
        required=True,
        help="Artifact directory to create; it must not exist yet.",
    )
    create.add_argument(
        "--timeout",
        type=_timeout_seconds,
        default=600,
        help="Seconds allowed for engine startup and capture (default: 600).",
    )
    create.add_argument(
        "server_args",
        nargs=argparse.REMAINDER,
        help="Server arguments after --.",
    )

    restore = commands.add_parser(
        "restore",
        help="Restore an artifact on the captured GPU in a fresh container.",
    )
    restore.add_argument(
        "--artifact",
        required=True,
        help="Artifact directory produced by `create`.",
    )
    restore.add_argument(
        "--timeout",
        type=_timeout_seconds,
        default=300,
        help="Seconds allowed for restore and readiness (default: 300).",
    )
    restore.add_argument(
        "--host",
        default=None,
        help="Listen address to use instead of the captured one.",
    )
    restore.add_argument(
        "--port",
        type=_port,
        default=None,
        help="Listen port to use instead of the captured one.",
    )


def execute(options):
    """Run the selected snapshot action and return the process exit code."""
    from sglang.srt.engine_snapshot.controller import (
        create_snapshot,
        restore_snapshot,
    )

    if options.action == "create":
        argv = list(options.server_args)
        if argv[:1] == ["--"]:
            argv = argv[1:]
        if not argv:
            raise SnapshotUsageError("create requires server arguments after --")
        create_snapshot(options.output, argv, options.timeout)
        print(f"Snapshot created: {options.output}")
    elif options.action == "restore":
        outcome = restore_snapshot(
            options.artifact, options.timeout, host=options.host, port=options.port
        )
        print(
            f"Engine restored: pid={outcome.root_pid} "
            f"address={outcome.host}:{outcome.port}"
        )
    return 0
