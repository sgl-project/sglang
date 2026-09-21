# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
"""The `sglang snapshot` command surface: argument declarations and dispatch."""

import argparse
import json

import msgspec

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
    """Declare the `create`, `restore` and `inspect` subcommands on `parser`."""
    commands = parser.add_subparsers(dest="action", required=True)

    create = commands.add_parser(
        "create", help="Initialize an engine and checkpoint it into an artifact."
    )
    create.add_argument(
        "--artifact",
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
        nargs="*",
        help="Server arguments, passed after a literal --.",
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

    inspect = commands.add_parser(
        "inspect",
        help="Report what an artifact contains and whether this host can restore it.",
    )
    inspect.add_argument(
        "--artifact",
        required=True,
        help="Artifact directory produced by `create`.",
    )
    inspect.add_argument(
        "--json",
        action="store_true",
        help="Print the report as JSON.",
    )


def require_separator(argv):
    """Reject `create` pass-through arguments that did not follow ``--``.

    ``argparse`` cannot report this mistake itself: without the separator an
    unrecognized option either lands inside ``server_args`` (the old REMAINDER
    behavior) or surfaces as a bare "unrecognized arguments", and neither
    points at the missing ``--`` as the problem.
    """
    if argv[:1] != ["create"] or "--" in argv or "-h" in argv or "--help" in argv:
        return
    raise SnapshotUsageError(
        "create takes the server arguments after a literal --, e.g. "
        "`sglang snapshot create --artifact DIR -- --model-path ...`"
    )


def execute(options):
    """Run the selected snapshot action and return the process exit code."""
    from sglang.srt.engine_snapshot.controller import (
        create_snapshot,
        inspect_snapshot,
        restore_snapshot,
    )

    if options.action == "create":
        argv = list(options.server_args)
        if not argv:
            raise SnapshotUsageError("create requires server arguments after --")
        create_snapshot(options.artifact, argv, options.timeout)
        print(f"Snapshot created: {options.artifact}")
        return 0
    if options.action == "restore":
        outcome = restore_snapshot(
            options.artifact, options.timeout, host=options.host, port=options.port
        )
        print(
            f"Engine restored: pid={outcome.root_pid} "
            f"address={outcome.host}:{outcome.port}"
        )
        return 0
    if options.action == "inspect":
        manifest, checks = inspect_snapshot(options.artifact)
        if options.json:
            print(
                json.dumps(
                    {
                        "manifest": msgspec.to_builtins(manifest),
                        "checks": checks,
                    },
                    indent=2,
                )
            )
        else:
            _print_report(manifest, checks)
        return 0 if checks["identity"] == "match" else 1
    raise SnapshotUsageError(f"unsupported snapshot action: {options.action}")


def _print_report(manifest, checks):
    fields = (
        ("artifact", manifest.artifact_path),
        ("format", manifest.format),
        ("created", manifest.created_at),
        ("model", manifest.model_path),
        ("engine", f"{manifest.host}:{manifest.port}"),
        ("gpu", f"{manifest.identity.gpu_name} {manifest.identity.gpu_uuid}"),
        (
            "process",
            f"root_pid={manifest.root_pid} pids={len(manifest.pids)} "
            f"cuda={len(manifest.cuda_pids)}",
        ),
        (
            "canary",
            f"token {manifest.canary.token_id} logprob {manifest.canary.logprob:.4f}",
        ),
        ("files", f"{len(manifest.files)} carried, {len(manifest.dev_shm)} /dev/shm"),
        ("bytes", manifest.artifact_bytes),
        ("identity", checks["identity"]),
        ("pids", "free" if not checks["occupied_pids"] else checks["occupied_pids"]),
        ("address", checks["listen_address"]),
    )
    width = max(len(name) for name, _ in fields)
    for name, value in fields:
        print(f"{name:>{width}}  {value}")
