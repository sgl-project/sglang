# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
"""`sglang snapshot` entry point: parse arguments, dispatch, map failures."""

import argparse


def snapshot(args, extra_argv):
    from sglang.srt.engine_snapshot.cli import (
        add_cli_args,
        execute,
        require_separator,
    )
    from sglang.srt.engine_snapshot.errors import SnapshotError, SnapshotUsageError

    parser = argparse.ArgumentParser(
        prog="sglang snapshot",
        description="Create, inspect, or restore an initialized engine snapshot.",
    )
    add_cli_args(parser)
    try:
        require_separator(extra_argv)
        options = parser.parse_args(extra_argv)
        code = execute(options)
    except SnapshotUsageError as exc:
        parser.error(str(exc))
    except SnapshotError as exc:
        parser.exit(1, f"error: {exc}\n")
    except (RuntimeError, OSError, ValueError) as exc:
        # A third-party failure that escaped the snapshot hierarchy still exits
        # with a message instead of a traceback.
        parser.exit(1, f"error: {exc}\n")
    parser.exit(code)
