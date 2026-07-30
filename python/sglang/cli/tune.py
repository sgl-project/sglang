"""`sglang tune` subcommand — house-style handler(args, extra_argv) + cli_main.

Wire-in (cli/main.py):
    subparsers.add_parser("tune", help="Auto-tune attention backends.", add_help=False)
    ...
    elif args.subcommand == "tune":
        from sglang.cli.tune import tune
        tune(args, extra_argv)
"""

from __future__ import annotations

import argparse
import logging
import sys

from sglang.tune.args import TuneArgs
from sglang.tune.device import detect_device
from sglang.tune.orchestrate import run_tune, summarize
from sglang.tune.shapes import AttnProfile


def _profile_from_args(a: TuneArgs) -> AttnProfile:
    return AttnProfile(
        num_qo_heads=a.qo_heads,
        num_kv_heads=a.kv_heads,
        head_dim=a.head_dim,
        dtype=a.dtype,
        kv_cache_dtype=a.kv_cache_dtype,
        is_mla=a.mla,
        tp_size=a.tp_size,
        ep_size=a.ep_size,
        dp_size=a.dp_size,
    )


def run(a: TuneArgs) -> dict:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    dev = (
        detect_device(mock_name=a.mock_device, mock_sm=a.mock_sm)
        if a.mock
        else detect_device()
    )
    # Real path: build AttnProfile from ServerArgs + ModelConfig instead of flags.
    profile = _profile_from_args(a)
    print(
        f"[attune] device={dev.name} {dev.sm_tag} cuda={dev.cuda_version} "
        f"profile={profile.family()} qo={profile.num_qo_heads} kv={profile.num_kv_heads}"
    )
    cfg = run_tune(
        dev,
        profile,
        packaged_dir=a.config_dir,
        local_cache_dir=a.local_cache_dir,
        mock=a.mock,
        isolate=a.isolate,
        phases=tuple(a.phases.split(",")),
        provenance={"attune_version": "0.1.0-prototype"},
    )
    print("\n[attune] summary:\n" + summarize(cfg))
    return cfg


def tune(args, extra_argv) -> None:
    """Subcommand entry from cli/main.py (args = top-level namespace, extra_argv = rest)."""
    parser = argparse.ArgumentParser(prog="sglang tune")
    TuneArgs.add_cli_args(parser)
    ns = parser.parse_args(extra_argv)
    run(TuneArgs.from_cli_args(ns))


def cli_main() -> None:
    """Entry for `python -m sglang.tune`."""
    tune(None, sys.argv[1:])


if __name__ == "__main__":
    cli_main()
