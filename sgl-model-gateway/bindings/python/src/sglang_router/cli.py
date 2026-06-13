#!/usr/bin/env python3
"""
SGLang Model Gateway CLI

Provides convenient command-line interface for launching the router and server.

Usage:
    smg launch [args]          # Launch router only
    smg server [args]          # Launch router + server
    smg --help                 # Show help
"""

import argparse
import os
import sys
from typing import List, Optional

from sglang_router.sglang_router_rs import (
    get_verbose_version_string,
    get_version_string,
)


def create_parser() -> argparse.ArgumentParser:
    """Create the main CLI parser with subcommands."""
    prog_name = os.path.basename(sys.argv[0]) if sys.argv else "smg"
    parser = argparse.ArgumentParser(
        prog=prog_name,
        description="SGLang Model Gateway - High-performance inference router",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Launch router subcommand
    launch_parser = subparsers.add_parser(
        "launch",
        help="Launch router only (requires existing worker URLs)",
        description="Launch the SGLang router with existing worker instances",
        add_help=False,  # Let router handle --help
    )

    # Launch server + router subcommand
    server_parser = subparsers.add_parser(
        "server",
        help="Launch router and server processes together",
        description="Launch both SGLang router and server processes",
        add_help=False,  # Let server handle --help
    )

    return parser


def main(argv: Optional[List[str]] = None) -> None:
    """Main CLI entry point."""
    if argv is None:
        argv = sys.argv[1:]

    # Handle version flags before parsing
    if argv and argv[0] in ["--version", "-V", "--version-verbose"]:
        if argv[0] == "--version-verbose":
            print(get_verbose_version_string())
        else:
            print(get_version_string())
        sys.exit(0)

    # Handle empty command - show help
    if not argv or argv[0] not in ["launch", "server", "-h", "--help"]:
        parser = create_parser()
        parser.print_help()
        sys.exit(1)

    parser = create_parser()
    args, unknown = parser.parse_known_args(argv)

    if args.command == "launch":
        # Import and call launch_router functions directly
        from sglang_router.launch_router import launch_router, parse_router_args

        # All router args are in unknown
        router_args = parse_router_args(unknown)
        launch_router(router_args)

    elif args.command == "server":
        # Import and call launch_server main with proper argv
        # Note: launch_server.main() uses argparse internally which reads sys.argv
        # We need to temporarily set sys.argv for compatibility
        import sglang_router.launch_server as launch_server_module

        # Preserve original sys.argv
        original_argv = sys.argv
        try:
            # All server args are in unknown
            prog_name = os.path.basename(sys.argv[0]) if sys.argv else "smg"
            sys.argv = [f"{prog_name} server"] + unknown
            launch_server_module.main()
        finally:
            # Restore original sys.argv
            sys.argv = original_argv

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
