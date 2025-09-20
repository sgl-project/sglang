"""SGLang CLI entry point for the 'sglang' console command.

Design note: CLI shape and future structure

- Top-level form: 'sglang <command> [args]'
- Current commands:
  - 'serve': launch the HTTP inference server. This replaces 'python -m sglang.launch_server'. Exact server flags and semantics remain owned by sglang.srt.server_args.prepare_server_args.
- Future commands (illustrative, not implemented here):
  - 'chat', 'eval', 'convert', 'cache', 'admin', 'doctor', 'version'
  Each command should live in this and be wired here via a thin dispatcher, e.g.:
      sglang.cli.bench:main(argv)      -> 'sglang bench ...'
"""

import os
import sys

from sglang.srt.entrypoints.http_server import launch_server
from sglang.srt.server_args import prepare_server_args
from sglang.srt.utils import kill_process_tree


def _print_usage(file):
    file.write("Usage: sglang serve [server-options]\n")
    file.write(
        "Run 'sglang serve --help' for server flags handled by sglang.srt.server_args.\n"
    )


def main(argv=None):
    """Entrypoint for the 'sglang' command dispatcher."""
    if argv is None:
        argv = sys.argv[1:]

    if not argv:
        _print_usage(sys.stderr)
        sys.exit(2)

    cmd, *rest = argv

    if cmd == "serve":
        server_args = prepare_server_args(rest)
        try:
            launch_server(server_args)
        finally:
            kill_process_tree(os.getpid(), include_parent=False)
        return

    # Unknown subcommand
    sys.stderr.write(f"Unknown command: {cmd}\n")
    _print_usage(sys.stderr)
    sys.exit(2)


if __name__ == "__main__":
    main()
