"""SGLang CLI entry point for the 'sglang' console command.

Design note: CLI shape and future structure

- Top-level form: 'sglang <command> [args]'
- Current commands:
  - 'serve': launch the HTTP inference server. This replaces 'python -m sglang.launch_server' and the previous 'sglang' without subcommand. Exact server flags and semantics remain owned by sglang.srt.server_args.prepare_server_args.
- Future commands (illustrative, not implemented here):
  - 'chat', 'eval', 'convert', 'cache', 'admin', 'doctor', 'version'
  Each command should live in its own module and be wired here via a thin dispatcher, e.g.:
      sglang.cli.chat:main(argv)      -> 'sglang chat ...'
      sglang.cli.eval:main(argv)      -> 'sglang eval ...'
      sglang.cli.convert:main(argv)   -> 'sglang convert ...'
  This file must stay a minimal router: no business logic or flag schemas beyond subcommand selection.

Behavioral contract

- 'sglang' without a subcommand prints usage and exits with code 2 (intentional to avoid ambiguity).
- Unknown commands also print usage and exit 2.
- 'sglang serve' delegates to prepare_server_args(rest) then sglang.srt.entrypoints.http_server.launch_server(args).
- The dispatcher must not alter how server arguments are parsed/validated; it only slices argv to strip the subcommand and passes the rest through unchanged.

Rationale

- Keeps a single executable name 'sglang' as the UX anchor, while allowing future subcommands to branch cleanly.
- Avoids ambiguity of invoking 'sglang' without a verb, while preserving the historical module entry point 'python -m sglang.launch_server'.

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
