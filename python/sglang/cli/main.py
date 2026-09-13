import argparse

from sglang.cli.utils import get_git_commit_hash
from sglang.version import __version__


def version(args, extra_argv):
    print(f"sglang version: {__version__}")
    print(f"git revision: {get_git_commit_hash()[:7]}")


def main():
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(dest="subcommand", required=True)
    subparsers.add_parser("serve", help="Launch an SGLang server.", add_help=False)
    subparsers.add_parser("generate", help="Run inference on a multimodal model.", add_help=False)
    subparsers.add_parser("auto-tune", help="Auto-tune kernel configurations.", add_help=False)
    version_parser = subparsers.add_parser("version", help="Show the version information.")
    version_parser.set_defaults(func=version)

    args, extra_argv = parser.parse_known_args()

    if args.subcommand == "serve":
        from sglang.cli.serve import serve
        serve(args, extra_argv)
    elif args.subcommand == "generate":
        from sglang.cli.generate import generate
        generate(args, extra_argv)
    elif args.subcommand == "auto-tune":
        from sglang.auto_tune.main import auto_tune
        auto_tune(args, extra_argv)
    elif args.subcommand == "version":
        version(args, extra_argv)
