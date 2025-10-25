import argparse


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    # serve subcommand
    from sglang.cli.serve import serve

    serve_parser = subparsers.add_parser(
        "serve",
        help="Launch the SGLang server.",
        add_help=False,  # Defer help to the specific parser
    )
    serve_parser.set_defaults(func=serve)

    args, extra_argv = parser.parse_known_args()
    args.func(args, extra_argv)
