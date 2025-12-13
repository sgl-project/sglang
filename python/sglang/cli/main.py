import argparse

from sglang.apps.sgl_diffusion_webui import sgl_diffusion_webui
from sglang.cli.generate import generate
from sglang.cli.serve import serve


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    serve_parser = subparsers.add_parser(
        "serve",
        help="Launch the SGLang server.",
        add_help=False,  # Defer help to the specific parser
    )
    serve_parser.set_defaults(func=serve)

    generate_parser = subparsers.add_parser(
        "generate",
        help="Run inference on a multimodal model.",
        add_help=False,  # Defer help to the specific parser
    )
    generate_parser.set_defaults(func=generate)

    webui_parser = subparsers.add_parser(
        "webui",
        help="Run inference on a multimodal model through webui.",
        add_help=False,  # Defer help to the specific parser
    )
    webui_parser.set_defaults(func=sgl_diffusion_webui)

    args, extra_argv = parser.parse_known_args()
    args.func(args, extra_argv)
