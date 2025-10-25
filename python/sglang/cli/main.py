import argparse


def get_is_diffusion_model(model_path: str):
    lowered_path = model_path.lower()
    return (
        "diffusion" in lowered_path
        or "wan" in lowered_path
        or "video" in lowered_path
        or "image" in lowered_path
        or "hunyuan" in lowered_path
        or "flux" in lowered_path
    )


def get_model_path(extra_argv):
    # Find the model_path argument
    model_path = None
    for i, arg in enumerate(extra_argv):
        if arg == "--model-path":
            if i + 1 < len(extra_argv):
                model_path = extra_argv[i + 1]
                break
        elif arg.startswith("--model-path="):
            model_path = arg.split("=", 1)[1]
            break

    if model_path is None:
        # Fallback for --help or other cases where model-path is not provided
        if any(h in extra_argv for h in ["-h", "--help"]):
            raise Exception(
                "Usage: sglang serve --model-path <model-name-or-path> [additional-arguments]\n\n"
                "This command can launch either a standard language model server or a diffusion model server.\n"
                "The server type is determined by the model path.\n"
                "For specific arguments, please provide a model_path."
            )
        else:
            raise Exception(
                "Error: --model-path is required. "
                "Please provide the path to the model."
            )
    return model_path


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

    # generate subcommand
    from sglang.cli.generate import generate

    generate_parser = subparsers.add_parser(
        "generate",
        help="Run inference on a multimodal model.",
        add_help=False,  # Defer help to the specific parser
    )
    generate_parser.set_defaults(func=generate)

    args, extra_argv = parser.parse_known_args()
    args.func(args, extra_argv)
