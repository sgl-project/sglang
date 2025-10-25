# SPDX-License-Identifier: Apache-2.0
# adapted from vllm: https://github.com/vllm-project/vllm/blob/v0.7.3/vllm/entrypoints/cli/main.py

import typer

from sglang.cli.serve import serve_command

app = typer.Typer()


@app.command()
def serve(
    ctx: typer.Context,
    model_path: str = typer.Option(
        ...,
        "--model-path",
        help="The path of the model weights. This can be a local folder or a Hugging Face repo ID.",
    ),
):
    serve_command(model_path, ctx)


if __name__ == "__main__":
    app()
