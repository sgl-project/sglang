"""
Save a model to remote storage using SGLang Engine with RemoteModelLoader.
This script initializes and launches an SGLang Engine (as defined in
sglang/srt/entrypoints/engine.py), then saves the loaded model weights
to a remote storage backend via Engine.save_remote_model().
The remote URL format depends on the connector backend, for example:
  - s3://bucket-name/model-name
  - redis://host:port/model-name
It accepts the same server arguments as launch_server.py, plus:
  --url       Remote storage URL to save the model to (required)
  --draft-url Remote storage URL for the draft model (optional, only
              needed when a speculative draft model is enabled)
Usage:
python3 -m sglang.save_model_remote_loader \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --url s3://my-bucket/llama-3.1-8b \
    --tp 1
Example with draft model:
python3 -m sglang.save_model_remote_loader \
    --model deepseek-ai/DeepSeek-V3 \
    --speculative-algorithm EAGLE \
    --speculative-draft-model-path /path/to/draft \
    --url s3://my-bucket/deepseek-v3 \
    --draft-url s3://my-bucket/deepseek-v3-draft \
    --tp 8 \
    --trust-remote-code
"""

import argparse
import dataclasses
import logging
from typing import Optional

from sglang.srt.entrypoints.engine import Engine
from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class SaveRemoteArgs:
    """Arguments specific to saving a model to remote storage."""

    url: str
    draft_url: Optional[str] = None

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser):
        parser.add_argument(
            "--url",
            type=str,
            required=True,
            help=(
                "Remote storage URL to save the model weights to. "
                "Format depends on the connector backend, e.g.: "
                "s3://bucket/model-name  or  redis://host:port/model-name"
            ),
        )
        parser.add_argument(
            "--draft-url",
            type=str,
            default=None,
            help=(
                "Remote storage URL for the speculative draft model weights. "
                "Only required when a draft model is enabled."
            ),
        )

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace) -> "SaveRemoteArgs":
        attrs = [(attr.name, type(attr.default)) for attr in dataclasses.fields(cls)]
        return cls(**{attr: getattr(args, attr) for attr, _ in attrs})


def run_save_remote(server_args: ServerArgs, save_args: SaveRemoteArgs):
    """Initialize the SGLang Engine and save the model to remote storage."""
    print(
        f"Initializing SGLang Engine for model: {server_args.model_path}\n"
        f"Target remote URL: {save_args.url}\n"
    )

    # Use info logging so model loading progress is visible in the CLI.
    server_args.log_level = "info"

    with Engine(server_args=server_args) as engine:
        print("Engine initialized successfully. Starting remote model save...\n")

        rpc_kwargs = {"url": save_args.url}
        if save_args.draft_url is not None:
            rpc_kwargs["draft_url"] = save_args.draft_url

        engine.save_remote_model(**rpc_kwargs)

        print(
            f"\nModel saved to remote storage successfully.\n"
            f"  Model  URL : {save_args.url}\n"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Initialize an SGLang Engine and save the model weights to remote storage."
        )
    )
    ServerArgs.add_cli_args(parser)
    SaveRemoteArgs.add_cli_args(parser)

    args = parser.parse_args()
    server_args = ServerArgs.from_cli_args(args)
    save_args = SaveRemoteArgs.from_cli_args(args)

    run_save_remote(server_args, save_args)
