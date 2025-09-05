# SPDX-License-Identifier: Apache-2.0
"""
Saves each worker's model state dict directly to a checkpoint, which enables a
fast load path for large tensor-parallel models where each worker only needs to
read its own shard rather than the entire checkpoint.

Example usage:

python save_remote_state.py \
    --model-path /path/to/load \
    --tensor-parallel-size 8 \
    --remote-model-save-url [protocol]://[host]:[port]/[model_name] \

Then, the model can be loaded with

llm = Engine(
    model_path="[protocol]://[host]:[port]/[model_name]",
    tensor_parallel_size=8,
)
"""
import dataclasses
from argparse import ArgumentParser
from pathlib import Path

from sglang import Engine, ServerArgs

parser = ArgumentParser()
ServerArgs.add_cli_args(parser)

parser.add_argument(
    "--remote-model-save-url",
    required=True,
    type=str,
    help="remote address to store model weights",
)
parser.add_argument(
    "--remote-draft-model-save-url",
    default=None,
    type=str,
    help="remote address to store draft model weights",
)


def main(args):
    engine_args = ServerArgs.from_cli_args(args)
    model_path = engine_args.model_path
    if not Path(model_path).is_dir():
        raise ValueError("model path must be a local directory")
    # Create LLM instance from arguments
    llm = Engine(**dataclasses.asdict(engine_args))
    llm.save_remote_model(
        url=args.remote_model_save_url, draft_url=args.remote_draft_model_save_url
    )
    print("save remote (draft) model successfully")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
