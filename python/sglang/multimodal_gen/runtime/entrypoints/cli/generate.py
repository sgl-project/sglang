# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
# adapted from vllm: https://github.com/vllm-project/vllm/blob/v0.7.3/vllm/entrypoints/cli/serve.py

import argparse
import dataclasses
import json
import os
import time
from datetime import datetime
from typing import cast

from dateutil.tz import UTC

from sglang.multimodal_gen import DiffGenerator
from sglang.multimodal_gen.configs.sample.base import (
    SamplingParams,
    generate_request_id,
)
from sglang.multimodal_gen.runtime.entrypoints.cli.cli_types import CLISubcommand
from sglang.multimodal_gen.runtime.entrypoints.cli.utils import (
    RaiseNotImplementedAction,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.runtime.utils.performance_logger import get_git_commit_hash
from sglang.multimodal_gen.utils import FlexibleArgumentParser

logger = init_logger(__name__)


def add_multimodal_gen_generate_args(parser: argparse.ArgumentParser):
    """Add the arguments for the generate command."""
    parser.add_argument(
        "--config",
        type=str,
        default="",
        required=False,
        help="Read CLI options from a config JSON or YAML file. If provided, --model-path and --prompt are optional.",
    )
    parser.add_argument(
        "--perf-dump-path",
        type=str,
        default=None,
        required=False,
        help="Path to dump the performance metrics (JSON) for the run.",
    )

    parser = ServerArgs.add_cli_args(parser)
    parser = SamplingParams.add_cli_args(parser)

    parser.add_argument(
        "--text-encoder-configs",
        action=RaiseNotImplementedAction,
        help="JSON array of text encoder configurations (NOT YET IMPLEMENTED)",
    )

    return parser


def generate_cmd(args: argparse.Namespace):
    """The entry point for the generate command."""
    # FIXME(mick): do not hard code
    args.request_id = generate_request_id()

    server_args = ServerArgs.from_cli_args(args)
    sampling_params = SamplingParams.from_cli_args(args)
    sampling_params.request_id = generate_request_id()
    generator = DiffGenerator.from_pretrained(
        model_path=server_args.model_path, server_args=server_args
    )

    start_time = time.monotonic()
    results = generator.generate(
        prompt=sampling_params.prompt, sampling_params=sampling_params
    )
    total_duration = time.monotonic() - start_time

    # Handle performance dumping
    if args.perf_dump_path and results:
        # Assuming results is a list of dicts or a single dict
        if isinstance(results, list):
            result = results[0] if results else {}
        else:
            result = results

        logging_info = result.get("logging_info", {})

        # Convert logging_info stages to the format expected by compare_perf
        formatted_steps = []
        if hasattr(logging_info, "stages"):
            stages = logging_info.stages
        elif isinstance(logging_info, dict) and "stages" in logging_info:
            stages = logging_info["stages"]
        else:
            stages = {}

        if isinstance(stages, dict):
            for name, info in stages.items():
                if not info:
                    continue
                exec_time = info.get("execution_time")
                if exec_time is not None:
                    formatted_steps.append(
                        {"name": name, "duration_ms": exec_time * 1000}
                    )

        # Construct report
        report = {
            "timestamp": datetime.now(UTC).isoformat(),
            "request_id": sampling_params.request_id,
            "commit_hash": get_git_commit_hash(),
            "tag": "cli_generate",
            "total_duration_ms": result.get("generation_time", total_duration) * 1000,
            "steps": formatted_steps,
            "meta": {
                "prompt": sampling_params.prompt,
                "model": server_args.model_path,
            },
        }

        try:
            dump_path = os.path.abspath(args.perf_dump_path)
            os.makedirs(os.path.dirname(dump_path), exist_ok=True)
            with open(dump_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
            print(f"[Performance] Metrics dumped to: {dump_path}")
        except Exception as e:
            logger.error(f"Failed to dump performance metrics: {e}")


class GenerateSubcommand(CLISubcommand):
    """The `generate` subcommand for the sglang-diffusion CLI"""

    def __init__(self) -> None:
        self.name = "generate"
        super().__init__()
        self.init_arg_names = self._get_init_arg_names()
        self.generation_arg_names = self._get_generation_arg_names()

    def _get_init_arg_names(self) -> list[str]:
        """Get names of arguments for DiffGenerator initialization"""
        return ["num_gpus", "tp_size", "sp_size", "model_path"]

    def _get_generation_arg_names(self) -> list[str]:
        """Get names of arguments for generate_video method"""
        return [field.name for field in dataclasses.fields(SamplingParams)]

    def cmd(self, args: argparse.Namespace) -> None:
        generate_cmd(args)

    def validate(self, args: argparse.Namespace) -> None:
        """Validate the arguments for this command"""
        if args.num_gpus is not None and args.num_gpus <= 0:
            raise ValueError("Number of gpus must be positive")

        if args.config and not os.path.exists(args.config):
            raise ValueError(f"Config file not found: {args.config}")

    def subparser_init(
        self, subparsers: argparse._SubParsersAction
    ) -> FlexibleArgumentParser:
        generate_parser = subparsers.add_parser(
            "generate",
            help="Run inference on a model",
            usage="sgl_diffusion generate (--model-path MODEL_PATH_OR_ID --prompt PROMPT) | --config CONFIG_FILE [OPTIONS]",
        )

        generate_parser = add_multimodal_gen_generate_args(generate_parser)

        return cast(FlexibleArgumentParser, generate_parser)
