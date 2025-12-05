# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
# adapted from vllm: https://github.com/vllm-project/vllm/blob/v0.7.3/vllm/entrypoints/cli/serve.py

import argparse
import dataclasses
import os
from typing import cast

import sglang.multimodal_gen.envs as envs
from sglang.multimodal_gen import DiffGenerator
from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
from sglang.multimodal_gen.runtime.entrypoints.cli.cli_types import CLISubcommand
from sglang.multimodal_gen.runtime.entrypoints.cli.utils import (
    RaiseNotImplementedAction,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.runtime.utils.perf_logger import (
    PerformanceLogger,
    RequestTimings,
)
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


def maybe_dump_performance(args: argparse.Namespace, server_args, prompt: str, results):
    """dump performance if necessary"""
    if not (args.perf_dump_path and results):
        return

    if isinstance(results, list):
        result = results[0] if results else {}
    else:
        result = results

    timings_dict = result.get("timings")
    if not (args.perf_dump_path and timings_dict):
        return

    timings = RequestTimings(request_id=timings_dict.get("request_id"))
    timings.stages = timings_dict.get("stages", {})
    timings.total_duration_ms = timings_dict.get("total_duration_ms", 0)

    PerformanceLogger.dump_benchmark_report(
        file_path=args.perf_dump_path,
        timings=timings,
        meta={
            "prompt": prompt,
            "model": server_args.model_path,
        },
        tag="cli_generate",
    )


def generate_cmd(args: argparse.Namespace):
    """The entry point for the generate command."""
    args.request_id = "mocked_fake_id_for_offline_generate"

    # Auto-enable stage logging if dump path is provided
    if args.perf_dump_path:
        os.environ["SGLANG_DIFFUSION_STAGE_LOGGING"] = "True"
        envs.SGLANG_DIFFUSION_STAGE_LOGGING = True

    server_args = ServerArgs.from_cli_args(args)
    sampling_params_kwargs = SamplingParams.get_cli_args(args)
    generator = DiffGenerator.from_pretrained(
        model_path=server_args.model_path, server_args=server_args
    )

    results = generator.generate(sampling_params_kwargs=sampling_params_kwargs)

    prompt = sampling_params_kwargs.get("prompt", None)
    maybe_dump_performance(args, server_args, prompt, results)


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
