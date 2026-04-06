# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""CLI argument definitions for runtime options."""

import argparse


class RuntimeArgs:
    """CLI argument definitions for runtime options."""

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        from sglang.srt.server_args import ServerArgs
        from sglang.srt.args.actions import DeprecatedStoreTrueAction

        # Runtime options
        parser.add_argument(
            "--device",
            type=str,
            default=ServerArgs.device,
            help="The device to use ('cuda', 'xpu', 'hpu', 'npu', 'cpu'). Defaults to auto-detection if not specified.",
        )
        parser.add_argument(
            "--tensor-parallel-size",
            "--tp-size",
            type=int,
            default=ServerArgs.tp_size,
            help="The tensor parallelism size.",
        )
        parser.add_argument(
            "--attention-context-parallel-size",
            "--attn-cp-size",
            type=int,
            default=ServerArgs.attn_cp_size,
            help="The attention context parallelism size.",
        )
        parser.add_argument(
            "--moe-data-parallel-size",
            "--moe-dp-size",
            type=int,
            default=ServerArgs.moe_dp_size,
            help="The moe data parallelism size.",
        )
        parser.add_argument(
            "--pipeline-parallel-size",
            "--pp-size",
            type=int,
            default=ServerArgs.pp_size,
            help="The pipeline parallelism size.",
        )
        parser.add_argument(
            "--pp-max-micro-batch-size",
            type=int,
            default=ServerArgs.pp_max_micro_batch_size,
            help="The maximum micro batch size in pipeline parallelism.",
        )
        parser.add_argument(
            "--pp-async-batch-depth",
            type=int,
            default=ServerArgs.pp_async_batch_depth,
            help="The async batch depth of pipeline parallelism.",
        )
        parser.add_argument(
            "--stream-interval",
            type=int,
            default=ServerArgs.stream_interval,
            help="The interval (or buffer size) for streaming in terms of the token length. A smaller value makes streaming smoother, while a larger value makes the throughput higher",
        )
        parser.add_argument(
            "--incremental-streaming-output",
            action="store_true",
            help="Whether to output as a sequence of disjoint segments.",
        )
        parser.add_argument(
            "--stream-response-default-include-usage",
            action="store_true",
            help="Include usage in every streaming response "
            "(even when stream_options is not specified).",
        )
        parser.add_argument(
            "--stream-output",
            action=DeprecatedStoreTrueAction,
            dest="incremental_streaming_output",
            new_flag="--incremental-streaming-output",
            help="[Deprecated] Use --incremental-streaming-output instead.",
        )
        parser.add_argument(
            "--enable-streaming-session",
            action="store_true",
            default=ServerArgs.enable_streaming_session,
            help="Enable streaming session mode and SessionAwareCache wrapper.",
        )
        parser.add_argument(
            "--random-seed",
            type=int,
            default=ServerArgs.random_seed,
            help="The random seed.",
        )
        parser.add_argument(
            "--constrained-json-whitespace-pattern",
            type=str,
            default=ServerArgs.constrained_json_whitespace_pattern,
            help="(outlines and llguidance backends only) Regex pattern for syntactic whitespaces allowed in JSON constrained output. For example, to allow the model generate consecutive whitespaces, set the pattern to [\n\t ]*",
        )
        parser.add_argument(
            "--constrained-json-disable-any-whitespace",
            action="store_true",
            help="(xgrammar and llguidance backends only) Enforce compact representation in JSON constrained output.",
        )
        parser.add_argument(
            "--watchdog-timeout",
            type=float,
            default=ServerArgs.watchdog_timeout,
            help="Set watchdog timeout in seconds. If a forward batch takes longer than this, the server will crash to prevent hanging.",
        )
        parser.add_argument(
            "--soft-watchdog-timeout",
            type=float,
            default=ServerArgs.soft_watchdog_timeout,
            help="Set soft watchdog timeout in seconds. If a forward batch takes longer than this, the server will dump information for debugging.",
        )
        parser.add_argument(
            "--dist-timeout",
            type=int,
            default=ServerArgs.dist_timeout,
            help="Set timeout for torch.distributed initialization.",
        )
        parser.add_argument(
            "--download-dir",
            type=str,
            default=ServerArgs.download_dir,
            help="Model download directory for huggingface.",
        )
        parser.add_argument(
            "--model-checksum",
            type=str,
            nargs="?",
            const="",
            default=None,
            help="Model file integrity verification. If provided without value, uses model-path as HF repo ID. Otherwise, provide checksums JSON file path or HuggingFace repo ID.",
        )
        parser.add_argument(
            "--base-gpu-id",
            type=int,
            default=ServerArgs.base_gpu_id,
            help="The base GPU ID to start allocating GPUs from. Useful when running multiple instances on the same machine.",
        )
        parser.add_argument(
            "--gpu-id-step",
            type=int,
            default=ServerArgs.gpu_id_step,
            help="The delta between consecutive GPU IDs that are used. For example, setting it to 2 will use GPU 0,2,4,...",
        )
        parser.add_argument(
            "--sleep-on-idle",
            action="store_true",
            help="Reduce CPU usage when sglang is idle.",
        )
        parser.add_argument(
            "--use-ray",
            action="store_true",
            help="Use Ray actors for scheduler process management.",
        )
        parser.add_argument(
            "--custom-sigquit-handler",
            help="Register a custom sigquit handler so you can do additional cleanup after the server is shutdown. This is only available for Engine, not for CLI.",
        )
