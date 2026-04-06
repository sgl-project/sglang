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
"""CLI argument definitions for LoRA."""

import argparse


class LoRAArgs:
    """CLI argument definitions for LoRA."""

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        from sglang.srt.server_args import ServerArgs
        from sglang.srt.args.constants import LORA_BACKEND_CHOICES
        from sglang.srt.args.actions import LoRAPathAction
        from sglang.srt.utils.common import (
            SUPPORTED_LORA_TARGET_MODULES,
            LORA_TARGET_ALL_MODULES,
        )

        # LoRA
        parser.add_argument(
            "--enable-lora",
            default=ServerArgs.enable_lora,
            action="store_true",
            help="Enable LoRA support for the model. This argument is automatically set to True if `--lora-paths` is provided for backward compatibility.",
        )
        parser.add_argument(
            "--enable-lora-overlap-loading",
            default=ServerArgs.enable_lora_overlap_loading,
            action="store_true",
            help="Enable asynchronous LoRA weight loading in order to overlap H2D transfers with GPU compute. This should be enabled if you find that your LoRA workloads are bottlenecked by adapter weight loading, for example when frequently loading large LoRA adapters.",
        )
        parser.add_argument(
            "--max-lora-rank",
            default=ServerArgs.max_lora_rank,
            type=int,
            help="The maximum rank of LoRA adapters. If not specified, it will be automatically inferred from the adapters provided in --lora-paths.",
        )
        parser.add_argument(
            "--lora-target-modules",
            type=str,
            choices=SUPPORTED_LORA_TARGET_MODULES + [LORA_TARGET_ALL_MODULES],
            nargs="*",
            default=None,
            help="The union set of all target modules where LoRA should be applied. If not specified, "
            "it will be automatically inferred from the adapters provided in --lora-paths. If 'all' is specified, "
            "all supported modules will be targeted.",
        )
        parser.add_argument(
            "--lora-paths",
            type=str,
            nargs="*",
            default=None,
            action=LoRAPathAction,
            help='The list of LoRA adapters to load. Each adapter must be specified in one of the following formats: <PATH> | <NAME>=<PATH> | JSON with schema {"lora_name":str,"lora_path":str,"pinned":bool}',
        )
        parser.add_argument(
            "--max-loras-per-batch",
            type=int,
            default=8,
            help="Maximum number of adapters for a running batch, include base-only request.",
        )
        parser.add_argument(
            "--max-loaded-loras",
            type=int,
            default=ServerArgs.max_loaded_loras,
            help="If specified, it limits the maximum number of LoRA adapters loaded in CPU memory at a time. The value must be greater than or equal to `--max-loras-per-batch`.",
        )
        parser.add_argument(
            "--lora-eviction-policy",
            type=str,
            default=ServerArgs.lora_eviction_policy,
            choices=["lru", "fifo"],
            help="LoRA adapter eviction policy when memory pool is full. 'lru': Least Recently Used (default, better cache efficiency). 'fifo': First-In-First-Out.",
        )
        parser.add_argument(
            "--lora-backend",
            type=str,
            choices=LORA_BACKEND_CHOICES,
            default=ServerArgs.lora_backend,
            help="Choose the kernel backend for multi-LoRA serving.",
        )
        parser.add_argument(
            "--max-lora-chunk-size",
            type=int,
            default=ServerArgs.max_lora_chunk_size,
            choices=[16, 32, 64, 128],
            help="Maximum chunk size for the ChunkedSGMV LoRA backend. Only used when --lora-backend is 'csgmv'. Choosing a larger value might improve performance.",
        )
        parser.add_argument(
            "--experts-shared-outer-loras",
            default=ServerArgs.experts_shared_outer_loras,
            action=argparse.BooleanOptionalAction,
            help="Force shared outer LoRA mode for MoE models. "
            "When set, w1/w3 lora_A and w2 lora_B are shared across experts "
            "(expert_dim=1). Use --no-experts-shared-outer-loras to force disable. "
            "By default this is auto-detected from adapter weights.",
        )
