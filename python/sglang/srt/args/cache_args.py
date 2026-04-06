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
"""CLI argument definitions for cache."""

import argparse


class CacheArgs:
    """CLI argument definitions for Mamba Cache, hierarchical cache, sparse attention, and LMCache."""

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        from sglang.srt.server_args import ServerArgs
        from sglang.srt.args.constants import (
            MAMBA_SSM_DTYPE_CHOICES,
            MAMBA_SCHEDULER_STRATEGY_CHOICES,
            MAMBA_BACKEND_CHOICES,
            LINEAR_ATTN_KERNEL_BACKEND_CHOICES,
        )

        # Mamba Cache
        parser.add_argument(
            "--max-mamba-cache-size",
            type=int,
            default=ServerArgs.max_mamba_cache_size,
            help="The maximum size of the mamba cache.",
        )
        parser.add_argument(
            "--mamba-ssm-dtype",
            type=str,
            default=None,
            choices=MAMBA_SSM_DTYPE_CHOICES,
            help="The data type of the SSM states in mamba cache. "
            "If not set, will be read from model config (mamba_ssm_dtype).",
        )
        parser.add_argument(
            "--mamba-full-memory-ratio",
            type=float,
            default=ServerArgs.mamba_full_memory_ratio,
            help="The ratio of mamba state memory to full kv cache memory.",
        )
        parser.add_argument(
            "--mamba-scheduler-strategy",
            type=str,
            choices=MAMBA_SCHEDULER_STRATEGY_CHOICES,
            default=ServerArgs.mamba_scheduler_strategy,
            help="The strategy to use for mamba radix cache.",
        )
        parser.add_argument(
            "--mamba-track-interval",
            type=int,
            default=ServerArgs.mamba_track_interval,
            help="The interval to track the mamba state during decode.",
        )
        parser.add_argument(
            "--mamba-backend",
            type=str,
            choices=MAMBA_BACKEND_CHOICES,
            default=ServerArgs.mamba_backend,
            help="Choose the kernel backend for Mamba SSM operations. Default is 'triton'. "
            "Options: 'triton' (default), 'flashinfer' (requires FlashInfer with Mamba support).",
        )
        parser.add_argument(
            "--linear-attn-backend",
            type=str,
            choices=LINEAR_ATTN_KERNEL_BACKEND_CHOICES,
            default=ServerArgs.linear_attn_backend,
            help="The default kernel backend for linear attention (GDN/KDA). "
            "Can be overridden per-mode by --linear-attn-decode-backend "
            "and --linear-attn-prefill-backend.",
        )
        parser.add_argument(
            "--linear-attn-decode-backend",
            type=str,
            choices=LINEAR_ATTN_KERNEL_BACKEND_CHOICES,
            default=ServerArgs.linear_attn_decode_backend,
            help="Override the kernel backend for linear attention decode. "
            "If not set, uses --linear-attn-backend.",
        )
        parser.add_argument(
            "--linear-attn-prefill-backend",
            type=str,
            choices=LINEAR_ATTN_KERNEL_BACKEND_CHOICES,
            default=ServerArgs.linear_attn_prefill_backend,
            help="Override the kernel backend for linear attention prefill/extend. "
            "If not set, uses --linear-attn-backend.",
        )

        # Hierarchical cache
        parser.add_argument(
            "--enable-hierarchical-cache",
            action="store_true",
            help="Enable hierarchical cache",
        )
        parser.add_argument(
            "--hicache-ratio",
            type=float,
            default=ServerArgs.hicache_ratio,
            help="The ratio of the size of host KV cache memory pool to the size of device pool.",
        )
        parser.add_argument(
            "--hicache-size",
            type=int,
            default=ServerArgs.hicache_size,
            help="The size of host KV cache memory pool in gigabytes, which will override the hicache_ratio if set.",
        )
        parser.add_argument(
            "--hicache-write-policy",
            type=str,
            choices=["write_back", "write_through", "write_through_selective"],
            default=ServerArgs.hicache_write_policy,
            help="The write policy of hierarchical cache.",
        )
        parser.add_argument(
            "--hicache-io-backend",
            type=str,
            choices=["direct", "kernel", "kernel_ascend"],
            default=ServerArgs.hicache_io_backend,
            help="The IO backend for KV cache transfer between CPU and GPU",
        )
        parser.add_argument(
            "--hicache-mem-layout",
            type=str,
            choices=[
                "layer_first",
                "page_first",
                "page_first_direct",
                "page_first_kv_split",
                "page_head",
            ],
            default=ServerArgs.hicache_mem_layout,
            help="The layout of host memory pool for hierarchical cache.",
        )
        parser.add_argument(
            "--hicache-storage-backend",
            type=str,
            choices=["file", "mooncake", "hf3fs", "nixl", "aibrix", "dynamic", "eic"],
            default=ServerArgs.hicache_storage_backend,
            help="The storage backend for hierarchical KV cache. "
            "Built-in backends: file, mooncake, hf3fs, nixl, aibrix. "
            "For dynamic backend, use --hicache-storage-backend-extra-config to specify: "
            "backend_name (custom name), module_path (Python module path), class_name (backend class name).",
        )
        parser.add_argument(
            "--hicache-storage-prefetch-policy",
            type=str,
            choices=["best_effort", "wait_complete", "timeout"],
            default=ServerArgs.hicache_storage_prefetch_policy,
            help="Control when prefetching from the storage backend should stop.",
        )
        parser.add_argument(
            "--hicache-storage-backend-extra-config",
            type=str,
            default=ServerArgs.hicache_storage_backend_extra_config,
            help="A dictionary in JSON string format, or a string starting with a leading '@' and a config file in JSON/YAML/TOML format, containing extra configuration for the storage backend.",
        )

        # Hierarchical sparse attention
        parser.add_argument(
            "--enable-hisparse",
            action="store_true",
            help="Enable hierarchical sparse attention",
        )

        parser.add_argument(
            "--hisparse-config",
            type=str,
            default=ServerArgs.hisparse_config,
            help="A dictionary in JSON string format for hierarchical sparse attention configuration. "
            'Example: \'{"top_k": 2048, "device_buffer_size": 4096}\'',
        )

        # LMCache
        parser.add_argument(
            "--enable-lmcache",
            action="store_true",
            help="Using LMCache as an alternative hierarchical cache solution",
        )
