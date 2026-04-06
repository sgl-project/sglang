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
"""CLI argument definitions for parallelism."""

import argparse


class ParallelArgs:
    """CLI argument definitions for data parallelism, multi-node, and expert parallelism."""

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        from sglang.srt.server_args import ServerArgs
        from sglang.srt.args.actions import DeprecatedAction
        from sglang.srt.args.constants import (
            MOE_A2A_BACKEND_CHOICES,
            MOE_RUNNER_BACKEND_CHOICES,
            NSA_PREFILL_CP_SPLIT_CHOICES,
            PREFILL_CP_SPLIT_CHOICES,
        )

        # Data parallelism
        parser.add_argument(
            "--data-parallel-size",
            "--dp-size",
            type=int,
            default=ServerArgs.dp_size,
            help="The data parallelism size.",
        )
        parser.add_argument(
            "--load-balance-method",
            type=str,
            default=ServerArgs.load_balance_method,
            help="The load balancing strategy for data parallelism.",
            choices=[
                "auto",
                "round_robin",
                "follow_bootstrap_room",
                "total_requests",
                "total_tokens",
            ],
        )
        parser.add_argument(
            "--prefill-round-robin-balance",
            action=DeprecatedAction,
            help="Note: --prefill-round-robin-balance is deprecated now.",
        )

        # Multi-node distributed serving
        parser.add_argument(
            "--dist-init-addr",
            "--nccl-init-addr",  # For backward compatibility. This will be removed in the future.
            type=str,
            help="The host address for initializing distributed backend (e.g., `192.168.0.2:25000`).",
        )
        parser.add_argument(
            "--nnodes", type=int, default=ServerArgs.nnodes, help="The number of nodes."
        )
        parser.add_argument(
            "--node-rank", type=int, default=ServerArgs.node_rank, help="The node rank."
        )

        # Expert parallelism
        parser.add_argument(
            "--expert-parallel-size",
            "--ep-size",
            "--ep",
            type=int,
            default=ServerArgs.ep_size,
            help="The expert parallelism size.",
        )
        parser.add_argument(
            "--moe-a2a-backend",
            type=str,
            choices=MOE_A2A_BACKEND_CHOICES,
            default=ServerArgs.moe_a2a_backend,
            help="Choose the backend for MoE A2A.",
        )
        parser.add_argument(
            "--moe-runner-backend",
            type=str,
            choices=MOE_RUNNER_BACKEND_CHOICES,
            default=ServerArgs.moe_runner_backend,
            help="Choose the runner backend for MoE.",
        )
        parser.add_argument(
            "--flashinfer-mxfp4-moe-precision",
            type=str,
            choices=["default", "bf16"],
            default=ServerArgs.flashinfer_mxfp4_moe_precision,
            help="Choose the computation precision of flashinfer mxfp4 moe",
        )
        parser.add_argument(
            "--enable-flashinfer-allreduce-fusion",
            action="store_true",
            help="Enable FlashInfer allreduce fusion with Residual RMSNorm.",
        )
        parser.add_argument(
            "--enforce-disable-flashinfer-allreduce-fusion",
            action="store_true",
            help="Enforce disable FlashInfer allreduce fusion.",
        )
        parser.add_argument(
            "--enable-aiter-allreduce-fusion",
            action="store_true",
            help="Enable Aiter AllReduce Fusion.",
        )
        parser.add_argument(
            "--deepep-mode",
            type=str,
            choices=["normal", "low_latency", "auto"],
            default="auto",
            help="Select the mode when enable DeepEP MoE, could be `normal`, `low_latency` or `auto`. Default is `auto`, which means `low_latency` for decode batch and `normal` for prefill batch.",
        )
        parser.add_argument(
            "--ep-num-redundant-experts",
            type=int,
            default=ServerArgs.ep_num_redundant_experts,
            help="Allocate this number of redundant experts in expert parallel.",
        )
        parser.add_argument(
            "--ep-dispatch-algorithm",
            type=str,
            default=ServerArgs.ep_dispatch_algorithm,
            help="The algorithm to choose ranks for redundant experts in expert parallel.",
        )
        parser.add_argument(
            "--init-expert-location",
            type=str,
            default=ServerArgs.init_expert_location,
            help="Initial location of EP experts.",
        )
        parser.add_argument(
            "--enable-eplb",
            action="store_true",
            help="Enable EPLB algorithm",
        )
        parser.add_argument(
            "--eplb-algorithm",
            type=str,
            default=ServerArgs.eplb_algorithm,
            help="Chosen EPLB algorithm",
        )
        parser.add_argument(
            "--eplb-rebalance-num-iterations",
            type=int,
            default=ServerArgs.eplb_rebalance_num_iterations,
            help="Number of iterations to automatically trigger a EPLB re-balance.",
        )
        parser.add_argument(
            "--eplb-rebalance-layers-per-chunk",
            type=int,
            default=ServerArgs.eplb_rebalance_layers_per_chunk,
            help="Number of layers to rebalance per forward pass.",
        )
        parser.add_argument(
            "--eplb-min-rebalancing-utilization-threshold",
            type=float,
            default=ServerArgs.eplb_min_rebalancing_utilization_threshold,
            help="Minimum threshold for GPU average utilization to trigger EPLB rebalancing. Must be in the range [0.0, 1.0].",
        )
        parser.add_argument(
            "--expert-distribution-recorder-mode",
            type=str,
            default=ServerArgs.expert_distribution_recorder_mode,
            help="Mode of expert distribution recorder.",
        )
        parser.add_argument(
            "--expert-distribution-recorder-buffer-size",
            type=int,
            default=ServerArgs.expert_distribution_recorder_buffer_size,
            help="Circular buffer size of expert distribution recorder. Set to -1 to denote infinite buffer.",
        )
        parser.add_argument(
            "--enable-expert-distribution-metrics",
            action="store_true",
            help="Enable logging metrics for expert balancedness",
        )
        parser.add_argument(
            "--deepep-config",
            type=str,
            default=ServerArgs.deepep_config,
            help="Tuned DeepEP config suitable for your own cluster. It can be either a string with JSON content or a file path.",
        )
        parser.add_argument(
            "--moe-dense-tp-size",
            type=int,
            default=ServerArgs.moe_dense_tp_size,
            help="TP size for MoE dense MLP layers. This flag is useful when, with large TP size, there are errors caused by weights in MLP layers having dimension smaller than the min dimension GEMM supports.",
        )
        parser.add_argument(
            "--elastic-ep-backend",
            type=str,
            default=ServerArgs.elastic_ep_backend,
            choices=["none", "mooncake", "nixl"],
            help="Specify the collective communication backend for elastic EP. Supports 'mooncake' and 'nixl'.",
        )
        parser.add_argument(
            "--enable-elastic-expert-backup",
            action="store_true",
            default=ServerArgs.enable_elastic_expert_backup,
            help="Enable elastic expert backup feature.",
        )
        parser.add_argument(
            "--mooncake-ib-device",
            type=str,
            default=ServerArgs.mooncake_ib_device,
            help="The InfiniBand devices for Mooncake Backend transfer, accepts multiple comma-separated devices "
            "(e.g., --mooncake-ib-device mlx5_0,mlx5_1). "
            "Default is None, which triggers automatic device detection when Mooncake Backend is enabled.",
        )

        # Context parallelism
        parser.add_argument(
            "--enable-nsa-prefill-context-parallel",
            action="store_true",
            help="Enable context parallelism used in the long sequence prefill phase of DeepSeek v3.2.",
        )
        parser.add_argument(
            "--nsa-prefill-cp-mode",
            type=str,
            default=ServerArgs.nsa_prefill_cp_mode,
            choices=NSA_PREFILL_CP_SPLIT_CHOICES,
            help="Token splitting mode for the prefill phase of DeepSeek v3.2 under context parallelism. Optional values: 'round-robin-split'(default), 'in-seq-split'  "
            "'round-robin-split' distributes tokens across ranks based on token_idx %% cp_size. It supports multi-batch prefill, fused MoE, and FP8 KV cache.",
        )
        parser.add_argument(
            "--enable-prefill-context-parallel",
            action="store_true",
            help="Enable context parallelism used in the prefill phase",
        )
        parser.add_argument(
            "--prefill-cp-mode",
            type=str,
            default=ServerArgs.prefill_cp_mode,
            choices=PREFILL_CP_SPLIT_CHOICES,
            help="Token splitting mode for the prefill phase under context parallelism. Optional values: 'in-seq-split' (default)",
        )
