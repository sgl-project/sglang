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
"""CLI argument definitions for miscellaneous options."""

import argparse
import json


class MiscArgs:
    """CLI argument definitions for miscellaneous server options."""

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        from sglang.srt.server_args import ServerArgs
        from sglang.srt.utils.common import json_list_type

        # Ktransformer server args
        parser.add_argument(
            "--kt-weight-path",
            type=str,
            help="[ktransformers parameter] The path of the quantized expert weights for amx kernel. A local folder.",
        )
        parser.add_argument(
            "--kt-method",
            type=str,
            default="AMXINT4",
            help="[ktransformers parameter] Quantization formats for CPU execution.",
        )
        parser.add_argument(
            "--kt-cpuinfer",
            type=int,
            help="[ktransformers parameter] The number of CPUInfer threads.",
        )
        parser.add_argument(
            "--kt-threadpool-count",
            type=int,
            default=2,
            help="[ktransformers parameter] One-to-one with the number of NUMA nodes (one thread pool per NUMA).",
        )
        parser.add_argument(
            "--kt-num-gpu-experts",
            type=int,
            help="[ktransformers parameter] The number of GPU experts.",
        )
        parser.add_argument(
            "--kt-max-deferred-experts-per-token",
            type=int,
            default=ServerArgs.kt_max_deferred_experts_per_token,
            help="[ktransformers parameter] Maximum number of experts deferred to CPU per token. All MoE layers except the final one use this value; the final layer always uses 0.",
        )

        # Diffusion LLM
        parser.add_argument(
            "--dllm-algorithm",
            type=str,
            default=ServerArgs.dllm_algorithm,
            help="The diffusion LLM algorithm, such as LowConfidence.",
        )
        parser.add_argument(
            "--dllm-algorithm-config",
            type=str,
            default=ServerArgs.dllm_algorithm_config,
            help="The diffusion LLM algorithm configurations. Must be a YAML file.",
        )

        # Double Sparsity
        parser.add_argument(
            "--enable-double-sparsity",
            action="store_true",
            help="Enable double sparsity attention",
        )
        parser.add_argument(
            "--ds-channel-config-path",
            type=str,
            default=ServerArgs.ds_channel_config_path,
            help="The path of the double sparsity channel config",
        )
        parser.add_argument(
            "--ds-heavy-channel-num",
            type=int,
            default=ServerArgs.ds_heavy_channel_num,
            help="The number of heavy channels in double sparsity attention",
        )
        parser.add_argument(
            "--ds-heavy-token-num",
            type=int,
            default=ServerArgs.ds_heavy_token_num,
            help="The number of heavy tokens in double sparsity attention",
        )
        parser.add_argument(
            "--ds-heavy-channel-type",
            type=str,
            default=ServerArgs.ds_heavy_channel_type,
            help="The type of heavy channels in double sparsity attention",
        )
        parser.add_argument(
            "--ds-sparse-decode-threshold",
            type=int,
            default=ServerArgs.ds_sparse_decode_threshold,
            help="The minimum decode sequence length required before the double-sparsity backend switches from the dense fallback to the sparse decode kernel.",
        )

        # Offloading
        parser.add_argument(
            "--cpu-offload-gb",
            type=int,
            default=ServerArgs.cpu_offload_gb,
            help="How many GBs of RAM to reserve for CPU offloading.",
        )
        parser.add_argument(
            "--offload-group-size",
            type=int,
            default=ServerArgs.offload_group_size,
            help="Number of layers per group in offloading.",
        )
        parser.add_argument(
            "--offload-num-in-group",
            type=int,
            default=ServerArgs.offload_num_in_group,
            help="Number of layers to be offloaded within a group.",
        )
        parser.add_argument(
            "--offload-prefetch-step",
            type=int,
            default=ServerArgs.offload_prefetch_step,
            help="Steps to prefetch in offloading.",
        )
        parser.add_argument(
            "--offload-mode",
            type=str,
            default=ServerArgs.offload_mode,
            help="Mode of offloading.",
        )

        # Args for multi-item-scoring
        parser.add_argument(
            "--multi-item-scoring-delimiter",
            type=int,
            default=ServerArgs.multi_item_scoring_delimiter,
            help="Delimiter token ID for multi-item scoring. Used to combine Query and Items into a single sequence: Query<delimiter>Item1<delimiter>Item2<delimiter>... This enables efficient batch processing of multiple items against a single query.",
        )

        # For PD-Multiplexing
        parser.add_argument(
            "--enable-pdmux",
            action="store_true",
            help="Enable PD-Multiplexing, PD running on greenctx stream.",
        )
        parser.add_argument(
            "--pdmux-config-path",
            type=str,
            default=None,
            help="The path of the PD-Multiplexing config file.",
        )
        parser.add_argument(
            "--sm-group-num",
            type=int,
            default=ServerArgs.sm_group_num,
            help="Number of sm partition groups.",
        )

        # Configuration file support
        parser.add_argument(
            "--config",
            type=str,
            help="Read CLI options from a config file. Must be a YAML file with configuration options.",
        )

        # For Multi-Modal
        parser.add_argument(
            "--enable-broadcast-mm-inputs-process",
            action="store_true",
            default=ServerArgs.enable_broadcast_mm_inputs_process,
            help="Enable broadcast mm-inputs process in scheduler.",
        )
        parser.add_argument(
            "--mm-process-config",
            type=json.loads,
            default=ServerArgs.mm_process_config,
            help="Multimodal preprocessing config, a json config contains keys: `image`, `video`, `audio`",
        )
        parser.add_argument(
            "--mm-enable-dp-encoder",
            action="store_true",
            default=ServerArgs.mm_enable_dp_encoder,
            help="Enabling data parallelism for mm encoder. The dp size will be set to the tp size automatically.",
        )
        parser.add_argument(
            "--limit-mm-data-per-request",
            type=json.loads,
            default=ServerArgs.limit_mm_data_per_request,
            help="Limit the number of multimodal inputs per request. "
            'e.g. \'{"image": 1, "video": 1, "audio": 1}\'',
        )

        # For checkpoint decryption
        parser.add_argument(
            "--decrypted-config-file",
            type=str,
            default=ServerArgs.decrypted_config_file,
            help="The path of the decrypted config file.",
        )
        parser.add_argument(
            "--decrypted-draft-config-file",
            type=str,
            default=ServerArgs.decrypted_draft_config_file,
            help="The path of the decrypted draft config file.",
        )
        parser.add_argument(
            "--enable-prefix-mm-cache",
            action="store_true",
            default=ServerArgs.enable_prefix_mm_cache,
            help="Enable prefix multimodal cache. Currently only supports mm-only.",
        )

        parser.add_argument(
            "--enable-mm-global-cache",
            action="store_true",
            default=ServerArgs.enable_mm_global_cache,
            help="Enable global multimodal embedding cache to skip redundant ViT inference.",
        )

        # For registering hooks
        parser.add_argument(
            "--forward-hooks",
            type=json_list_type,
            default=ServerArgs.forward_hooks,
            help="JSON-formatted forward hook specifications to attach to the model.",
        )
