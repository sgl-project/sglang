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
"""The arguments of the server."""

import argparse
import dataclasses
import json
import logging
import os
import random
import tempfile
from typing import List, Literal, Optional, Union

from sglang.srt.hf_transformers_utils import check_gguf_file, get_config
from sglang.srt.reasoning_parser import ReasoningParser
from sglang.srt.utils import (
    configure_ipv6,
    get_device,
    get_device_memory_capacity,
    is_flashinfer_available,
    is_hip,
    is_port_available,
    is_remote_url,
    is_valid_ipv6_address,
    nullable_str,
)

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class ServerArgs:
    # Model and tokenizer
    model_path: str
    tokenizer_path: Optional[str] = None
    tokenizer_mode: str = "auto"
    skip_tokenizer_init: bool = False
    skip_server_warmup: bool = False
    load_format: str = "auto"
    model_loader_extra_config: str = "{}"
    trust_remote_code: bool = False
    dtype: str = "auto"
    kv_cache_dtype: str = "auto"
    quantization: Optional[str] = None
    quantization_param_path: Optional[str] = None
    context_length: Optional[int] = None
    device: Optional[str] = None
    served_model_name: Optional[str] = None
    chat_template: Optional[str] = None
    completion_template: Optional[str] = None
    is_embedding: bool = False
    enable_multimodal: Optional[bool] = None
    revision: Optional[str] = None
    hybrid_kvcache_ratio: Optional[float] = None
    impl: str = "auto"

    # Port for the HTTP server
    host: str = "127.0.0.1"
    port: int = 30000

    # Memory and scheduling
    mem_fraction_static: Optional[float] = None
    max_running_requests: Optional[int] = None
    max_total_tokens: Optional[int] = None
    chunked_prefill_size: Optional[int] = None
    max_prefill_tokens: int = 16384
    schedule_policy: str = "fcfs"
    schedule_conservativeness: float = 1.0
    cpu_offload_gb: int = 0
    page_size: int = 1

    # Other runtime options
    tp_size: int = 1
    pp_size: int = 1
    max_micro_batch_size: Optional[int] = None
    stream_interval: int = 1
    stream_output: bool = False
    random_seed: Optional[int] = None
    constrained_json_whitespace_pattern: Optional[str] = None
    watchdog_timeout: float = 300
    dist_timeout: Optional[int] = None  # timeout for torch.distributed
    download_dir: Optional[str] = None
    base_gpu_id: int = 0
    gpu_id_step: int = 1
    sleep_on_idle: bool = False

    # Logging
    log_level: str = "info"
    log_level_http: Optional[str] = None
    log_requests: bool = False
    log_requests_level: int = 0
    crash_dump_folder: Optional[str] = None
    show_time_cost: bool = False
    enable_metrics: bool = False
    bucket_time_to_first_token: Optional[List[float]] = None
    bucket_e2e_request_latency: Optional[List[float]] = None
    bucket_inter_token_latency: Optional[List[float]] = None
    collect_tokens_histogram: bool = False
    decode_log_interval: int = 40
    enable_request_time_stats_logging: bool = False
    kv_events_config: Optional[str] = None

    # API related
    api_key: Optional[str] = None
    file_storage_path: str = "sglang_storage"
    enable_cache_report: bool = False
    reasoning_parser: Optional[str] = None
    tool_call_parser: Optional[str] = None

    # Data parallelism
    dp_size: int = 1
    load_balance_method: str = "round_robin"

    # Multi-node distributed serving
    dist_init_addr: Optional[str] = None
    nnodes: int = 1
    node_rank: int = 0

    # Model override args in JSON
    json_model_override_args: str = "{}"
    preferred_sampling_params: Optional[str] = None

    # LoRA
    lora_paths: Optional[Union[dict[str, str], List[str]]] = None
    max_loras_per_batch: int = 8
    lora_backend: str = "triton"

    # Kernel backend
    attention_backend: Optional[str] = None
    sampling_backend: Optional[str] = None
    grammar_backend: Optional[str] = None
    mm_attention_backend: Optional[str] = None

    # Speculative decoding
    speculative_algorithm: Optional[str] = None
    speculative_draft_model_path: Optional[str] = None
    speculative_num_steps: Optional[int] = None
    speculative_eagle_topk: Optional[int] = None
    speculative_num_draft_tokens: Optional[int] = None
    speculative_accept_threshold_single: float = 1.0
    speculative_accept_threshold_acc: float = 1.0
    speculative_token_map: Optional[str] = None

    # Expert parallelism
    ep_size: int = 1
    enable_ep_moe: bool = False
    enable_deepep_moe: bool = False
    enable_flashinfer_moe: bool = False
    deepep_mode: Optional[Literal["auto", "normal", "low_latency"]] = "auto"
    ep_num_redundant_experts: int = 0
    ep_dispatch_algorithm: Optional[Literal["static", "dynamic", "fake"]] = None
    init_expert_location: str = "trivial"
    enable_eplb: bool = False
    eplb_algorithm: str = "auto"
    eplb_rebalance_num_iterations: int = 1000
    eplb_rebalance_layers_per_chunk: Optional[int] = None
    expert_distribution_recorder_mode: Optional[
        Literal["stat", "stat_approx", "per_pass", "per_token"]
    ] = None
    expert_distribution_recorder_buffer_size: Optional[int] = None
    enable_expert_distribution_metrics: bool = False
    deepep_config: Optional[str] = None
    moe_dense_tp_size: Optional[int] = None

    # Double Sparsity
    enable_double_sparsity: bool = False
    ds_channel_config_path: Optional[str] = None
    ds_heavy_channel_num: int = 32
    ds_heavy_token_num: int = 256
    ds_heavy_channel_type: str = "qk"
    ds_sparse_decode_threshold: int = 4096

    # Optimization/debug options
    disable_radix_cache: bool = False
    cuda_graph_max_bs: Optional[int] = None
    cuda_graph_bs: Optional[List[int]] = None
    disable_cuda_graph: bool = False
    disable_cuda_graph_padding: bool = False
    enable_profile_cuda_graph: bool = False
    enable_nccl_nvls: bool = False
    enable_tokenizer_batch_encode: bool = False
    disable_outlines_disk_cache: bool = False
    disable_custom_all_reduce: bool = False
    enable_mscclpp: bool = False
    disable_overlap_schedule: bool = False
    disable_overlap_cg_plan: bool = False
    enable_mixed_chunk: bool = False
    enable_dp_attention: bool = False
    enable_dp_lm_head: bool = False
    enable_two_batch_overlap: bool = False
    enable_torch_compile: bool = False
    torch_compile_max_bs: int = 32
    torchao_config: str = ""
    enable_nan_detection: bool = False
    enable_p2p_check: bool = False
    triton_attention_reduce_in_fp32: bool = False
    triton_attention_num_kv_splits: int = 8
    num_continuous_decode_steps: int = 1
    delete_ckpt_after_loading: bool = False
    enable_memory_saver: bool = False
    allow_auto_truncate: bool = False
    enable_custom_logit_processor: bool = False
    enable_hierarchical_cache: bool = False
    hicache_ratio: float = 2.0
    hicache_size: int = 0
    hicache_write_policy: str = "write_through_selective"
    flashinfer_mla_disable_ragged: bool = False
    disable_shared_experts_fusion: bool = False
    disable_chunked_prefix_cache: bool = False
    disable_fast_image_processor: bool = False
    enable_return_hidden_states: bool = False
    warmups: Optional[str] = None

    # Debug tensor dumps
    debug_tensor_dump_output_folder: Optional[str] = None
    debug_tensor_dump_input_file: Optional[str] = None
    debug_tensor_dump_inject: bool = False
    debug_tensor_dump_prefill_only: bool = False

    # For PD disaggregation: can be "null" (not disaggregated), "prefill" (prefill-only), or "decode" (decode-only)
    disaggregation_mode: str = "null"
    disaggregation_transfer_backend: str = "mooncake"
    disaggregation_bootstrap_port: int = 8998
    disaggregation_decode_tp: Optional[int] = None
    disaggregation_decode_dp: Optional[int] = None
    disaggregation_prefill_pp: Optional[int] = 1
    disaggregation_ib_device: Optional[str] = None
    num_reserved_decode_tokens: int = 512  # used for decode kv cache offload in PD
    pdlb_url: Optional[str] = None

    # For model weight update
    custom_weight_loader: Optional[List[str]] = None
    weight_loader_disable_mmap: bool = False

    def __post_init__(self):
        # Expert parallelism
        if self.enable_ep_moe:
            self.ep_size = self.tp_size
            logger.warning(
                f"EP MoE is enabled. The expert parallel size is adjusted to be the same as the tensor parallel size[{self.tp_size}]."
            )
        if self.enable_flashinfer_moe:
            assert (
                self.quantization == "modelopt_fp4"
            ), "modelopt_fp4 quantization is required for Flashinfer MOE"
            os.environ["TRTLLM_ENABLE_PDL"] = "1"
            self.disable_shared_experts_fusion = True
            logger.warning(
                f"Flashinfer MoE is enabled. Shared expert fusion is disabled."
            )
        # Set missing default values
        if self.tokenizer_path is None:
            self.tokenizer_path = self.model_path

        if self.device is None:
            self.device = get_device()

        if self.served_model_name is None:
            self.served_model_name = self.model_path

        if self.random_seed is None:
            self.random_seed = random.randint(0, 1 << 30)

        gpu_mem = get_device_memory_capacity(self.device)

        # Set mem fraction static
        if self.mem_fraction_static is None:
            if gpu_mem is not None:
                # GPU memory capacity = model weights + KV cache pool + activations + cuda graph buffers
                # mem_fraction_static = (model weights + KV cache pool) / GPU memory capacity.

                # We want mem_fraction_static to be as large as possible but still has enough room
                # for activations and cuda graph buffers. We use the following heuristic to
                # compute the needed size for activations and cuda graph buffers:
                # - The size of the activation depends on the chunked_prefill_size and model size.
                # - The size of cuda graph buffers depends on the cuda graph capture range and model size.
                # For GPUs with more memory, we use a larger chunked_prefill_size and
                # capture more cuda graphs, so they need to reserve more memory.
                parallel_size = self.tp_size * self.pp_size

                if gpu_mem < 20 * 1024:
                    # T4, 4080. (chunked_prefill_size 2k, cuda_graph_max_bs 8)
                    reserved_mem = (2.8 + parallel_size / 10) * 1024
                elif gpu_mem < 35 * 1024:
                    # A10, L40, 4090, 5090. (chunked_prefill_size 2k, cuda_graph_max_bs 8)
                    reserved_mem = (2.8 + parallel_size / 10) * 1024
                elif gpu_mem < 90 * 1024:
                    # H100, A100. (chunked_prefill_size 8k, cuda_graph_max_bs 160)
                    reserved_mem = (9.5 + parallel_size / 2) * 1024
                elif gpu_mem < 100 * 1024:
                    # H20. (chunked_prefill_size 8k, cuda_graph_max_bs 256)
                    reserved_mem = (12 + parallel_size / 2) * 1024
                elif gpu_mem < 160 * 1024:
                    # H200. (chunked_prefill_size 8k, cuda_graph_max_bs 256)
                    reserved_mem = (12 + parallel_size / 2) * 1024
                else:
                    # B200, MI300. (chunked_prefill_size 16k, cuda_graph_max_bs 512)
                    reserved_mem = 32 * 1024

                if self.speculative_algorithm is not None:
                    # draft model and larger cuda graph buffers
                    reserved_mem += 2 * 1024
                if self.enable_dp_attention:
                    reserved_mem += 4 * 1024

                self.mem_fraction_static = round((gpu_mem - reserved_mem) / gpu_mem, 3)
            else:
                self.mem_fraction_static = 0.88

        # Set chunked prefill size, which depends on the gpu memory capacity
        if self.chunked_prefill_size is None:
            if gpu_mem is not None:
                if gpu_mem < 35 * 1024:  # A10, L40, 4090
                    self.chunked_prefill_size = 2048
                elif gpu_mem < 160 * 1024:  # H100, H200, A100, H20
                    self.chunked_prefill_size = 8192
                else:  # B200, MI300
                    self.chunked_prefill_size = 16384
            else:
                self.chunked_prefill_size = 4096
        assert self.chunked_prefill_size % self.page_size == 0

        # Set cuda graph max batch size
        if self.cuda_graph_max_bs is None:
            # Based on detailed statistics, when serving TP1/TP2 models on lower-end GPUs with HBM<25G, you can either disable cuda graph or set `cuda_graph_max_bs` to a very small value to reduce the memory overhead of creating cuda graphs, with almost no impact on performance. However, when serving models with TP4 or TP8, we need to enable cuda graph to maintain high performance. In this case, we can set `cuda_graph_max_bs` to 80 (half of the default value 160) to reduce the memory overhead of creating cuda graphs. Looking at the logs from TP4 serving of qwen2-72b, a value of 80 is sufficient and can reduce the memory overhead of creating cuda graphs on lower-end GPUs compared to the original 160, avoiding OOM issues.
            if gpu_mem is not None and gpu_mem < 35 * 1024:
                if self.tp_size < 4:
                    self.cuda_graph_max_bs = 8
                else:
                    self.cuda_graph_max_bs = 80

        assert self.moe_dense_tp_size in {
            1,
            None,
        }, "moe_dense_tp_size only support 1 and None currently"

        if self.attention_backend == "flashmla":
            logger.warning(
                "FlashMLA only supports a page_size of 64, change page_size to 64."
            )
            self.page_size = 64

        if self.attention_backend == "cutlass_mla":
            logger.warning(
                "Cutlass MLA only supports a page_size of 128, change page_size to 128."
            )
            self.page_size = 128

        # Set kernel backends for hpu device
        if self.device == "hpu":
            self.attention_backend = "torch_native"
            self.sampling_backend = "pytorch"

        # Set kernel backends
        if self.device == "cpu":
            if self.attention_backend is None:
                self.attention_backend = "intel_amx"
            self.sampling_backend = "pytorch"

        if self.sampling_backend is None:
            self.sampling_backend = (
                "flashinfer" if is_flashinfer_available() else "pytorch"
            )

        if self.attention_backend == "torch_native":
            logger.warning(
                "Cuda graph is disabled because of using torch native attention backend"
            )
            self.disable_cuda_graph = True

        # Choose grammar backend
        if self.grammar_backend is None:
            self.grammar_backend = "xgrammar"

        # Data parallelism attention
        if self.enable_dp_attention:
            self.schedule_conservativeness = self.schedule_conservativeness * 0.3
            assert (
                self.dp_size > 1
            ), "Please set a dp-size > 1. You can use 1 < dp-size <= tp-size "
            assert self.tp_size % self.dp_size == 0
            self.chunked_prefill_size = self.chunked_prefill_size // self.dp_size
            logger.warning(
                f"DP attention is enabled. The chunked prefill size is adjusted to {self.chunked_prefill_size} to avoid MoE kernel issues. "
            )

        if self.enable_dp_lm_head:
            assert (
                self.enable_dp_attention
            ), "Please enable dp attention when setting enable_dp_attention. "

        # DeepEP MoE
        if self.enable_deepep_moe:
            if self.deepep_mode == "auto":
                assert (
                    not self.enable_dp_attention
                ), "DeepEP MoE `auto` mode is not supported with DP Attention."
            if self.deepep_mode == "normal":
                logger.warning("Cuda graph is disabled because deepep_mode=`normal`")
                self.disable_cuda_graph = True
            self.ep_size = self.tp_size
            logger.warning(
                f"DeepEP MoE is enabled. The expert parallel size is adjusted to be the same as the tensor parallel size[{self.tp_size}]."
            )

        if self.pp_size > 1:
            self.disable_overlap_schedule = True
            logger.warning(
                "Pipeline parallelism is incompatible with overlap schedule."
            )

        if self.enable_eplb and (self.expert_distribution_recorder_mode is None):
            self.expert_distribution_recorder_mode = "stat"
            logger.info(
                "EPLB is enabled. The expert_distribution_recorder_mode is automatically set."
            )

        if (self.enable_eplb or (self.init_expert_location is not None)) and (
            self.ep_dispatch_algorithm is None
        ):
            self.ep_dispatch_algorithm = "static"
            logger.info(
                "EPLB is enabled or init_expert_location is provided. ep_dispatch_algorithm is configured."
            )

        if self.enable_expert_distribution_metrics and (
            self.expert_distribution_recorder_mode is None
        ):
            self.expert_distribution_recorder_mode = "stat"

        if self.expert_distribution_recorder_buffer_size is None:
            if (x := self.eplb_rebalance_num_iterations) is not None:
                self.expert_distribution_recorder_buffer_size = x
            elif self.expert_distribution_recorder_mode is not None:
                self.expert_distribution_recorder_buffer_size = 1000

        # Speculative Decoding
        if self.speculative_algorithm == "NEXTN":
            # NEXTN shares the same implementation of EAGLE
            self.speculative_algorithm = "EAGLE"

        if self.speculative_algorithm in ("EAGLE", "EAGLE3"):
            if self.max_running_requests is None:
                self.max_running_requests = 48
            self.disable_overlap_schedule = True
            logger.warning(
                "Overlap scheduler is disabled because of using "
                "eagle speculative decoding."
            )
            if self.enable_mixed_chunk:
                self.enable_mixed_chunk = False
                logger.warning(
                    "Mixed chunked prefill is disabled because of using "
                    "eagle speculative decoding."
                )

            model_arch = get_model_arch(self)

            # Auto set draft_model_path DeepSeek-V3/R1
            if model_arch == "DeepseekV3ForCausalLM":
                if self.speculative_draft_model_path is None:
                    self.speculative_draft_model_path = self.model_path
                else:
                    logger.warning(
                        "DeepSeek MTP does not require setting speculative_draft_model_path."
                    )

            # Auto choose parameters
            if self.speculative_num_steps is None:
                assert (
                    self.speculative_eagle_topk is None
                    and self.speculative_num_draft_tokens is None
                )
                (
                    self.speculative_num_steps,
                    self.speculative_eagle_topk,
                    self.speculative_num_draft_tokens,
                ) = auto_choose_speculative_params(self)

            if (
                self.speculative_eagle_topk == 1
                and self.speculative_num_draft_tokens != self.speculative_num_steps + 1
            ):
                logger.warning(
                    "speculative_num_draft_tokens is adjusted to speculative_num_steps + 1 when speculative_eagle_topk == 1"
                )
                self.speculative_num_draft_tokens = self.speculative_num_steps + 1

            # The token generated from the verify step is counted.
            # If sepculative_num_steps >= speculative_num_draft_tokens, the additional tokens will definitely be discarded.
            # assert self.speculative_num_steps < self.speculative_num_draft_tokens

        # GGUF
        if (
            self.load_format == "auto" or self.load_format == "gguf"
        ) and check_gguf_file(self.model_path):
            self.quantization = self.load_format = "gguf"

        if is_remote_url(self.model_path):
            self.load_format = "remote"

        # AMD-specific Triton attention KV splits default number
        if is_hip():
            self.triton_attention_num_kv_splits = 16

        # PD disaggregation
        if self.disaggregation_mode == "decode":
            assert (
                self.disaggregation_decode_tp is None
            ), "Cannot set --disaggregation-decode-tp for the decode engine."
            assert (
                self.disaggregation_decode_dp is None
            ), "Cannot set --disaggregation-decode-dp for the decode engine."

            self.disable_radix_cache = True
            logger.warning("KV cache is forced as chunk cache for decode server")
        elif self.disaggregation_mode == "prefill":
            if self.disaggregation_decode_tp is None:
                self.disaggregation_decode_tp = self.tp_size
            if self.disaggregation_decode_dp is None:
                self.disaggregation_decode_dp = self.dp_size

            self.disaggregation_prefill_pp = self.pp_size
            self.validate_disagg_tp_size(self.tp_size, self.disaggregation_decode_tp)

            self.disable_cuda_graph = True
            logger.warning("Cuda graph is disabled for prefill server")

        os.environ["SGLANG_ENABLE_TORCH_COMPILE"] = (
            "1" if self.enable_torch_compile else "0"
        )
        # Set env var before grammar backends init
        os.environ["SGLANG_DISABLE_OUTLINES_DISK_CACHE"] = (
            "1" if self.disable_outlines_disk_cache else "0"
        )

        if self.custom_weight_loader is None:
            self.custom_weight_loader = []

    def validate_disagg_tp_size(self, prefill_tp: int, decode_tp: int):
        larger_tp = max(decode_tp, prefill_tp)
        smaller_tp = min(decode_tp, prefill_tp)
        assert larger_tp % smaller_tp == 0, (
            "Different tp size is supported only when one tp is multiple of the other. "
            f"decode_tp={decode_tp}, prefill_tp={prefill_tp}"
        )

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser):
        # Model and port args
        parser.add_argument(
            "--model-path",
            "--model",
            type=str,
            help="The path of the model weights. This can be a local folder or a Hugging Face repo ID.",
            required=True,
        )
        parser.add_argument(
            "--tokenizer-path",
            type=str,
            default=ServerArgs.tokenizer_path,
            help="The path of the tokenizer.",
        )
        parser.add_argument(
            "--host",
            type=str,
            default=ServerArgs.host,
            help="The host of the HTTP server.",
        )
        parser.add_argument(
            "--port",
            type=int,
            default=ServerArgs.port,
            help="The port of the HTTP server.",
        )
        parser.add_argument(
            "--tokenizer-mode",
            type=str,
            default=ServerArgs.tokenizer_mode,
            choices=["auto", "slow"],
            help="Tokenizer mode. 'auto' will use the fast "
            "tokenizer if available, and 'slow' will "
            "always use the slow tokenizer.",
        )
        parser.add_argument(
            "--skip-tokenizer-init",
            action="store_true",
            help="If set, skip init tokenizer and pass input_ids in generate request.",
        )
        parser.add_argument(
            "--skip-server-warmup",
            action="store_true",
            help="If set, skip warmup.",
        )
        parser.add_argument(
            "--load-format",
            type=str,
            default=ServerArgs.load_format,
            choices=[
                "auto",
                "pt",
                "safetensors",
                "npcache",
                "dummy",
                "sharded_state",
                "gguf",
                "bitsandbytes",
                "layered",
                "remote",
            ],
            help="The format of the model weights to load. "
            '"auto" will try to load the weights in the safetensors format '
            "and fall back to the pytorch bin format if safetensors format "
            "is not available. "
            '"pt" will load the weights in the pytorch bin format. '
            '"safetensors" will load the weights in the safetensors format. '
            '"npcache" will load the weights in pytorch format and store '
            "a numpy cache to speed up the loading. "
            '"dummy" will initialize the weights with random values, '
            "which is mainly for profiling."
            '"gguf" will load the weights in the gguf format. '
            '"bitsandbytes" will load the weights using bitsandbytes '
            "quantization."
            '"layered" loads weights layer by layer so that one can quantize a '
            "layer before loading another to make the peak memory envelope "
            "smaller.",
        )
        parser.add_argument(
            "--model-loader-extra-config",
            type=str,
            help="Extra config for model loader. "
            "This will be passed to the model loader corresponding to the chosen load_format.",
            default=ServerArgs.model_loader_extra_config,
        )
        parser.add_argument(
            "--trust-remote-code",
            action="store_true",
            help="Whether or not to allow for custom models defined on the Hub in their own modeling files.",
        )
        parser.add_argument(
            "--dtype",
            type=str,
            default=ServerArgs.dtype,
            choices=["auto", "half", "float16", "bfloat16", "float", "float32"],
            help="Data type for model weights and activations.\n\n"
            '* "auto" will use FP16 precision for FP32 and FP16 models, and '
            "BF16 precision for BF16 models.\n"
            '* "half" for FP16. Recommended for AWQ quantization.\n'
            '* "float16" is the same as "half".\n'
            '* "bfloat16" for a balance between precision and range.\n'
            '* "float" is shorthand for FP32 precision.\n'
            '* "float32" for FP32 precision.',
        )
        parser.add_argument(
            "--kv-cache-dtype",
            type=str,
            default=ServerArgs.kv_cache_dtype,
            choices=["auto", "fp8_e5m2", "fp8_e4m3"],
            help='Data type for kv cache storage. "auto" will use model data type. "fp8_e5m2" and "fp8_e4m3" is supported for CUDA 11.8+.',
        )
        parser.add_argument(
            "--quantization",
            type=str,
            default=ServerArgs.quantization,
            choices=[
                "awq",
                "fp8",
                "gptq",
                "marlin",
                "gptq_marlin",
                "awq_marlin",
                "bitsandbytes",
                "gguf",
                "modelopt",
                "modelopt_fp4",
                "w8a8_int8",
                "w8a8_fp8",
                "moe_wna16",
                "qoq",
            ],
            help="The quantization method.",
        )
        parser.add_argument(
            "--quantization-param-path",
            type=nullable_str,
            default=None,
            help="Path to the JSON file containing the KV cache "
            "scaling factors. This should generally be supplied, when "
            "KV cache dtype is FP8. Otherwise, KV cache scaling factors "
            "default to 1.0, which may cause accuracy issues. ",
        )
        parser.add_argument(
            "--context-length",
            type=int,
            default=ServerArgs.context_length,
            help="The model's maximum context length. Defaults to None (will use the value from the model's config.json instead).",
        )
        parser.add_argument(
            "--device",
            type=str,
            default=ServerArgs.device,
            help="The device to use ('cuda', 'xpu', 'hpu', 'npu', 'cpu'). Defaults to auto-detection if not specified.",
        )
        parser.add_argument(
            "--served-model-name",
            type=str,
            default=ServerArgs.served_model_name,
            help="Override the model name returned by the v1/models endpoint in OpenAI API server.",
        )
        parser.add_argument(
            "--chat-template",
            type=str,
            default=ServerArgs.chat_template,
            help="The buliltin chat template name or the path of the chat template file. This is only used for OpenAI-compatible API server.",
        )
        parser.add_argument(
            "--completion-template",
            type=str,
            default=ServerArgs.completion_template,
            help="The buliltin completion template name or the path of the completion template file. This is only used for OpenAI-compatible API server. only for code completion currently.",
        )
        parser.add_argument(
            "--is-embedding",
            action="store_true",
            help="Whether to use a CausalLM as an embedding model.",
        )
        parser.add_argument(
            "--enable-multimodal",
            default=ServerArgs.enable_multimodal,
            action="store_true",
            help="Enable the multimodal functionality for the served model. If the model being served is not multimodal, nothing will happen",
        )
        parser.add_argument(
            "--revision",
            type=str,
            default=None,
            help="The specific model version to use. It can be a branch "
            "name, a tag name, or a commit id. If unspecified, will use "
            "the default version.",
        )
        parser.add_argument(
            "--impl",
            type=str,
            default=ServerArgs.impl,
            help="Which implementation of the model to use.\n\n"
            '* "auto" will try to use the SGLang implementation if it exists '
            "and fall back to the Transformers implementation if no SGLang "
            "implementation is available.\n"
            '* "sglang" will use the SGLang model implementation.\n'
            '* "transformers" will use the Transformers model '
            "implementation.\n",
        )

        # Memory and scheduling
        parser.add_argument(
            "--mem-fraction-static",
            type=float,
            default=ServerArgs.mem_fraction_static,
            help="The fraction of the memory used for static allocation (model weights and KV cache memory pool). Use a smaller value if you see out-of-memory errors.",
        )
        parser.add_argument(
            "--max-running-requests",
            type=int,
            default=ServerArgs.max_running_requests,
            help="The maximum number of running requests.",
        )
        parser.add_argument(
            "--max-total-tokens",
            type=int,
            default=ServerArgs.max_total_tokens,
            help="The maximum number of tokens in the memory pool. If not specified, it will be automatically calculated based on the memory usage fraction. "
            "This option is typically used for development and debugging purposes.",
        )
        parser.add_argument(
            "--chunked-prefill-size",
            type=int,
            default=ServerArgs.chunked_prefill_size,
            help="The maximum number of tokens in a chunk for the chunked prefill. Setting this to -1 means disabling chunked prefill.",
        )
        parser.add_argument(
            "--max-prefill-tokens",
            type=int,
            default=ServerArgs.max_prefill_tokens,
            help="The maximum number of tokens in a prefill batch. The real bound will be the maximum of this value and the model's maximum context length.",
        )
        parser.add_argument(
            "--schedule-policy",
            type=str,
            default=ServerArgs.schedule_policy,
            choices=["lpm", "random", "fcfs", "dfs-weight"],
            help="The scheduling policy of the requests.",
        )
        parser.add_argument(
            "--schedule-conservativeness",
            type=float,
            default=ServerArgs.schedule_conservativeness,
            help="How conservative the schedule policy is. A larger value means more conservative scheduling. Use a larger value if you see requests being retracted frequently.",
        )
        parser.add_argument(
            "--cpu-offload-gb",
            type=int,
            default=ServerArgs.cpu_offload_gb,
            help="How many GBs of RAM to reserve for CPU offloading.",
        )
        parser.add_argument(
            "--page-size",
            type=int,
            default=ServerArgs.page_size,
            help="The number of tokens in a page.",
        )
        parser.add_argument(
            "--hybrid-kvcache-ratio",
            nargs="?",
            const=0.5,
            type=float,
            default=ServerArgs.hybrid_kvcache_ratio,
            help=(
                "Mix ratio in [0,1] between uniform and hybrid kv buffers "
                "(0.0 = pure uniform: swa_size / full_size = 1)"
                "(1.0 = pure hybrid: swa_size / full_size = local_attention_size / context_length)"
            ),
        )

        # Other runtime options
        parser.add_argument(
            "--tensor-parallel-size",
            "--tp-size",
            type=int,
            default=ServerArgs.tp_size,
            help="The tensor parallelism size.",
        )
        parser.add_argument(
            "--pipeline-parallel-size",
            "--pp-size",
            type=int,
            default=ServerArgs.pp_size,
            help="The pipeline parallelism size.",
        )
        parser.add_argument(
            "--max-micro-batch-size",
            type=int,
            default=ServerArgs.max_micro_batch_size,
            help="The maximum micro batch size in pipeline parallelism.",
        )
        parser.add_argument(
            "--stream-interval",
            type=int,
            default=ServerArgs.stream_interval,
            help="The interval (or buffer size) for streaming in terms of the token length. A smaller value makes streaming smoother, while a larger value makes the throughput higher",
        )
        parser.add_argument(
            "--stream-output",
            action="store_true",
            help="Whether to output as a sequence of disjoint segments.",
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
            help=r"Regex pattern for syntactic whitespaces allowed in JSON constrained output. For example, to allow the model generate consecutive whitespaces, set the pattern to [\n\t ]*",
        )
        parser.add_argument(
            "--watchdog-timeout",
            type=float,
            default=ServerArgs.watchdog_timeout,
            help="Set watchdog timeout in seconds. If a forward batch takes longer than this, the server will crash to prevent hanging.",
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

        # Logging
        parser.add_argument(
            "--log-level",
            type=str,
            default=ServerArgs.log_level,
            help="The logging level of all loggers.",
        )
        parser.add_argument(
            "--log-level-http",
            type=str,
            default=ServerArgs.log_level_http,
            help="The logging level of HTTP server. If not set, reuse --log-level by default.",
        )
        parser.add_argument(
            "--log-requests",
            action="store_true",
            help="Log metadata, inputs, outputs of all requests. The verbosity is decided by --log-requests-level",
        )
        parser.add_argument(
            "--log-requests-level",
            type=int,
            default=0,
            help="0: Log metadata (no sampling parameters). 1: Log metadata and sampling parameters. 2: Log metadata, sampling parameters and partial input/output. 3: Log every input/output.",
            choices=[0, 1, 2, 3],
        )
        parser.add_argument(
            "--crash-dump-folder",
            type=str,
            default=ServerArgs.crash_dump_folder,
            help="Folder path to dump requests from the last 5 min before a crash (if any). If not specified, crash dumping is disabled.",
        )
        parser.add_argument(
            "--show-time-cost",
            action="store_true",
            help="Show time cost of custom marks.",
        )
        parser.add_argument(
            "--enable-metrics",
            action="store_true",
            help="Enable log prometheus metrics.",
        )
        parser.add_argument(
            "--bucket-time-to-first-token",
            type=float,
            nargs="+",
            default=ServerArgs.bucket_time_to_first_token,
            help="The buckets of time to first token, specified as a list of floats.",
        )
        parser.add_argument(
            "--bucket-inter-token-latency",
            type=float,
            nargs="+",
            default=ServerArgs.bucket_inter_token_latency,
            help="The buckets of inter-token latency, specified as a list of floats.",
        )
        parser.add_argument(
            "--bucket-e2e-request-latency",
            type=float,
            nargs="+",
            default=ServerArgs.bucket_e2e_request_latency,
            help="The buckets of end-to-end request latency, specified as a list of floats.",
        )
        parser.add_argument(
            "--collect-tokens-histogram",
            action="store_true",
            default=ServerArgs.collect_tokens_histogram,
            help="Collect prompt/generation tokens histogram.",
        )
        parser.add_argument(
            "--kv-events-config",
            type=str,
            default=None,
            help="Config in json format for NVIDIA dynamo KV event publishing. Publishing will be enabled if this flag is used.",
        )
        parser.add_argument(
            "--decode-log-interval",
            type=int,
            default=ServerArgs.decode_log_interval,
            help="The log interval of decode batch.",
        )
        parser.add_argument(
            "--enable-request-time-stats-logging",
            action="store_true",
            default=ServerArgs.enable_request_time_stats_logging,
            help="Enable per request time stats logging",
        )

        # API related
        parser.add_argument(
            "--api-key",
            type=str,
            default=ServerArgs.api_key,
            help="Set API key of the server. It is also used in the OpenAI API compatible server.",
        )
        parser.add_argument(
            "--file-storage-path",
            type=str,
            default=ServerArgs.file_storage_path,
            help="The path of the file storage in backend.",
        )
        parser.add_argument(
            "--enable-cache-report",
            action="store_true",
            help="Return number of cached tokens in usage.prompt_tokens_details for each openai request.",
        )
        parser.add_argument(
            "--reasoning-parser",
            type=str,
            choices=list(ReasoningParser.DetectorMap.keys()),
            default=ServerArgs.reasoning_parser,
            help=f"Specify the parser for reasoning models, supported parsers are: {list(ReasoningParser.DetectorMap.keys())}.",
        )
        parser.add_argument(
            "--tool-call-parser",
            type=str,
            choices=["qwen25", "mistral", "llama3", "deepseekv3", "pythonic"],
            default=ServerArgs.tool_call_parser,
            help="Specify the parser for handling tool-call interactions. Options include: 'qwen25', 'mistral', 'llama3', 'deepseekv3', and 'pythonic'.",
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
                "round_robin",
                "shortest_queue",
            ],
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

        # Model override args
        parser.add_argument(
            "--json-model-override-args",
            type=str,
            help="A dictionary in JSON string format used to override default model configurations.",
            default=ServerArgs.json_model_override_args,
        )
        parser.add_argument(
            "--preferred-sampling-params",
            type=str,
            help="json-formatted sampling settings that will be returned in /get_model_info",
        )

        # LoRA
        parser.add_argument(
            "--lora-paths",
            type=str,
            nargs="*",
            default=None,
            action=LoRAPathAction,
            help="The list of LoRA adapters. You can provide a list of either path in str or renamed path in the format {name}={path}.",
        )
        parser.add_argument(
            "--max-loras-per-batch",
            type=int,
            default=8,
            help="Maximum number of adapters for a running batch, include base-only request.",
        )
        parser.add_argument(
            "--lora-backend",
            type=str,
            default="triton",
            help="Choose the kernel backend for multi-LoRA serving.",
        )

        # Kernel backend
        parser.add_argument(
            "--attention-backend",
            type=str,
            choices=[
                "aiter",
                "cutlass_mla",
                "fa3",
                "flashinfer",
                "flashmla",
                "intel_amx",
                "torch_native",
                "triton",
            ],
            default=ServerArgs.attention_backend,
            help="Choose the kernels for attention layers.",
        )
        parser.add_argument(
            "--sampling-backend",
            type=str,
            choices=["flashinfer", "pytorch"],
            default=ServerArgs.sampling_backend,
            help="Choose the kernels for sampling layers.",
        )
        parser.add_argument(
            "--grammar-backend",
            type=str,
            choices=["xgrammar", "outlines", "llguidance", "none"],
            default=ServerArgs.grammar_backend,
            help="Choose the backend for grammar-guided decoding.",
        )

        # Speculative decoding
        parser.add_argument(
            "--speculative-algorithm",
            type=str,
            choices=["EAGLE", "EAGLE3", "NEXTN"],
            help="Speculative algorithm.",
        )
        parser.add_argument(
            "--speculative-draft-model-path",
            type=str,
            help="The path of the draft model weights. This can be a local folder or a Hugging Face repo ID.",
        )
        parser.add_argument(
            "--speculative-num-steps",
            type=int,
            help="The number of steps sampled from draft model in Speculative Decoding.",
            default=ServerArgs.speculative_num_steps,
        )
        parser.add_argument(
            "--speculative-eagle-topk",
            type=int,
            help="The number of tokens sampled from the draft model in eagle2 each step.",
            default=ServerArgs.speculative_eagle_topk,
        )
        parser.add_argument(
            "--speculative-num-draft-tokens",
            type=int,
            help="The number of tokens sampled from the draft model in Speculative Decoding.",
            default=ServerArgs.speculative_num_draft_tokens,
        )
        parser.add_argument(
            "--speculative-accept-threshold-single",
            type=float,
            help="Accept a draft token if its probability in the target model is greater than this threshold.",
            default=ServerArgs.speculative_accept_threshold_single,
        )
        parser.add_argument(
            "--speculative-accept-threshold-acc",
            type=float,
            help="The accept probability of a draft token is raised from its target probability p to min(1, p / threshold_acc).",
            default=ServerArgs.speculative_accept_threshold_acc,
        )
        parser.add_argument(
            "--speculative-token-map",
            type=str,
            help="The path of the draft model's small vocab table.",
            default=ServerArgs.speculative_token_map,
        )
        parser.add_argument(
            "--mm-attention-backend",
            type=str,
            choices=["sdpa", "fa3", "triton_attn"],
            default=ServerArgs.mm_attention_backend,
            help="Set multimodal attention backend.",
        )

        # Expert parallelism
        parser.add_argument(
            "--expert-parallel-size",
            "--ep-size",
            type=int,
            default=ServerArgs.ep_size,
            help="The expert parallelism size.",
        )
        parser.add_argument(
            "--enable-ep-moe",
            action="store_true",
            help="Enabling expert parallelism for moe. The ep size is equal to the tp size.",
        )
        parser.add_argument(
            "--enable-flashinfer-moe",
            action="store_true",
            help="Enable FlashInfer CUTLASS MoE backend for modelopt_fp4 quant on Blackwell. Supports MoE-EP with --enable-ep-moe",
        )
        parser.add_argument(
            "--enable-deepep-moe",
            action="store_true",
            help="Enabling DeepEP MoE implementation for EP MoE.",
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
            help="The type of heavy channels in double sparsity attention",
        )

        # Optimization/debug options
        parser.add_argument(
            "--disable-radix-cache",
            action="store_true",
            help="Disable RadixAttention for prefix caching.",
        )
        parser.add_argument(
            "--cuda-graph-max-bs",
            type=int,
            default=ServerArgs.cuda_graph_max_bs,
            help="Set the maximum batch size for cuda graph. It will extend the cuda graph capture batch size to this value.",
        )
        parser.add_argument(
            "--cuda-graph-bs",
            type=int,
            nargs="+",
            help="Set the list of batch sizes for cuda graph.",
        )
        parser.add_argument(
            "--disable-cuda-graph",
            action="store_true",
            help="Disable cuda graph.",
        )
        parser.add_argument(
            "--disable-cuda-graph-padding",
            action="store_true",
            help="Disable cuda graph when padding is needed. Still uses cuda graph when padding is not needed.",
        )
        parser.add_argument(
            "--enable-profile-cuda-graph",
            action="store_true",
            help="Enable profiling of cuda graph capture.",
        )
        parser.add_argument(
            "--enable-nccl-nvls",
            action="store_true",
            help="Enable NCCL NVLS for prefill heavy requests when available.",
        )
        parser.add_argument(
            "--enable-tokenizer-batch-encode",
            action="store_true",
            help="Enable batch tokenization for improved performance when processing multiple text inputs. Do not use with image inputs, pre-tokenized input_ids, or input_embeds.",
        )
        parser.add_argument(
            "--disable-outlines-disk-cache",
            action="store_true",
            help="Disable disk cache of outlines to avoid possible crashes related to file system or high concurrency.",
        )
        parser.add_argument(
            "--disable-custom-all-reduce",
            action="store_true",
            help="Disable the custom all-reduce kernel and fall back to NCCL.",
        )
        parser.add_argument(
            "--enable-mscclpp",
            action="store_true",
            help="Enable using mscclpp for small messages for all-reduce kernel and fall back to NCCL.",
        )
        parser.add_argument(
            "--disable-overlap-schedule",
            action="store_true",
            help="Disable the overlap scheduler, which overlaps the CPU scheduler with GPU model worker.",
        )
        parser.add_argument(
            "--disable-overlap-cg-plan",
            action="store_true",
            help="Disable the overlap optimization for cudagraph preparation in eagle verify.",
        )
        parser.add_argument(
            "--enable-mixed-chunk",
            action="store_true",
            help="Enabling mixing prefill and decode in a batch when using chunked prefill.",
        )
        parser.add_argument(
            "--enable-dp-attention",
            action="store_true",
            help="Enabling data parallelism for attention and tensor parallelism for FFN. The dp size should be equal to the tp size. Currently DeepSeek-V2 and Qwen 2/3 MoE models are supported.",
        )
        parser.add_argument(
            "--enable-dp-lm-head",
            action="store_true",
            help="Enable vocabulary parallel across the attention TP group to avoid all-gather across DP groups, optimizing performance under DP attention.",
        )
        parser.add_argument(
            "--enable-two-batch-overlap",
            action="store_true",
            help="Enabling two micro batches to overlap.",
        )
        parser.add_argument(
            "--enable-torch-compile",
            action="store_true",
            help="Optimize the model with torch.compile. Experimental feature.",
        )
        parser.add_argument(
            "--torch-compile-max-bs",
            type=int,
            default=ServerArgs.torch_compile_max_bs,
            help="Set the maximum batch size when using torch compile.",
        )
        parser.add_argument(
            "--torchao-config",
            type=str,
            default=ServerArgs.torchao_config,
            help="Optimize the model with torchao. Experimental feature. Current choices are: int8dq, int8wo, int4wo-<group_size>, fp8wo, fp8dq-per_tensor, fp8dq-per_row",
        )
        parser.add_argument(
            "--enable-nan-detection",
            action="store_true",
            help="Enable the NaN detection for debugging purposes.",
        )
        parser.add_argument(
            "--enable-p2p-check",
            action="store_true",
            help="Enable P2P check for GPU access, otherwise the p2p access is allowed by default.",
        )
        parser.add_argument(
            "--triton-attention-reduce-in-fp32",
            action="store_true",
            help="Cast the intermediate attention results to fp32 to avoid possible crashes related to fp16."
            "This only affects Triton attention kernels.",
        )
        parser.add_argument(
            "--triton-attention-num-kv-splits",
            type=int,
            default=ServerArgs.triton_attention_num_kv_splits,
            help="The number of KV splits in flash decoding Triton kernel. Larger value is better in longer context scenarios. The default value is 8.",
        )
        parser.add_argument(
            "--num-continuous-decode-steps",
            type=int,
            default=ServerArgs.num_continuous_decode_steps,
            help="Run multiple continuous decoding steps to reduce scheduling overhead. "
            "This can potentially increase throughput but may also increase time-to-first-token latency. "
            "The default value is 1, meaning only run one decoding step at a time.",
        )
        parser.add_argument(
            "--delete-ckpt-after-loading",
            action="store_true",
            help="Delete the model checkpoint after loading the model.",
        )
        parser.add_argument(
            "--enable-memory-saver",
            action="store_true",
            help="Allow saving memory using release_memory_occupation and resume_memory_occupation",
        )
        parser.add_argument(
            "--allow-auto-truncate",
            action="store_true",
            help="Allow automatically truncating requests that exceed the maximum input length instead of returning an error.",
        )
        parser.add_argument(
            "--enable-custom-logit-processor",
            action="store_true",
            help="Enable users to pass custom logit processors to the server (disabled by default for security)",
        )
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
            "--flashinfer-mla-disable-ragged",
            action="store_true",
            help="Not using ragged prefill wrapper when running flashinfer mla",
        )
        parser.add_argument(
            "--disable-shared-experts-fusion",
            action="store_true",
            help="Disable shared experts fusion optimization for deepseek v3/r1.",
        )
        parser.add_argument(
            "--disable-chunked-prefix-cache",
            action="store_true",
            help="Disable chunked prefix cache feature for deepseek, which should save overhead for short sequences.",
        )
        parser.add_argument(
            "--disable-fast-image-processor",
            action="store_true",
            help="Adopt base image processor instead of fast image processor.",
        )
        parser.add_argument(
            "--enable-return-hidden-states",
            action="store_true",
            help="Enable returning hidden states with responses.",
        )
        parser.add_argument(
            "--warmups",
            type=str,
            required=False,
            help="Specify custom warmup functions (csv) to run before server starts eg. --warmups=warmup_name1,warmup_name2 "
            "will run the functions `warmup_name1` and `warmup_name2` specified in warmup.py before the server starts listening for requests",
        )

        # Debug tensor dumps
        parser.add_argument(
            "--debug-tensor-dump-output-folder",
            type=str,
            default=ServerArgs.debug_tensor_dump_output_folder,
            help="The output folder for dumping tensors.",
        )
        parser.add_argument(
            "--debug-tensor-dump-input-file",
            type=str,
            default=ServerArgs.debug_tensor_dump_input_file,
            help="The input filename for dumping tensors",
        )
        parser.add_argument(
            "--debug-tensor-dump-inject",
            type=str,
            default=ServerArgs.debug_tensor_dump_inject,
            help="Inject the outputs from jax as the input of every layer.",
        )
        parser.add_argument(
            "--debug-tensor-dump-prefill-only",
            action="store_true",
            help="Only dump the tensors for prefill requests (i.e. batch size > 1).",
        )

        # Disaggregation
        parser.add_argument(
            "--disaggregation-mode",
            type=str,
            default="null",
            choices=["null", "prefill", "decode"],
            help='Only used for PD disaggregation. "prefill" for prefill-only server, and "decode" for decode-only server. If not specified, it is not PD disaggregated',
        )
        parser.add_argument(
            "--disaggregation-transfer-backend",
            type=str,
            default=ServerArgs.disaggregation_transfer_backend,
            choices=["mooncake", "nixl"],
            help="The backend for disaggregation transfer. Default is mooncake.",
        )
        parser.add_argument(
            "--disaggregation-bootstrap-port",
            type=int,
            default=ServerArgs.disaggregation_bootstrap_port,
            help="Bootstrap server port on the prefill server. Default is 8998.",
        )
        parser.add_argument(
            "--disaggregation-decode-tp",
            type=int,
            default=ServerArgs.disaggregation_decode_tp,
            help="Decode tp size. If not set, it matches the tp size of the current engine. This is only set on the prefill server.",
        )
        parser.add_argument(
            "--disaggregation-decode-dp",
            type=int,
            default=ServerArgs.disaggregation_decode_dp,
            help="Decode dp size. If not set, it matches the dp size of the current engine. This is only set on the prefill server.",
        )
        parser.add_argument(
            "--disaggregation-prefill-pp",
            type=int,
            default=ServerArgs.disaggregation_prefill_pp,
            help="Prefill pp size. If not set, it is default to 1. This is only set on the decode server.",
        )
        parser.add_argument(
            "--disaggregation-ib-device",
            type=str,
            default=ServerArgs.disaggregation_ib_device,
            help="The InfiniBand devices for disaggregation transfer, accepts single device (e.g., --disaggregation-ib-device mlx5_0) "
            "or multiple comma-separated devices (e.g., --disaggregation-ib-device mlx5_0,mlx5_1). "
            "Default is None, which triggers automatic device detection when mooncake backend is enabled.",
        )
        parser.add_argument(
            "--num-reserved-decode-tokens",
            type=int,
            default=ServerArgs.num_reserved_decode_tokens,
            help="Number of decode tokens that will have memory reserved when adding new request to the running batch.",
        )
        parser.add_argument(
            "--pdlb-url",
            type=str,
            default=None,
            help="The URL of the PD disaggregation load balancer. If set, the prefill/decode server will register with the load balancer.",
        )
        parser.add_argument(
            "--custom-weight-loader",
            type=str,
            nargs="*",
            default=None,
            help="The custom dataloader which used to update the model. Should be set with a valid import path, such as my_package.weight_load_func",
        )
        parser.add_argument(
            "--weight-loader-disable-mmap",
            action="store_true",
            help="Disable mmap while loading weight using safetensors.",
        )

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace):
        args.tp_size = args.tensor_parallel_size
        args.pp_size = args.pipeline_parallel_size
        args.dp_size = args.data_parallel_size
        args.ep_size = args.expert_parallel_size
        attrs = [attr.name for attr in dataclasses.fields(cls)]
        return cls(**{attr: getattr(args, attr) for attr in attrs})

    def url(self):
        if is_valid_ipv6_address(self.host):
            return f"http://[{self.host}]:{self.port}"
        else:
            return f"http://{self.host}:{self.port}"

    def check_server_args(self):
        assert (
            self.tp_size * self.pp_size
        ) % self.nnodes == 0, "tp_size must be divisible by number of nodes"

        # FIXME pp constraints
        if self.pp_size > 1:
            assert (
                self.disable_overlap_schedule
                and self.speculative_algorithm is None
                and not self.enable_mixed_chunk
            ), "Pipeline parallelism is not compatible with overlap schedule, speculative decoding, mixed chunked prefill."

        assert not (
            self.dp_size > 1 and self.nnodes != 1 and not self.enable_dp_attention
        ), "multi-node data parallel is not supported unless dp attention!"
        assert (
            self.max_loras_per_batch > 0
            # FIXME
            and (self.lora_paths is None or self.disable_radix_cache)
        ), "compatibility of lora and radix attention is in progress"
        assert self.base_gpu_id >= 0, "base_gpu_id must be non-negative"
        assert self.gpu_id_step >= 1, "gpu_id_step must be positive"

        if isinstance(self.lora_paths, list):
            lora_paths = self.lora_paths
            self.lora_paths = {}
            for lora_path in lora_paths:
                if "=" in lora_path:
                    name, path = lora_path.split("=", 1)
                    self.lora_paths[name] = path
                else:
                    self.lora_paths[lora_path] = lora_path


def prepare_server_args(argv: List[str]) -> ServerArgs:
    """
    Prepare the server arguments from the command line arguments.

    Args:
        args: The command line arguments. Typically, it should be `sys.argv[1:]`
            to ensure compatibility with `parse_args` when no arguments are passed.

    Returns:
        The server arguments.
    """
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    raw_args = parser.parse_args(argv)
    server_args = ServerArgs.from_cli_args(raw_args)
    return server_args


ZMQ_TCP_PORT_DELTA = 233


@dataclasses.dataclass
class PortArgs:
    # The ipc filename for tokenizer to receive inputs from detokenizer (zmq)
    tokenizer_ipc_name: str
    # The ipc filename for scheduler (rank 0) to receive inputs from tokenizer (zmq)
    scheduler_input_ipc_name: str
    # The ipc filename for detokenizer to receive inputs from scheduler (zmq)
    detokenizer_ipc_name: str

    # The port for nccl initialization (torch.dist)
    nccl_port: int

    # The ipc filename for rpc call between Engine and Scheduler
    rpc_ipc_name: str

    # The ipc filename for Scheduler to send metrics
    metrics_ipc_name: str

    @staticmethod
    def init_new(server_args, dp_rank: Optional[int] = None) -> "PortArgs":
        port = server_args.port + random.randint(100, 1000)
        while True:
            if is_port_available(port):
                break
            if port < 60000:
                port += 42
            else:
                port -= 43

        if not server_args.enable_dp_attention:
            # Normal case, use IPC within a single node
            return PortArgs(
                tokenizer_ipc_name=f"ipc://{tempfile.NamedTemporaryFile(delete=False).name}",
                scheduler_input_ipc_name=f"ipc://{tempfile.NamedTemporaryFile(delete=False).name}",
                detokenizer_ipc_name=f"ipc://{tempfile.NamedTemporaryFile(delete=False).name}",
                nccl_port=port,
                rpc_ipc_name=f"ipc://{tempfile.NamedTemporaryFile(delete=False).name}",
                metrics_ipc_name=f"ipc://{tempfile.NamedTemporaryFile(delete=False).name}",
            )
        else:
            # DP attention. Use TCP + port to handle both single-node and multi-node.
            if server_args.nnodes == 1 and server_args.dist_init_addr is None:
                dist_init_addr = ("127.0.0.1", server_args.port + ZMQ_TCP_PORT_DELTA)
            elif server_args.dist_init_addr.startswith("["):  # ipv6 address
                port_num, host = configure_ipv6(server_args.dist_init_addr)
                dist_init_addr = (host, str(port_num))
            else:
                dist_init_addr = server_args.dist_init_addr.split(":")

            assert (
                len(dist_init_addr) == 2
            ), "please provide --dist-init-addr as host:port of head node"

            dist_init_host, dist_init_port = dist_init_addr
            port_base = int(dist_init_port) + 1
            if dp_rank is None:
                # TokenizerManager to DataParallelController
                scheduler_input_port = port_base + 4
            else:
                scheduler_input_port = port_base + 4 + 1 + dp_rank

            return PortArgs(
                tokenizer_ipc_name=f"tcp://{dist_init_host}:{port_base}",
                scheduler_input_ipc_name=f"tcp://{dist_init_host}:{scheduler_input_port}",
                detokenizer_ipc_name=f"tcp://{dist_init_host}:{port_base + 1}",
                nccl_port=port,
                rpc_ipc_name=f"tcp://{dist_init_host}:{port_base + 2}",
                metrics_ipc_name=f"tcp://{dist_init_host}:{port_base + 3}",
            )


class LoRAPathAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, {})
        for lora_path in values:
            if "=" in lora_path:
                name, path = lora_path.split("=", 1)
                getattr(namespace, self.dest)[name] = path
            else:
                getattr(namespace, self.dest)[lora_path] = lora_path


class DeprecatedAction(argparse.Action):
    def __init__(self, option_strings, dest, nargs=0, **kwargs):
        super(DeprecatedAction, self).__init__(
            option_strings, dest, nargs=nargs, **kwargs
        )

    def __call__(self, parser, namespace, values, option_string=None):
        raise ValueError(self.help)


def get_model_arch(args: ServerArgs):
    hf_config = get_config(
        args.model_path,
        trust_remote_code=args.trust_remote_code,
        revision=args.revision,
        model_override_args=json.loads(args.json_model_override_args),
    )
    return hf_config.architectures[0]


def auto_choose_speculative_params(self: ServerArgs):
    """
    Automatically choose the parameters for speculative decoding.

    You can tune them on your own models and prompts with scripts/playground/bench_speculative.py
    """
    kwargs = {}

    hf_config = get_config(
        self.model_path,
        trust_remote_code=self.trust_remote_code,
        revision=self.revision,
        model_override_args=json.loads(self.json_model_override_args),
        **kwargs,
    )
    arch = hf_config.architectures[0]

    if arch in ["LlamaForCausalLM"]:
        # The default value for llama
        return (5, 4, 8)
    elif arch in ["DeepseekV3ForCausalLM", "DeepseekV2ForCausalLM"]:
        # The default value for deepseek
        return (3, 1, 4)
    elif arch in ["Grok1ForCausalLM", "Grok1VForCausalLM"]:
        return (5, 4, 8)
    else:
        # The default value for all other models
        return (5, 4, 8)
