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

from __future__ import annotations

import argparse
import dataclasses
import importlib
import importlib.util
import json
import logging
import os
import random
import tempfile
from typing import Any, Callable, Dict, List, Literal, Optional, Union

from sglang.srt.connector import ConnectorType
from sglang.srt.environ import envs
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.srt.layers.attention.fla.chunk_delta_h import CHUNK_SIZE as FLA_CHUNK_SIZE
from sglang.srt.lora.lora_registry import LoRARef
from sglang.srt.parser.reasoning_parser import ReasoningParser
from sglang.srt.utils.common import (
    LORA_TARGET_ALL_MODULES,
    SUPPORTED_LORA_TARGET_MODULES,
    check_pkg_version_at_least,
    configure_ipv6,
    cpu_has_amx_support,
    get_bool_env_var,
    get_device,
    get_device_memory_capacity,
    get_device_name,
    get_device_sm,
    get_int_env_var,
    get_quantization_config,
    is_blackwell_supported,
    is_cuda,
    is_flashinfer_available,
    is_hip,
    is_hopper_with_cuda_12_3,
    is_no_spec_infer_or_topk_one,
    is_npu,
    is_port_available,
    is_remote_url,
    is_sm90_supported,
    is_sm100_supported,
    is_sm120_supported,
    is_triton_kernels_available,
    is_valid_ipv6_address,
    json_list_type,
    nullable_str,
    parse_connector_type,
    torch_release,
    wait_port_available,
    xpu_has_xmx_support,
)
from sglang.srt.utils.hf_transformers_utils import check_gguf_file
from sglang.utils import is_in_ci

logger = logging.getLogger(__name__)

# Define constants
DEFAULT_UVICORN_ACCESS_LOG_EXCLUDE_PREFIXES = ()
SAMPLING_BACKEND_CHOICES = {"flashinfer", "pytorch", "ascend"}
LOAD_FORMAT_CHOICES = [
    "auto",
    "pt",
    "safetensors",
    "npcache",
    "dummy",
    "sharded_state",
    "gguf",
    "bitsandbytes",
    "layered",
    "flash_rl",
    "remote",
    "remote_instance",
    "fastsafetensors",
    "private",
]

QUANTIZATION_CHOICES = [
    "awq",
    "fp8",
    "mxfp8",
    "gptq",
    "marlin",
    "gptq_marlin",
    "awq_marlin",
    "bitsandbytes",
    "gguf",
    "modelopt",
    "modelopt_fp8",
    "modelopt_fp4",
    "petit_nvfp4",
    "w8a8_int8",
    "w8a8_fp8",
    "moe_wna16",
    "qoq",
    "w4afp8",
    "mxfp4",
    "auto-round",
    "compressed-tensors",  # for Ktransformers
    "modelslim",  # for NPU
    "quark_int4fp8_moe",
]

SPECULATIVE_DRAFT_MODEL_QUANTIZATION_CHOICES = [*QUANTIZATION_CHOICES, "unquant"]

ATTENTION_BACKEND_CHOICES = [
    # Common
    "triton",
    "torch_native",
    "flex_attention",
    "nsa",
    # NVIDIA specific
    "cutlass_mla",
    "fa3",
    "fa4",
    "flashinfer",
    "flashmla",
    "trtllm_mla",
    "trtllm_mha",
    "dual_chunk_flash_attn",
    # AMD specific
    "aiter",
    "wave",
    # Other platforms
    "intel_amx",
    "ascend",
    "intel_xpu",
]

LORA_BACKEND_CHOICES = ["triton", "csgmv", "ascend", "torch_native"]

DISAGG_TRANSFER_BACKEND_CHOICES = ["mooncake", "nixl", "ascend", "fake", "mori"]

ENCODER_TRANSFER_BACKEND_CHOICES = ["zmq_to_scheduler", "zmq_to_tokenizer", "mooncake"]

GRAMMAR_BACKEND_CHOICES = ["xgrammar", "outlines", "llguidance", "none"]

DETERMINISTIC_ATTENTION_BACKEND_CHOICES = ["flashinfer", "fa3", "triton"]

RADIX_SUPPORTED_DETERMINISTIC_ATTENTION_BACKEND = ["fa3", "triton"]

NSA_PREFILL_CP_SPLIT_CHOICES = ["in-seq-split", "round-robin-split"]

DEFAULT_LORA_EVICTION_POLICY = "lru"

NSA_CHOICES = [
    "flashmla_sparse",
    "flashmla_kv",
    "flashmla_auto",
    "fa3",
    "tilelang",
    "aiter",
    "trtllm",
]

RADIX_EVICTION_POLICY_CHOICES = ["lru", "lfu"]

RL_ON_POLICY_TARGET_CHOICES = ["fsdp"]

MOE_RUNNER_BACKEND_CHOICES = [
    "auto",
    "deep_gemm",
    "triton",
    "triton_kernel",
    "flashinfer_trtllm",
    "flashinfer_cutlass",
    "flashinfer_mxfp4",
    "flashinfer_cutedsl",
    "cutlass",
]

MOE_A2A_BACKEND_CHOICES = [
    "none",
    "deepep",
    "mooncake",
    "mori",
    "ascend_fuseep",
    "flashinfer",
]

FP8_GEMM_RUNNER_BACKEND_CHOICES = [
    "auto",
    "deep_gemm",
    "flashinfer_trtllm",
    "flashinfer_deepgemm",
    "cutlass",
    "triton",
    "aiter",
]

FP4_GEMM_RUNNER_BACKEND_CHOICES = [
    "auto",
    "flashinfer_cudnn",
    "flashinfer_cutlass",
    "flashinfer_trtllm",
]

MAMBA_SSM_DTYPE_CHOICES = ["float32", "bfloat16"]

MAMBA_SCHEDULER_STRATEGY_CHOICES = ["auto", "no_buffer", "extra_buffer"]


# Allow external code to add more choices
def add_load_format_choices(choices):
    LOAD_FORMAT_CHOICES.extend(choices)


def add_quantization_method_choices(choices):
    QUANTIZATION_CHOICES.extend(choices)


def add_attention_backend_choices(choices):
    ATTENTION_BACKEND_CHOICES.extend(choices)


def add_disagg_transfer_backend_choices(choices):
    DISAGG_TRANSFER_BACKEND_CHOICES.extend(choices)


def add_grammar_backend_choices(choices):
    GRAMMAR_BACKEND_CHOICES.extend(choices)


def add_moe_runner_backend_choices(choices):
    MOE_RUNNER_BACKEND_CHOICES.extend(choices)


def add_fp8_gemm_runner_backend_choices(choices):
    FP8_GEMM_RUNNER_BACKEND_CHOICES.extend(choices)


def add_fp4_gemm_runner_backend_choices(choices):
    FP4_GEMM_RUNNER_BACKEND_CHOICES.extend(choices)


def add_deterministic_attention_backend_choices(choices):
    DETERMINISTIC_ATTENTION_BACKEND_CHOICES.extend(choices)


def add_radix_supported_deterministic_attention_backend_choices(choices):
    RADIX_SUPPORTED_DETERMINISTIC_ATTENTION_BACKEND.extend(choices)


def add_radix_eviction_policy_choices(choices):
    RADIX_EVICTION_POLICY_CHOICES.extend(choices)


def add_rl_on_policy_target_choices(choices):
    RL_ON_POLICY_TARGET_CHOICES.extend(choices)


def add_mamba_ssm_dtype_choices(choices):
    MAMBA_SSM_DTYPE_CHOICES.extend(choices)


@dataclasses.dataclass
class ServerArgs:
    """
    The arguments of the server.

    NOTE: When you add new arguments, please make sure the order
    in this class definition the same as the order in the the function
    `ServerArgs.add_cli_args`.
    Please follow the existing style to group the new arguments into related groups or create new groups.
    """

    # Model and tokenizer
    model_path: str
    tokenizer_path: Optional[str] = None
    tokenizer_mode: str = "auto"
    tokenizer_worker_num: int = 1
    skip_tokenizer_init: bool = False
    load_format: str = "auto"
    model_loader_extra_config: str = "{}"
    trust_remote_code: bool = False
    context_length: Optional[int] = None
    is_embedding: bool = False
    enable_multimodal: Optional[bool] = None
    revision: Optional[str] = None
    model_impl: str = "auto"

    # HTTP server
    host: str = "127.0.0.1"
    port: int = 30000
    fastapi_root_path: str = ""
    grpc_mode: bool = False
    skip_server_warmup: bool = False
    warmups: Optional[str] = None
    nccl_port: Optional[int] = None
    checkpoint_engine_wait_weights_before_ready: bool = False

    # Quantization and data type
    dtype: str = "auto"
    quantization: Optional[str] = None
    quantization_param_path: Optional[str] = None
    kv_cache_dtype: str = "auto"
    enable_fp32_lm_head: bool = False
    modelopt_quant: Optional[Union[str, Dict]] = None
    modelopt_checkpoint_restore_path: Optional[str] = None
    modelopt_checkpoint_save_path: Optional[str] = None
    modelopt_export_path: Optional[str] = None
    quantize_and_serve: bool = False
    rl_quant_profile: Optional[str] = None  # For flash_rl load format

    # Memory and scheduling
    mem_fraction_static: Optional[float] = None
    max_running_requests: Optional[int] = None
    max_queued_requests: Optional[int] = None
    max_total_tokens: Optional[int] = None
    chunked_prefill_size: Optional[int] = None
    enable_dynamic_chunking: bool = False
    max_prefill_tokens: int = 16384
    prefill_max_requests: Optional[int] = None
    schedule_policy: str = "fcfs"
    enable_priority_scheduling: bool = False
    abort_on_priority_when_disabled: bool = False
    schedule_low_priority_values_first: bool = False
    priority_scheduling_preemption_threshold: int = 10
    schedule_conservativeness: float = 1.0
    page_size: Optional[int] = None
    swa_full_tokens_ratio: float = 0.8
    disable_hybrid_swa_memory: bool = False
    radix_eviction_policy: str = "lru"
    enable_prefill_delayer: bool = False
    prefill_delayer_max_delay_passes: int = 30
    prefill_delayer_token_usage_low_watermark: Optional[float] = None
    prefill_delayer_forward_passes_buckets: Optional[List[float]] = None
    prefill_delayer_wait_seconds_buckets: Optional[List[float]] = None

    # Runtime options
    device: Optional[str] = None
    tp_size: int = 1
    pp_size: int = 1
    pp_max_micro_batch_size: Optional[int] = None
    pp_async_batch_depth: int = 0
    stream_interval: int = 1
    stream_output: bool = False
    random_seed: Optional[int] = None
    constrained_json_whitespace_pattern: Optional[str] = None
    constrained_json_disable_any_whitespace: bool = False
    watchdog_timeout: float = 300
    soft_watchdog_timeout: Optional[float] = None
    dist_timeout: Optional[int] = None  # timeout for torch.distributed
    download_dir: Optional[str] = None
    model_checksum: Optional[str] = None
    base_gpu_id: int = 0
    gpu_id_step: int = 1
    sleep_on_idle: bool = False
    custom_sigquit_handler: Optional[Callable] = None

    # Logging
    log_level: str = "info"
    log_level_http: Optional[str] = None
    log_requests: bool = False
    log_requests_level: int = 2
    log_requests_format: str = "text"
    log_requests_target: Optional[List[str]] = None
    uvicorn_access_log_exclude_prefixes: List[str] = dataclasses.field(
        default_factory=lambda: list(DEFAULT_UVICORN_ACCESS_LOG_EXCLUDE_PREFIXES)
    )
    crash_dump_folder: Optional[str] = None
    show_time_cost: bool = False
    enable_metrics: bool = False
    enable_metrics_for_all_schedulers: bool = False
    tokenizer_metrics_custom_labels_header: str = "x-custom-labels"
    tokenizer_metrics_allowed_custom_labels: Optional[List[str]] = None
    extra_metric_labels: Optional[Dict[str, str]] = None
    bucket_time_to_first_token: Optional[List[float]] = None
    bucket_inter_token_latency: Optional[List[float]] = None
    bucket_e2e_request_latency: Optional[List[float]] = None
    collect_tokens_histogram: bool = False
    prompt_tokens_buckets: Optional[List[str]] = None
    generation_tokens_buckets: Optional[List[str]] = None
    gc_warning_threshold_secs: float = 0.0
    decode_log_interval: int = 40
    enable_request_time_stats_logging: bool = False
    kv_events_config: Optional[str] = None
    enable_trace: bool = False
    otlp_traces_endpoint: str = "localhost:4317"

    # RequestMetricsExporter configuration
    export_metrics_to_file: bool = False
    export_metrics_to_file_dir: Optional[str] = None

    # API related
    api_key: Optional[str] = None
    admin_api_key: Optional[str] = None
    served_model_name: Optional[str] = None
    weight_version: str = "default"
    chat_template: Optional[str] = None
    hf_chat_template_name: Optional[str] = None
    completion_template: Optional[str] = None
    file_storage_path: str = "sglang_storage"
    enable_cache_report: bool = False
    reasoning_parser: Optional[str] = None
    tool_call_parser: Optional[str] = None
    tool_server: Optional[str] = None
    sampling_defaults: str = "model"

    # Data parallelism
    dp_size: int = 1
    load_balance_method: str = "auto"

    # Multi-node distributed serving
    dist_init_addr: Optional[str] = None
    nnodes: int = 1
    node_rank: int = 0

    # Model override args in JSON
    json_model_override_args: str = "{}"
    preferred_sampling_params: Optional[str] = None

    # LoRA
    enable_lora: Optional[bool] = None
    enable_lora_overlap_loading: Optional[bool] = None
    max_lora_rank: Optional[int] = None
    lora_target_modules: Optional[Union[set[str], List[str]]] = None
    lora_paths: Optional[
        Union[dict[str, str], List[dict[str, str]], List[str], List[LoRARef]]
    ] = None
    max_loaded_loras: Optional[int] = None
    max_loras_per_batch: int = 8
    lora_eviction_policy: str = "lru"
    lora_backend: str = "csgmv"
    max_lora_chunk_size: Optional[int] = 16

    # Kernel backend
    attention_backend: Optional[str] = None
    decode_attention_backend: Optional[str] = None
    prefill_attention_backend: Optional[str] = None
    sampling_backend: Optional[str] = None
    grammar_backend: Optional[str] = None
    mm_attention_backend: Optional[str] = None
    fp8_gemm_runner_backend: str = "auto"
    fp4_gemm_runner_backend: str = "auto"
    nsa_prefill_backend: Optional[str] = (
        None  # None = auto-detect based on hardware/kv_cache_dtype
    )
    nsa_decode_backend: Optional[str] = (
        None  # auto-detect based on hardware/kv_cache_dtype
    )
    disable_flashinfer_autotune: bool = False

    # Speculative decoding
    speculative_algorithm: Optional[str] = None
    speculative_draft_model_path: Optional[str] = None
    speculative_draft_model_revision: Optional[str] = None
    speculative_draft_load_format: Optional[str] = None
    speculative_num_steps: Optional[int] = None
    speculative_eagle_topk: Optional[int] = None
    speculative_num_draft_tokens: Optional[int] = None
    speculative_accept_threshold_single: float = 1.0
    speculative_accept_threshold_acc: float = 1.0
    speculative_token_map: Optional[str] = None
    speculative_attention_mode: str = "prefill"
    speculative_draft_attention_backend: Optional[str] = None
    speculative_moe_runner_backend: Optional[str] = None
    speculative_moe_a2a_backend: Optional[str] = None
    speculative_draft_model_quantization: Optional[str] = None

    # Speculative decoding (ngram)
    speculative_ngram_min_match_window_size: int = 1
    speculative_ngram_max_match_window_size: int = 12
    speculative_ngram_min_bfs_breadth: int = 1
    speculative_ngram_max_bfs_breadth: int = 10
    speculative_ngram_match_type: Literal["BFS", "PROB"] = "BFS"
    speculative_ngram_branch_length: int = 18
    speculative_ngram_capacity: int = 10 * 1000 * 1000
    enable_multi_layer_eagle: bool = False

    # Expert parallelism
    ep_size: int = 1
    moe_a2a_backend: Literal[
        "none", "deepep", "mooncake", "mori", "ascend_fuseep", "flashinfer"
    ] = "none"
    moe_runner_backend: str = "auto"
    flashinfer_mxfp4_moe_precision: Literal["default", "bf16"] = "default"
    enable_flashinfer_allreduce_fusion: bool = False
    deepep_mode: Literal["auto", "normal", "low_latency"] = "auto"
    ep_num_redundant_experts: int = 0
    ep_dispatch_algorithm: Optional[Literal["static", "dynamic", "fake"]] = None
    init_expert_location: str = "trivial"
    enable_eplb: bool = False
    eplb_algorithm: str = "auto"
    eplb_rebalance_num_iterations: int = 1000
    eplb_rebalance_layers_per_chunk: Optional[int] = None
    eplb_min_rebalancing_utilization_threshold: float = 1.0
    expert_distribution_recorder_mode: Optional[
        Literal["stat", "stat_approx", "per_pass", "per_token"]
    ] = None
    expert_distribution_recorder_buffer_size: Optional[int] = None
    enable_expert_distribution_metrics: bool = False
    deepep_config: Optional[str] = None
    moe_dense_tp_size: Optional[int] = None
    elastic_ep_backend: Literal[None, "mooncake"] = None
    mooncake_ib_device: Optional[str] = None

    # Mamba cache
    max_mamba_cache_size: Optional[int] = None
    mamba_ssm_dtype: str = "float32"
    mamba_full_memory_ratio: float = 0.9
    mamba_scheduler_strategy: str = "auto"
    mamba_track_interval: int = 256

    # Hierarchical cache
    enable_hierarchical_cache: bool = False
    hicache_ratio: float = 2.0
    hicache_size: int = 0
    hicache_write_policy: str = "write_through"
    hicache_io_backend: str = "kernel"
    hicache_mem_layout: str = "layer_first"
    disable_hicache_numa_detect: bool = False
    hicache_storage_backend: Optional[str] = None
    hicache_storage_prefetch_policy: str = "best_effort"
    hicache_storage_backend_extra_config: Optional[str] = None

    # Hierarchical sparse attention
    hierarchical_sparse_attention_extra_config: Optional[str] = None

    # LMCache
    enable_lmcache: bool = False

    # Ktransformers/AMX expert parallelism
    kt_weight_path: Optional[str] = None
    kt_method: Optional[str] = None
    kt_cpuinfer: Optional[int] = None
    kt_threadpool_count: Optional[int] = None
    kt_num_gpu_experts: Optional[int] = None
    kt_max_deferred_experts_per_token: Optional[int] = None

    # Diffusion LLM
    dllm_algorithm: Optional[str] = None
    dllm_algorithm_config: Optional[str] = None

    # Double Sparsity
    enable_double_sparsity: bool = False
    ds_channel_config_path: Optional[str] = None
    ds_heavy_channel_num: int = 32
    ds_heavy_token_num: int = 256
    ds_heavy_channel_type: str = "qk"
    ds_sparse_decode_threshold: int = 4096

    # Offloading
    cpu_offload_gb: int = 0
    offload_group_size: int = -1
    offload_num_in_group: int = 1
    offload_prefetch_step: int = 1
    offload_mode: str = "cpu"

    # Scoring configuration
    # Delimiter token ID used to combine Query and Items into a single sequence for multi-item scoring.
    # Format: Query<delimiter>Item1<delimiter>Item2<delimiter>...
    # This enables efficient batch processing of multiple items against a single query.
    multi_item_scoring_delimiter: Optional[Union[int]] = None

    # Optimization/debug options
    disable_radix_cache: bool = False
    cuda_graph_max_bs: Optional[int] = None
    cuda_graph_bs: Optional[List[int]] = None
    disable_cuda_graph: bool = False
    disable_cuda_graph_padding: bool = False
    enable_profile_cuda_graph: bool = False
    enable_cudagraph_gc: bool = False
    enable_layerwise_nvtx_marker: bool = False
    enable_nccl_nvls: bool = False
    enable_symm_mem: bool = False
    disable_flashinfer_cutlass_moe_fp4_allgather: bool = False
    enable_tokenizer_batch_encode: bool = False
    disable_tokenizer_batch_decode: bool = False
    disable_outlines_disk_cache: bool = False
    disable_custom_all_reduce: bool = False
    enable_mscclpp: bool = False
    enable_torch_symm_mem: bool = False
    disable_overlap_schedule: bool = False
    enable_mixed_chunk: bool = False
    enable_dp_attention: bool = False
    enable_dp_lm_head: bool = False
    enable_two_batch_overlap: bool = False
    enable_single_batch_overlap: bool = False
    tbo_token_distribution_threshold: float = 0.48
    enable_torch_compile: bool = False
    enable_piecewise_cuda_graph: bool = False
    enable_torch_compile_debug_mode: bool = False
    torch_compile_max_bs: int = 32
    piecewise_cuda_graph_max_tokens: Optional[int] = None
    piecewise_cuda_graph_tokens: Optional[List[int]] = None
    piecewise_cuda_graph_compiler: str = "eager"
    torchao_config: str = ""
    enable_nan_detection: bool = False
    enable_p2p_check: bool = False
    triton_attention_reduce_in_fp32: bool = False
    triton_attention_num_kv_splits: int = 8
    triton_attention_split_tile_size: Optional[int] = None
    num_continuous_decode_steps: int = 1
    delete_ckpt_after_loading: bool = False
    enable_memory_saver: bool = False
    enable_weights_cpu_backup: bool = False
    enable_draft_weights_cpu_backup: bool = False
    allow_auto_truncate: bool = False
    enable_custom_logit_processor: bool = False
    flashinfer_mla_disable_ragged: bool = False
    disable_shared_experts_fusion: bool = False
    disable_chunked_prefix_cache: bool = False
    disable_fast_image_processor: bool = False
    keep_mm_feature_on_device: bool = False
    enable_return_hidden_states: bool = False
    enable_return_routed_experts: bool = False
    scheduler_recv_interval: int = 1
    numa_node: Optional[List[int]] = None
    enable_deterministic_inference: bool = False
    rl_on_policy_target: Optional[str] = None
    enable_attn_tp_input_scattered: bool = False
    # Context parallelism used in the long sequence prefill phase of DeepSeek v3.2
    enable_nsa_prefill_context_parallel: bool = False
    nsa_prefill_cp_mode: str = "in-seq-split"
    enable_fused_qk_norm_rope: bool = False
    enable_precise_embedding_interpolation: bool = False

    # Dynamic batch tokenizer
    enable_dynamic_batch_tokenizer: bool = False
    dynamic_batch_tokenizer_batch_size: int = 32
    dynamic_batch_tokenizer_batch_timeout: float = 0.002

    # Debug tensor dumps
    debug_tensor_dump_output_folder: Optional[str] = None
    # None means dump all layers.
    debug_tensor_dump_layers: Optional[List[int]] = None
    # TODO(guoyuhong): clean the old dumper code.
    debug_tensor_dump_input_file: Optional[str] = None
    debug_tensor_dump_inject: bool = False

    # PD disaggregation: can be "null" (not disaggregated), "prefill" (prefill-only), or "decode" (decode-only)
    disaggregation_mode: Literal["null", "prefill", "decode"] = "null"
    disaggregation_transfer_backend: str = "mooncake"
    disaggregation_bootstrap_port: int = 8998
    disaggregation_decode_tp: Optional[int] = None
    disaggregation_decode_dp: Optional[int] = None
    disaggregation_prefill_pp: Optional[int] = 1
    disaggregation_ib_device: Optional[str] = None
    disaggregation_decode_enable_offload_kvcache: bool = False
    # Enable auto FAKE mode for decode node testing, no need to pass bootstrap_host in request
    disaggregation_decode_enable_fake_auto: bool = False
    num_reserved_decode_tokens: int = 512  # used for decode kv cache offload in PD
    # FIXME: hack to reduce ITL when decode bs is small
    disaggregation_decode_polling_interval: int = 1

    # Encode prefill disaggregation
    encoder_only: bool = False
    language_only: bool = False
    encoder_transfer_backend: str = ENCODER_TRANSFER_BACKEND_CHOICES[0]
    encoder_urls: List[str] = dataclasses.field(default_factory=list)

    # For model weight update and weight loading
    custom_weight_loader: Optional[List[str]] = None
    weight_loader_disable_mmap: bool = False
    remote_instance_weight_loader_seed_instance_ip: Optional[str] = None
    remote_instance_weight_loader_seed_instance_service_port: Optional[int] = None
    remote_instance_weight_loader_send_weights_group_ports: Optional[List[int]] = None
    remote_instance_weight_loader_backend: Literal["transfer_engine", "nccl"] = "nccl"
    remote_instance_weight_loader_start_seed_via_transfer_engine: bool = False

    # For PD-Multiplexing
    enable_pdmux: bool = False
    pdmux_config_path: Optional[str] = None
    sm_group_num: int = 8

    # For Multi-Modal
    mm_max_concurrent_calls: int = 32
    mm_per_request_timeout: float = 10.0
    enable_broadcast_mm_inputs_process: bool = False
    enable_prefix_mm_cache: bool = False
    mm_enable_dp_encoder: bool = False
    mm_process_config: Optional[Dict[str, Any]] = None
    limit_mm_data_per_request: Optional[Union[str, Dict[str, int]]] = None

    # For checkpoint decryption
    decrypted_config_file: Optional[str] = None
    decrypted_draft_config_file: Optional[str] = None

    # For forward hooks
    forward_hooks: Optional[List[dict[str, Any]]] = None

    def __post_init__(self):
        """
        Orchestrates the handling of various server arguments, ensuring proper configuration and validation.
        """

        # Normalize load balancing defaults early (before dummy-model short-circuit).
        self._handle_load_balance_method()

        if self.model_path.lower() in ["none", "dummy"]:
            # Skip for dummy models
            return

        # Handle deprecated arguments.
        self._handle_deprecated_args()

        # Handle deprecated environment variables for prefill delayer.
        self._handle_prefill_delayer_env_compat()

        # Set missing default values.
        self._handle_missing_default_values()

        # Handle device-specific backends.
        self._handle_hpu_backends()
        self._handle_cpu_backends()
        self._handle_npu_backends()

        # Get GPU memory capacity, which is a common dependency for several configuration steps.
        gpu_mem = get_device_memory_capacity(self.device)

        # Handle memory-related, chunked prefill, and CUDA graph batch size configurations.
        self._handle_gpu_memory_settings(gpu_mem)

        # Apply model-specific adjustments.
        self._handle_model_specific_adjustments()

        # Set kernel backends.
        self._handle_sampling_backend()
        self._handle_attention_backend_compatibility()
        self._handle_kv4_compatibility()
        self._handle_page_size()
        self._handle_amd_specifics()
        self._handle_grammar_backend()

        # Handle Hicache settings.
        self._handle_hicache()

        # Handle data parallelism.
        self._handle_data_parallelism()

        # Handle MoE configurations.
        self._handle_moe_kernel_config()
        self._handle_a2a_moe()
        self._handle_eplb_and_dispatch()
        self._handle_expert_distribution_metrics()
        self._handle_elastic_ep()

        # Handle pipeline parallelism.
        self._handle_pipeline_parallelism()

        # Handle speculative decoding logic.
        self._handle_speculative_decoding()

        # Handle model loading format.
        self._handle_load_format()

        # Handle PD disaggregation.
        self._handle_pd_disaggregation()

        # Handle Encoder disaggregation.
        self._handle_encoder_disaggregation()

        # Validate tokenizer settings.
        self._handle_tokenizer_batching()

        # Propagate environment variables.
        self._handle_environment_variables()

        # Validate cache settings.
        self._handle_cache_compatibility()

        # Handle deterministic inference.
        self._handle_deterministic_inference()

        # Handle diffusion LLM inference.
        self._handle_dllm_inference()

        # Handle debug utilities.
        self._handle_debug_utils()

        # Handle any other necessary validations.
        self._handle_other_validations()

    def _handle_load_balance_method(self):
        if self.disaggregation_mode not in ("null", "prefill", "decode"):
            raise ValueError(
                f"Invalid disaggregation_mode={self.disaggregation_mode!r}"
            )

        if self.load_balance_method == "auto":
            # Default behavior:
            # - non-PD: round_robin
            # - PD prefill: follow_bootstrap_room
            # - PD decode: round_robin
            self.load_balance_method = (
                "follow_bootstrap_room"
                if self.disaggregation_mode == "prefill"
                else "round_robin"
            )
            return

        # Backward compat: in PD prefill, legacy "round_robin" means `bootstrap_room` routing.
        if (
            self.disaggregation_mode == "prefill"
            and self.load_balance_method == "round_robin"
        ):
            logger.warning(
                "In PD-disaggregation prefill mode, the 'round_robin' load balancing method "
                "means `bootstrap_room` routing (use 'follow_bootstrap_room' instead). "
                "Falling back to 'follow_bootstrap_room' for backward compatibility."
            )
            self.load_balance_method = "follow_bootstrap_room"

    def _handle_deprecated_args(self):
        # Handle deprecated tool call parsers
        deprecated_tool_call_parsers = {"qwen25": "qwen", "glm45": "glm"}
        if self.tool_call_parser in deprecated_tool_call_parsers:
            logger.warning(
                f"The tool_call_parser '{self.tool_call_parser}' is deprecated. Please use '{deprecated_tool_call_parsers[self.tool_call_parser]}' instead."
            )
            self.tool_call_parser = deprecated_tool_call_parsers[self.tool_call_parser]

    def _handle_prefill_delayer_env_compat(self):
        if envs.SGLANG_SCHEDULER_DECREASE_PREFILL_IDLE.get():
            self.enable_prefill_delayer = True
        if x := envs.SGLANG_PREFILL_DELAYER_MAX_DELAY_PASSES.get():
            self.prefill_delayer_max_delay_passes = x
        if x := envs.SGLANG_PREFILL_DELAYER_TOKEN_USAGE_LOW_WATERMARK.get():
            self.prefill_delayer_token_usage_low_watermark = x

    def _handle_missing_default_values(self):
        if self.tokenizer_path is None:
            self.tokenizer_path = self.model_path
        if self.served_model_name is None:
            self.served_model_name = self.model_path
        if self.device is None:
            self.device = get_device()
        if self.random_seed is None:
            self.random_seed = random.randint(0, 1 << 30)
        if self.mm_process_config is None:
            self.mm_process_config = {}

        # Handle ModelScope model downloads
        if get_bool_env_var("SGLANG_USE_MODELSCOPE"):
            if not os.path.exists(self.model_path):
                from modelscope import snapshot_download

                self.model_path = snapshot_download(
                    self.model_path, cache_dir=self.download_dir, revision=self.revision
                )
                self.tokenizer_path = snapshot_download(
                    self.tokenizer_path,
                    cache_dir=self.download_dir,
                    revision=self.revision,
                    ignore_patterns=["*.bin", "*.safetensors"],
                )

        # Mamba scheduler strategy
        if self.mamba_scheduler_strategy == "auto":
            # TODO: when extra_buffer is more verified, we can set the default path based on
            #       [overlap, non-overlap]
            self.mamba_scheduler_strategy = "no_buffer"

        # In speculative scenario:
        # - If `speculative_draft_model_quantization` is specified, the draft model uses this quantization method.
        # - Otherwise, the draft model defaults to the same quantization as the target model.
        if self.speculative_draft_model_quantization is None:
            self.speculative_draft_model_quantization = self.quantization
        elif self.speculative_draft_model_quantization == "unquant":
            self.speculative_draft_model_quantization = None

    def _handle_hpu_backends(self):
        if self.device == "hpu":
            self.attention_backend = "torch_native"
            self.sampling_backend = "pytorch"

    def _handle_cpu_backends(self):
        if self.device == "cpu":
            if self.attention_backend is None:
                self.attention_backend = "intel_amx"
            self.sampling_backend = "pytorch"

    def _handle_npu_backends(self):
        if self.device == "npu":
            from sglang.srt.hardware_backend.npu.utils import set_default_server_args

            set_default_server_args(self)

            if self.piecewise_cuda_graph_compiler != "eager":
                logger.warning(
                    "At this moment Ascend platform only support prefill graph compilation with "
                    "piecewise_cuda_graph_compiler='eager', change piecewise_cuda_graph_compiler to 'eager'."
                )
                self.piecewise_cuda_graph_compiler = "eager"

    def _handle_gpu_memory_settings(self, gpu_mem):
        """
        Configure GPU memory-dependent settings including
        chunked_prefill_size, cuda_graph_max_bs, and mem_fraction_static.

        Here are our heuristics:
        - Set chunked_prefill_size and cuda_graph_max_bs based on the GPU memory capacity.
          This is because GPUs with more memory are generally more powerful, we need to use a larger
          chunked_prefill_size and a larger cuda_graph_max_bs to fully utilize the GPU.
        - Then set mem_fraction_static based on chunked_prefill_size and cuda_graph_max_bs.

          GPU memory capacity = model weights + KV cache pool + activations + cuda graph buffers

          The argument mem_fraction_static is defined as (model weights + KV cache pool) / GPU memory capacity,
          or equivalently, mem_fraction_static = (GPU memory capacity - activations - cuda graph buffers) / GPU memory capacity.

          In order to compute mem_fraction_static, we need to estimate the size of activations and cuda graph buffers.
          The activation memory is proportional to the chunked_prefill_size.
          The cuda graph memory is proportional to the cuda_graph_max_bs.
          We use reserved_mem = chunked_prefill_size * 1.5 + cuda_graph_max_bs * 2 to estimate the size of activations and cuda graph buffers in GB.
          and set mem_fraction_static = (GPU memory capacity - reserved_mem) / GPU memory capacity.

          The coefficient 1.5 is a heuristic value, in the future, we can do better estimation by looking at the model types, hidden sizes or even do a dummy run.
        """
        if gpu_mem is not None:
            if gpu_mem < 20 * 1024:
                # T4, 4080
                # (chunked_prefill_size 2k, cuda_graph_max_bs 8)
                if self.chunked_prefill_size is None:
                    self.chunked_prefill_size = 2048
                if self.cuda_graph_max_bs is None:
                    self.cuda_graph_max_bs = 8
            elif gpu_mem < 35 * 1024:
                # A10, 4090, 5090
                # (chunked_prefill_size 2k, cuda_graph_max_bs 24 if tp < 4 else 80)
                if self.chunked_prefill_size is None:
                    self.chunked_prefill_size = 2048
                if self.cuda_graph_max_bs is None:
                    # Based on detailed statistics, when serving TP1/TP2 models on lower-end GPUs with HBM < 35GB, you can either disable cuda graph or set `cuda_graph_max_bs` to a very small value to reduce the memory overhead of creating cuda graphs, with almost no impact on performance.
                    # However, when serving models with TP4 or TP8, we need to enable cuda graph to maintain high performance. In this case, we can set `cuda_graph_max_bs` to 80 (half of the default value 160) to reduce the memory overhead of creating cuda graphs. Looking at the logs
                    # from TP4 serving of qwen2-72b, a value of 80 is sufficient and can reduce the memory overhead of creating cuda graphs on lower-end GPUs compared to the original 160, avoiding OOM issues.
                    if self.tp_size < 4:
                        self.cuda_graph_max_bs = 24
                    else:
                        self.cuda_graph_max_bs = 80
            elif gpu_mem < 60 * 1024:
                # A100 (40GB), L40,
                # (chunked_prefill_size 4k, cuda_graph_max_bs 32 if tp < 4 else 160)
                if self.chunked_prefill_size is None:
                    self.chunked_prefill_size = 4096
                if self.cuda_graph_max_bs is None:
                    if self.tp_size < 4:
                        self.cuda_graph_max_bs = 32
                    else:
                        self.cuda_graph_max_bs = 160
            elif gpu_mem < 90 * 1024:
                # H100, A100
                # (chunked_prefill_size 8k, cuda_graph_max_bs 256 if tp < 4 else 512)
                if self.chunked_prefill_size is None:
                    self.chunked_prefill_size = 8192
                if self.cuda_graph_max_bs is None:
                    if self.tp_size < 4:
                        self.cuda_graph_max_bs = 256
                    else:
                        self.cuda_graph_max_bs = 512
            elif gpu_mem < 160 * 1024:
                # H20, H200
                # (chunked_prefill_size 8k, cuda_graph_max_bs 256 if tp < 4 else 512)
                if self.chunked_prefill_size is None:
                    self.chunked_prefill_size = 8192
                if self.cuda_graph_max_bs is None:
                    if self.tp_size < 4:
                        self.cuda_graph_max_bs = 256
                    else:
                        self.cuda_graph_max_bs = 512
            else:
                # B200, MI300
                # (chunked_prefill_size 16k, cuda_graph_max_bs 512)
                if self.chunked_prefill_size is None:
                    self.chunked_prefill_size = 16384
                if self.cuda_graph_max_bs is None:
                    self.cuda_graph_max_bs = 512
        else:
            # Fallback defaults when gpu_mem is None
            if self.chunked_prefill_size is None:
                self.chunked_prefill_size = 4096
            if self.cuda_graph_max_bs is None:
                self.cuda_graph_max_bs = 160

        # Set cuda graph batch sizes
        if self.cuda_graph_bs is None:
            self.cuda_graph_bs = self._generate_cuda_graph_batch_sizes()
        else:
            self.cuda_graph_max_bs = max(self.cuda_graph_bs)

        if self.piecewise_cuda_graph_max_tokens is None:
            # Refer to pr #15927, by default we set the piecewise cuda graph max tokens to the chunked prefill size by default.
            # For MLA backend, the introduction of piecewise cuda graph will influence the kernel dispatch difference compared to the original mode.
            # To avoid the performance regression, we set the max tokens to 2048 by default.
            if not self.use_mla_backend():
                self.piecewise_cuda_graph_max_tokens = self.chunked_prefill_size
            else:
                self.piecewise_cuda_graph_max_tokens = 2048

        if self.piecewise_cuda_graph_tokens is None:
            self.piecewise_cuda_graph_tokens = (
                self._generate_piecewise_cuda_graph_tokens()
            )

        if self.mem_fraction_static is None:
            # Constant meta data (e.g., from attention backend)
            reserved_mem = 512
            # For activation during large prefill
            if self.chunked_prefill_size > 0:
                reserved_mem += max(self.chunked_prefill_size, 2048) * 1.5
            else:
                reserved_mem += max(self.max_prefill_tokens, 2048) * 1.5
            # For cuda graphs
            reserved_mem += self.cuda_graph_max_bs * 2
            # Some adjustments for large parallel size
            reserved_mem += self.tp_size * self.pp_size / 8 * 1024

            if self.enable_dp_attention:
                # DP attention needs more padding for some operations
                reserved_mem += self.cuda_graph_max_bs * self.dp_size * 3

                # DP attention uses much more memory for large cuda graph max bs,
                # likely due to some inefficiencies in torch allocator or our implementation.
                # So we need to reserve more memory.
                if self.cuda_graph_max_bs > 300:
                    reserved_mem += self.cuda_graph_max_bs * self.dp_size * 1.5

            # For piecewise cuda graphs
            if self.enable_piecewise_cuda_graph:
                # Only calculate the memory overhead for Non-Torch Memory use since the Torch Memory can be reused with Cuda Graph Capture
                reserved_mem += len(self.piecewise_cuda_graph_tokens) * 8

            if gpu_mem is not None and gpu_mem > 60 * 1024:
                reserved_mem = max(reserved_mem, 10 * 1024)

            if self.speculative_algorithm is not None:
                if self.speculative_algorithm == "STANDALONE":
                    # standalonedraft model and cuda graphs
                    reserved_mem += 6 * 1024
                elif self.speculative_algorithm != "NGRAM":
                    # eagle draft models and cuda graphs
                    reserved_mem += 2 * 1024

            self.mem_fraction_static = (
                round((gpu_mem - reserved_mem) / gpu_mem, 3)
                if gpu_mem is not None
                else 0.88
            )

            # Multimodal models need more memory for the image processing,
            # so we adjust the mem_fraction_static accordingly.
            model_config = self.get_model_config()
            if model_config.is_multimodal and not self.language_only:
                self.adjust_mem_fraction_for_vlm(model_config)

            # If symm mem is enabled and prealloc size is not set, set it to 4GB
            if (
                self.enable_symm_mem
                and not envs.SGLANG_SYMM_MEM_PREALLOC_GB_SIZE.is_set()
            ):
                envs.SGLANG_SYMM_MEM_PREALLOC_GB_SIZE.set(4)
                logger.warning(
                    "Symmetric memory is enabled, setting symmetric memory prealloc size to 4GB as default."
                    "Use environment variable SGLANG_SYMM_MEM_PREALLOC_GB_SIZE to change the prealloc size."
                )

    def _generate_cuda_graph_batch_sizes(self):
        """
        Generate the list of batch sizes for CUDA graph capture based on cuda_graph_max_bs.
        This integrates the logic from cuda_graph_runner.py.
        """
        # Handle disable_cuda_graph_padding as the first condition for both spec and non-spec
        if self.disable_cuda_graph_padding:
            capture_bs = list(range(1, self.cuda_graph_max_bs + 1))
        elif self.speculative_algorithm is None:
            # Normal case:
            capture_bs = (
                [1, 2, 4, 8, 12]
                + list(range(16, 257, 8))
                + list(range(272, 512, 16))
                + list(range(512, self.cuda_graph_max_bs + 1, 32))
            )
        else:
            # Spec decoding case: less padding for smaller batch sizes
            capture_bs = (
                list(range(1, 9, 1))
                + list(range(10, 33, 2))
                + list(range(40, 65, 4))
                + list(range(72, 257, 8))
                + list(range(272, self.cuda_graph_max_bs + 1, 16))
            )

        capture_bs = [bs for bs in capture_bs if bs <= self.cuda_graph_max_bs]

        return capture_bs

    def _generate_piecewise_cuda_graph_tokens(self):
        """
        Generate the list of batch sizes for piecewise CUDA graph capture
        based on piecewise_cuda_graph_max_tokens.
        """
        capture_sizes = (
            list(range(4, 33, 4))
            + list(range(48, 257, 16))
            + list(range(288, 513, 32))
            + list(range(576, 1024 + 1, 64))
            + list(range(1280, 4096 + 1, 256))
            + list(range(4608, self.piecewise_cuda_graph_max_tokens + 1, 512))
        )

        capture_sizes = [
            s for s in capture_sizes if s <= self.piecewise_cuda_graph_max_tokens
        ]

        return capture_sizes

    def _set_default_nsa_kv_cache_dtype(self, major: int) -> str:
        user_set_prefill = self.nsa_prefill_backend is not None
        user_set_decode = self.nsa_decode_backend is not None

        # If user specified a backend but didn't explicitly set kv_cache_dtype,
        # suggest them to be explicit about kv_cache_dtype to avoid surprises
        if (user_set_prefill or user_set_decode) and self.kv_cache_dtype == "auto":
            logger.warning(
                f"When specifying --nsa-prefill-backend or --nsa-decode-backend, "
                f"you should also explicitly set --kv-cache-dtype (e.g., 'fp8_e4m3' or 'bfloat16'). "
                f"DeepSeek V3.2 defaults to FP8 KV cache which may not be compatible with all backends."
            )

        if self.kv_cache_dtype == "auto":
            self.kv_cache_dtype = "fp8_e4m3" if major >= 10 else "bfloat16"
            logger.warning(
                f"Setting KV cache dtype to {self.kv_cache_dtype} for DeepSeek DSA on SM{major} device."
            )
        if self.kv_cache_dtype == "bf16":
            self.kv_cache_dtype = "bfloat16"
        assert self.kv_cache_dtype in [
            "bfloat16",
            "fp8_e4m3",
        ], "DeepSeek DSA only supports bf16/bfloat16 or fp8_e4m3 kv_cache_dtype"

    def _set_default_nsa_backends(self, kv_cache_dtype: str, major: int) -> str:
        user_set_prefill = self.nsa_prefill_backend is not None
        user_set_decode = self.nsa_decode_backend is not None

        if kv_cache_dtype == "fp8_e4m3":
            # flashmla_auto dispatches to flashmla_sparse/flashmla_kv based on hardware and heuristics
            if not user_set_prefill:
                self.nsa_prefill_backend = "flashmla_auto"
            if not user_set_decode:
                self.nsa_decode_backend = "flashmla_kv"
        else:
            # set prefill/decode backends based on hardware architecture.
            if major >= 10:
                if not user_set_prefill:
                    self.nsa_prefill_backend = "flashmla_sparse"
                if not user_set_decode:
                    self.nsa_decode_backend = "trtllm"
            else:
                # Hopper defaults for bfloat16
                if not user_set_prefill:
                    self.nsa_prefill_backend = "flashmla_sparse"
                if not user_set_decode:
                    self.nsa_decode_backend = "fa3"

        logger.warning(
            f"Set NSA backends for {self.kv_cache_dtype} KV Cache: prefill={self.nsa_prefill_backend}, decode={self.nsa_decode_backend}."
        )

    def _handle_model_specific_adjustments(self):
        from sglang.srt.configs.model_config import is_deepseek_nsa

        if parse_connector_type(self.model_path) == ConnectorType.INSTANCE:
            return

        hf_config = self.get_model_config().hf_config
        model_arch = hf_config.architectures[0]

        if model_arch in [
            "MistralLarge3ForCausalLM",
            "PixtralForConditionalGeneration",
        ]:
            self.dtype = "bfloat16"

        if model_arch in [
            "DeepseekV3ForCausalLM",
            "KimiK25ForConditionalGeneration",
            "MistralLarge3ForCausalLM",
            "PixtralForConditionalGeneration",
        ]:
            # Set attention backend for DeepSeek
            if is_deepseek_nsa(hf_config):  # DeepSeek 3.2
                if self.is_attention_backend_not_set():
                    self.attention_backend = "nsa"
                    logger.info("Use nsa attention backend for DeepSeek with DSA.")

                if not is_npu():  # CUDA or ROCm GPU
                    if self.enable_nsa_prefill_context_parallel:
                        logger.warning(
                            f"Context parallel feature is still under experiment. It has only been verified on Hopper platform."
                        )
                        if self.nsa_prefill_cp_mode == "in-seq-split":
                            # TODO Supports moe_dense_tp_size != 1, kv cache dtype = "fp8",moe_a2a_backend non-deepep and cross-machine operation .
                            self.enable_dp_attention = True
                            self.moe_dense_tp_size = 1
                            self.moe_a2a_backend = "deepep"
                            self.ep_size = self.tp_size
                            self.kv_cache_dtype = "bf16"
                            logger.warning(
                                f"For in-seq split mode, we have the following restrictions: moe_dense_tp_size == 1, moe_a2a_backend == deepep, ep_size == tp_size, kv_cache_dtype == bf16, batch_size == 1"
                            )
                        else:
                            self.enable_dp_attention = True
                            self.moe_dense_tp_size = 1
                            assert (
                                self.dp_size == 1
                            ), "For round-robin split mode, dp attention is not supported."
                        assert (
                            self.tp_size == 8
                        ), "Current multi-machine CP support suffers from precision issues. So context parallel only support Single machine(tp_size == 8)"

                        logger.warning(
                            f"Enable Context Parallel opt for deeeseekv3.2-DSA, Setting dp_size == {self.dp_size} and moe_dense_tp_size == {self.moe_dense_tp_size}, ep_size == {self.ep_size}, tp_size == {self.tp_size}, kv_cache_dtype == {self.kv_cache_dtype}, moe_a2a_backend {self.moe_a2a_backend} "
                        )
                    else:
                        # Pure TP and partial DP Attention mode is active for NSA, logging a warning
                        if self.dp_size < self.tp_size:
                            logger.warning(
                                f"DSA with TP mode is active, dp_size={self.dp_size}, tp_size={self.tp_size}, "
                                f"attn_tp_size={self.tp_size}, attention weights will be sharded across {self.tp_size} ranks."
                            )

                    if is_hip():
                        self.page_size = 1
                        logger.warning(
                            "Setting page size to 1 for DeepSeek DSA on ROCm."
                        )
                    else:
                        # For CUDA GPU
                        self.page_size = 64
                        logger.warning("Setting page size to 64 for DeepSeek DSA.")

                    import torch

                    major, _ = torch.cuda.get_device_capability()
                    self._set_default_nsa_kv_cache_dtype(major)
                    self._set_default_nsa_backends(self.kv_cache_dtype, major)

                if self.enable_nsa_prefill_context_parallel:
                    assert (
                        self.disaggregation_mode != "decode"
                    ), "CP is only supported for prefill when PD disaggregation, please remove --enable-nsa-prefill-context-parallel."

            else:
                # DeepSeek V3/R1/V3.1
                if self.enable_piecewise_cuda_graph:
                    logger.info("Piecewise CUDA graph is enabled, use MLA for prefill.")

                if is_sm100_supported():
                    if (
                        self.attention_backend is None
                        and self.prefill_attention_backend is None
                        and self.decode_attention_backend is None
                    ):
                        self.attention_backend = "trtllm_mla"
                        logger.info(
                            "Use trtllm_mla as attention backend on sm100 for DeepseekV3ForCausalLM"
                        )

            # Set moe backend for DeepSeek
            if is_sm100_supported():
                quant_method = get_quantization_config(hf_config)
                if self.quantization is None:
                    # Default DeepSeek V3/R1 native FP8 when not explicitly set,
                    # Because we need this condition for an assertion in
                    # flashinfer_trtllm MoE runner backend.
                    if quant_method is None and model_arch in ["DeepseekV3ForCausalLM"]:
                        self.quantization = "fp8"
                        logger.info(
                            "Quantization not specified, default to fp8 for DeepSeek on sm100"
                        )
                    else:
                        self.quantization = quant_method
                if (
                    self.moe_a2a_backend == "none"
                    and self.moe_runner_backend == "auto"
                    and self.quantization in ["fp8", "modelopt_fp8", "modelopt_fp4"]
                ):
                    self.moe_runner_backend = "flashinfer_trtllm"
                    logger.info(
                        "Use flashinfer_trtllm as MoE runner backend on sm100 for DeepseekV3ForCausalLM"
                    )

                if (
                    self.quantization == "modelopt_fp4"
                    and self.speculative_algorithm == "EAGLE"
                    and (
                        self.speculative_moe_runner_backend is None
                        or self.speculative_moe_a2a_backend is None
                    )
                ):
                    if envs.SGLANG_NVFP4_CKPT_FP8_NEXTN_MOE.get():
                        self.speculative_moe_runner_backend = "deep_gemm"
                        self.speculative_moe_a2a_backend = "deepep"
                        logger.info(
                            "Use deep_gemm moe runner and deepep a2a backend for bf16 nextn layer in deepseek fp4 checkpoint."
                        )
                    else:
                        self.speculative_moe_runner_backend = "triton"
                        self.speculative_moe_a2a_backend = "none"
                        logger.info(
                            "Use triton fused moe by default for bf16 nextn layer in deepseek fp4 checkpoint."
                        )

        elif model_arch in ["GptOssForCausalLM"]:
            # Set attention backend for GPT-OSS
            if self.is_attention_backend_not_set():
                if is_sm100_supported():
                    self.attention_backend = "trtllm_mha"
                elif is_sm90_supported():
                    self.attention_backend = "fa3"
                else:
                    self.attention_backend = "triton"

            supported_backends = ["triton", "trtllm_mha", "fa3", "fa4", "ascend"]
            prefill_attn_backend, decode_attn_backend = self.get_attention_backends()
            assert (
                prefill_attn_backend in supported_backends
                and decode_attn_backend in supported_backends
            ), (
                f"GptOssForCausalLM requires one of {supported_backends} attention backend, but got the following backends\n"
                f"- Prefill: {prefill_attn_backend}\n"
                f"- Decode: {decode_attn_backend}\n"
            )

            if (
                prefill_attn_backend == "trtllm_mha"
                or decode_attn_backend == "trtllm_mha"
            ):
                # TODO: support swa kv indices translation for trtllm_mha attention backend
                self.disable_hybrid_swa_memory = True
                logger.warning(
                    "Disable hybrid SWA memory for GPT-OSS model with trtllm_mha attention backend."
                )

            quant_method = get_quantization_config(hf_config)
            is_mxfp4_quant_format = quant_method == "mxfp4"
            if is_mxfp4_quant_format:
                # use bf16 for mxfp4 triton kernels
                self.dtype = "bfloat16"

            if self.moe_runner_backend == "auto":
                if self.enable_piecewise_cuda_graph:
                    self.moe_runner_backend = "auto"
                    logger.warning(
                        "Enable piecewise CUDA graph, enabling auto MOE kernel."
                    )
                elif is_blackwell_supported() and is_mxfp4_quant_format:
                    self.moe_runner_backend = "flashinfer_mxfp4"
                    logger.warning(
                        "Detected SM100 and MXFP4 quantization format for GPT-OSS model, enabling FlashInfer MXFP4 MOE kernel."
                    )
                elif (
                    is_hip() and get_bool_env_var("SGLANG_USE_AITER")
                ) and is_mxfp4_quant_format:
                    self.moe_runner_backend = "auto"
                    logger.warning(
                        "Detected ROCm and MXFP4 quantization format for GPT-OSS model, enabling aiter MXFP4 MOE kernel."
                    )
                elif self.ep_size == 1 and is_triton_kernels_available():
                    self.moe_runner_backend = "triton_kernel"
                    logger.warning(
                        "Detected GPT-OSS model, enabling triton_kernels MOE kernel."
                    )

            if self.moe_runner_backend == "triton_kernel":
                assert (
                    self.ep_size == 1
                ), "Triton kernel MoE is only supported when ep_size == 1"

        elif "MiMoV2FlashForCausalLM" in model_arch:
            if self.speculative_algorithm == "EAGLE":
                self.enable_multi_layer_eagle = True
                logger.info(
                    "Enable multi-layer EAGLE speculative decoding for MiMoV2FlashForCausalLM model."
                )
                if not envs.SGLANG_ENABLE_SPEC_V2.get():
                    envs.SGLANG_ENABLE_SPEC_V2.set(True)
                    logger.warning(
                        "Spec v2 is enabled for multi-layer EAGLE speculative decoding."
                    )

            if self.enable_hierarchical_cache:
                self.swa_full_tokens_ratio = 1.0
                logger.warning(
                    "Reset swa_full_tokens_ratio to 1.0 for MiMoV2FlashForCausalLM model with hierarchical cache"
                )
                self.disable_hybrid_swa_memory = True
                logger.warning(
                    "Disable hybrid SWA memory for MiMoV2FlashForCausalLM model with hierarchical cache"
                )
        elif "Step3p5ForCausalLM" in model_arch:
            if self.speculative_algorithm == "EAGLE":
                self.enable_multi_layer_eagle = True
                logger.info(
                    "Enable multi-layer EAGLE speculative decoding for Step3p5ForCausalLM model."
                )
                if not envs.SGLANG_ENABLE_SPEC_V2.get():
                    envs.SGLANG_ENABLE_SPEC_V2.set(True)
                    logger.warning(
                        "Spec v2 is enabled for multi-layer EAGLE speculative decoding."
                    )
            if self.enable_hierarchical_cache:
                self.swa_full_tokens_ratio = 1.0
                logger.warning(
                    "Reset swa_full_tokens_ratio to 1.0 for Step3p5ForCausalLM model with hierarchical cache"
                )
                self.disable_hybrid_swa_memory = True
                logger.warning(
                    "Disable hybrid SWA memory for Step3p5ForCausalLM model with hierarchical cache"
                )
        elif "Llama4" in model_arch and self.device != "cpu":
            # Auto-select attention backend for Llama4 if not specified
            if self.attention_backend is None:
                if is_sm100_supported():
                    self.attention_backend, platform = "trtllm_mha", "sm100"
                elif is_sm90_supported():
                    self.attention_backend, platform = "fa3", "sm90"
                elif is_hip():
                    self.attention_backend, platform = "aiter", "hip"
                elif self.device == "xpu":
                    self.attention_backend, platform = "intel_xpu", "xpu"
                else:
                    self.attention_backend, platform = "triton", "other platforms"
                logger.warning(
                    f"Use {self.attention_backend} as attention backend on {platform} for Llama4 model"
                )
            assert self.attention_backend in {
                "fa3",
                "aiter",
                "triton",
                "trtllm_mha",
                "intel_xpu",
            }, f"fa3, aiter, triton, trtllm_mha or intel_xpu is required for Llama4 model but got {self.attention_backend}"
            if is_sm100_supported() and self.moe_runner_backend == "auto":
                if self.quantization in {"fp8", "modelopt_fp8"}:
                    self.moe_runner_backend = "flashinfer_trtllm"
                    logger.info(
                        "Use flashinfer_trtllm as MoE runner backend on SM100 for Llama4"
                    )
        elif model_arch in [
            "Gemma2ForCausalLM",
            "Gemma3ForCausalLM",
            "Gemma3ForConditionalGeneration",
            "Gemma3nForCausalLM",
            "Gemma3nForConditionalGeneration",
        ]:
            # FIXME: https://github.com/sgl-project/sglang/pull/7367 is not compatible with gemma2 model.
            # It failed at this test: https://github.com/sgl-project/sglang/actions/runs/16255155597/job/45890331952#step:4:736
            logger.warning(
                f"Disable hybrid SWA memory for {model_arch} as it is not yet supported."
            )
            self.disable_hybrid_swa_memory = True
        elif model_arch in ["Exaone4ForCausalLM", "ExaoneMoEForCausalLM"]:
            if hf_config.sliding_window_pattern is not None:
                logger.warning(
                    f"Disabling hybrid SWA memory for {model_arch} as it is not yet supported."
                )
                self.disable_hybrid_swa_memory = True
                # https://docs.sglang.ai/advanced_features/attention_backend.html
                accepted_backends = ["fa3", "triton", "trtllm_mha"]
                assert (
                    self.attention_backend in accepted_backends
                ), f"One of the attention backends in {accepted_backends} is required for {model_arch}, but got {self.attention_backend}"
        elif model_arch in ["Olmo2ForCausalLM"]:
            # FIXME: https://github.com/sgl-project/sglang/pull/7367 is not compatible with Olmo3 model.
            logger.warning(
                f"Disabling hybrid SWA memory for {model_arch} as it is not yet supported."
            )
            self.disable_hybrid_swa_memory = True

            if self.attention_backend is None:
                if is_cuda() and is_sm100_supported():
                    self.attention_backend = "trtllm_mha"
                elif is_cuda() and get_device_sm() >= 80:
                    self.attention_backend = "fa3"
                else:
                    self.attention_backend = "triton"

            # Flashinfer appears to degrade performance when sliding window attention
            # is used for the Olmo2 architecture. Olmo2 does not use sliding window attention
            # but Olmo3 does.
            assert (
                self.attention_backend != "flashinfer"
            ), "FlashInfer backend can significantly degrade the performance of Olmo3 models."

            logger.info(
                f"Using {self.attention_backend} as attention backend for {model_arch}."
            )
        elif model_arch in ["KimiLinearForCausalLM"]:
            self._handle_mamba_radix_cache(
                model_arch=model_arch,
                support_mamba_cache=False,
            )
        elif model_arch in ["NemotronHForCausalLM"]:
            model_config = self.get_model_config()
            if model_config.quantization in [
                "modelopt",
                "modelopt_fp8",
                "modelopt_fp4",
            ]:
                assert model_config.hf_config.mlp_hidden_act == "relu2"
                if model_config.quantization == "modelopt":
                    self.quantization = (
                        "modelopt_fp4"
                        if model_config.hf_config.quantization_config["quant_algo"]
                        == "NVFP4"
                        else "modelopt_fp8"
                    )
                else:
                    self.quantization = model_config.quantization
                self.moe_runner_backend = "flashinfer_cutlass"

            self._handle_mamba_radix_cache(
                model_arch=model_arch,
                support_mamba_cache_extra_buffer=False,
                sm100_default_attention_backend="flashinfer",
            )
            assert self.attention_backend != "triton", (
                "NemotronHForCausalLM does not support triton attention backend,"
                "as the first layer might not be an attention layer"
            )
        elif model_arch in [
            "Qwen3MoeForCausalLM",
            "Qwen3VLMoeForConditionalGeneration",
        ]:
            if is_sm100_supported():
                quant_method = get_quantization_config(hf_config)
                if self.quantization is None and quant_method is not None:
                    self.quantization = quant_method
                if (
                    (
                        self.quantization in ("fp8", "modelopt_fp4")
                        or self.quantization is None
                    )
                    and self.moe_a2a_backend == "none"
                    and self.moe_runner_backend == "auto"
                ):
                    self.moe_runner_backend = "flashinfer_trtllm"
                    logger.info(
                        "Use flashinfer_trtllm as MoE runner backend on sm100 for "
                        f"{model_arch}"
                    )
        elif model_arch in ["Qwen3NextForCausalLM"]:
            if is_sm100_supported():
                quant_method = get_quantization_config(hf_config)
                if self.quantization is None and quant_method is not None:
                    self.quantization = quant_method
                if (
                    (
                        self.quantization in ("fp8", "modelopt_fp4")
                        or self.quantization is None
                    )
                    and self.moe_a2a_backend == "none"
                    and self.moe_runner_backend == "auto"
                ):
                    self.moe_runner_backend = "flashinfer_trtllm"
                    logger.info(
                        "Use flashinfer_trtllm as MoE runner backend on sm100 for Qwen3NextForCausalLM"
                    )
            self._handle_mamba_radix_cache(
                model_arch=model_arch,
                support_mamba_cache_extra_buffer=True,
                sm100_default_attention_backend="triton",
            )

        elif model_arch in ["Glm4MoeForCausalLM"]:
            if is_sm100_supported():
                quantization_config = getattr(hf_config, "quantization_config", None)
                quant_method = (
                    quantization_config.get("quant_method")
                    if quantization_config is not None
                    else None
                )
                if self.quantization is None and quant_method is not None:
                    self.quantization = quant_method
                if (
                    self.quantization == "modelopt_fp4"
                    and self.moe_a2a_backend == "none"
                    and self.moe_runner_backend == "auto"
                ):
                    # Only enable flashinfer_trtllm if flashinfer-python version is >= 0.6.2
                    if check_pkg_version_at_least("flashinfer-python", "0.6.2"):
                        self.moe_runner_backend = "flashinfer_trtllm"
                        logger.info(
                            "Use flashinfer_trtllm as MoE runner backend on sm100 for Glm4MoeForCausalLM"
                        )

        elif model_arch in [
            "FalconH1ForCausalLM",
            "JetNemotronForCausalLM",
            "JetVLMForConditionalGeneration",
        ]:
            self._handle_mamba_radix_cache(
                model_arch=model_arch,
                support_mamba_cache_extra_buffer=False,
                sm100_default_attention_backend="triton",
            )

        elif model_arch in ["Lfm2ForCausalLM"]:
            self._handle_mamba_radix_cache(
                model_arch=model_arch,
                support_mamba_cache_extra_buffer=False,
                sm100_default_attention_backend="flashinfer",
            )
            assert self.attention_backend != "triton", (
                f"{model_arch} does not support triton attention backend, "
                "as the first layer might not be an attention layer"
            )

        if envs.SGLANG_EMBEDDINGS_SPARSE_HEAD.is_set():
            self.disable_overlap_schedule = True
            logger.warning(
                f"Overlap scheduler is disabled when using sparse head for embedding model."
            )

        # TRTLLM AllReduce Fusion supports SM90/100, enable it by default
        # for models with explicit support (DeepseekV3, GptOss, Glm4Moe, Qwen3Moe)
        # TODO: currently, it is only supported in the single node scenario. https://github.com/flashinfer-ai/flashinfer/issues/2006
        # TODO: there is currently a bug on H20 device specifically, https://github.com/flashinfer-ai/flashinfer/issues/2204
        device_name = get_device_name()
        is_h20_device = (
            device_name and "H20" in device_name and "H200" not in device_name
        )
        if (
            not self.enable_flashinfer_allreduce_fusion
            and model_arch
            in [
                "DeepseekV3ForCausalLM",
                "GptOssForCausalLM",
                "Glm4MoeForCausalLM",
                "Glm4MoeLiteForCausalLM",
                "Qwen3MoeForCausalLM",
                "KimiK25ForConditionalGeneration",
            ]
            and (is_sm90_supported() or is_sm100_supported())
            and not self.enable_dp_attention
            and self.nnodes == 1
            and not is_h20_device
            and self.moe_a2a_backend == "none"
        ):
            self.enable_flashinfer_allreduce_fusion = True

    def _handle_mamba_radix_cache(
        self,
        model_arch: str,
        support_mamba_cache: bool = True,
        support_mamba_cache_extra_buffer: bool = True,
        sm100_default_attention_backend: str = None,
    ):
        if (
            is_sm100_supported()
            and self.attention_backend is None
            and sm100_default_attention_backend is not None
        ):
            self.attention_backend = sm100_default_attention_backend
            logger.info(
                f"Use {sm100_default_attention_backend} as attention backend on sm100 for {model_arch}"
            )

        if not support_mamba_cache:
            logger.warning(
                f"Disabling Radix Cache for {model_arch} as it is not yet supported."
            )
            self.disable_radix_cache = True
            return

        if not support_mamba_cache_extra_buffer:
            assert (
                not self.enable_mamba_extra_buffer()
            ), f"mamba extra_buffer is not supported for {model_arch} model"
        elif self.enable_mamba_extra_buffer():  # extra_buffer
            assert (
                is_cuda()
            ), "Mamba extra_buffer is only supported on CUDA devices with FLA backend"
            if self.speculative_num_draft_tokens is not None:
                assert (
                    self.mamba_track_interval >= self.speculative_num_draft_tokens
                ), f"mamba_track_interval {self.mamba_track_interval} must be greater than or equal to speculative_num_draft_tokens {self.speculative_num_draft_tokens}"

            if self.page_size is not None:
                assert (
                    self.mamba_track_interval % self.page_size == 0
                ), f"mamba_track_interval {self.mamba_track_interval} must be divisible by page_size {self.page_size}"
                assert (
                    max(FLA_CHUNK_SIZE, self.page_size)
                    % min(FLA_CHUNK_SIZE, self.page_size)
                    == 0
                ), f"For SSM models with extra buffer, either FLA_CHUNK_SIZE or page_size must be divisible by the other, got {FLA_CHUNK_SIZE=}, {self.page_size=}"
        elif not self.disable_radix_cache:  # no_buffer
            if self.speculative_algorithm is None:
                logger.warning(
                    "Disabling overlap schedule since mamba no_buffer is not compatible with "
                    "overlap schedule, try to use --disable-radix-cache if overlap schedule is necessary"
                )
                self.disable_overlap_schedule = True
                if self.attention_backend == "trtllm_mha":
                    logger.warning(
                        "Disabling radix cache since trtllm_mha does not support page_size = 1, which is required by MambaRadixCache. "
                        "Try to use --attention-backend triton if radix cache is necessary."
                    )
                    self.disable_radix_cache = True
                    self.disable_overlap_schedule = False
            else:
                logger.warning(
                    f"Disabling radix cache since speculative decoding for {model_arch} is not supported with radix cache yet."
                )
                self.disable_radix_cache = True

    def _handle_sampling_backend(self):
        if self.sampling_backend is None:
            self.sampling_backend = (
                "flashinfer" if is_flashinfer_available() else "pytorch"
            )

    def _handle_attention_backend_compatibility(self):
        model_config = self.get_model_config()
        use_mla_backend = self.use_mla_backend()

        if self.prefill_attention_backend is not None and (
            self.prefill_attention_backend == self.decode_attention_backend
        ):  # override the default attention backend
            self.attention_backend = self.prefill_attention_backend

        # Pick the default attention backend if not specified
        if self.attention_backend is None:
            """
            Auto select the fastest attention backend.

            1. Models with MHA Architecture (e.g: Llama, QWen)
                1.1 We will turn on FA3 on hopper unless user use spec decode with topk > 1 or page_size > 1.
                1.2 Use trtllm_mha for SM100/SM103 (Blackwell B200/GB200/B300) excluding spec with topk > 1.
                   Note: trtllm_mha does not support SM120, which will fall back to flashinfer.
                1.3 In other cases, we will use flashinfer if available, otherwise use triton.
            2. Models with MLA Architecture and using FA3
                2.1 We will use FA3 backend on hopper.
                2.2 We will use Flashinfer backend on blackwell.
                2.3 Otherwise, we will use triton backend.
            """

            if not use_mla_backend:
                # MHA architecture
                if is_hopper_with_cuda_12_3() and is_no_spec_infer_or_topk_one(self):
                    # Note: flashinfer 0.6.1 caused performance regression on Hopper attention kernel
                    # Before the kernel is fixed, we choose fa3 as the default backend on Hopper MHA
                    # ref: https://github.com/sgl-project/sglang/issues/17411
                    self.attention_backend = "fa3"
                elif (
                    is_sm100_supported()
                    and is_no_spec_infer_or_topk_one(self)
                    and (
                        self.speculative_algorithm is None
                        or self.speculative_eagle_topk is not None
                    )
                ):
                    self.attention_backend = "trtllm_mha"
                elif is_hip():
                    self.attention_backend = "aiter"
                else:
                    self.attention_backend = (
                        "flashinfer" if is_flashinfer_available() else "triton"
                    )
            else:
                # MLA architecture
                if is_hopper_with_cuda_12_3():
                    self.attention_backend = "fa3"
                elif is_sm100_supported():
                    self.attention_backend = "flashinfer"
                elif is_hip():
                    head_num = model_config.get_num_kv_heads(self.tp_size)
                    # TODO current aiter only support head number 16 or 128 head number
                    if head_num == 128 or head_num == 16:
                        self.attention_backend = "aiter"
                    else:
                        self.attention_backend = "triton"
                else:
                    self.attention_backend = "triton"

            logger.info(
                f"Attention backend not specified. Use {self.attention_backend} backend by default."
            )

        # Torch native and flex attention backends
        if self.attention_backend == "torch_native":
            logger.warning(
                "Cuda graph is disabled because of using torch native attention backend"
            )
            self.disable_cuda_graph = True

        if self.attention_backend == "flex_attention":
            logger.warning(
                "Cuda graph is disabled because of using torch Flex Attention backend"
            )
            self.disable_cuda_graph = True
            assert (
                self.speculative_algorithm is None
            ), "Speculative decoding is currently not supported with Flex Attention backend"

        # Major NVIDIA platforms backends
        if (
            self.attention_backend == "flashmla"
            or self.decode_attention_backend == "flashmla"
        ):
            logger.warning(
                "FlashMLA only supports a page_size of 64, change page_size to 64."
            )
            self.page_size = 64

        if (
            self.attention_backend == "cutlass_mla"
            or self.decode_attention_backend == "cutlass_mla"
        ):
            logger.warning(
                "Cutlass MLA only supports a page_size of 128, change page_size to 128."
            )
            self.page_size = 128

        if (
            self.attention_backend == "trtllm_mla"
            or self.decode_attention_backend == "trtllm_mla"
        ):
            if not is_blackwell_supported():
                raise ValueError(
                    "TRTLLM MLA backend is only supported on Blackwell GPUs (SM100). Please use a different backend."
                )

            if self.page_size not in [32, 64]:
                logger.warning(
                    f"TensorRT-LLM MLA only supports page_size of 32 or 64, changing page_size from {self.page_size} to 64."
                )
                self.page_size = 64

            if self.kv_cache_dtype not in ["fp8_e4m3", "fp4_e2m1", "bf16", "auto"]:
                raise ValueError(
                    "TensorRT-LLM MLA backend only supports kv-cache-dtype of fp8_e4m3, fp4_e2m1, bf16, or auto."
                )

        if (
            self.attention_backend == "trtllm_mha"
            or self.decode_attention_backend == "trtllm_mha"
            or self.prefill_attention_backend == "trtllm_mha"
        ):
            if not is_sm100_supported():
                raise ValueError(
                    "TRTLLM MHA backend is only supported on Blackwell GPUs (SM100). Please use a different backend."
                )

            if self.page_size not in [16, 32, 64]:
                logger.warning(
                    f"TensorRT-LLM MHA only supports page_size of 16, 32 or 64, changing page_size from {self.page_size} to 64."
                )
                self.page_size = 64

        if self.attention_backend == "fa3" and self.kv_cache_dtype == "fp8_e5m2":
            logger.warning(
                "FlashAttention3 only supports fp8_e4m3 if using FP8; "
                "Setting attention backend to triton."
            )
            self.attention_backend = "triton"

        if self.prefill_attention_backend == "fa4" and not self.use_mla_backend():
            logger.warning(
                f"FA4 backend only supports page size 128 for non-MLA model architectures, changing page_size from {self.page_size} to 128."
            )
            self.page_size = 128

        # AMD platforms backends
        if self.attention_backend == "aiter":
            if model_config.context_len > 8192:
                self.mem_fraction_static *= 0.85

        # Other platforms backends
        if (
            self.attention_backend == "intel_amx"
            and self.device == "cpu"
            and not cpu_has_amx_support()
        ):
            logger.warning(
                "The current platform does not support Intel AMX, will fallback to torch_native backend."
            )
            self.attention_backend = "torch_native"

        if (
            self.attention_backend == "intel_xpu"
            and self.device == "xpu"
            and not xpu_has_xmx_support()
        ):
            logger.warning(
                "The current platform does not support Intel XMX, will fallback to triton backend."
            )
            self.attention_backend = "triton"

        if self.attention_backend == "intel_xpu":
            if self.page_size not in [32, 64, 128]:
                logger.warning(
                    f"Intel XPU attention backend only supports page_size of 32, 64 or 128, changing page_size from {self.page_size} to 128."
                )
                self.page_size = 128

        # Dual chunk flash attention backend
        if (
            getattr(model_config.hf_config, "dual_chunk_attention_config", None)
            is not None
        ):
            if self.attention_backend is None:
                self.attention_backend = "dual_chunk_flash_attn"
                logger.info("Dual chunk attention is turned on by default.")
            elif self.attention_backend != "dual_chunk_flash_attn":
                raise ValueError(
                    "Dual chunk attention is enabled, but attention backend is set to "
                    f"{self.attention_backend}. Please set it to 'dual_chunk_flash_attn'."
                )
        if self.attention_backend == "dual_chunk_flash_attn":
            logger.warning(
                "Mixed chunk and radix cache are disabled when using dual-chunk flash attention backend"
            )
            self.enable_mixed_chunk = False
            self.disable_radix_cache = True

    def _handle_kv4_compatibility(self):
        """Check FP4 KV cache compatibility with the attention backend"""
        if self.kv_cache_dtype != "fp4_e2m1":
            return

        use_mla_backend = self.use_mla_backend()
        # self.attention_backend didn't overwrite self.prefill/decode_attention_backend yet
        self.prefill_attention_backend_str, self.decode_attention_backend_str = (
            self.get_attention_backends()
        )

        if is_cuda():
            if (
                self.prefill_attention_backend_str != self.decode_attention_backend_str
                and self.prefill_attention_backend_str != "fa4"
            ):  # Take care of prefill=fa4 later
                logger.warning(
                    f"Attention: Using KV4 with PREFILL = {self.prefill_attention_backend_str} "
                    f"and DECODE = {self.decode_attention_backend_str}. "
                    f"Compatibility issues are unlikely, but may occur in rare edge cases."
                )
            else:
                if self.prefill_attention_backend_str == "fa4":
                    if use_mla_backend:  # FA4 + MLA
                        KV4_FA4_MLA_BACKEND_CHOICES = [
                            "cutlass_mla",
                            "flashinfer",
                            "trtllm_mla",
                        ]
                        assert (
                            self.decode_attention_backend_str
                            in KV4_FA4_MLA_BACKEND_CHOICES
                        ), (
                            f"KV4 FA4 MLA expects decode_attention_backend to be one of "
                            f"{KV4_FA4_MLA_BACKEND_CHOICES}, but got {self.decode_attention_backend_str}"
                        )
                    else:  # FA4 + MHA
                        KV4_FA4_MHA_BACKEND_CHOICES = [
                            "triton",
                            "torch_native",
                            "flex_attention",
                        ]
                        assert (
                            self.decode_attention_backend_str
                            in KV4_FA4_MHA_BACKEND_CHOICES
                        ), (
                            f"KV4 FA4 MHA expects decode_attention_backend to be one of "
                            f"{KV4_FA4_MHA_BACKEND_CHOICES}, but got {self.decode_attention_backend_str}"
                        )
                else:
                    if use_mla_backend:  # !FA4 + MLA
                        KV4_ATTENTION_MLA_BACKEND_CHOICES = [
                            "cutlass_mla",
                            "flashinfer",
                            "trtllm_mla",
                            "flashmla",
                        ]
                        assert (
                            self.attention_backend in KV4_ATTENTION_MLA_BACKEND_CHOICES
                        ), (
                            f"KV4 MLA expects attention_backend to be one of "
                            f"{KV4_ATTENTION_MLA_BACKEND_CHOICES}, but got {self.attention_backend}"
                        )
                    else:  # !FA4 + MHA
                        KV4_ATTENTION_MHA_BACKEND_CHOICES = [
                            "triton",
                            "torch_native",
                            "flex_attention",
                            "trtllm_mha",
                        ]
                        assert (
                            self.attention_backend in KV4_ATTENTION_MHA_BACKEND_CHOICES
                        ), (
                            f"KV4 MHA expects attention_backend to be one of "
                            f"{KV4_ATTENTION_MHA_BACKEND_CHOICES}, but got {self.attention_backend}"
                        )
        else:
            raise RuntimeError("KV4 is not tested on non-CUDA platforms.")

    def _handle_page_size(self):
        if self.page_size is None:
            self.page_size = 1

    def _handle_amd_specifics(self):
        if is_hip():
            self.triton_attention_num_kv_splits = 16

    def _handle_grammar_backend(self):
        if self.grammar_backend is None:
            self.grammar_backend = "xgrammar"

    def _handle_data_parallelism(self):
        if self.dp_size == 1:
            self.enable_dp_attention = False
            self.enable_dp_lm_head = False

        if self.enable_dp_attention:
            self.schedule_conservativeness = self.schedule_conservativeness * 0.3
            assert self.tp_size % self.dp_size == 0
            self.chunked_prefill_size = self.chunked_prefill_size // self.dp_size
            logger.warning(
                f"DP attention is enabled. The chunked prefill size is adjusted to {self.chunked_prefill_size} to avoid MoE kernel issues. "
            )

        if self.enable_dp_lm_head:
            assert (
                self.enable_dp_attention
            ), "Please enable dp attention when setting enable_dp_lm_head. "

    def _handle_moe_kernel_config(self):
        if self.quantization == "mxfp8":
            if self.moe_runner_backend not in ["auto", "cutlass"]:
                logger.warning(
                    "mxfp8 quantization forces --moe-runner-backend=cutlass. "
                    f"Overriding {self.moe_runner_backend!r}."
                )
            self.moe_runner_backend = "cutlass"

        if self.moe_runner_backend == "flashinfer_cutlass":
            assert self.quantization in [
                "modelopt_fp4",
                "modelopt_fp8",
                None,
            ], f"Invalid quantization '{self.quantization}'. \nFlashInfer Cutlass MOE supports only: 'modelopt_fp4', 'modelopt_fp8', or bfloat16 (None)."
            assert self.ep_size in [
                1,
                self.tp_size,
            ], "The expert parallel size must be 1 or the same as the tensor parallel size"

        if self.moe_runner_backend == "flashinfer_trtllm":
            assert self.quantization in [
                "modelopt_fp4",
                "fp8",
                "modelopt_fp8",
                "compressed-tensors",
                None,
            ], f"Invalid quantization '{self.quantization}'. \nFlashInfer TRTLLM MOE supports only: 'modelopt_fp4', 'fp8', 'modelopt_fp8', 'compressed-tensors', or bfloat16 (None)."
            self.disable_shared_experts_fusion = True
            logger.warning(
                "FlashInfer TRTLLM MoE is enabled. --disable-shared-experts-fusion is automatically set."
            )

        if get_bool_env_var("SGLANG_CUTLASS_MOE"):
            logger.warning(
                "SGLANG_CUTLASS_MOE is deprecated, use --moe-runner-backend=cutlass and/or --speculative-moe-runner-backend=cutlass instead"
            )
            assert self.quantization in [
                "fp8",
                "mxfp8",
            ], "cutlass MoE is only supported with fp8/mxfp8 quantization"
            self.moe_runner_backend = "cutlass"
        if self.moe_runner_backend == "cutlass" and self.quantization in [
            "fp8",
            "mxfp8",
        ]:
            assert (
                self.ep_size == 1
            ), "FP8/MXFP8 Cutlass MoE is only supported with ep_size == 1"

    def _handle_a2a_moe(self):
        if self.moe_a2a_backend == "deepep":
            if self.deepep_mode == "normal":
                logger.warning("Cuda graph is disabled because deepep_mode=`normal`")
                self.disable_cuda_graph = True
            self.ep_size = self.tp_size
            logger.warning(
                f"DeepEP MoE is enabled. The expert parallel size is adjusted to be the same as the tensor parallel size[{self.tp_size}]."
            )

        if self.moe_a2a_backend == "mooncake":
            self.ep_size = self.tp_size
            logger.warning(
                f"Mooncake MoE is enabled. The expert parallel size is adjusted to be the same as the tensor parallel size[{self.tp_size}]."
            )

        if self.moe_a2a_backend == "ascend_fuseep":
            self.ep_size = self.tp_size
            logger.warning(
                f"Ascend fused EP MoE is enabled. The expert parallel size is adjusted to be the same as the tensor parallel size[{self.tp_size}]."
            )
        if self.moe_a2a_backend == "flashinfer":
            self.ep_size = self.tp_size
            logger.warning(
                f"Flashinfer MoE A2A is enabled. The expert parallel size is adjusted to be the same as the tensor parallel size[{self.tp_size}]."
            )
            self.disable_shared_experts_fusion = True
            logger.warning(
                "Flashinfer MoE A2A is enabled. --disable-shared-experts-fusion is automatically set."
            )
            if self.deepep_mode != "auto":
                logger.warning("--deepep-mode is ignored for Flashinfer MoE A2A")
            if os.environ.get("SGLANG_MOE_NVFP4_DISPATCH") is None:
                envs.SGLANG_MOE_NVFP4_DISPATCH.set(True)
                logger.warning(
                    "SGLANG_MOE_NVFP4_DISPATCH is set to True for Flashinfer MoE A2A"
                )
            assert self.moe_runner_backend in [
                "flashinfer_cutlass"
            ], "Flashinfer MoE A2A is only supported with flashinfer_cutlass moe runner backend"

        if self.moe_a2a_backend == "mori":
            self.ep_size = self.tp_size
            self.deepep_mode = "normal"
            logger.warning("auto set deepep_mode=`normal` for MORI EP")
            logger.warning(
                f"MoRI MoE is enabled. The expert parallel size is adjusted to be the same as the tensor parallel size[{self.tp_size}]."
            )

            assert (self.chunked_prefill_size) <= get_int_env_var(
                "SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK", 4096
            ), "SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK (default 4096) must be larger or equal to chunked_prefill_size"

    def _handle_eplb_and_dispatch(self):
        if self.enable_eplb and (self.expert_distribution_recorder_mode is None):
            self.expert_distribution_recorder_mode = "stat"
            logger.warning(
                "EPLB is enabled. The expert_distribution_recorder_mode is automatically set."
            )

        if (self.enable_eplb or (self.init_expert_location != "trivial")) and (
            self.ep_dispatch_algorithm is None
        ):
            self.ep_dispatch_algorithm = "static"

        if self.enable_eplb:
            assert self.ep_size > 1

    def _handle_elastic_ep(self):
        if self.elastic_ep_backend is not None:
            if self.enable_eplb:
                if self.eplb_algorithm == "auto":
                    self.eplb_algorithm = "elasticity_aware"
                assert (
                    self.eplb_algorithm == "elasticity_aware"
                ), "Elastic EP requires eplb_algorithm to be set to 'auto' or 'elasticity_aware'."

            if self.elastic_ep_backend == "mooncake":
                self.mooncake_ib_device = self._validate_ib_devices(
                    self.mooncake_ib_device
                )

    def _handle_expert_distribution_metrics(self):
        if self.enable_expert_distribution_metrics and (
            self.expert_distribution_recorder_mode is None
        ):
            self.expert_distribution_recorder_mode = "stat"

        if self.expert_distribution_recorder_buffer_size is None:
            if (x := self.eplb_rebalance_num_iterations) is not None:
                self.expert_distribution_recorder_buffer_size = x
            elif self.expert_distribution_recorder_mode is not None:
                self.expert_distribution_recorder_buffer_size = 1000

    def _handle_pipeline_parallelism(self):
        if self.pp_size > 1:
            self.disable_overlap_schedule = True
            logger.warning(
                "Pipeline parallelism is incompatible with overlap schedule."
            )

    def _handle_hicache(self):
        if (
            self.hicache_mem_layout == "page_first_direct"
            and self.hicache_io_backend == "kernel"
        ):
            self.hicache_io_backend = "direct"
            logger.warning(
                "Kernel io backend does not support page first direct layout"
            )

        if (
            self.enable_hierarchical_cache
            or self.disaggregation_decode_enable_offload_kvcache
        ) and self.hicache_io_backend == "kernel":
            # fix for the compatibility issue with FlashAttention3 decoding and HiCache kernel backend
            # Only override when the *effective* decode backend would be FA3.
            # Otherwise, respect the user's chosen attention backend (e.g., aiter on ROCm).
            effective_decode_backend = (
                self.decode_attention_backend
                if self.decode_attention_backend is not None
                else self.attention_backend
            )
            if effective_decode_backend == "fa3":
                if self.decode_attention_backend is None:
                    # If decode backend wasn't explicitly set, pick a safe default that works with HiCache kernel IO.
                    if not self.use_mla_backend():
                        self.decode_attention_backend = (
                            "flashinfer" if is_flashinfer_available() else "triton"
                        )
                    else:
                        self.decode_attention_backend = (
                            "flashinfer" if is_sm100_supported() else "triton"
                        )
                else:
                    # If user explicitly requested FA3 decode, fall back to direct IO.
                    self.hicache_io_backend = "direct"
                    logger.warning(
                        "FlashAttention3 decode backend is not compatible with hierarchical cache. "
                        "Setting hicache_io_backend to vanilla I/O, which may lead to suboptimal performance with small page sizes."
                    )

        if self.hicache_storage_backend == "mooncake":
            if self.hicache_mem_layout == "layer_first":
                if self.hicache_io_backend == "direct":
                    self.hicache_mem_layout = "page_first_direct"
                elif self.hicache_io_backend == "kernel":
                    self.hicache_mem_layout = "page_first"
                logger.warning(
                    f"Mooncake storage backend does not support layer_first layout, "
                    f"switching to {self.hicache_mem_layout} layout for {self.hicache_io_backend} io backend"
                )

    def _handle_speculative_decoding(self):
        if (
            self.speculative_draft_model_path is not None
            and self.speculative_draft_model_revision is None
        ):
            self.speculative_draft_model_revision = "main"

        # Avoid using flashinfer_trtllm for speculative MoE runner backend by default
        # TODO: Remove this block after verifying no accuracy regression with flashinfer_trtllm speculative backend
        from sglang.srt.layers.moe.utils import MoeRunnerBackend

        if self.speculative_moe_runner_backend is None:
            self.speculative_moe_runner_backend = (
                "auto"
                if self.moe_runner_backend == "flashinfer_trtllm"
                else self.moe_runner_backend
            )
        else:
            assert not MoeRunnerBackend(
                self.speculative_moe_runner_backend
            ).is_flashinfer_trtllm(), "Currently speculative MoE runner backend cannot be flashinfer_trtllm for risk in some draft models."

        if self.speculative_algorithm == "NEXTN":
            self.speculative_algorithm = "EAGLE"

        if self.speculative_algorithm in ("EAGLE", "EAGLE3", "STANDALONE"):
            if self.speculative_algorithm == "STANDALONE" and self.enable_dp_attention:
                # TODO: support dp attention for standalone speculative decoding
                raise ValueError(
                    "Currently standalone speculative decoding does not support dp attention."
                )

            if self.max_running_requests is None:
                self.max_running_requests = 48
                logger.warning(
                    "Max running requests is reset to 48 for speculative decoding. You can override this by explicitly setting --max-running-requests."
                )

            if (
                self.speculative_algorithm in ["EAGLE", "EAGLE3", "STANDALONE"]
                and envs.SGLANG_ENABLE_SPEC_V2.get()
            ):
                self.disable_overlap_schedule = False
                logger.warning(
                    "Spec v2 is enabled for eagle/eagle3 speculative decoding and overlap schedule is turned on."
                )
                if (
                    self.speculative_eagle_topk is not None
                    and self.speculative_eagle_topk > 1
                ):
                    raise ValueError(
                        "Spec v2 currently only supports topk = 1 for speculative decoding."
                    )
            else:
                self.disable_overlap_schedule = True
                logger.warning(
                    "Overlap scheduler is disabled when spec v2 is off or using unsupported speculative algorithm. "
                    "You can set env SGLANG_ENABLE_SPEC_V2=True to enable the experimental overlap scheduler. "
                )

            if self.enable_mixed_chunk:
                self.enable_mixed_chunk = False
                logger.warning(
                    "Mixed chunked prefill is disabled because of using "
                    "eagle speculative decoding."
                )

            model_arch = self.get_model_config().hf_config.architectures[0]
            if model_arch in [
                "DeepseekV32ForCausalLM",
                "DeepseekV3ForCausalLM",
                "Glm4MoeForCausalLM",
                "Glm4MoeLiteForCausalLM",
                "BailingMoeForCausalLM",
                "BailingMoeV2ForCausalLM",
                "MistralLarge3ForCausalLM",
                "PixtralForConditionalGeneration",
            ]:
                if self.speculative_draft_model_path is None:
                    self.speculative_draft_model_path = self.model_path
                    self.speculative_draft_model_revision = self.revision
                else:
                    if model_arch not in [
                        "MistralLarge3ForCausalLM",
                        "PixtralForConditionalGeneration",
                    ]:
                        logger.warning(
                            "DeepSeek MTP does not require setting speculative_draft_model_path."
                        )

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
                self.attention_backend == "trtllm_mha"
                or self.decode_attention_backend == "trtllm_mha"
                or self.prefill_attention_backend == "trtllm_mha"
            ):
                if self.speculative_eagle_topk > 1:
                    raise ValueError(
                        "trtllm_mha backend only supports topk = 1 for speculative decoding."
                    )

            if (
                self.speculative_eagle_topk == 1
                and self.speculative_num_draft_tokens != self.speculative_num_steps + 1
            ):
                logger.warning(
                    "speculative_num_draft_tokens is adjusted to speculative_num_steps + 1 when speculative_eagle_topk == 1"
                )
                self.speculative_num_draft_tokens = self.speculative_num_steps + 1

            if (
                self.speculative_eagle_topk > 1
                and self.page_size > 1
                and self.attention_backend not in ["flashinfer", "fa3"]
            ):
                raise ValueError(
                    "speculative_eagle_topk > 1 with page_size > 1 is unstable and produces incorrect results for paged attention backends. This combination is only supported for the 'flashinfer' backend."
                )

        if self.speculative_algorithm == "NGRAM":
            if not self.device.startswith("cuda"):
                raise ValueError(
                    "Ngram speculative decoding only supports CUDA device."
                )

            if self.max_running_requests is None:
                self.max_running_requests = 48
                logger.warning(
                    "Max running requests is reset to 48 for speculative decoding. You can override this by explicitly setting --max-running-requests."
                )

            self.disable_overlap_schedule = True
            self.enable_mixed_chunk = False
            self.speculative_eagle_topk = self.speculative_ngram_max_bfs_breadth
            if self.speculative_num_draft_tokens is None:
                self.speculative_num_draft_tokens = (
                    self.speculative_ngram_max_match_window_size
                )
            logger.warning(
                "The overlap scheduler and mixed chunked prefill are disabled because of "
                "using ngram speculative decoding."
            )

            if (
                self.speculative_eagle_topk > 1
                and self.page_size > 1
                and self.attention_backend != "flashinfer"
            ):
                raise ValueError(
                    f"speculative_eagle_topk({self.speculative_eagle_topk}) > 1 "
                    f"with page_size({self.page_size}) > 1 is unstable "
                    "and produces incorrect results for paged attention backends. "
                    "This combination is only supported for the 'flashinfer' backend."
                )
            if self.enable_dp_attention:
                # TODO: support dp attention for ngram speculative decoding
                raise ValueError(
                    "Currently ngram speculative decoding does not support dp attention."
                )

    def _handle_load_format(self):
        if (
            self.load_format == "auto" or self.load_format == "gguf"
        ) and check_gguf_file(self.model_path):
            self.quantization = self.load_format = "gguf"

        if is_remote_url(self.model_path):
            self.load_format = "remote"

        if self.custom_weight_loader is None:
            self.custom_weight_loader = []

        if self.load_format == "remote_instance":
            if (
                self.remote_instance_weight_loader_seed_instance_ip is None
                or self.remote_instance_weight_loader_seed_instance_service_port is None
            ):
                logger.warning(
                    "Fallback load_format to 'auto' due to incomplete remote instance weight loader settings."
                )
                self.load_format = "auto"
            elif (
                self.remote_instance_weight_loader_send_weights_group_ports is None
                and self.remote_instance_weight_loader_backend == "nccl"
            ):
                logger.warning(
                    "Fallback load_format to 'auto' due to incomplete remote instance weight loader NCCL group ports settings."
                )
                self.load_format = "auto"
            elif (
                not self.validate_transfer_engine()
                and self.remote_instance_weight_loader_backend == "transfer_engine"
            ):
                logger.warning(
                    "Fallback load_format to 'auto' due to 'transfer_engine' backend is not supported."
                )
                self.load_format = "auto"

        # Check whether TransferEngine can be used when users want to start seed service that supports TransferEngine backend.
        if self.remote_instance_weight_loader_start_seed_via_transfer_engine:
            self.remote_instance_weight_loader_start_seed_via_transfer_engine = (
                self.validate_transfer_engine()
            )

    def _handle_pd_disaggregation(self):
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

            if not self.enable_piecewise_cuda_graph:
                self.disable_cuda_graph = True
                logger.warning(
                    "Cuda graph is disabled for prefill server when piecewise cuda graph is not enabled."
                )

    def _handle_encoder_disaggregation(self):
        if self.enable_prefix_mm_cache and not self.encoder_only:
            raise ValueError(
                "--enable-prefix-mm-cache requires --encoder-only to be enabled"
            )
        if self.encoder_only and self.language_only:
            raise ValueError("Cannot set --encoder-only and --language-only together")
        if self.encoder_only and not self.disaggregation_mode == "null":
            raise ValueError(
                "Cannot set --encoder-only and --disaggregation-mode prefill/decode together"
            )

        if self.language_only and len(self.encoder_urls) == 0:
            raise ValueError(
                "requires at least one encoder urls to be set via --encoder-urls"
            )

        # Validate IB devices when mooncake backend is used
        if (
            self.disaggregation_transfer_backend == "mooncake"
            and self.disaggregation_mode in ("prefill", "decode")
        ) or self.encoder_transfer_backend == "mooncake":
            self.disaggregation_ib_device = self._validate_ib_devices(
                self.disaggregation_ib_device
            )

    def _validate_ib_devices(self, device_str: str) -> Optional[str]:
        """
        Validate IB devices before passing to mooncake.

        Args:
            device_str: Comma-separated IB device names (e.g., "mlx5_0,mlx5_1")

        Returns:
            Normalized comma-separated string of validated device names, or None if input is None.
        """
        if device_str is None:
            logger.warning(
                "No IB devices specified for Mooncake backend, falling back to auto discovery."
            )
            return None

        # Strip whitespace from device names
        devices = [d.strip() for d in device_str.split(",") if d.strip()]
        if len(devices) == 0:
            raise ValueError("No valid IB devices specified")

        # Check for duplicates
        if len(devices) != len(set(devices)):
            raise ValueError(f"Duplicate IB devices specified: {device_str}")

        # Get available IB devices from sysfs
        ib_sysfs_path = "/sys/class/infiniband"
        if not os.path.isdir(ib_sysfs_path):
            raise RuntimeError(
                f"InfiniBand sysfs path not found: {ib_sysfs_path}. "
                "Please ensure InfiniBand drivers are installed."
            )

        available_devices = set(os.listdir(ib_sysfs_path))
        if len(available_devices) == 0:
            raise RuntimeError(f"No IB devices found in {ib_sysfs_path}")

        # Check for invalid devices
        invalid_devices = [d for d in devices if d not in available_devices]
        if len(invalid_devices) != 0:
            raise ValueError(
                f"Invalid IB devices specified: {invalid_devices}. "
                f"Available devices: {sorted(available_devices)}"
            )

        return ",".join(devices)

    def _handle_tokenizer_batching(self):
        if self.enable_tokenizer_batch_encode and self.enable_dynamic_batch_tokenizer:
            raise ValueError(
                "Cannot enable both --enable-tokenizer-batch-encode and --enable-dynamic-batch-tokenizer. "
                "Please choose one tokenizer batching approach."
            )

        if self.skip_tokenizer_init:
            if self.tokenizer_worker_num != 1:
                logger.warning(
                    "skip_tokenizer_init=True disables tokenizer workers; forcing tokenizer_worker_num=1 "
                    f"(requested {self.tokenizer_worker_num})."
                )
                self.tokenizer_worker_num = 1

            if self.enable_tokenizer_batch_encode:
                logger.warning(
                    "skip_tokenizer_init=True ignores --enable-tokenizer-batch-encode; disabling it."
                )
                self.enable_tokenizer_batch_encode = False

            if self.enable_dynamic_batch_tokenizer:
                logger.warning(
                    "skip_tokenizer_init=True ignores --enable-dynamic-batch-tokenizer; disabling it."
                )
                self.enable_dynamic_batch_tokenizer = False

    def _handle_environment_variables(self):
        envs.SGLANG_ENABLE_TORCH_COMPILE.set("1" if self.enable_torch_compile else "0")
        envs.SGLANG_MAMBA_SSM_DTYPE.set(self.mamba_ssm_dtype)
        envs.SGLANG_DISABLE_OUTLINES_DISK_CACHE.set(
            "1" if self.disable_outlines_disk_cache else "0"
        )
        envs.SGLANG_ENABLE_DETERMINISTIC_INFERENCE.set(
            "1" if self.enable_deterministic_inference else "0"
        )

    def _handle_cache_compatibility(self):
        if self.enable_hierarchical_cache and self.disable_radix_cache:
            raise ValueError(
                "The arguments enable-hierarchical-cache and disable-radix-cache are mutually exclusive "
                "and cannot be used at the same time. Please use only one of them."
            )

        if self.disaggregation_decode_enable_offload_kvcache:
            if self.disaggregation_mode != "decode":
                raise ValueError(
                    "The argument disaggregation-decode-enable-offload-kvcache is only supported for decode side."
                )
            if (
                self.disaggregation_mode == "decode"
                and envs.SGLANG_ENABLE_SPEC_V2.get()
            ):
                raise ValueError(
                    "Spec v2 and decode offload kv cache are incompatible and cannot be enabled together."
                )
        if not (0 < self.swa_full_tokens_ratio <= 1.0):
            raise ValueError("--swa-full-tokens-ratio should be in range (0, 1.0].")

    def _handle_deterministic_inference(self):
        if self.rl_on_policy_target is not None:
            logger.warning(
                "Enable deterministic inference because of rl_on_policy_target."
            )
            self.enable_deterministic_inference = True

            # For VLM
            os.environ["SGLANG_VLM_CACHE_SIZE_MB"] = "0"
            # TODO remove this environment variable as a whole
            os.environ["SGLANG_ENABLE_DETERMINISTIC_INFERENCE"] = "1"

        if self.enable_deterministic_inference:
            # Check sampling backend
            self.sampling_backend = "pytorch"
            logger.warning(
                "Sampling backend is set to pytorch for deterministic inference."
            )
            is_deepseek_model = False
            if parse_connector_type(self.model_path) != ConnectorType.INSTANCE:
                try:
                    hf_config = self.get_model_config().hf_config
                    model_arch = hf_config.architectures[0]
                    is_deepseek_model = model_arch in [
                        "DeepseekV2ForCausalLM",
                        "DeepseekV3ForCausalLM",
                        "DeepseekV32ForCausalLM",
                        "MistralLarge3ForCausalLM",
                        "PixtralForConditionalGeneration",
                    ]
                except Exception:
                    pass

            # Check attention backend
            if self.attention_backend is None:
                # User didn't specify attention backend, fallback based on GPU architecture
                if is_sm100_supported() or is_sm120_supported():
                    # Blackwell and newer architectures
                    if is_deepseek_model:
                        # fallback to triton for DeepSeek models because flashinfer doesn't support deterministic inference for DeepSeek models yet
                        self.attention_backend = "triton"
                    else:
                        # fallback to flashinfer on Blackwell for non-DeepSeek models
                        self.attention_backend = "flashinfer"
                else:
                    # Hopper (SM90) and older architectures
                    self.attention_backend = "fa3"
                logger.warning(
                    f"Attention backend not specified. Falling back to '{self.attention_backend}' for deterministic inference. "
                    f"You can explicitly set --attention-backend to one of {DETERMINISTIC_ATTENTION_BACKEND_CHOICES}."
                )
            elif self.attention_backend not in DETERMINISTIC_ATTENTION_BACKEND_CHOICES:
                # User explicitly specified an incompatible attention backend
                raise ValueError(
                    f"Currently only {DETERMINISTIC_ATTENTION_BACKEND_CHOICES} attention backends are supported for deterministic inference, "
                    f"but you explicitly specified '{self.attention_backend}'."
                )

            if is_deepseek_model:
                if self.attention_backend not in ["fa3", "triton"]:
                    raise ValueError(
                        f"Currently only {RADIX_SUPPORTED_DETERMINISTIC_ATTENTION_BACKEND} attention backends are supported for deterministic inference with DeepSeek models. But you're using {self.attention_backend}."
                    )

            if (
                self.attention_backend
                not in RADIX_SUPPORTED_DETERMINISTIC_ATTENTION_BACKEND
            ):
                # Currently, only certain backends support radix cache. Support for other backends is in progress
                self.disable_radix_cache = True
                logger.warning(
                    f"Currently radix cache is not compatible with {self.attention_backend} attention backend for deterministic inference. It will be supported in the future."
                )

            # Check TP size
            if self.tp_size > 1:
                if is_hip():
                    # AMD: use 1-stage all-reduce kernel which is inherently deterministic
                    # (each GPU reads all data from all GPUs, reduces locally in fixed order)
                    logger.info(
                        "AMD/ROCm: Using 1-stage all-reduce kernel (deterministic)"
                    )
                else:
                    # CUDA: use NCCL tree algorithm
                    os.environ["NCCL_ALGO"] = "allreduce:tree"
                    self.disable_custom_all_reduce = True
                    logger.warning(
                        "NCCL_ALGO is set to 'allreduce:tree' and custom all reduce is disabled for deterministic inference when TP size > 1."
                    )

    def _handle_dllm_inference(self):
        if self.dllm_algorithm is None:
            return
        # On AMD/HIP, disable cuda graph for DLLM and use triton backend
        if is_hip():
            if not self.disable_cuda_graph:
                logger.warning(
                    "Cuda graph is disabled for diffusion LLM inference on AMD GPUs"
                )
                self.disable_cuda_graph = True
            if self.attention_backend not in ["triton", "aiter"]:
                logger.warning(
                    "Attention backend is set to triton for diffusion LLM inference on AMD GPUs"
                )
                self.attention_backend = "triton"
        elif not self.disable_cuda_graph:
            if self.attention_backend != "flashinfer":
                logger.warning(
                    "Attention backend is set to flashinfer because of enabling cuda graph in diffusion LLM inference"
                )
                self.attention_backend = "flashinfer"
        if not self.disable_overlap_schedule:
            logger.warning(
                "Overlap schedule is disabled because of using diffusion LLM inference"
            )
            self.disable_overlap_schedule = True
        if not self.disable_radix_cache:
            logger.warning(
                "Radix cache is disabled because of using diffusion LLM inference"
            )
            self.disable_radix_cache = True
        if not self.pp_size > 1:
            logger.warning(
                "Pipeline parallelism is disabled because of using diffusion LLM inference"
            )
            self.pp_size = 1

    def _handle_other_validations(self):
        # Handle model inference tensor dump.
        if self.debug_tensor_dump_output_folder is not None:
            logger.warning(
                "Cuda graph and server warmup are disabled because of using tensor dump mode"
            )
            self.disable_cuda_graph = True
            self.skip_server_warmup = True

        # Validate limit_mm_per_prompt modalities
        if self.limit_mm_data_per_request:
            if isinstance(self.limit_mm_data_per_request, str):
                self.limit_mm_data_per_request = json.loads(
                    self.limit_mm_data_per_request
                )

            if isinstance(self.limit_mm_data_per_request, dict):
                allowed_modalities = {"image", "video", "audio"}
                for modality in self.limit_mm_data_per_request.keys():
                    if modality not in allowed_modalities:
                        raise ValueError(
                            f"Invalid modality '{modality}' in --limit-mm-data-per-request."
                            f"Allowed modalities are: {list(allowed_modalities)}"
                        )

        # Validate preferred_sampling_params
        if self.preferred_sampling_params:
            if isinstance(self.preferred_sampling_params, str):
                self.preferred_sampling_params = json.loads(
                    self.preferred_sampling_params
                )

    def _handle_debug_utils(self):
        if is_in_ci() and self.soft_watchdog_timeout is None:
            logger.info("Set soft_watchdog_timeout since in CI")
            self.soft_watchdog_timeout = 300

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser):

        # Model and tokenizer
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
            "--tokenizer-mode",
            type=str,
            default=ServerArgs.tokenizer_mode,
            choices=["auto", "slow"],
            help="Tokenizer mode. 'auto' will use the fast "
            "tokenizer if available, and 'slow' will "
            "always use the slow tokenizer.",
        )
        parser.add_argument(
            "--tokenizer-worker-num",
            type=int,
            default=ServerArgs.tokenizer_worker_num,
            help="The worker num of the tokenizer manager.",
        )
        parser.add_argument(
            "--skip-tokenizer-init",
            action="store_true",
            help="If set, skip init tokenizer and pass input_ids in generate request.",
        )
        parser.add_argument(
            "--load-format",
            type=str,
            default=ServerArgs.load_format,
            choices=LOAD_FORMAT_CHOICES,
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
            "--context-length",
            type=int,
            default=ServerArgs.context_length,
            help="The model's maximum context length. Defaults to None (will use the value from the model's config.json instead).",
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
            "--model-impl",
            type=str,
            default=ServerArgs.model_impl,
            help="Which implementation of the model to use.\n\n"
            '* "auto" will try to use the SGLang implementation if it exists '
            "and fall back to the Transformers implementation if no SGLang "
            "implementation is available.\n"
            '* "sglang" will use the SGLang model implementation.\n'
            '* "transformers" will use the Transformers model '
            '* "mindspore" will use the MindSpore model '
            "implementation.\n",
        )

        # HTTP server
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
            "--fastapi-root-path",
            type=str,
            default=ServerArgs.fastapi_root_path,
            help="App is behind a path based routing proxy.",
        )
        parser.add_argument(
            "--grpc-mode",
            action="store_true",
            help="If set, use gRPC server instead of HTTP server.",
        )
        parser.add_argument(
            "--skip-server-warmup",
            action="store_true",
            help="If set, skip warmup.",
        )
        parser.add_argument(
            "--warmups",
            type=str,
            required=False,
            help="Specify custom warmup functions (csv) to run before server starts eg. --warmups=warmup_name1,warmup_name2 "
            "will run the functions `warmup_name1` and `warmup_name2` specified in warmup.py before the server starts listening for requests",
        )
        parser.add_argument(
            "--nccl-port",
            type=int,
            default=ServerArgs.nccl_port,
            help="The port for NCCL distributed environment setup. Defaults to a random port.",
        )
        parser.add_argument(
            "--checkpoint-engine-wait-weights-before-ready",
            action="store_true",
            help="If set, the server will wait for initial weights to be loaded via checkpoint-engine or other update methods "
            "before serving inference requests.",
        )

        # Quantization and data type
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
            "--quantization",
            type=str,
            default=ServerArgs.quantization,
            choices=QUANTIZATION_CHOICES,
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
            "--kv-cache-dtype",
            type=str,
            default=ServerArgs.kv_cache_dtype,
            choices=["auto", "fp8_e5m2", "fp8_e4m3", "bf16", "bfloat16", "fp4_e2m1"],
            help='Data type for kv cache storage. "auto" will use model data type. "bf16" or "bfloat16" for BF16 KV cache. "fp8_e5m2" and "fp8_e4m3" are supported for CUDA 11.8+. "fp4_e2m1" (only mxfp4) is supported for CUDA 12.8+ and PyTorch 2.8.0+',
        )
        parser.add_argument(
            "--enable-fp32-lm-head",
            action="store_true",
            help="If set, the LM head outputs (logits) are in FP32.",
        )
        parser.add_argument(
            "--modelopt-quant",
            type=str,
            default=ServerArgs.modelopt_quant,
            help="The ModelOpt quantization configuration. "
            "Supported values: 'fp8', 'int4_awq', 'w4a8_awq', 'nvfp4', 'nvfp4_awq'. "
            "This requires the NVIDIA Model Optimizer library to be installed: pip install nvidia-modelopt",
        )
        parser.add_argument(
            "--modelopt-checkpoint-restore-path",
            type=str,
            default=ServerArgs.modelopt_checkpoint_restore_path,
            help="Path to restore a previously saved ModelOpt quantized checkpoint. "
            "If provided, the quantization process will be skipped and the model "
            "will be loaded from this checkpoint.",
        )
        parser.add_argument(
            "--modelopt-checkpoint-save-path",
            type=str,
            default=ServerArgs.modelopt_checkpoint_save_path,
            help="Path to save the ModelOpt quantized checkpoint after quantization. "
            "This allows reusing the quantized model in future runs.",
        )
        parser.add_argument(
            "--modelopt-export-path",
            type=str,
            default=ServerArgs.modelopt_export_path,
            help="Path to export the quantized model in HuggingFace format after ModelOpt quantization. "
            "The exported model can then be used directly with SGLang for inference. "
            "If not provided, the model will not be exported.",
        )
        parser.add_argument(
            "--quantize-and-serve",
            action="store_true",
            default=ServerArgs.quantize_and_serve,
            help="Quantize the model with ModelOpt and immediately serve it without exporting. "
            "This is useful for development and prototyping. For production, it's recommended "
            "to use separate quantization and deployment steps.",
        )
        parser.add_argument(
            "--rl-quant-profile",
            type=str,
            default=ServerArgs.rl_quant_profile,
            help="Path to the FlashRL quantization profile. Required when using --load-format flash_rl.",
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
            "--max-queued-requests",
            type=int,
            default=ServerArgs.max_queued_requests,
            help="The maximum number of queued requests. This option is ignored when using disaggregation-mode.",
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
            "--prefill-max-requests",
            type=int,
            default=ServerArgs.prefill_max_requests,
            help="The maximum number of requests in a prefill batch. If not specified, there is no limit.",
        )
        parser.add_argument(
            "--enable-dynamic-chunking",
            action="store_true",
            default=ServerArgs.enable_dynamic_chunking,
            help="Enable dynamic chunk size adjustment for pipeline parallelism. When enabled, chunk sizes are dynamically calculated based on fitted function to maintain consistent execution time across chunks.",
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
            choices=[
                "lpm",
                "random",
                "fcfs",
                "dfs-weight",
                "lof",
                "priority",
                "routing-key",
            ],
            help="The scheduling policy of the requests.",
        )
        parser.add_argument(
            "--enable-priority-scheduling",
            action="store_true",
            default=ServerArgs.enable_priority_scheduling,
            help="Enable priority scheduling. Requests with higher priority integer values will be scheduled first by default.",
        )
        parser.add_argument(
            "--abort-on-priority-when-disabled",
            action="store_true",
            default=ServerArgs.abort_on_priority_when_disabled,
            help="If set, abort requests that specify a priority when priority scheduling is disabled.",
        )
        parser.add_argument(
            "--schedule-low-priority-values-first",
            action="store_true",
            default=ServerArgs.schedule_low_priority_values_first,
            help="If specified with --enable-priority-scheduling, the scheduler will schedule requests with lower priority integer values first.",
        )
        parser.add_argument(
            "--priority-scheduling-preemption-threshold",
            type=int,
            default=ServerArgs.priority_scheduling_preemption_threshold,
            help="Minimum difference in priorities for an incoming request to have to preempt running request(s).",
        )
        parser.add_argument(
            "--schedule-conservativeness",
            type=float,
            default=ServerArgs.schedule_conservativeness,
            help="How conservative the schedule policy is. A larger value means more conservative scheduling. Use a larger value if you see requests being retracted frequently.",
        )
        parser.add_argument(
            "--page-size",
            type=int,
            default=ServerArgs.page_size,
            help="The number of tokens in a page.",
        )
        parser.add_argument(
            "--hybrid-kvcache-ratio",
            action=DeprecatedAction,
            help="Note: --hybrid-kvcache-ratio is deprecated now. Please use --swa-full-tokens-ratio instead.",
        )
        parser.add_argument(
            "--swa-full-tokens-ratio",
            type=float,
            default=ServerArgs.swa_full_tokens_ratio,
            help="The ratio of SWA layer KV tokens / full layer KV tokens, regardless of the number of swa:full layers. It should be between 0 and 1. "
            "E.g. 0.5 means if each swa layer has 50 tokens, then each full layer has 100 tokens.",
        )
        parser.add_argument(
            "--disable-hybrid-swa-memory",
            action="store_true",
            help="Disable the hybrid SWA memory pool.",
        )
        parser.add_argument(
            "--radix-eviction-policy",
            type=str,
            choices=RADIX_EVICTION_POLICY_CHOICES,
            default=ServerArgs.radix_eviction_policy,
            help="The eviction policy of radix trees. 'lru' stands for Least Recently Used, 'lfu' stands for Least Frequently Used.",
        )
        parser.add_argument(
            "--enable-prefill-delayer",
            action="store_true",
            help="Enable prefill delayer for DP attention to reduce idle time.",
        )
        parser.add_argument(
            "--prefill-delayer-max-delay-passes",
            type=int,
            default=ServerArgs.prefill_delayer_max_delay_passes,
            help="Maximum forward passes to delay prefill.",
        )
        parser.add_argument(
            "--prefill-delayer-token-usage-low-watermark",
            type=float,
            default=None,
            help="Token usage low watermark for prefill delayer.",
        )
        parser.add_argument(
            "--prefill-delayer-forward-passes-buckets",
            type=float,
            nargs="+",
            default=None,
            help="Custom buckets for prefill delayer forward passes histogram. 0 and max_delay_passes-1 will be auto-added.",
        )
        parser.add_argument(
            "--prefill-delayer-wait-seconds-buckets",
            type=float,
            nargs="+",
            default=None,
            help="Custom buckets for prefill delayer wait seconds histogram. 0 will be auto-added.",
        )

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
            "--custom-sigquit-handler",
            help="Register a custom sigquit handler so you can do additional cleanup after the server is shutdown. This is only available for Engine, not for CLI.",
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
            default=ServerArgs.log_requests_level,
            help="0: Log metadata (no sampling parameters). 1: Log metadata and sampling parameters. 2: Log metadata, sampling parameters and partial input/output. 3: Log every input/output.",
            choices=[0, 1, 2, 3],
        )
        parser.add_argument(
            "--log-requests-format",
            type=str,
            default=ServerArgs.log_requests_format,
            choices=["text", "json"],
            help="Format for request logging: 'text' (human-readable) or 'json' (structured)",
        )
        parser.add_argument(
            "--log-requests-target",
            type=str,
            nargs="+",
            default=ServerArgs.log_requests_target,
            help="Target(s) for request logging: 'stdout' and/or directory path(s) for file output. "
            "Can specify multiple targets, e.g., '--log-requests-target stdout /my/path'. ",
        )
        parser.add_argument(
            "--uvicorn-access-log-exclude-prefixes",
            type=str,
            nargs="*",
            default=list(DEFAULT_UVICORN_ACCESS_LOG_EXCLUDE_PREFIXES),
            help="Exclude uvicorn access logs whose request path starts with any of these prefixes. "
            "Defaults to empty (disabled). "
            "Example: --uvicorn-access-log-exclude-prefixes /metrics /health",
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
            "--enable-metrics-for-all-schedulers",
            action="store_true",
            help="Enable --enable-metrics-for-all-schedulers when you want schedulers on all TP ranks (not just TP 0) "
            "to record request metrics separately. This is especially useful when dp_attention is enabled, as "
            "otherwise all metrics appear to come from TP 0.",
        )
        parser.add_argument(
            "--tokenizer-metrics-custom-labels-header",
            type=str,
            default=ServerArgs.tokenizer_metrics_custom_labels_header,
            help="Specify the HTTP header for passing custom labels for tokenizer metrics.",
        )
        parser.add_argument(
            "--tokenizer-metrics-allowed-custom-labels",
            type=str,
            nargs="+",
            default=ServerArgs.tokenizer_metrics_allowed_custom_labels,
            help="The custom labels allowed for tokenizer metrics. The labels are specified via a dict in "
            "'--tokenizer-metrics-custom-labels-header' field in HTTP requests, e.g., {'label1': 'value1', 'label2': "
            "'value2'} is allowed if '--tokenizer-metrics-allowed-custom-labels label1 label2' is set.",
        )
        parser.add_argument(
            "--extra-metric-labels",
            type=json.loads,
            default=ServerArgs.extra_metric_labels,
            help="The custom labels for metrics. "
            'e.g. \'{"label1": "value1", "label2": "value2"}\'',
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
        bucket_rule = (
            "Supports 3 rule types: 'default' uses predefined buckets; 'tse <middle> <base> <count>' "
            "generates two sides exponential distributed buckets (e.g., 'tse 1000 2 8' generates buckets "
            "[984.0, 992.0, 996.0, 998.0, 1000.0, 1002.0, 1004.0, 1008.0, 1016.0]).); 'custom <value1> "
            "<value2> ...' uses custom bucket values (e.g., 'custom 10 50 100 500')."
        )
        parser.add_argument(
            "--prompt-tokens-buckets",
            type=str,
            nargs="+",
            default=ServerArgs.prompt_tokens_buckets,
            help=f"The buckets rule of prompt tokens. {bucket_rule}",
        )
        parser.add_argument(
            "--generation-tokens-buckets",
            type=str,
            nargs="+",
            default=ServerArgs.generation_tokens_buckets,
            help=f"The buckets rule for generation tokens histogram. {bucket_rule}",
        )
        parser.add_argument(
            "--gc-warning-threshold-secs",
            type=float,
            default=ServerArgs.gc_warning_threshold_secs,
            help="The threshold for long GC warning. If a GC takes longer than this, a warning will be logged. Set to 0 to disable.",
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
        parser.add_argument(
            "--kv-events-config",
            type=str,
            default=None,
            help="Config in json format for NVIDIA dynamo KV event publishing. Publishing will be enabled if this flag is used.",
        )
        parser.add_argument(
            "--enable-trace",
            action="store_true",
            help="Enable opentelemetry trace",
        )
        parser.add_argument(
            "--otlp-traces-endpoint",
            type=str,
            default="localhost:4317",
            help="Config opentelemetry collector endpoint if --enable-trace is set. format: <ip>:<port>",
        )

        # RequestMetricsExporter configuration
        parser.add_argument(
            "--export-metrics-to-file",
            action="store_true",
            help="Export performance metrics for each request to local file (e.g. for forwarding to external systems).",
        )
        parser.add_argument(
            "--export-metrics-to-file-dir",
            type=str,
            default=ServerArgs.export_metrics_to_file_dir,
            help="Directory path for writing performance metrics files (required when --export-metrics-to-file is enabled).",
        )

        # API related
        parser.add_argument(
            "--api-key",
            type=str,
            default=ServerArgs.api_key,
            help="Set API key of the server. It is also used in the OpenAI API compatible server.",
        )
        parser.add_argument(
            "--admin-api-key",
            type=str,
            default=ServerArgs.admin_api_key,
            help=(
                "Set admin API key for sensitive management endpoints (e.g. /clear_hicache_storage_backend). "
                "When set, admin endpoints require this key and do NOT accept --api-key."
            ),
        )
        parser.add_argument(
            "--served-model-name",
            type=str,
            default=ServerArgs.served_model_name,
            help="Override the model name returned by the v1/models endpoint in OpenAI API server.",
        )
        parser.add_argument(
            "--weight-version",
            type=str,
            default=ServerArgs.weight_version,
            help="Version identifier for the model weights. Defaults to 'default' if not specified.",
        )
        parser.add_argument(
            "--chat-template",
            type=str,
            default=ServerArgs.chat_template,
            help="The buliltin chat template name or the path of the chat template file. This is only used for OpenAI-compatible API server.",
        )
        parser.add_argument(
            "--hf-chat-template-name",
            type=str,
            default=ServerArgs.hf_chat_template_name,
            help="When the HuggingFace tokenizer has multiple chat templates (e.g., 'default', 'tool_use', 'rag'), "
            "specify which named template to use. If not set, the first available template is used.",
        )
        parser.add_argument(
            "--completion-template",
            type=str,
            default=ServerArgs.completion_template,
            help="The buliltin completion template name or the path of the completion template file. This is only used for OpenAI-compatible API server. only for code completion currently.",
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
        tool_call_parser_choices = list(FunctionCallParser.ToolCallParserEnum.keys())
        parser.add_argument(
            "--tool-call-parser",
            type=str,
            choices=tool_call_parser_choices,
            default=ServerArgs.tool_call_parser,
            help=f"Specify the parser for handling tool-call interactions. Options include: {tool_call_parser_choices}.",
        )
        parser.add_argument(
            "--tool-server",
            type=str,
            default=None,
            help="Either 'demo' or a comma-separated list of tool server urls to use for the model. If not specified, no tool server will be used.",
        )
        parser.add_argument(
            "--sampling-defaults",
            type=str,
            choices=["openai", "model"],
            default=ServerArgs.sampling_defaults,
            help="Where to get default sampling parameters. "
            "'openai' uses SGLang/OpenAI defaults (temperature=1.0, top_p=1.0, etc.). "
            "'model' uses the model's generation_config.json to get the recommended "
            "sampling parameters if available. Default is 'model'.",
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

        # Model override args
        parser.add_argument(
            "--json-model-override-args",
            type=str,
            help="A dictionary in JSON string format used to override default model configurations.",
            default=ServerArgs.json_model_override_args,
        )
        parser.add_argument(
            "--preferred-sampling-params",
            type=json.loads,
            help="json-formatted sampling settings that will be returned in /get_model_info",
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

        # Kernel backend
        parser.add_argument(
            "--attention-backend",
            type=str,
            choices=ATTENTION_BACKEND_CHOICES,
            default=ServerArgs.attention_backend,
            help="Choose the kernels for attention layers.",
        )
        parser.add_argument(
            "--prefill-attention-backend",
            type=str,
            choices=ATTENTION_BACKEND_CHOICES,
            default=ServerArgs.prefill_attention_backend,
            help="Choose the kernels for prefill attention layers (have priority over --attention-backend).",
        )
        parser.add_argument(
            "--decode-attention-backend",
            type=str,
            choices=ATTENTION_BACKEND_CHOICES,
            default=ServerArgs.decode_attention_backend,
            help="Choose the kernels for decode attention layers (have priority over --attention-backend).",
        )
        parser.add_argument(
            "--sampling-backend",
            type=str,
            choices=SAMPLING_BACKEND_CHOICES,
            default=ServerArgs.sampling_backend,
            help="Choose the kernels for sampling layers.",
        )
        parser.add_argument(
            "--grammar-backend",
            type=str,
            choices=GRAMMAR_BACKEND_CHOICES,
            default=ServerArgs.grammar_backend,
            help="Choose the backend for grammar-guided decoding.",
        )
        parser.add_argument(
            "--mm-attention-backend",
            type=str,
            choices=["sdpa", "fa3", "fa4", "triton_attn", "ascend_attn", "aiter_attn"],
            default=ServerArgs.mm_attention_backend,
            help="Set multimodal attention backend.",
        )
        parser.add_argument(
            "--nsa-prefill-backend",
            default=ServerArgs.nsa_prefill_backend,
            type=str,
            choices=NSA_CHOICES,
            help="NSA prefill backend. If not specified, auto-detects based on hardware and kv_cache_dtype.",
        )
        parser.add_argument(
            "--nsa-decode-backend",
            default=ServerArgs.nsa_decode_backend,
            type=str,
            choices=NSA_CHOICES,
            help="NSA decode backend. If not specified, auto-detects based on hardware and kv_cache_dtype.",
        )
        parser.add_argument(
            "--fp8-gemm-backend",
            type=str,
            choices=FP8_GEMM_RUNNER_BACKEND_CHOICES,
            default=ServerArgs.fp8_gemm_runner_backend,
            dest="fp8_gemm_runner_backend",
            help="Choose the runner backend for Blockwise FP8 GEMM operations. "
            "Options: 'auto' (default, auto-selects based on hardware), "
            "'deep_gemm' (JIT-compiled; enabled by default on NVIDIA Hopper (SM90) and Blackwell (SM100) when DeepGEMM is installed), "
            "'flashinfer_trtllm' (optimal for Blackwell and low-latency), "
            "'flashinfer_deepgemm' (Hopper SM90 only; uses swapAB optimization for small M dimensions in decoding), "
            "'cutlass' (optimal for Hopper/Blackwell GPUs and high-throughput), "
            "'triton' (fallback, widely compatible), "
            "'aiter' (ROCm only). "
            "NOTE: This replaces the deprecated environment variables "
            "SGLANG_ENABLE_FLASHINFER_FP8_GEMM and SGLANG_SUPPORT_CUTLASS_BLOCK_FP8.",
        )
        parser.add_argument(
            "--fp4-gemm-backend",
            type=str,
            choices=FP4_GEMM_RUNNER_BACKEND_CHOICES,
            default=ServerArgs.fp4_gemm_runner_backend,
            dest="fp4_gemm_runner_backend",
            help="Choose the runner backend for NVFP4 GEMM operations. "
            "Options: 'auto' (default, selects between flashinfer_cudnn/flashinfer_cutlass based on CUDA/cuDNN version), "
            "'flashinfer_cudnn' (FlashInfer cuDNN backend, optimal on CUDA 13+ with cuDNN 9.15+), "
            "'flashinfer_cutlass' (FlashInfer CUTLASS backend, optimal on CUDA 12), "
            "'flashinfer_trtllm' (FlashInfer TensorRT-LLM backend, requires different weight preparation with shuffling). "
            "NOTE: This replaces the deprecated environment variable "
            "SGLANG_FLASHINFER_FP4_GEMM_BACKEND.",
        )
        parser.add_argument(
            "--disable-flashinfer-autotune",
            default=ServerArgs.disable_flashinfer_autotune,
            action="store_true",
            help="Disable FlashInfer autotuning.",
        )

        # Speculative decoding
        parser.add_argument(
            "--speculative-algorithm",
            type=str,
            choices=["EAGLE", "EAGLE3", "NEXTN", "STANDALONE", "NGRAM"],
            help="Speculative algorithm.",
        )
        parser.add_argument(
            "--speculative-draft-model-path",
            "--speculative-draft-model",
            type=str,
            help="The path of the draft model weights. This can be a local folder or a Hugging Face repo ID.",
        )
        parser.add_argument(
            "--speculative-draft-model-revision",
            type=str,
            default=None,
            help="The specific draft model version to use. It can be a branch "
            "name, a tag name, or a commit id. If unspecified, will use "
            "the default version.",
        )
        parser.add_argument(
            "--speculative-draft-load-format",
            type=str,
            default=ServerArgs.speculative_draft_load_format,
            choices=LOAD_FORMAT_CHOICES,
            help="The format of the draft model weights to load. "
            "If not specified, will use the same format as --load-format. "
            "Use 'dummy' to initialize draft model weights with random values for profiling.",
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
            "--speculative-attention-mode",
            type=str,
            choices=["prefill", "decode"],
            help="Attention backend for speculative decoding operations (both target verify and draft extend). Can be one of 'prefill' (default) or 'decode'.",
            default=ServerArgs.speculative_attention_mode,
        )
        parser.add_argument(
            "--speculative-draft-attention-backend",
            type=str,
            help="Attention backend for speculative decoding drafting.",
            default=ServerArgs.speculative_draft_attention_backend,
        )
        parser.add_argument(
            "--speculative-moe-runner-backend",
            type=str,
            choices=MOE_RUNNER_BACKEND_CHOICES,
            default=ServerArgs.speculative_moe_runner_backend,
            help="Choose the runner backend for MoE in speculative decoding.",
        )
        parser.add_argument(
            "--speculative-moe-a2a-backend",
            type=str,
            choices=MOE_A2A_BACKEND_CHOICES,
            default=ServerArgs.speculative_moe_a2a_backend,
            help="Choose the backend for MoE A2A in speculative decoding",
        )
        parser.add_argument(
            "--speculative-draft-model-quantization",
            type=str,
            choices=SPECULATIVE_DRAFT_MODEL_QUANTIZATION_CHOICES,
            default=ServerArgs.speculative_draft_model_quantization,
            help="The quantization method for speculative model.",
        )

        # Speculative decoding (ngram)
        parser.add_argument(
            "--speculative-ngram-min-match-window-size",
            type=int,
            default=ServerArgs.speculative_ngram_min_match_window_size,
            help="The minimum window size for pattern matching in ngram speculative decoding.",
        )
        parser.add_argument(
            "--speculative-ngram-max-match-window-size",
            type=int,
            default=ServerArgs.speculative_ngram_max_match_window_size,
            help="The maximum window size for pattern matching in ngram speculative decoding.",
        )
        parser.add_argument(
            "--speculative-ngram-min-bfs-breadth",
            type=int,
            default=ServerArgs.speculative_ngram_min_bfs_breadth,
            help="The minimum breadth for BFS (Breadth-First Search) in ngram speculative decoding.",
        )
        parser.add_argument(
            "--speculative-ngram-max-bfs-breadth",
            type=int,
            default=ServerArgs.speculative_ngram_max_bfs_breadth,
            help="The maximum breadth for BFS (Breadth-First Search) in ngram speculative decoding.",
        )
        parser.add_argument(
            "--speculative-ngram-match-type",
            type=str,
            choices=["BFS", "PROB"],
            default=ServerArgs.speculative_ngram_match_type,
            help="The match type for cache tree.",
        )
        parser.add_argument(
            "--speculative-ngram-branch-length",
            type=int,
            default=ServerArgs.speculative_ngram_branch_length,
            help="The branch length for ngram speculative decoding.",
        )
        parser.add_argument(
            "--speculative-ngram-capacity",
            type=int,
            default=ServerArgs.speculative_ngram_capacity,
            help="The cache capacity for ngram speculative decoding.",
        )

        # Multi-layer Eagle speculative decoding
        parser.add_argument(
            "--enable-multi-layer-eagle",
            action="store_true",
            help="Enable multi-layer Eagle speculative decoding.",
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
            choices=["none", "mooncake"],
            help="Specify the collective communication backend for elastic EP. Currently supports 'mooncake'.",
        )
        parser.add_argument(
            "--mooncake-ib-device",
            type=str,
            default=ServerArgs.mooncake_ib_device,
            help="The InfiniBand devices for Mooncake Backend transfer, accepts multiple comma-separated devices "
            "(e.g., --mooncake-ib-device mlx5_0,mlx5_1). "
            "Default is None, which triggers automatic device detection when Mooncake Backend is enabled.",
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
            default=ServerArgs.mamba_ssm_dtype,
            choices=MAMBA_SSM_DTYPE_CHOICES,
            help="The data type of the SSM states in mamba cache.",
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
            "--disable-hicache-numa-detect",
            action="store_true",
            help="Disable binding the process to the NUMA node closest to the active CUDA device when hierarchical cache is enabled.",
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
            "--hierarchical-sparse-attention-extra-config",
            type=str,
            default=ServerArgs.hierarchical_sparse_attention_extra_config,
            help="A dictionary in JSON string format for hierarchical sparse attention configuration. "
            "Required fields: algorithm (str), backend (str). "
            "All other fields are algorithm-specific and passed to the algorithm constructor. "
            'Example: \'{"algorithm": "quest", "backend": "flashattention", "sparsity_ratio": 0.7, "min_sparse_prompt_len": 2048}\'',
        )

        # LMCache
        parser.add_argument(
            "--enable-lmcache",
            action="store_true",
            help="Using LMCache as an alternative hierarchical cache solution",
        )

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
            "--enable-cudagraph-gc",
            action="store_true",
            help="Enable garbage collection during CUDA graph capture. If disabled (default), GC is frozen during capture to speed up the process.",
        )
        parser.add_argument(
            "--enable-layerwise-nvtx-marker",
            action="store_true",
            help="Enable layerwise NVTX profiling annotations for the model.",
        )
        parser.add_argument(
            "--enable-nccl-nvls",
            action="store_true",
            help="Enable NCCL NVLS for prefill heavy requests when available.",
        )
        parser.add_argument(
            "--enable-symm-mem",
            action="store_true",
            help="Enable NCCL symmetric memory for fast collectives.",
        )
        parser.add_argument(
            "--disable-flashinfer-cutlass-moe-fp4-allgather",
            action="store_true",
            help="Disables quantize before all-gather for flashinfer cutlass moe.",
        )
        parser.add_argument(
            "--enable-tokenizer-batch-encode",
            action="store_true",
            help="Enable batch tokenization for improved performance when processing multiple text inputs. Do not use with image inputs, pre-tokenized input_ids, or input_embeds.",
        )
        parser.add_argument(
            "--disable-tokenizer-batch-decode",
            action="store_true",
            help="Disable batch decoding when decoding multiple completions.",
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
            "--enable-torch-symm-mem",
            action="store_true",
            help="Enable using torch symm mem for all-reduce kernel and fall back to NCCL. Only supports CUDA device SM90 and above. SM90 supports world size 4, 6, 8. SM100 supports world size 6, 8.",
        )
        parser.add_argument(
            "--disable-overlap-schedule",
            action="store_true",
            help="Disable the overlap scheduler, which overlaps the CPU scheduler with GPU model worker.",
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
            "--enable-single-batch-overlap",
            action="store_true",
            help="Let computation and communication overlap within one micro batch.",
        )
        parser.add_argument(
            "--tbo-token-distribution-threshold",
            type=float,
            default=ServerArgs.tbo_token_distribution_threshold,
            help="The threshold of token distribution between two batches in micro-batch-overlap, determines whether to two-batch-overlap or two-chunk-overlap. Set to 0 denote disable two-chunk-overlap.",
        )
        parser.add_argument(
            "--enable-torch-compile",
            action="store_true",
            help="Optimize the model with torch.compile. Experimental feature.",
        )
        parser.add_argument(
            "--enable-torch-compile-debug-mode",
            action="store_true",
            help="Enable debug mode for torch compile",
        )
        parser.add_argument(
            "--enable-piecewise-cuda-graph",
            action="store_true",
            help="Optimize the model with piecewise cuda graph for extend/prefill only. Experimental feature.",
        )
        parser.add_argument(
            "--piecewise-cuda-graph-tokens",
            type=int,
            nargs="+",
            help="Set the list of token lengths for piecewise cuda graph capture.",
        )
        parser.add_argument(
            "--piecewise-cuda-graph-compiler",
            type=str,
            default=ServerArgs.piecewise_cuda_graph_compiler,
            help="Set the compiler for piecewise cuda graph. Choices are: eager, inductor.",
            choices=["eager", "inductor"],
        )
        parser.add_argument(
            "--torch-compile-max-bs",
            type=int,
            default=ServerArgs.torch_compile_max_bs,
            help="Set the maximum batch size when using torch compile.",
        )
        parser.add_argument(
            "--piecewise-cuda-graph-max-tokens",
            type=int,
            default=ServerArgs.piecewise_cuda_graph_max_tokens,
            help="Set the maximum tokens when using piecewise cuda graph.",
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
            "--triton-attention-split-tile-size",
            type=int,
            default=ServerArgs.triton_attention_split_tile_size,
            help="The size of split KV tile in flash decoding Triton kernel. Used for deterministic inference.",
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
            "--enable-weights-cpu-backup",
            action="store_true",
            help="Save model weights (both main model and draft model, if any) to CPU memory during release_weights_occupation and resume_weights_occupation",
        )
        parser.add_argument(
            "--enable-draft-weights-cpu-backup",
            action="store_true",
            help="Save draft model weights to CPU memory during release_weights_occupation and resume_weights_occupation",
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
            "--keep-mm-feature-on-device",
            action="store_true",
            help="Keep multimodal feature tensors on device after processing to save D2H copy.",
        )
        parser.add_argument(
            "--enable-return-hidden-states",
            action="store_true",
            help="Enable returning hidden states with responses.",
        )
        parser.add_argument(
            "--enable-return-routed-experts",
            action="store_true",
            help="Enable returning routed experts of each layer with responses.",
        )
        parser.add_argument(
            "--scheduler-recv-interval",
            type=int,
            default=ServerArgs.scheduler_recv_interval,
            help="The interval to poll requests in scheduler. Can be set to >1 to reduce the overhead of this.",
        )
        parser.add_argument(
            "--numa-node",
            type=int,
            nargs="+",
            help="Sets the numa node for the subprocesses. i-th element corresponds to i-th subprocess.",
        )
        parser.add_argument(
            "--enable-deterministic-inference",
            action="store_true",
            help="Enable deterministic inference mode with batch invariant ops.",
        )
        parser.add_argument(
            "--rl-on-policy-target",
            type=str,
            default=ServerArgs.rl_on_policy_target,
            choices=RL_ON_POLICY_TARGET_CHOICES,
            help="The training system that SGLang needs to match for true on-policy.",
        )
        parser.add_argument(
            "--enable-attn-tp-input-scattered",
            action="store_true",
            help="Allow input of attention to be scattered when only using tensor parallelism, to reduce the computational load of operations such as qkv latent.",
        )
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
            help="Token splitting mode for the prefill phase of DeepSeek v3.2 under context parallelism. Optional values: 'in-seq-split' (default), 'round-robin-split'. "
            "'round-robin-split' distributes tokens across ranks based on token_idx %% cp_size. It supports multi-batch prefill, fused MoE, and FP8 KV cache.",
        )
        parser.add_argument(
            "--enable-fused-qk-norm-rope",
            action="store_true",
            help="Enable fused qk normalization and rope rotary embedding.",
        )
        parser.add_argument(
            "--enable-precise-embedding-interpolation",
            action="store_true",
            help="Enable corner alignment for resize of embeddings grid to ensure more accurate(but slower) evaluation of interpolated embedding values.",
        )

        # Dynamic batch tokenizer
        parser.add_argument(
            "--enable-dynamic-batch-tokenizer",
            action="store_true",
            help="Enable async dynamic batch tokenizer for improved performance when multiple requests arrive concurrently.",
        )
        parser.add_argument(
            "--dynamic-batch-tokenizer-batch-size",
            type=int,
            default=ServerArgs.dynamic_batch_tokenizer_batch_size,
            help="[Only used if --enable-dynamic-batch-tokenizer is set] Maximum batch size for dynamic batch tokenizer.",
        )
        parser.add_argument(
            "--dynamic-batch-tokenizer-batch-timeout",
            type=float,
            default=ServerArgs.dynamic_batch_tokenizer_batch_timeout,
            help="[Only used if --enable-dynamic-batch-tokenizer is set] Timeout in seconds for batching tokenization requests.",
        )

        # Debug tensor dumps
        parser.add_argument(
            "--debug-tensor-dump-output-folder",
            type=str,
            default=ServerArgs.debug_tensor_dump_output_folder,
            help="The output folder for dumping tensors.",
        )
        parser.add_argument(
            "--debug-tensor-dump-layers",
            type=int,
            nargs="+",
            help="The layer ids to dump. Dump all layers if not specified.",
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

        # PD disaggregation
        parser.add_argument(
            "--disaggregation-mode",
            type=str,
            default=ServerArgs.disaggregation_mode,
            choices=["null", "prefill", "decode"],
            help='Only used for PD disaggregation. "prefill" for prefill-only server, and "decode" for decode-only server. If not specified, it is not PD disaggregated',
        )
        parser.add_argument(
            "--disaggregation-transfer-backend",
            type=str,
            default=ServerArgs.disaggregation_transfer_backend,
            choices=DISAGG_TRANSFER_BACKEND_CHOICES,
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
            "--disaggregation-decode-enable-offload-kvcache",
            action="store_true",
            help="Enable async KV cache offloading on decode server (PD mode).",
        )
        parser.add_argument(
            "--disaggregation-decode-enable-fake-auto",
            action="store_true",
            help="Auto enable FAKE mode for decode node testing, "
            "no need to pass bootstrap_host and bootstrap_room in request.",
        )
        parser.add_argument(
            "--num-reserved-decode-tokens",
            type=int,
            default=ServerArgs.num_reserved_decode_tokens,
            help="Number of decode tokens that will have memory reserved when adding new request to the running batch.",
        )
        parser.add_argument(
            "--disaggregation-decode-polling-interval",
            type=int,
            default=ServerArgs.disaggregation_decode_polling_interval,
            help="The interval to poll requests in decode server. Can be set to >1 to reduce the overhead of this.",
        )

        # Encode prefill disaggregation
        parser.add_argument(
            "--encoder-only",
            action="store_true",
            help="For MLLM with an encoder, launch an encoder-only server",
        )
        parser.add_argument(
            "--language-only",
            action="store_true",
            help="For VLM, load weights for the language model only.",
        )
        parser.add_argument(
            "--encoder-transfer-backend",
            type=str,
            default=ServerArgs.encoder_transfer_backend,
            choices=ENCODER_TRANSFER_BACKEND_CHOICES,
            help="The backend for encoder disaggregation transfer. Default is zmq_to_scheduler.",
        )
        parser.add_argument(
            "--encoder-urls",
            nargs="+",
            type=str,
            default=[],
            help="List of encoder server urls.",
        )

        # Custom weight loader
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
        parser.add_argument(
            "--remote-instance-weight-loader-seed-instance-ip",
            type=str,
            default=ServerArgs.remote_instance_weight_loader_seed_instance_ip,
            help="The ip of the seed instance for loading weights from remote instance.",
        )
        parser.add_argument(
            "--remote-instance-weight-loader-seed-instance-service-port",
            type=int,
            default=ServerArgs.remote_instance_weight_loader_seed_instance_service_port,
            help="The service port of the seed instance for loading weights from remote instance.",
        )
        parser.add_argument(
            "--remote-instance-weight-loader-send-weights-group-ports",
            type=json_list_type,
            default=ServerArgs.remote_instance_weight_loader_send_weights_group_ports,
            help="The communication group ports for loading weights from remote instance.",
        )
        parser.add_argument(
            "--remote-instance-weight-loader-backend",
            type=str,
            choices=["transfer_engine", "nccl"],
            default=ServerArgs.remote_instance_weight_loader_backend,
            help="The backend for loading weights from remote instance. Can be 'transfer_engine' or 'nccl'. Default is 'nccl'.",
        )
        parser.add_argument(
            "--remote-instance-weight-loader-start-seed-via-transfer-engine",
            action="store_true",
            help="Start seed server via transfer engine backend for remote instance weight loader.",
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
            "--mm-max-concurrent-calls",
            type=int,
            default=ServerArgs.mm_max_concurrent_calls,
            help="The max concurrent calls for async mm data processing.",
        )
        parser.add_argument(
            "--mm-per-request-timeout",
            type=int,
            default=ServerArgs.mm_per_request_timeout,
            help="The timeout for each multi-modal request in seconds.",
        )
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

        # For registering hooks
        parser.add_argument(
            "--forward-hooks",
            type=json_list_type,
            default=ServerArgs.forward_hooks,
            help="JSON-formatted forward hook specifications to attach to the model.",
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

    def get_model_config(self):
        # Lazy init to avoid circular import
        from sglang.srt.configs.model_config import ModelConfig

        if hasattr(self, "model_config"):
            return self.model_config
        self.model_config = ModelConfig.from_server_args(self)
        return self.model_config

    def get_attention_backends(self):
        prefill_attention_backend_str = (
            self.prefill_attention_backend
            if self.prefill_attention_backend
            else self.attention_backend
        )
        decode_attention_backend_str = (
            self.decode_attention_backend
            if self.decode_attention_backend
            else self.attention_backend
        )
        return prefill_attention_backend_str, decode_attention_backend_str

    def use_mla_backend(self):
        from sglang.srt.configs.model_config import AttentionArch

        model_config = self.get_model_config()
        return model_config.attention_arch == AttentionArch.MLA

    def is_attention_backend_not_set(self):
        return (
            self.attention_backend is None
            and self.prefill_attention_backend is None
            and self.decode_attention_backend is None
        )

    def enable_mamba_extra_buffer(self) -> bool:
        return self.mamba_scheduler_strategy == "extra_buffer"

    @property
    def mamba_cache_chunk_size(self) -> int:
        # For mamba cache with extra buffer, the chunk size is the max of FLA_CHUNK_SIZE and page_size.
        # It is used to determine the caching point in a sequence during prefill.
        return max(FLA_CHUNK_SIZE, self.page_size)

    def check_server_args(self):
        # Check parallel size constraints
        assert (
            self.tp_size * self.pp_size
        ) % self.nnodes == 0, "tp_size must be divisible by number of nodes"

        if self.pp_size > 1:
            assert (
                self.disable_overlap_schedule
                and self.speculative_algorithm is None
                and not self.enable_mixed_chunk
            ), "Pipeline parallelism is not compatible with overlap schedule, speculative decoding, mixed chunked prefill."

        assert not (
            self.dp_size > 1 and self.nnodes != 1 and not self.enable_dp_attention
        ), "multi-node data parallel is not supported unless dp attention!"

        assert self.base_gpu_id >= 0, "base_gpu_id must be non-negative"
        assert self.gpu_id_step >= 1, "gpu_id_step must be positive"

        assert self.moe_dense_tp_size in {
            1,
            None,
        }, "moe_dense_tp_size only support 1 and None currently"

        # Check served model name to not have colon as it is reserved for LoRA adapter syntax
        assert ":" not in self.served_model_name, (
            "served_model_name cannot contain a colon (':') character. "
            "The colon is reserved for the 'model:adapter' syntax used in LoRA adapter specification. "
            f"Invalid value: '{self.served_model_name}'"
        )

        # Check LoRA
        self.check_lora_server_args()

        # torch 2.9.1 has compatibility issues with cuDNN 9.14 and below,
        # causing extremely slow nn.Conv3d performance.
        # TODO(yhyang201): Remove this check when sglang no longer uses torch 2.9.1.
        self.check_torch_2_9_1_cudnn_compatibility()

        # Check speculative decoding
        if self.speculative_algorithm is not None:
            assert (
                not self.enable_mixed_chunk
            ), "enable_mixed_chunk is required for speculative decoding"

        # Check chunked prefill
        # Skip validation if chunked prefill is disabled (i.e., size <= 0).
        # Skip validation if disaggregation mode is decode.
        if self.chunked_prefill_size > 0 and self.disaggregation_mode != "decode":
            assert (
                self.chunked_prefill_size % self.page_size == 0
            ), "chunked_prefill_size must be divisible by page_size"

        # Check pdmux
        if self.enable_pdmux:
            assert (
                self.pp_size == 1
            ), "PD-Multiplexing is only supported with pipeline parallelism disabled (pp_size=1)."
            assert (
                self.chunked_prefill_size == -1
            ), "PD-Multiplexing is not compatible with chunked prefill."
            assert (
                self.disaggregation_mode == "null"
            ), "PD-Multiplexing is not compatible with disaggregation mode."
            assert (
                self.disable_overlap_schedule
            ), "PD-Multiplexing is not compatible with overlap schedule."

            # NOTE: CUDA Green Context may encounter potential issues with CudaGraph on torch 2.7.x – 2.8.x, leading to performance degradation.
            import torch

            if torch_release >= (2, 7):
                logger.warning(
                    "WARNING: PD-Multiplexing may experience performance degradation with torch versions > 2.6.x.\n"
                    f"  Current torch version is {torch.__version__}.\n"
                    "  Please manually install torch 2.6.x."
                )

        assert self.tokenizer_worker_num > 0, "Tokenizer worker num must >= 1"
        self.validate_buckets_rule(
            "--prompt-tokens-buckets", self.prompt_tokens_buckets
        )
        self.validate_buckets_rule(
            "--generation-tokens-buckets", self.generation_tokens_buckets
        )

        # Check scheduling policy
        if self.enable_priority_scheduling:
            assert self.schedule_policy in [
                "fcfs",
                "lof",
            ], f"To use priority scheduling, schedule_policy must be 'fcfs' or 'lof'. '{self.schedule_policy}' is not supported."

        # Check multi-item scoring
        if self.multi_item_scoring_delimiter is not None:
            assert self.disable_radix_cache, (
                "Multi-item scoring requires radix cache to be disabled. "
                "Please set --disable-radix-cache when using --multi-item-scoring-delimiter."
            )
            assert self.chunked_prefill_size == -1, (
                "Multi-item scoring requires chunked prefill to be disabled. "
                "Please set --chunked-prefill-size -1 when using --multi-item-scoring-delimiter."
            )

        assert (
            self.schedule_conservativeness >= 0
        ), "schedule_conservativeness must be non-negative"

        if self.model_impl == "mindspore":
            assert is_npu(), "MindSpore model impl is only supported on Ascend npu."

        # Check metrics labels
        if (
            not self.tokenizer_metrics_custom_labels_header
            and self.tokenizer_metrics_allowed_custom_labels
        ):
            raise ValueError(
                "Please set --tokenizer-metrics-custom-labels-header when setting --tokenizer-metrics-allowed-custom-labels."
            )

        # Check metrics exporters
        if self.export_metrics_to_file and self.export_metrics_to_file_dir is None:
            raise ValueError(
                "--export-metrics-to-file-dir is required when --export-metrics-to-file is enabled"
            )

        # Check two batch overlap
        if self.enable_two_batch_overlap and self.moe_a2a_backend == "none":
            raise ValueError(
                "When enabling two batch overlap, moe_a2a_backend cannot be 'none'."
            )

    def check_torch_2_9_1_cudnn_compatibility(self):
        if get_bool_env_var("SGLANG_DISABLE_CUDNN_CHECK"):
            return

        if self.get_model_config().is_multimodal:
            import torch

            if torch_release[:3] == (2, 9, 1):
                cudnn_version = None
                try:
                    cudnn_version = torch.backends.cudnn.version()
                except Exception:
                    cudnn_version = None
                if cudnn_version is not None:
                    version_float = float(str(cudnn_version)[:3]) / 100
                    if version_float < 9.15:
                        RED = "\033[91m"
                        BOLD = "\033[1m"
                        RESET = "\033[0m"
                        msg = (
                            f"{RED}{BOLD}"
                            "CRITICAL WARNING: PyTorch 2.9.1 & CuDNN Compatibility Issue Detected\n"
                            "--------------------------------------------------------------------------------\n"
                            f"Current Environment: PyTorch {torch.__version__} | CuDNN {version_float:.2f}\n\n"
                            "Issue:     There is a KNOWN BUG in PyTorch 2.9.1's `nn.Conv3d` implementation\n"
                            "           when used with CuDNN versions older than 9.15. This can cause\n"
                            "           SEVERE PERFORMANCE DEGRADATION and EXCESSIVE MEMORY USAGE.\n\n"
                            "Reference: https://github.com/pytorch/pytorch/issues/168167\n\n"
                            "Solution:  You MUST upgrade CuDNN to version 9.15+ to ensure correctness.\n\n"
                            "Run the following command immediately to fix:\n"
                            "    pip install nvidia-cudnn-cu12==9.16.0.29\n\n"
                            "Or you can disable this check by setting env var SGLANG_DISABLE_CUDNN_CHECK=1\n"
                            "--------------------------------------------------------------------------------\n"
                            f"{RESET}"
                        )
                        raise RuntimeError(msg)
                else:
                    RED = "\033[91m"
                    RESET = "\033[0m"
                    logger.warning(
                        f"{RED}WARNING: Could not determine CuDNN version for torch==2.9.1. Please ensure CuDNN >= 9.15 to avoid nn.Conv3d bugs.{RESET}"
                    )

    def check_lora_server_args(self):
        assert self.max_loras_per_batch > 0, "max_loras_per_batch must be positive"

        # Enable LoRA if any LoRA paths are provided for backward compatibility.
        if self.lora_paths:
            if self.enable_lora is None:
                self.enable_lora = True
                logger.warning(
                    "--enable-lora is set to True because --lora-paths is provided."
                )
            elif self.enable_lora is False:
                logger.warning(
                    "--enable-lora is set to False, any provided lora_paths will be ignored."
                )

        if self.enable_lora:
            if self.enable_lora_overlap_loading is None:
                self.enable_lora_overlap_loading = False

            if self.enable_lora_overlap_loading:
                # TODO (glenliu21): use some sort of buffer with eviction instead of enforcing a limit
                max_loaded_loras_limit = self.max_loras_per_batch * 2
                assert (
                    self.max_loaded_loras is not None
                    and self.max_loaded_loras <= max_loaded_loras_limit
                ), (
                    "Enabling LoRA overlap loading requires pinning LoRA adapter weights in CPU memory, "
                    f"so --max-loaded-loras must be less than or equal to double --max-loras-per-batch: {max_loaded_loras_limit}"
                )

            # Validate compatibility with speculative decoding
            if self.speculative_algorithm not in ["NGRAM", None]:
                raise ValueError(
                    "Currently LoRA is only compatible with NGRAM speculative decoding."
                )

            # Parse lora_paths
            if isinstance(self.lora_paths, list):
                lora_paths = self.lora_paths
                self.lora_paths = []
                for lora_path in lora_paths:
                    if isinstance(lora_path, str):
                        if "=" in lora_path:
                            name, path = lora_path.split("=", 1)
                            lora_ref = LoRARef(
                                lora_name=name, lora_path=path, pinned=False
                            )
                        else:
                            lora_ref = LoRARef(
                                lora_name=lora_path, lora_path=lora_path, pinned=False
                            )
                    elif isinstance(lora_path, dict):
                        assert (
                            "lora_name" in lora_path and "lora_path" in lora_path
                        ), f"When providing LoRA paths as a list of dict, each dict should contain 'lora_name' and 'lora_path' keys. Got: {lora_path}"
                        lora_ref = LoRARef(
                            lora_name=lora_path["lora_name"],
                            lora_path=lora_path["lora_path"],
                            pinned=lora_path.get("pinned", False),
                        )
                    else:
                        raise ValueError(
                            f"Invalid type for item in --lora-paths list: {type(lora_path)}. "
                            "Expected a string or a dictionary."
                        )
                    self.lora_paths.append(lora_ref)
            elif isinstance(self.lora_paths, dict):
                self.lora_paths = [
                    LoRARef(lora_name=k, lora_path=v, pinned=False)
                    for k, v in self.lora_paths.items()
                ]
            elif self.lora_paths is None:
                self.lora_paths = []
            else:
                raise ValueError(
                    f"Invalid type for --lora-paths: {type(self.lora_paths)}. "
                    "Expected a list or a dictionary."
                )

            # Expand target modules
            if self.lora_target_modules:
                self.lora_target_modules = set(self.lora_target_modules)
                if "all" in self.lora_target_modules:
                    assert (
                        len(self.lora_target_modules) == 1
                    ), "If 'all' is specified in --lora-target-modules, it should be the only module specified."
                    self.lora_target_modules = set(SUPPORTED_LORA_TARGET_MODULES)

                    # When using the chunked SGMV backend, skip embedding / lm_head layers for now,
                    # since it does not support these yet (TODO: implement embedding / lm_head support)
                    if self.lora_backend == "csgmv":
                        logger.warning(
                            "LoRA backend 'csgmv' does not yet support embedding or lm_head layers; "
                            "dropping 'embed_tokens' and 'lm_head' from --lora-target-modules=all. "
                            "To apply LoRA to these, use --lora-backend triton."
                        )
                        self.lora_target_modules.discard("embed_tokens")
                        self.lora_target_modules.discard("lm_head")

            # Ensure sufficient information is provided for LoRA initialization.
            assert self.lora_paths or (
                self.max_lora_rank and self.lora_target_modules
            ), "When no initial --lora-paths is provided, you need to specify both --max-lora-rank and --lora-target-modules for LoRA initialization."

            # Validate max_loaded_loras
            if self.max_loaded_loras is not None:
                assert self.max_loaded_loras >= self.max_loras_per_batch, (
                    "max_loaded_loras should be greater than or equal to max_loras_per_batch. "
                    f"max_loaded_loras={self.max_loaded_loras}, max_loras_per_batch={self.max_loras_per_batch}"
                )
                assert len(self.lora_paths) <= self.max_loaded_loras, (
                    "The number of LoRA paths should not exceed max_loaded_loras. "
                    f"max_loaded_loras={self.max_loaded_loras}, lora_paths={len(self.lora_paths)}"
                )

            if self.max_lora_chunk_size is not None:
                assert (
                    16 <= self.max_lora_chunk_size <= 128
                    and (self.max_lora_chunk_size & (self.max_lora_chunk_size - 1)) == 0
                ), "--max-lora-chunk-size must be a power of 2 between 16 and 128."

    def validate_disagg_tp_size(self, prefill_tp: int, decode_tp: int):
        larger_tp = max(decode_tp, prefill_tp)
        smaller_tp = min(decode_tp, prefill_tp)
        assert larger_tp % smaller_tp == 0, (
            "Different tp size is supported only when one tp is multiple of the other. "
            f"decode_tp={decode_tp}, prefill_tp={prefill_tp}"
        )

    def validate_buckets_rule(self, arg_name: str, buckets_rule: List[str]):
        if not buckets_rule:
            return

        assert len(buckets_rule) > 0, f"{arg_name} cannot be empty list"
        rule = buckets_rule[0]
        assert rule in [
            "tse",
            "default",
            "custom",
        ], f"Unsupported {arg_name} rule type: '{rule}'. Must be one of: 'tse', 'default', 'custom'"

        if rule == "tse":
            assert (
                len(buckets_rule) == 4
            ), f"{arg_name} TSE rule requires exactly 4 parameters: ['tse', middle, base, count], got {len(buckets_rule)}"
            try:
                middle = float(buckets_rule[1])
                base = float(buckets_rule[2])
                count = int(buckets_rule[3])
            except (ValueError, IndexError):
                assert (
                    False
                ), f"{arg_name} TSE rule parameters must be: ['tse', <float:middle>, <float:base>, <int:count>]"
            assert base > 1, f"{arg_name} TSE base must be larger than 1, got: {base}"
            assert count > 0, f"{arg_name} TSE count must be positive, got: {count}"
            assert middle > 0, f"{arg_name} TSE middle must be positive, got: {middle}"

        elif rule == "default":
            assert (
                len(buckets_rule) == 1
            ), f"{arg_name} default rule should only have one parameter: ['default'], got {len(buckets_rule)}"

        elif rule == "custom":
            assert (
                len(buckets_rule) >= 2
            ), f"{arg_name} custom rule requires at least one bucket value: ['custom', value1, ...]"
            try:
                bucket_values = [float(x) for x in buckets_rule[1:]]
            except ValueError:
                assert False, f"{arg_name} custom rule bucket values must be numeric"
            assert len(set(bucket_values)) == len(
                bucket_values
            ), f"{arg_name} custom rule bucket values should not contain duplicates"
            assert all(
                val >= 0 for val in bucket_values
            ), f"{arg_name} custom rule bucket values should be non-negative"

    def adjust_mem_fraction_for_vlm(self, model_config):
        vision_config = getattr(model_config.hf_config, "vision_config", None)
        if vision_config is None:
            return

        # roughly reduce the mem_fraction_static base on params of Vit
        original_server_arg_mem_fraction = self.mem_fraction_static
        # a base mem_fraction_static factor for regular Vit
        base_mem_fraction_reduction_ratio = 0.95

        vit_num_layers = getattr(vision_config, "num_hidden_layers", 24)
        vit_hidden_size = getattr(vision_config, "hidden_size", 1024)

        # baseline ViT params (ViT-L/14)
        baseline_vit_layers = 24
        baseline_vit_hidden_size = 1024

        # weight params count
        current_complexity_score = vit_num_layers * (vit_hidden_size**2)
        baseline_complexity_score = baseline_vit_layers * (baseline_vit_hidden_size**2)
        complexity_ratio = (
            current_complexity_score / baseline_complexity_score
            if baseline_complexity_score > 0
            else 1.0
        )

        # every time the complexity grows 100%, adjust final factor for 10%
        sensitivity_scale = 0.1
        dynamic_adjustment_factor = 1.0 - sensitivity_scale * (complexity_ratio - 1.0)
        dynamic_adjustment_factor = max(0.8, min(1.05, dynamic_adjustment_factor))

        final_overall_factor = (
            base_mem_fraction_reduction_ratio * dynamic_adjustment_factor
        )
        self.mem_fraction_static = (
            original_server_arg_mem_fraction * final_overall_factor
        )

    def validate_transfer_engine(self):
        if importlib.util.find_spec("mooncake.engine") is None:
            logger.warning(
                f"Failed to import mooncake.engine. Does not support using TransferEngine as remote instance weight loader backend."
            )
            return False
        elif self.enable_memory_saver:
            logger.warning(
                "Memory saver is enabled, which is not compatible with TransferEngine. Does not support using TransferEngine as remote instance weight loader backend."
            )
            return False
        else:
            return True

    def remote_instance_weight_loader_use_transfer_engine(self):
        # Use TransferEngine as seed backend.
        if self.remote_instance_weight_loader_start_seed_via_transfer_engine:
            return True
        # Use TransferEngine as client backend.
        elif (
            self.load_format == "remote_instance"
            and self.remote_instance_weight_loader_backend == "transfer_engine"
        ):
            return True
        else:
            return False


# NOTE: This is a global variable to hold the server args for scheduler.
_global_server_args: Optional[ServerArgs] = None


def set_global_server_args_for_scheduler(server_args: ServerArgs):
    global _global_server_args
    _global_server_args = server_args


set_global_server_args_for_tokenizer = set_global_server_args_for_scheduler


def get_global_server_args() -> ServerArgs:
    if _global_server_args is None:
        raise ValueError("Global server args is not set yet!")

    return _global_server_args


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

    # Check for config file and merge arguments if present
    if "--config" in argv:
        # Import here to avoid circular imports
        from sglang.srt.server_args_config_parser import ConfigArgumentMerger

        # Extract boolean actions from the parser to handle them correctly
        config_merger = ConfigArgumentMerger(parser)
        argv = config_merger.merge_config_with_args(argv)

    raw_args = parser.parse_args(argv)
    return ServerArgs.from_cli_args(raw_args)


ZMQ_TCP_PORT_DELTA = 233
DP_ATTENTION_HANDSHAKE_PORT_DELTA = 13


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

    # The ipc filename for Tokenizer and worker tokenizer
    tokenizer_worker_ipc_name: Optional[str]

    @staticmethod
    def init_new(
        server_args: ServerArgs,
        dp_rank: Optional[int] = None,
        worker_ports: Optional[List[int]] = None,
    ) -> PortArgs:
        if server_args.nccl_port is None:
            nccl_port = server_args.port + random.randint(100, 1000)
            while True:
                if is_port_available(nccl_port):
                    break
                if nccl_port < 60000:
                    nccl_port += 42
                else:
                    nccl_port -= 43
        else:
            nccl_port = server_args.nccl_port

        if server_args.tokenizer_worker_num == 1:
            tokenizer_worker_ipc_name = None
        else:
            tokenizer_worker_ipc_name = (
                f"ipc://{tempfile.NamedTemporaryFile(delete=False).name}"
            )

        if not server_args.enable_dp_attention:
            # Normal case, use IPC within a single node
            return PortArgs(
                tokenizer_ipc_name=f"ipc://{tempfile.NamedTemporaryFile(delete=False).name}",
                scheduler_input_ipc_name=f"ipc://{tempfile.NamedTemporaryFile(delete=False).name}",
                detokenizer_ipc_name=f"ipc://{tempfile.NamedTemporaryFile(delete=False).name}",
                nccl_port=nccl_port,
                rpc_ipc_name=f"ipc://{tempfile.NamedTemporaryFile(delete=False).name}",
                metrics_ipc_name=f"ipc://{tempfile.NamedTemporaryFile(delete=False).name}",
                tokenizer_worker_ipc_name=tokenizer_worker_ipc_name,
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
            dist_init_port = int(dist_init_port)
            port_base = dist_init_port + 1
            detokenizer_port = port_base + 1
            rpc_port = port_base + 2
            metrics_ipc_name = port_base + 3
            if dp_rank is None:
                # TokenizerManager to DataParallelController
                scheduler_input_port = port_base + 4
            else:
                assert worker_ports is not None
                scheduler_input_port = worker_ports[dp_rank]

            try:
                if dp_rank is None:
                    wait_port_available(dist_init_port, "dist_init_port")
                    wait_port_available(port_base, "port_base")
                    wait_port_available(detokenizer_port, "detokenizer_port")
                    wait_port_available(nccl_port, "nccl_port")
                    wait_port_available(rpc_port, "rpc_port")
                    wait_port_available(metrics_ipc_name, "metrics_ipc_name")
                # Check scheduler_input_port only for dp.
                # Skip check when using worker_ports since the port is already bound by our ZMQ socket
                if dp_rank is None or worker_ports is None:
                    wait_port_available(scheduler_input_port, "scheduler_input_port")
            except ValueError as e:
                logger.exception(
                    f"Port is already in use. {dist_init_port=} {port_base=} {detokenizer_port=} {nccl_port=} {scheduler_input_port=}"
                )
                raise

            return PortArgs(
                tokenizer_ipc_name=f"tcp://{dist_init_host}:{port_base}",
                scheduler_input_ipc_name=f"tcp://{dist_init_host}:{scheduler_input_port}",
                detokenizer_ipc_name=f"tcp://{dist_init_host}:{detokenizer_port}",
                nccl_port=nccl_port,
                rpc_ipc_name=f"tcp://{dist_init_host}:{rpc_port}",
                metrics_ipc_name=f"tcp://{dist_init_host}:{metrics_ipc_name}",
                tokenizer_worker_ipc_name=tokenizer_worker_ipc_name,
            )


class LoRAPathAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        lora_paths = []
        if values:
            assert isinstance(values, list), "Expected a list of LoRA paths."
            for lora_path in values:
                lora_path = lora_path.strip()
                if lora_path.startswith("{") and lora_path.endswith("}"):
                    obj = json.loads(lora_path)
                    assert "lora_path" in obj and "lora_name" in obj, (
                        f"{repr(lora_path)} looks like a JSON str, "
                        "but it does not contain 'lora_name' and 'lora_path' keys."
                    )
                    lora_paths.append(obj)
                else:
                    lora_paths.append(lora_path)

        setattr(namespace, self.dest, lora_paths)


def print_deprecated_warning(message: str):
    logger.warning(f"\033[1;33m{message}\033[0m")


class DeprecatedAction(argparse.Action):
    def __init__(self, option_strings, dest, nargs=0, **kwargs):
        super(DeprecatedAction, self).__init__(
            option_strings, dest, nargs=nargs, **kwargs
        )

    def __call__(self, parser, namespace, values, option_string=None):
        print_deprecated_warning(
            f"The command line argument '{option_string}' is deprecated and will be removed in future versions."
        )


def auto_choose_speculative_params(self: ServerArgs):
    """
    Automatically choose the parameters for speculative decoding.

    You can tune them on your own models and prompts with scripts/playground/bench_speculative.py
    """
    hf_config = self.get_model_config().hf_config
    arch = hf_config.architectures[0]
    if self.speculative_algorithm == "STANDALONE":
        # The default value for standalone speculative decoding
        return (3, 1, 4)
    if arch in ["LlamaForCausalLM"]:
        # The default value for llama
        return (5, 4, 8)
    elif arch in [
        "DeepseekV32ForCausalLM",
        "DeepseekV3ForCausalLM",
        "DeepseekV2ForCausalLM",
        "GptOssForCausalLM",
        "Glm4MoeForCausalLM",
        "Glm4MoeLiteForCausalLM",
        "BailingMoeForCausalLM",
        "BailingMoeV2ForCausalLM",
        "MistralLarge3ForCausalLM",
        "PixtralForConditionalGeneration",
        "MiMoV2FlashForCausalLM",
    ]:
        return (3, 1, 4)
    elif arch in ["Grok1ForCausalLM", "Grok1VForCausalLM"]:
        return (5, 4, 8)
    else:
        # The default value for all other models
        return (3, 1, 4)
