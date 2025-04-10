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
from typing import List, Literal, Optional

from sglang.srt.hf_transformers_utils import check_gguf_file
from sglang.srt.reasoning_parser import ReasoningParser
from sglang.srt.utils import (
    configure_ipv6,
    get_amdgpu_memory_capacity,
    get_device,
    get_hpu_memory_capacity,
    get_nvgpu_memory_capacity,
    is_cuda,
    is_flashinfer_available,
    is_hip,
    is_hpu,
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
    load_format: str = "auto"
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
    revision: Optional[str] = None

    # Port for the HTTP server
    host: str = "127.0.0.1"
    port: int = 30000

    # Memory and scheduling
    mem_fraction_static: Optional[float] = None
    max_running_requests: Optional[int] = None
    max_total_tokens: Optional[int] = None
    chunked_prefill_size: Optional[int] = None
    max_prefill_tokens: Optional[int] = None
    schedule_policy: str = "fcfs"
    schedule_conservativeness: float = 1.0
    cpu_offload_gb: int = 0
    page_size: int = 1

    # Other runtime options
    tp_size: int = 1
    stream_interval: int = 1
    stream_output: bool = False
    random_seed: Optional[int] = None
    constrained_json_whitespace_pattern: Optional[str] = None
    watchdog_timeout: float = 300
    dist_timeout: Optional[int] = None  # timeout for torch.distributed
    download_dir: Optional[str] = None
    base_gpu_id: int = 0
    gpu_id_step: int = 1

    # Logging
    log_level: str = "info"
    log_level_http: Optional[str] = None
    log_requests: bool = False
    log_requests_level: int = 0
    show_time_cost: bool = False
    enable_metrics: bool = False
    decode_log_interval: int = 40

    # API related
    api_key: Optional[str] = None
    file_storage_path: str = "sglang_storage"
    enable_cache_report: bool = False
    reasoning_parser: Optional[str] = None

    # Data parallelism
    dp_size: int = 1
    load_balance_method: str = "round_robin"

    # Expert parallelism
    ep_size: int = 1

    # Multi-node distributed serving
    dist_init_addr: Optional[str] = None
    nnodes: int = 1
    node_rank: int = 0

    # Model override args in JSON
    json_model_override_args: str = "{}"

    # LoRA
    lora_paths: Optional[List[str]] = None
    max_loras_per_batch: int = 8
    lora_backend: str = "triton"

    # Kernel backend
    attention_backend: Optional[str] = None
    sampling_backend: Optional[str] = None
    grammar_backend: Optional[str] = None

    # Speculative decoding
    speculative_algorithm: Optional[str] = None
    speculative_draft_model_path: Optional[str] = None
    speculative_num_steps: Optional[int] = None
    speculative_eagle_topk: Optional[int] = None
    speculative_num_draft_tokens: Optional[int] = None
    speculative_accept_threshold_single: float = 1.0
    speculative_accept_threshold_acc: float = 1.0
    speculative_token_map: Optional[str] = None

    # Double Sparsity
    enable_double_sparsity: bool = False
    ds_channel_config_path: Optional[str] = None
    ds_heavy_channel_num: int = 32
    ds_heavy_token_num: int = 256
    ds_heavy_channel_type: str = "qk"
    ds_sparse_decode_threshold: int = 4096

    # Optimization/debug options
    disable_radix_cache: bool = False
    disable_cuda_graph: bool = False
    disable_cuda_graph_padding: bool = False
    enable_nccl_nvls: bool = False
    disable_outlines_disk_cache: bool = False
    disable_custom_all_reduce: bool = False
    disable_mla: bool = False
    disable_overlap_schedule: bool = False
    enable_mixed_chunk: bool = False
    enable_dp_attention: bool = False
    enable_ep_moe: bool = False
    enable_deepep_moe: bool = False
    deepep_mode: Optional[Literal["auto", "normal", "low_latency"]] = "auto"
    enable_torch_compile: bool = False
    torch_compile_max_bs: int = 32
    cuda_graph_max_bs: Optional[int] = None
    cuda_graph_bs: Optional[List[int]] = None
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
    tool_call_parser: Optional[str] = None
    enable_hierarchical_cache: bool = False
    hicache_ratio: float = 2.0
    enable_flashinfer_mla: bool = False  # TODO: remove this argument
    enable_flashmla: bool = False
    flashinfer_mla_disable_ragged: bool = False
    warmups: Optional[str] = None
    n_share_experts_fusion: int = 0
    disable_shared_experts_fusion: bool = False

    # Debug tensor dumps
    debug_tensor_dump_output_folder: Optional[str] = None
    debug_tensor_dump_input_file: Optional[str] = None
    debug_tensor_dump_inject: bool = False

    # For PD disaggregation: can be "null" (not disaggregated), "prefill" (prefill-only), or "decode" (decode-only)
    disaggregation_mode: str = "null"
    disaggregation_bootstrap_port: int = 8998

    def __post_init__(self):
        # Expert parallelism
        if self.enable_ep_moe:
            self.ep_size = self.tp_size
            logger.info(
                f"EP MoE is enabled. The expert parallel size is adjusted to be the same as the tensor parallel size[{self.tp_size}]."
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

        if is_cuda():
            gpu_mem = get_nvgpu_memory_capacity()
        elif is_hip():
            gpu_mem = get_amdgpu_memory_capacity()
        elif self.device == "hpu":
            gpu_mem = get_hpu_memory_capacity()
        else:
            # GPU memory is not known yet or no GPU is available.
            gpu_mem = None

        if is_hip():
            self.disable_shared_experts_fusion = True

        # Set mem fraction static, which depends on the tensor parallelism size
        if self.mem_fraction_static is None:
            if self.tp_size >= 16:
                self.mem_fraction_static = 0.79
            elif self.tp_size >= 8:
                self.mem_fraction_static = 0.81
            elif self.tp_size >= 4:
                self.mem_fraction_static = 0.85
            elif self.tp_size >= 2:
                self.mem_fraction_static = 0.87
            else:
                self.mem_fraction_static = 0.88

        # Set chunked prefill size, which depends on the gpu memory capacity
        if self.chunked_prefill_size is None:
            if gpu_mem is not None and gpu_mem < 25_000:
                self.chunked_prefill_size = 2048
            else:
                self.chunked_prefill_size = 8192

        assert self.chunked_prefill_size % self.page_size == 0

        if self.enable_flashmla is True:
            logger.warning(
                "FlashMLA only supports a page_size of 64, change page_size to 64."
            )
            self.page_size = 64
        # Set cuda graph max batch size
        if self.cuda_graph_max_bs is None:
            # Based on detailed statistics, when serving TP1/TP2 models on lower-end GPUs with HBM<25G, you can either disable cuda graph or set `cuda_graph_max_bs` to a very small value to reduce the memory overhead of creating cuda graphs, with almost no impact on performance. However, when serving models with TP4 or TP8, we need to enable cuda graph to maintain high performance. In this case, we can set `cuda_graph_max_bs` to 80 (half of the default value 160) to reduce the memory overhead of creating cuda graphs. Looking at the logs from TP4 serving of qwen2-72b, a value of 80 is sufficient and can reduce the memory overhead of creating cuda graphs on lower-end GPUs compared to the original 160, avoiding OOM issues.
            if gpu_mem is not None and gpu_mem < 25_000:
                if self.tp_size < 4:
                    self.cuda_graph_max_bs = 8
                else:
                    self.cuda_graph_max_bs = 80
            else:
                self.cuda_graph_max_bs = 160

        # Set kernel backends for hpu device
        if self.device == "hpu":
            self.attention_backend = "hpu"
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

        # Expert parallelism
        if self.enable_ep_moe:
            self.ep_size = self.tp_size
            logger.info(
                f"EP MoE is enabled. The expert parallel size is adjusted to be the same as the tensor parallel size[{self.tp_size}]."
            )

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

        self.enable_sp_layernorm = False
        # DeepEP MoE
        if self.enable_deepep_moe:
            if self.deepep_mode == "auto":
                assert (
                    not self.enable_dp_attention
                ), "DeepEP MoE `auto` mode is not supported with DP Attention."
            self.ep_size = self.tp_size
            self.enable_sp_layernorm = (
                self.dp_size < self.tp_size if self.enable_dp_attention else True
            )
            logger.info(
                f"DeepEP MoE is enabled. The expert parallel size is adjusted to be the same as the tensor parallel size[{self.tp_size}]."
            )

        # Speculative Decoding
        if self.speculative_algorithm == "NEXTN":
            # NEXTN shares the same implementation of EAGLE
            self.speculative_algorithm = "EAGLE"

        if (
            self.speculative_algorithm == "EAGLE"
            or self.speculative_algorithm == "EAGLE3"
        ):
            if self.max_running_requests is None:
                self.max_running_requests = 48
            self.disable_overlap_schedule = True
            logger.info(
                "Overlap scheduler is disabled because of using "
                "eagle speculative decoding."
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

            if self.page_size > 1 and self.speculative_eagle_topk > 1:
                self.speculative_eagle_topk = 1
                logger.info("speculative_eagle_topk is changed to 1 when page_size > 1")

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
        if self.disaggregation_mode == "prefill":
            self.disable_cuda_graph = True
            logger.warning("KV cache is forced as chunk cache for decode server")
            self.disable_overlap_schedule = True
            logger.warning("Overlap scheduler is disabled for prefill server")
        elif self.disaggregation_mode == "decode":
            self.disable_radix_cache = True
            logger.warning("Cuda graph is disabled for prefill server")
            self.disable_overlap_schedule = True
            logger.warning("Overlap scheduler is disabled for decode server")

        os.environ["SGLANG_ENABLE_TORCH_COMPILE"] = (
            "1" if self.enable_torch_compile else "0"
        )

        if is_hpu():
            if self.max_running_requests is None:
                self.max_running_requests = 128

            self.page_size = 128  # Currently, HPU only supports page size of 128
            self.disable_radix_cache = (
                True  # Currently, HPU does not support radix cache
            )
            if self.max_prefill_tokens is None:
                self.max_prefill_tokens = (
                    4096  # Currently, HPU prefer smaller prefill batches
                )
        else:
            self.max_prefill_tokens = 16384

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser):
        # Model and port args
        parser.add_argument(
            "--model-path",
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
            "--host", type=str, default=ServerArgs.host, help="The host of the server."
        )
        parser.add_argument(
            "--port", type=int, default=ServerArgs.port, help="The port of the server."
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
            help="If set, skip init tokenizer and pass input_ids in generate request",
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
            help="The device to use ('cuda', 'xpu', 'hpu', 'cpu'). Defaults to auto-detection if not specified.",
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
            "--revision",
            type=str,
            default=None,
            help="The specific model version to use. It can be a branch "
            "name, a tag name, or a commit id. If unspecified, will use "
            "the default version.",
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

        # Other runtime options
        parser.add_argument(
            "--tensor-parallel-size",
            "--tp-size",
            type=int,
            default=ServerArgs.tp_size,
            help="The tensor parallelism size.",
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
            help="0: Log metadata. 1. Log metadata and partial input/output. 2. Log every input/output.",
            choices=[0, 1, 2],
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
            "--decode-log-interval",
            type=int,
            default=ServerArgs.decode_log_interval,
            help="The log interval of decode batch.",
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

        # Expert parallelism
        parser.add_argument(
            "--expert-parallel-size",
            "--ep-size",
            type=int,
            default=ServerArgs.ep_size,
            help="The expert parallelism size.",
        )

        # Multi-node distributed serving
        parser.add_argument(
            "--dist-init-addr",
            "--nccl-init-addr",  # For backward compatbility. This will be removed in the future.
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
            choices=["flashinfer", "triton", "torch_native", "fa3"],
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
        parser.add_argument(
            "--enable-flashinfer-mla",
            action="store_true",
            help="Enable FlashInfer MLA optimization. This argument will be deprecated soon! Please use '--attention-backend flashinfer' instead for switching on flashfiner mla!",
        )
        parser.add_argument(
            "--enable-flashmla",
            action="store_true",
            help="Enable FlashMLA decode optimization",
        )
        parser.add_argument(
            "--flashinfer-mla-disable-ragged",
            action="store_true",
            help="Not using ragged prefill wrapper when running flashinfer mla",
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
            "--enable-nccl-nvls",
            action="store_true",
            help="Enable NCCL NVLS for prefill heavy requests when available.",
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
            "--disable-mla",
            action="store_true",
            help="Disable Multi-head Latent Attention (MLA) for DeepSeek V2/V3/R1 series models.",
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
            help="Enabling data parallelism for attention and tensor parallelism for FFN. The dp size should be equal to the tp size. Currently only DeepSeek-V2 is supported.",
        )
        parser.add_argument(
            "--enable-ep-moe",
            action="store_true",
            help="Enabling expert parallelism for moe. The ep size is equal to the tp size.",
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
            "--cuda-graph-max-bs",
            type=int,
            default=ServerArgs.cuda_graph_max_bs,
            help="Set the maximum batch size for cuda graph.",
        )
        parser.add_argument(
            "--cuda-graph-bs",
            type=int,
            nargs="+",
            help="Set the list of batch sizes for cuda graph.",
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
            help="Cast the intermidiate attention results to fp32 to avoid possible crashes related to fp16."
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
            "--tool-call-parser",
            type=str,
            choices=["qwen25", "mistral", "llama3"],
            default=ServerArgs.tool_call_parser,
            help="Specify the parser for handling tool-call interactions. Options include: 'qwen25', 'mistral', and 'llama3'.",
        )
        parser.add_argument(
            "--enable-hierarchical-cache",
            action="store_true",
            help="Enable hierarchical cache",
        )
        parser.add_argument(
            "--hicache-ratio",
            type=float,
            required=False,
            default=ServerArgs.hicache_ratio,
            help="The ratio of the size of host KV cache memory pool to the size of device pool.",
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
            "--n-share-experts-fusion",
            type=int,
            default=0,
            help="The number of shared_experts need to be replica to fuse with normal experts in deepseek v3/r1 "
            "we use tp_size by default.",
        )
        parser.add_argument(
            "--disable-shared-experts-fusion",
            action="store_true",
            help="Disable shared experts fusion by setting n_share_experts_fusion to 0.",
        )

        # Server warmups
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

        # Disaggregation
        parser.add_argument(
            "--disaggregation-mode",
            type=str,
            default="null",
            choices=["null", "prefill", "decode"],
            help='Only used for PD disaggregation. "prefill" for prefill-only server, and "decode" for decode-only server. If not specified, it is not PD disaggregated',
        )
        parser.add_argument(
            "--disaggregation-bootstrap-port",
            type=int,
            default=ServerArgs.disaggregation_bootstrap_port,
            help="Bootstrap server port on the prefill server. Default is 8998.",
        )

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace):
        args.tp_size = args.tensor_parallel_size
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
            self.tp_size % self.nnodes == 0
        ), "tp_size must be divisible by number of nodes"
        assert not (
            self.dp_size > 1 and self.nnodes != 1 and not self.enable_dp_attention
        ), "multi-node data parallel is not supported unless dp attention!"
        assert (
            self.max_loras_per_batch > 0
            # FIXME
            and (self.lora_paths is None or self.disable_cuda_graph)
            and (self.lora_paths is None or self.disable_radix_cache)
        ), "compatibility of lora and cuda graph and radix attention is in progress"
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
                scheduler_input_port = (
                    port_base + 3
                )  # TokenizerManager to DataParallelController
            else:
                scheduler_input_port = port_base + 3 + 1 + dp_rank

            return PortArgs(
                tokenizer_ipc_name=f"tcp://{dist_init_host}:{port_base}",
                scheduler_input_ipc_name=f"tcp://{dist_init_host}:{scheduler_input_port}",
                detokenizer_ipc_name=f"tcp://{dist_init_host}:{port_base + 1}",
                nccl_port=port,
                rpc_ipc_name=f"tcp://{dist_init_host}:{port_base + 2}",
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


def auto_choose_speculative_params(self: ServerArgs):
    """
    Automatically choose the parameters for speculative decoding.

    You can tune them on your own models and prompts with scripts/playground/bench_speculative.py
    """
    if self.decrypted_config_file:
        config_path = self.decrypted_config_file
    else:
        config_path = os.path.join(self.model_path, "config.json")
    if not os.path.exists(config_path):
        raise ValueError(f"{config_path} is not found.")

    config = json.load(open(config_path))

    arch = config.get("architectures", ["Unknown"])[0]

    if arch in ["LlamaForCausalLM"]:
        # The default value for llama
        return (5, 4, 8)
    elif arch in ["DeepseekV3ForCausalLM", "DeepseekV2ForCausalLM"]:
        # The default value for deepseek
        return (5, 4, 8)
    elif arch in ["Grok1ForCausalLM", "Grok1VForCausalLM"]:
        return (5, 4, 8)
    else:
        # The default value for all other models
        return (5, 4, 8)
