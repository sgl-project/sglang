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
import glob
import importlib
import importlib.util
import json
import logging
import math
import os
import random
import socket
import tempfile
import uuid
from functools import cached_property
from typing import Any, Callable, Dict, List, Literal, Optional, Union

from sglang.jit_kernel.kv_canary.consts import RealKvHashMode
from sglang.srt.arg_groups.arg_utils import A, Arg, add_cli_args_from_dataclass
from sglang.srt.arg_groups.argparse_actions import (
    DeprecatedAction,
    DeprecatedAliasStoreAction,
    DeprecatedStoreConstAction,
    DeprecatedStoreTrueAction,
    LoRAPathAction,
)
from sglang.srt.configs.linear_attn_model_registry import get_linear_attn_spec_by_arch
from sglang.srt.connector import ConnectorType
from sglang.srt.distributed.device_communicators.mooncake_transfer_engine import (
    parse_ib_device_config,
)
from sglang.srt.environ import envs
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.srt.layers.attention.fla.chunk_delta_h import CHUNK_SIZE as FLA_CHUNK_SIZE
from sglang.srt.lora.lora_registry import LoRARef
from sglang.srt.model_executor.cuda_graph_config import (
    ALLOWED_BACKENDS_PER_PHASE,
    Backend,
    CudaGraphConfig,
    Phase,
    default_cuda_graph_config,
    parse_cuda_graph_config_arg,
)
from sglang.srt.parser.reasoning_parser import ReasoningParser
from sglang.srt.platforms import current_platform
from sglang.srt.speculative.decoupled_spec_io import DecoupledSpecIpcConfig
from sglang.srt.utils.common import (
    LORA_TARGET_ALL_MODULES,
    SUPPORTED_LORA_TARGET_MODULES,
    cpu_has_amx_support,
    get_device,
    get_device_memory_capacity,
    get_device_sm,
    get_int_env_var,
    get_nvidia_driver_version,
    get_quantization_config,
    has_fp8_weights_in_checkpoint,
    human_readable_int,
    is_blackwell_supported,
    is_cpu,
    is_cuda,
    is_flashinfer_available,
    is_hip,
    is_hopper_with_cuda_12_3,
    is_host_cpu_arm64,
    is_mps,
    is_musa,
    is_no_spec_infer_or_topk_one,
    is_npu,
    is_remote_url,
    is_sm90_supported,
    is_sm100_supported,
    is_sm120_supported,
    is_triton_kernels_available,
    is_xpu,
    json_list_type,
    nullable_str,
    parse_connector_type,
    torch_release,
    xpu_has_xmx_support,
)
from sglang.srt.utils.hf_transformers_utils import check_gguf_file
from sglang.srt.utils.network import NetworkAddress, get_free_port, wait_port_available
from sglang.srt.utils.runai_utils import ObjectStorageModel, is_runai_obj_uri
from sglang.srt.utils.tensor_bridge import use_mlx
from sglang.utils import is_in_ci

logger = logging.getLogger(__name__)

# Define constants
DEFAULT_UVICORN_ACCESS_LOG_EXCLUDE_PREFIXES = ()
MIMO_V2_MODEL_ARCHS = (
    "MiMoV2ForCausalLM",
    "MiMoV2FlashForCausalLM",
)
LLAMA4_MODEL_ARCHS = (
    "Llama4ForConditionalGeneration",
    "Llama4ForCausalLM",
)

SAMPLING_BACKEND_CHOICES = {"flashinfer", "pytorch", "ascend"}
if envs.SGLANG_KV_CANARY_ENABLE_TOKEN_ORACLE.get():
    SAMPLING_BACKEND_CHOICES.add("token_oracle")

LOAD_FORMAT_CHOICES = [
    "auto",
    "pt",
    "safetensors",
    "npcache",
    "dummy",
    "sharded_state",
    "gguf",
    "bitsandbytes",
    "mistral",
    "layered",
    "flash_rl",
    "remote",
    "remote_instance",
    "fastsafetensors",
    "private",
    "runai_streamer",
]

# TODO: this list should likely contain only methods that support online quantization, or that support using custom quantization classes compatible with a given `quant_method` in config.json.
# Some of the choices here do NOT support online quantization.
QUANTIZATION_CHOICES = [
    "awq",
    "fp8",  # MOE + linear online quantization.
    "mxfp8",  # MOE + linear online quantization.
    "gptq",
    "marlin",
    "gptq_marlin",
    "awq_marlin",
    "bitsandbytes",
    "gguf",
    # Modelopt has some online quantization support through ModelOptModelLoader.
    "modelopt",
    "modelopt_fp8",
    "modelopt_fp4",
    "nvfp4_online",
    "modelopt_mixed",
    "petit_nvfp4",
    "w8a8_int8",  # mentioned in quantization.md documentation, supporting compressed-tensors quant_method.
    "w8a8_fp8",  # mentioned in quantization.md documentation, supporting compressed-tensors quant_method.
    "moe_wna16",  # custom loading logic for gptq/awq checkpoints (likely untested/unused)
    "qoq",
    "w4afp8",
    "mxfp4",  # MOE-only.
    "auto-round",
    "auto-round-int8",
    "compressed-tensors",  # for Ktransformers
    "modelslim",  # for NPU
    "quark",  # AMD Quark quantizer (FP8 / MXFP4 / Int4FP8 etc.)
    "quark_int4fp8_moe",
    "quark_mxfp4",  # Online MOE + linear quantization.
    # Apple Silicon MLX backend — on-the-fly quantization of fp16 weights at load
    # time via mlx.nn.quantize. Only takes effect when SGLANG_USE_MLX=1.
    "mlx_q4",  # 4 bits, group_size=64 (mlx-community default)
    "mlx_q8",  # 8 bits, group_size=64
    "unquant",
]


SPECULATIVE_DRAFT_MODEL_QUANTIZATION_CHOICES = QUANTIZATION_CHOICES

ATTENTION_BACKEND_CHOICES = [
    # Common
    "triton",
    "torch_native",
    "flex_attention",
    "dsa",
    "nsa",  # Deprecated alias for "dsa"
    "dsv4",
    "compressed",  # Deprecated alias for "dsv4"
    # NVIDIA specific
    "cutlass_mla",
    "fa3",
    "fa4",
    "flashinfer",
    "flashmla",
    "trtllm_mla",
    "cutedsl_mla",
    "tokenspeed_mla",
    "trtllm_mha",
    "dual_chunk_flash_attn",
    # AMD specific
    "aiter",
    "wave",
    # Other platforms
    "intel_amx",
    "ascend",
    "intel_xpu",
    "mlu",
]

DETERMINISTIC_ATTENTION_BACKEND_CHOICES = ["flashinfer", "fa3", "triton", "ascend"]

RADIX_SUPPORTED_DETERMINISTIC_ATTENTION_BACKEND = ["fa3", "triton", "ascend"]

DISAGG_TRANSFER_BACKEND_CHOICES = [
    "mooncake",
    "nixl",
    "ascend",
    "fake",
    "mori",
    "mooncake_tcp",
]

GRAMMAR_BACKEND_CHOICES = ["xgrammar", "outlines", "llguidance", "none"]

# Placeholder token inserted between items in Multi-Item Scoring sequences:
# query<delim>item1<delim>item2<delim>... Positions are pre-computed from item
# lengths (multi_item_delimiter_indices); the token only exists for FlashInfer
# attention mask compat and logprob column indexing. Will be removed once the
# attention backend supports position-only MIS.
MIS_DELIMITER_TOKEN_ID = 9999

MOE_RUNNER_BACKEND_CHOICES = [
    "auto",
    "deep_gemm",
    "triton",
    "triton_kernel",
    "flashinfer_trtllm",
    "experimental_sgl_trtllm",
    "flashinfer_trtllm_routed",
    "flashinfer_cutlass",
    "flashinfer_mxfp4",
    "flashinfer_cutedsl",
    "cutlass",
    "aiter",
    "marlin",
]

MOE_A2A_BACKEND_CHOICES = [
    "none",
    "deepep",
    "mooncake",
    "nixl",
    "mori",
    "ascend_fuseep",
    "flashinfer",
    "megamoe",
]

FP8_GEMM_RUNNER_BACKEND_CHOICES = [
    "auto",
    "deep_gemm",
    "flashinfer_trtllm",
    "flashinfer_cutlass",
    "flashinfer_deepgemm",
    "cutlass",
    "triton",
    "aiter",
]

FP4_GEMM_RUNNER_BACKEND_CHOICES = [
    "auto",
    "cutlass",
    "flashinfer_cudnn",
    "flashinfer_cutedsl",
    "flashinfer_cutlass",
    "flashinfer_trtllm",
    "marlin",
]

RADIX_EVICTION_POLICY_CHOICES = ["lru", "lfu", "slru", "priority"]

RL_ON_POLICY_TARGET_CHOICES = ["fsdp"]

LORA_BACKEND_CHOICES = ["triton", "csgmv", "ascend", "torch_native"]

ENCODER_TRANSFER_BACKEND_CHOICES = ["zmq_to_scheduler", "zmq_to_tokenizer", "mooncake"]

DSA_PREFILL_CP_SPLIT_CHOICES = ["in-seq-split", "round-robin-split"]
NSA_PREFILL_CP_SPLIT_CHOICES = DSA_PREFILL_CP_SPLIT_CHOICES  # deprecated alias

PREFILL_CP_SPLIT_CHOICES = ["in-seq-split"]

DEFAULT_LORA_EVICTION_POLICY = "lru"

DSA_CHOICES = [
    "flashmla_sparse",
    "flashmla_kv",
    "flashmla_auto",
    "fa3",
    "tilelang",
    "aiter",
    "trtllm",
]
NSA_CHOICES = DSA_CHOICES  # deprecated alias

DSA_TOPK_BACKEND_CHOICES = ["sgl-kernel", "torch", "flashinfer"]

MAMBA_RADIX_CACHE_STRATEGY_CHOICES = [
    "auto",
    "no_buffer",
    "extra_buffer",
    "extra_buffer_lazy",
]

MAMBA_BACKEND_CHOICES = ["triton", "flashinfer"]

LINEAR_ATTN_KERNEL_BACKEND_CHOICES = ["triton", "cutedsl", "flashinfer"]


# Allow external code to add more choices
def add_load_format_choices(choices):
    LOAD_FORMAT_CHOICES.extend(choices)


def add_quantization_method_choices(choices):
    QUANTIZATION_CHOICES.extend(choices)


def add_attention_backend_choices(choices):
    ATTENTION_BACKEND_CHOICES.extend(choices)


def add_deterministic_attention_backend_choices(choices):
    DETERMINISTIC_ATTENTION_BACKEND_CHOICES.extend(choices)


def add_radix_supported_deterministic_attention_backend_choices(choices):
    RADIX_SUPPORTED_DETERMINISTIC_ATTENTION_BACKEND.extend(choices)


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


def add_radix_eviction_policy_choices(choices):
    RADIX_EVICTION_POLICY_CHOICES.extend(choices)


def add_rl_on_policy_target_choices(choices):
    RL_ON_POLICY_TARGET_CHOICES.extend(choices)


def add_linear_attn_kernel_backend_choices(choices):
    LINEAR_ATTN_KERNEL_BACKEND_CHOICES.extend(choices)


@dataclasses.dataclass
class ServerArgs:
    """Server-wide configuration for SGLang.

    Adding new arguments
    --------------------
    1. **Place the field in the right section.** Arguments are grouped by
       comment blocks (``# Model and tokenizer``, ``# LoRA``, etc.).
       Add new fields to the matching section, or create a new section
       with a ``# ---`` banner when none fits.

    2. **Use the ``A[T, ...]`` annotation.**  ``A`` is an alias for
       ``typing.Annotated``.  The primary CLI flag is auto-derived from the
       field name (``tp_size`` → ``--tp-size``).  Use ``aliases`` for
       longer alternate names
       (``aliases=["--tensor-parallel-size"]``)::

           # Bare string — simplest form (just help text):
           host: A[str, "The host of the HTTP server."] = "127.0.0.1"
           trust_remote_code: A[bool, "Whether to allow custom models."] = False

           # Arg(...) — when you need choices, aliases, type_parser, etc.:
           load_format: A[str, Arg(help="...", choices=CHOICES)] = "auto"
           model_path: A[str, Arg(help="...", aliases=["--model"])]

       See ``Arg`` in ``arg_groups/arg_utils.py`` for the full list of
       supported metadata (``choices``, ``aliases``, ``type_parser``,
       ``nargs``, ``const``, ``action``, ``no_cli``, …).

    3. **Manual entries in ``add_cli_args`` — only for special cases.**
       A few arguments cannot use the annotation style and must be
       registered manually in ``add_cli_args``:

       - **Deprecated flags** that redirect to another field via
         ``DeprecatedAction`` / ``DeprecatedAliasStoreAction`` / etc.
       - **Dynamic choices** computed at runtime (e.g. ``reasoning_parser``
         whose choices come from a plugin registry).
       - The ``--config`` meta-argument (not a dataclass field).

       Everything else should use the ``A[T, ...]`` annotation.
    """

    # -------------------------------------------------------------------------
    # Model and tokenizer
    # -------------------------------------------------------------------------
    model_path: A[
        str,
        Arg(
            help="The path of the model weights. This can be a local folder or a Hugging Face repo ID.",
            aliases=["--model"],
        ),
    ]
    tokenizer_path: A[Optional[str], "The path of the tokenizer."] = None
    tokenizer_mode: A[
        str,
        Arg(
            help="Tokenizer mode. 'auto' will use the fast tokenizer if available, "
            "and 'slow' will always use the slow tokenizer.",
            choices=["auto", "slow"],
        ),
    ] = "auto"
    tokenizer_backend: A[
        str,
        Arg(
            help="Tokenizer backend. 'huggingface' uses the default HuggingFace "
            "tokenizers library, and 'fastokens' uses the fastokens library "
            "for faster tokenization. Requires the fastokens package to be installed.",
            choices=["huggingface", "fastokens"],
        ),
    ] = "huggingface"
    tokenizer_worker_num: A[int, "The worker num of the tokenizer manager."] = 1
    detokenizer_worker_num: A[int, "The worker num of the detokenizer manager."] = 1
    skip_tokenizer_init: A[
        bool, "If set, skip init tokenizer and pass input_ids in generate request."
    ] = False
    load_format: A[
        str,
        Arg(
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
            choices=LOAD_FORMAT_CHOICES,
        ),
    ] = "auto"
    model_loader_extra_config: A[
        str,
        "Extra config for model loader. This will be passed to the model loader corresponding to the chosen load_format.",
    ] = "{}"
    trust_remote_code: A[
        bool,
        "Whether or not to allow for custom models defined on the Hub in their own modeling files.",
    ] = False
    context_length: A[
        Optional[int],
        Arg(
            help="The model's maximum context length. Defaults to None (will use the value from the model's config.json instead)."
            f"\n\n{human_readable_int.__doc__}",
            type_parser=human_readable_int,
        ),
    ] = None
    is_embedding: A[bool, "Whether to use a CausalLM as an embedding model."] = False
    enable_multimodal: A[
        Optional[bool],
        "Enable the multimodal functionality for the served model. If the model being served is not multimodal, nothing will happen",
    ] = None
    revision: A[
        Optional[str],
        "The specific model version to use. It can be a branch name, a tag name, or a commit id. If unspecified, will use the default version.",
    ] = None
    model_impl: A[
        str,
        Arg(
            help=(
                "Which implementation of the model to use.\n\n"
                '* "auto" will try to use the SGLang implementation if it exists '
                "and fall back to the Transformers implementation if no SGLang "
                "implementation is available.\n"
                '* "sglang" will use the SGLang model implementation.\n'
                '* "transformers" will use the Transformers model '
                '* "mindspore" will use the MindSpore model '
                "implementation.\n"
            )
        ),
    ] = "auto"
    model_config_parser: A[
        str,
        Arg(
            help=(
                'Which model-config parser to use. "auto" picks "mistral" '
                'via the is_mistral_model name heuristic, else "hf" '
                "(AutoConfig over config.json). Plugins can register additional "
                "parsers via @register_model_config_parser."
            )
        ),
    ] = "auto"
    json_model_override_args: A[
        str,
        "A dictionary in JSON string format used to override default model configurations.",
    ] = "{}"

    # -------------------------------------------------------------------------
    # HTTP server
    # -------------------------------------------------------------------------
    host: A[str, "The host of the HTTP server."] = "127.0.0.1"
    port: A[int, "The port of the HTTP server."] = 30000
    fastapi_root_path: A[str, "App is behind a path based routing proxy."] = ""
    grpc_mode: A[bool, "If set, use gRPC server instead of HTTP server."] = False
    skip_server_warmup: A[bool, "If set, skip warmup."] = False
    warmups: A[
        Optional[str],
        "Specify custom warmup functions (csv) to run before server starts eg. --warmups=warmup_name1,warmup_name2 will run the functions `warmup_name1` and `warmup_name2` specified in warmup.py before the server starts listening for requests",
    ] = None
    enable_http2: A[
        bool,
        "Use Granian instead of Uvicorn as the ASGI server, enabling HTTP/1.1 and HTTP/2 auto-negotiation. Clients may use h2c (cleartext HTTP/2) or plain HTTP/1.1. Requires 'pip install sglang[http2]'.",
    ] = False

    # -------------------------------------------------------------------------
    # SSL/TLS
    # -------------------------------------------------------------------------
    ssl_keyfile: A[Optional[str], "The file path to the SSL key file."] = None
    ssl_certfile: A[Optional[str], "The file path to the SSL certificate file."] = None
    ssl_ca_certs: A[Optional[str], "The CA certificates file."] = None
    ssl_keyfile_password: A[
        Optional[str], "The password to decrypt the SSL keyfile."
    ] = None
    enable_ssl_refresh: A[
        bool,
        "Enable automatic SSL certificate hot-reloading when cert/key files change on disk. Requires --ssl-certfile and --ssl-keyfile.",
    ] = False

    # -------------------------------------------------------------------------
    # Quantization and data type
    # -------------------------------------------------------------------------
    dtype: A[
        str,
        Arg(
            help=(
                "Data type for model weights and activations.\n\n"
                '* "auto" will use FP16 precision for FP32 and FP16 models, and '
                "BF16 precision for BF16 models.\n"
                '* "half" for FP16. Recommended for AWQ quantization.\n'
                '* "float16" is the same as "half".\n'
                '* "bfloat16" for a balance between precision and range.\n'
                '* "float" is shorthand for FP32 precision.\n'
                '* "float32" for FP32 precision.'
            ),
            choices=["auto", "half", "float16", "bfloat16", "float", "float32"],
        ),
    ] = "auto"
    quantization: A[
        Optional[str],
        Arg(help="The quantization method.", choices=QUANTIZATION_CHOICES),
    ] = None
    quantization_param_path: A[
        Optional[str],
        Arg(
            help=(
                "Path to the JSON file containing the KV cache scaling factors. "
                "This should generally be supplied, when KV cache dtype is FP8. "
                "Otherwise, KV cache scaling factors default to 1.0, which may "
                "cause accuracy issues. "
            ),
            type_parser=nullable_str,
        ),
    ] = None
    kv_cache_dtype: A[
        str,
        Arg(
            help=(
                'Data type for kv cache storage. "auto" will use model data type. '
                '"bf16" or "bfloat16" for BF16 KV cache. "fp8_e5m2" and '
                '"fp8_e4m3" are supported for CUDA 11.8+. "fp4_e2m1" (only '
                "mxfp4) is supported for CUDA 12.8+ and PyTorch 2.8.0+"
            ),
            choices=["auto", "fp8_e5m2", "fp8_e4m3", "bf16", "bfloat16", "fp4_e2m1"],
        ),
    ] = "auto"
    enable_fp32_lm_head: A[
        bool, "If set, the LM head outputs (logits) are in FP32."
    ] = False
    modelopt_quant: A[
        Optional[Union[str, Dict]],
        (
            "The ModelOpt quantization configuration. Supported values: 'fp8', "
            "'int4_awq', 'w4a8_awq', 'nvfp4', 'nvfp4_awq'. This requires the "
            "NVIDIA Model Optimizer library to be installed: pip install "
            "nvidia-modelopt"
        ),
    ] = None
    modelopt_checkpoint_restore_path: A[
        Optional[str],
        (
            "Path to restore a previously saved ModelOpt quantized checkpoint. "
            "If provided, the quantization process will be skipped and the model "
            "will be loaded from this checkpoint."
        ),
    ] = None
    modelopt_checkpoint_save_path: A[
        Optional[str],
        (
            "Path to save the ModelOpt quantized checkpoint after quantization. "
            "This allows reusing the quantized model in future runs."
        ),
    ] = None
    modelopt_export_path: A[
        Optional[str],
        (
            "Path to export the quantized model in HuggingFace format after "
            "ModelOpt quantization. The exported model can then be used directly "
            "with SGLang for inference. If not provided, the model will not be "
            "exported."
        ),
    ] = None
    quantize_and_serve: A[
        bool,
        (
            "Quantize the model with ModelOpt and immediately serve it without "
            "exporting. This is useful for development and prototyping. For "
            "production, it's recommended to use separate quantization and "
            "deployment steps."
        ),
    ] = False
    rl_quant_profile: A[
        Optional[str],
        "Path to the FlashRL quantization profile. Required when using --load-format flash_rl.",
    ] = None  # For flash_rl load format
    enable_tf32_matmul: A[
        bool,
        "Enable float32 matmuls to use TensorFloat32 precision for better performance (via torch.set_float32_matmul_precision). CUDA only.",
    ] = False

    # -------------------------------------------------------------------------
    # Memory and scheduling
    # -------------------------------------------------------------------------
    mem_fraction_static: A[
        Optional[float],
        "The fraction of the memory used for static allocation (model weights and KV cache memory pool). Use a smaller value if you see out-of-memory errors.",
    ] = None
    max_running_requests: A[
        Optional[int], "The maximum number of running requests."
    ] = None
    max_queued_requests: A[
        Optional[int],
        "The maximum number of queued requests. This option is ignored when using disaggregation-mode.",
    ] = None
    max_total_tokens: A[
        Optional[int],
        Arg(
            help=(
                "The maximum number of tokens in the memory pool. If not "
                "specified, it will be automatically calculated based on the "
                "memory usage fraction. This option is typically used for "
                "development and debugging purposes."
                + f"\n\n{human_readable_int.__doc__}"
            ),
            type_parser=human_readable_int,
        ),
    ] = None
    chunked_prefill_size: A[
        Optional[int],
        "The maximum number of tokens in a chunk for the chunked prefill. Setting this to -1 means disabling chunked prefill.",
    ] = None
    enable_dynamic_chunking: A[
        bool,
        "Enable dynamic chunk size adjustment for pipeline parallelism. When enabled, chunk sizes are dynamically calculated based on fitted function to maintain consistent execution time across chunks.",
    ] = False
    max_prefill_tokens: A[
        int,
        Arg(
            help=(
                "The maximum number of tokens in a prefill batch. The real bound "
                "will be the maximum of this value and the model's maximum "
                "context length." + f"\n\n{human_readable_int.__doc__}"
            ),
            type_parser=human_readable_int,
        ),
    ] = 16384
    prefill_max_requests: A[
        Optional[int],
        "The maximum number of requests in a prefill batch. If not specified, there is no limit.",
    ] = None
    schedule_policy: A[
        str,
        Arg(
            help="The scheduling policy of the requests.",
            choices=[
                "lpm",
                "random",
                "fcfs",
                "dfs-weight",
                "lof",
                "priority",
                "routing-key",
            ],
        ),
    ] = "fcfs"
    enable_priority_scheduling: A[
        bool,
        "Enable priority scheduling. Requests with higher priority integer values will be scheduled first by default.",
    ] = False
    disable_priority_preemption: A[bool, "Disable priority scheduling preemption."] = (
        False
    )
    default_priority_value: A[
        Optional[int], "Default priority for requests without explicit priority."
    ] = None
    abort_on_priority_when_disabled: A[
        bool,
        "If set, abort requests that specify a priority when priority scheduling is disabled.",
    ] = False
    schedule_low_priority_values_first: A[
        bool,
        "If specified with --enable-priority-scheduling, the scheduler will schedule requests with lower priority integer values first.",
    ] = False
    priority_scheduling_preemption_threshold: A[
        int,
        "Minimum difference in priorities for an incoming request to have to preempt running request(s).",
    ] = 10
    schedule_conservativeness: A[
        float,
        "How conservative the schedule policy is. A larger value means more conservative scheduling. Use a larger value if you see requests being retracted frequently.",
    ] = 1.0
    page_size: A[Optional[int], "The number of tokens in a page."] = None
    swa_full_tokens_ratio: A[
        float,
        (
            "The ratio of SWA layer KV tokens / full layer KV tokens, regardless "
            "of the number of swa:full layers. It should be between 0 and 1. "
            "E.g. 0.5 means if each swa layer has 50 tokens, then each full "
            "layer has 100 tokens."
        ),
    ] = 0.8
    disable_hybrid_swa_memory: A[bool, "Disable the hybrid SWA memory pool."] = False
    radix_eviction_policy: A[
        str,
        Arg(
            help=(
                "The eviction policy of radix trees. 'lru' stands for Least "
                "Recently Used, 'lfu' stands for Least Frequently Used, 'slru' "
                "stands for Segmented Least Recently Used, and 'priority' evicts "
                "lower-priority requests first."
            ),
            choices=RADIX_EVICTION_POLICY_CHOICES,
        ),
    ] = "lru"
    prefill_only_disable_kv_cache: A[
        bool,
        "Skip the physical KV cache allocation for embedding-mode prefill-only workloads. Currently only valid with --is-embedding, --chunked-prefill-size=-1, --disable-radix-cache, an FA prefill backend, and non-FP4 KV cache so the fa_skip_kv_cache path is active (no layer reads or writes the cache). Other prefill-only workloads such as scoring/MIS may benefit from this later once their attention paths stop using paged KV. Scheduler admission accounting is unchanged; per-layer K/V tensors are sized to (page_size, head_num, head_dim) placeholders so GPU memory is not wasted.",
    ] = False
    disable_radix_cache: A[bool, "Disable RadixAttention for prefix caching."] = False
    enable_page_major_kv_layout: A[
        bool,
        "Enable the page-major KV layout: lay out the Mamba state and full/SWA "
        "KV caches in a page-granularity envelope (page is the outermost axis, "
        "layer-major within a page) instead of the default per-layer "
        "(layer-major) layout. Requires the Triton attention / linear-attn / "
        "Mamba backends.",
    ] = False
    disable_chunked_prefix_cache: A[
        bool,
        "Disable chunked prefix cache feature for deepseek, which should save overhead for short sequences.",
    ] = False
    disable_overlap_schedule: A[
        bool,
        "Disable the overlap scheduler, which overlaps the CPU scheduler with GPU model worker.",
    ] = False
    num_continuous_decode_steps: A[
        int,
        "Run multiple continuous decoding steps to reduce scheduling overhead. This can potentially increase throughput but may also increase time-to-first-token latency. The default value is 1, meaning only run one decoding step at a time.",
    ] = 1
    scheduler_recv_interval: A[
        int,
        "The interval to poll requests in scheduler. Can be set to >1 to reduce the overhead of this.",
    ] = 1
    enable_mixed_chunk: A[
        bool,
        "Enabling mixing prefill and decode in a batch when using chunked prefill.",
    ] = False

    # -------------------------------------------------------------------------
    # Device info and server timeout
    # -------------------------------------------------------------------------
    device: A[
        Optional[str],
        "The device to use ('cuda', 'xpu', 'hpu', 'npu', 'cpu', 'musa'). Defaults to auto-detection if not specified.",
    ] = None
    base_gpu_id: A[
        int,
        "The base GPU ID to start allocating GPUs from. Useful when running multiple instances on the same machine.",
    ] = 0
    gpu_id_step: A[
        int,
        "The delta between consecutive GPU IDs that are used. For example, setting it to 2 will use GPU 0,2,4,...",
    ] = 1
    random_seed: A[Optional[int], "The random seed."] = None
    watchdog_timeout: A[
        float,
        "Set watchdog timeout in seconds. If a forward batch takes longer than this, the server will crash to prevent hanging.",
    ] = 300
    soft_watchdog_timeout: A[
        Optional[float],
        "Set soft watchdog timeout in seconds. If a forward batch takes longer than this, the server will dump information for debugging.",
    ] = None
    sleep_on_idle: A[bool, "Reduce CPU usage when sglang is idle."] = False
    use_ray: A[bool, "Use Ray actors for scheduler process management."] = False
    custom_sigquit_handler: Optional[Callable] = None
    numa_node: A[
        Optional[List[int]],
        "Sets the numa node for the subprocesses. i-th element corresponds to i-th subprocess. If unset, will be automatically detected on NUMA systems.",
    ] = None
    gc_threshold: A[
        Optional[List[int]],
        "Set the garbage collection thresholds (the collection frequency). Accepts 1 to 3 integers.",
    ] = None

    # -------------------------------------------------------------------------
    # Distributed topology and parallelism (TP, PP, DP, CP)
    # -------------------------------------------------------------------------
    nccl_port: A[
        Optional[int],
        "The port for NCCL distributed environment setup. Defaults to a random port.",
    ] = None
    dist_timeout: A[
        Optional[int],
        "Set timeout for torch.distributed initialization.",
    ] = None
    dist_init_addr: A[
        Optional[str],
        Arg(
            help="The host address for initializing distributed backend (e.g., `192.168.0.2:25000`).",
            aliases=["--nccl-init-addr"],
        ),
    ] = None
    nnodes: A[int, "The number of nodes."] = 1
    node_rank: A[int, "The node rank."] = 0
    tp_size: A[
        int,
        Arg(
            help="The tensor parallelism size.",
            aliases=["--tensor-parallel-size"],
        ),
    ] = 1
    dcp_size: A[
        int,
        Arg(
            help="The decode context parallelism size.",
            aliases=["--decode-context-parallel-size"],
        ),
    ] = 1
    pp_size: A[
        int,
        Arg(
            help="The pipeline parallelism size.",
            aliases=["--pipeline-parallel-size"],
        ),
    ] = 1
    pp_max_micro_batch_size: A[
        Optional[int],
        "The maximum micro batch size in pipeline parallelism.",
    ] = None
    pp_async_batch_depth: A[int, "The async batch depth of pipeline parallelism."] = 0
    dp_size: A[
        int,
        Arg(
            help="The data parallelism size.",
            aliases=["--data-parallel-size"],
        ),
    ] = 1
    load_balance_method: A[
        str,
        Arg(
            help="The load balancing strategy for data parallelism.",
            choices=[
                "auto",
                "round_robin",
                "follow_bootstrap_room",
                "total_requests",
                "total_tokens",
            ],
        ),
    ] = "auto"
    attn_cp_size: A[
        int,
        Arg(
            help="The attention context parallelism size.",
            aliases=["--attention-context-parallel-size"],
        ),
    ] = 1
    moe_dp_size: A[
        int,
        Arg(
            help="The moe data parallelism size.",
            aliases=["--moe-data-parallel-size"],
        ),
    ] = 1
    dcp_size: A[
        int,
        Arg(
            help="The decode context parallelism size.",
            aliases=["--decode-context-parallel-size"],
        ),
    ] = 1
    enable_prefill_cp: A[
        bool,
        "Enable context parallelism for the prefill phase. Select the layout with --cp-strategy.",
    ] = False
    cp_strategy: A[
        Optional[str],
        Arg(
            help="Sharding strategy for prefill CP. 'zigzag' is the former in-seq-split mode; 'interleave' is the former round-robin-split mode.",
            choices=("zigzag", "interleave"),
        ),
    ] = None
    enable_dsa_prefill_context_parallel: A[bool, Arg(no_cli=True)] = False
    dsa_prefill_cp_mode: A[str, Arg(no_cli=True)] = "round-robin-split"
    enable_prefill_context_parallel: A[bool, Arg(no_cli=True)] = False
    prefill_cp_mode: A[str, Arg(no_cli=True)] = "in-seq-split"
    # DP attention
    enable_dp_attention: A[
        bool,
        "Enabling data parallelism for attention and tensor parallelism for FFN. The dp size should be equal to the tp size. Currently DeepSeek-V2 and Qwen 2/3 MoE models are supported.",
    ] = False
    enable_dp_attention_local_control_broadcast: A[
        bool,
        "With DP-attention, send control messages to every DP group leader and broadcast within attn_tp_group instead of the full tp_group. Eliminates a costly all-ranks gloo sync on every scheduler iteration.",
    ] = False
    enable_dp_lm_head: A[
        bool,
        "Enable vocabulary parallel across the attention TP group to avoid all-gather across DP groups, optimizing performance under DP attention.",
    ] = False
    enable_attn_tp_input_scattered: A[
        bool,
        "Allow input of attention to be scattered when only using tensor parallelism, to reduce the computational load of operations such as qkv latent.",
    ] = False
    disable_attn_tp_gather: A[
        bool,
        "Disable scheduler-side attn_tp_gather (the upstream SP path "
        "that pads num_tokens to attn_tp_size and pre-allocates a gathered "
        "buffer). Use for models that manage SP scatter/gather at the "
        "model level (e.g., perform their own all_gather/reduce_scatter "
        "inside attention) and do not consume the upstream gathered_buffer. "
        "Without this, the cuda graph runner pads num_tokens to attn_tp_size, "
        "which can cause kernel autotuners to select wrong-sized variants "
        "at small batches.",
    ] = False
    enable_p2p_check: A[
        bool,
        "Enable P2P check for GPU access, otherwise the p2p access is allowed by default.",
    ] = False

    # -------------------------------------------------------------------------
    # Streaming
    # -------------------------------------------------------------------------
    stream_interval: A[
        int,
        "The interval (or buffer size) for streaming in terms of the token length. A smaller value makes streaming smoother, while a larger value makes the throughput higher",
    ] = 1
    batch_notify_size: A[
        int,
        "Number of streaming notifications to batch before yielding to the event loop. Reduces asyncio wakeup overhead under high concurrency.",
    ] = 16
    stream_response_default_include_usage: A[
        bool,
        "Include usage in every streaming response (even when stream_options is not specified).",
    ] = False
    incremental_streaming_output: A[
        bool,
        "Whether to output as a sequence of disjoint segments.",
    ] = False
    enable_streaming_session: A[
        bool,
        "Enable streaming session mode and StreamingSession wrapper.",
    ] = False
    enable_session_radix_cache: A[
        bool,
        "Hold per-session KV as ordinary evictable radix entries, tagged by session id and bulk-evicted on close. Requires --radix-eviction-policy priority.",
    ] = False

    # -------------------------------------------------------------------------
    # Constrained decoding
    # -------------------------------------------------------------------------
    constrained_json_whitespace_pattern: A[
        Optional[str],
        "(outlines and llguidance backends only) Regex pattern for syntactic whitespaces allowed in JSON constrained output. For example, to allow the model generate consecutive whitespaces, set the pattern to [\n\t ]*",
    ] = None
    constrained_json_disable_any_whitespace: A[
        bool,
        "(xgrammar and llguidance backends only) Enforce compact representation in JSON constrained output.",
    ] = False

    # -------------------------------------------------------------------------
    # Logging, metrics, and tracing
    # -------------------------------------------------------------------------
    log_level: A[str, "The logging level of all loggers."] = "info"
    log_level_http: A[
        Optional[str],
        "The logging level of HTTP server. If not set, reuse --log-level by default.",
    ] = None
    log_requests: A[
        bool,
        "Log metadata, inputs, outputs of all requests. The verbosity is decided by --log-requests-level",
    ] = False
    log_requests_level: A[
        int,
        Arg(
            help="0: Log metadata (no sampling parameters). 1: Log metadata and sampling parameters. 2: Log metadata, sampling parameters and partial input/output. 3: Log every input/output.",
            choices=[0, 1, 2, 3],
        ),
    ] = 2
    log_requests_format: A[
        str,
        Arg(
            help="Format for request logging: 'text' (human-readable) or 'json' (structured)",
            choices=["text", "json"],
        ),
    ] = "text"
    log_requests_target: A[
        Optional[List[str]],
        "Target(s) for request logging: 'stdout' and/or directory path(s) for file output. Can specify multiple targets, e.g., '--log-requests-target stdout /my/path'. ",
    ] = None
    uvicorn_access_log_exclude_prefixes: A[
        List[str],
        Arg(
            help="Exclude uvicorn access logs whose request path starts with any of these prefixes. Defaults to empty (disabled). Example: --uvicorn-access-log-exclude-prefixes /metrics /health",
            nargs="*",
        ),
    ] = dataclasses.field(
        default_factory=lambda: list(DEFAULT_UVICORN_ACCESS_LOG_EXCLUDE_PREFIXES)
    )
    crash_dump_folder: A[
        Optional[str],
        "Folder path to dump requests from the last 5 min before a crash (if any). If not specified, crash dumping is disabled.",
    ] = None
    show_time_cost: A[bool, "Show time cost of custom marks."] = False
    enable_metrics: A[bool, "Enable log prometheus metrics."] = False
    grpc_http_sidecar_port: A[
        Optional[int],
        "Port for the HTTP sidecar server in gRPC mode (--grpc-mode). Serves Prometheus metrics and profiling endpoints. Defaults to --port + 1. Not used in HTTP mode.",
    ] = None
    enable_mfu_metrics: A[bool, "Enable estimated MFU-related prometheus metrics."] = (
        False
    )
    enable_metrics_for_all_schedulers: A[
        bool,
        "Enable --enable-metrics-for-all-schedulers when you want schedulers on all TP ranks (not just TP 0) to record request metrics separately. This is especially useful when dp_attention is enabled, as otherwise all metrics appear to come from TP 0.",
    ] = False
    load_snapshot_publish_interval: A[
        int,
        "Publish load snapshot to shared memory every N decode iterations. Prefill and idle always publish immediately.",
    ] = 15
    tokenizer_metrics_custom_labels_header: A[
        str,
        "Specify the HTTP header for passing custom labels for tokenizer metrics.",
    ] = "x-custom-labels"
    tokenizer_metrics_allowed_custom_labels: A[
        Optional[List[str]],
        "The custom labels allowed for tokenizer metrics. The labels are specified via a dict in '--tokenizer-metrics-custom-labels-header' field in HTTP requests, e.g., {'label1': 'value1', 'label2': 'value2'} is allowed if '--tokenizer-metrics-allowed-custom-labels label1 label2' is set.",
    ] = None
    extra_metric_labels: A[
        Optional[Dict[str, str]],
        Arg(
            help='The custom labels for metrics. e.g. \'{"label1": "value1", "label2": "value2"}\'',
            type_parser=json.loads,
        ),
    ] = None
    bucket_time_to_first_token: A[
        Optional[List[float]],
        "The buckets of time to first token, specified as a list of floats.",
    ] = None
    bucket_inter_token_latency: A[
        Optional[List[float]],
        "The buckets of inter-token latency, specified as a list of floats.",
    ] = None
    bucket_e2e_request_latency: A[
        Optional[List[float]],
        "The buckets of end-to-end request latency, specified as a list of floats.",
    ] = None
    prompt_tokens_buckets: A[
        Optional[List[str]],
        "The buckets rule of prompt tokens. "
        "Supports 3 rule types: 'default' uses predefined buckets; 'tse <middle> <base> <count>' "
        "generates two sides exponential distributed buckets (e.g., 'tse 1000 2 8' generates buckets "
        "[984.0, 992.0, 996.0, 998.0, 1000.0, 1002.0, 1004.0, 1008.0, 1016.0]).); 'custom <value1> "
        "<value2> ...' uses custom bucket values (e.g., 'custom 10 50 100 500').",
    ] = None
    generation_tokens_buckets: A[
        Optional[List[str]],
        "The buckets rule for generation tokens histogram. "
        "Supports 3 rule types: 'default' uses predefined buckets; 'tse <middle> <base> <count>' "
        "generates two sides exponential distributed buckets (e.g., 'tse 1000 2 8' generates buckets "
        "[984.0, 992.0, 996.0, 998.0, 1000.0, 1002.0, 1004.0, 1008.0, 1016.0]).); 'custom <value1> "
        "<value2> ...' uses custom bucket values (e.g., 'custom 10 50 100 500').",
    ] = None
    gc_warning_threshold_secs: A[
        float,
        "The threshold for long GC warning. If a GC takes longer than this, a warning will be logged. Set to 0 to disable.",
    ] = 0.0
    decode_log_interval: A[
        int,
        "The log and metrics reporting interval (in decode iterations) for decode batches.",
    ] = 40
    enable_request_time_stats_logging: A[
        bool, "Enable per request time stats logging"
    ] = False
    kv_events_config: A[
        Optional[str],
        "Config in json format for NVIDIA dynamo KV event publishing. Publishing will be enabled if this flag is used.",
    ] = None
    enable_forward_pass_metrics: A[
        bool,
        "Enable per-iteration forward pass metrics via ZMQ IPC. External consumers (e.g. Dynamo planner) subscribe to the IPC endpoint exposed in server_args.forward_pass_metrics_ipc_name.",
    ] = False
    forward_pass_metrics_worker_id: A[str, Arg(help=argparse.SUPPRESS)] = ""
    forward_pass_metrics_ipc_name: A[Optional[str], Arg(help=argparse.SUPPRESS)] = None
    enable_trace: A[bool, "Enable opentelemetry trace"] = False
    trace_modules: A[
        str,
        "Select the components to trace. Available options are 'request' and 'mooncake'. Format: <module1 name>,<module2 name>,...",
    ] = "request"
    otlp_traces_endpoint: A[
        str,
        "Config opentelemetry collector endpoint if --enable-trace is set. format: <ip>:<port>",
    ] = "localhost:4317"
    # RequestMetricsExporter configuration
    export_metrics_to_file: A[
        bool,
        "Export performance metrics for each request to local file (e.g. for forwarding to external systems).",
    ] = False
    export_metrics_to_file_dir: A[
        Optional[str],
        "Directory path for writing performance metrics files (required when --export-metrics-to-file is enabled).",
    ] = None
    # Class-level DI for the five *MetricsCollector classes. Maps collector role
    # (one of: "scheduler", "tokenizer", "storage", "radix_cache", "expert_dispatch")
    # to a subclass of the matching base collector. The five instantiation sites
    # read from this map and fall back to the base class. Class-object only (no
    # CLI surface) since this exists for embedded use cases that pass a Python
    # class directly. Default None preserves existing behavior.
    stat_loggers: Optional[Dict[str, type]] = None

    # -------------------------------------------------------------------------
    # API related
    # -------------------------------------------------------------------------
    api_key: A[
        Optional[str],
        "Set API key of the server. It is also used in the OpenAI API compatible server.",
    ] = None
    admin_api_key: A[
        Optional[str],
        "Set admin API key for sensitive management endpoints (e.g. /clear_hicache_storage_backend). When set, admin endpoints require this key and do NOT accept --api-key.",
    ] = None
    served_model_name: A[
        Optional[str],
        "Override the model name returned by the v1/models endpoint in OpenAI API server.",
    ] = None
    weight_version: A[
        str,
        "Version identifier for the model weights. Defaults to 'default' if not specified.",
    ] = "default"
    chat_template: A[
        Optional[str],
        "The buliltin chat template name or the path of the chat template file. This is only used for OpenAI-compatible API server.",
    ] = None
    hf_chat_template_name: A[
        Optional[str],
        "When the HuggingFace tokenizer has multiple chat templates (e.g., 'default', 'tool_use', 'rag'), specify which named template to use. If not set, the first available template is used.",
    ] = None
    completion_template: A[
        Optional[str],
        "The buliltin completion template name or the path of the completion template file. This is only used for OpenAI-compatible API server. only for code completion currently.",
    ] = None
    file_storage_path: A[str, "The path of the file storage in backend."] = (
        "sglang_storage"
    )
    enable_cache_report: A[
        bool,
        "Return number of cached tokens in usage.prompt_tokens_details for each openai request.",
    ] = False
    reasoning_parser: Optional[str] = None
    strip_thinking_cache: A[
        bool,
        "Skip caching reasoning-model output (thinking + answer) in the radix tree on finish; keep only the prompt prefix. Opt-in: changes cache contents.",
    ] = False
    enable_strict_thinking: A[
        bool,
        "Enable strict token filtering during the thinking phase. Blocks model-specific excluded tokens (e.g., tool call markers) during reasoning. Requires a grammar backend that supports token filtering.",
    ] = False
    tool_call_parser: Optional[str] = None
    tool_server: A[
        Optional[str],
        "Either 'demo' or a comma-separated list of tool server urls to use for the model. If not specified, no tool server will be used.",
    ] = None
    sampling_defaults: A[
        str,
        Arg(
            help="Where to get default sampling parameters. 'openai' uses SGLang/OpenAI defaults (temperature=1.0, top_p=1.0, etc.). 'model' uses the model's generation_config.json to get the recommended sampling parameters if available. Default is 'model'.",
            choices=["openai", "model"],
        ),
    ] = "model"
    asr_max_buffer_seconds: A[
        int,
        "Maximum seconds of PCM audio the streaming ASR WebSocket handler will accumulate before closing the session with a buffer_overflow error. Guards against OOM when a client streams audio faster than inference can consume it. Default 60s.",
    ] = 60
    asr_max_concurrent_sessions: A[
        int,
        "Maximum number of concurrent realtime ASR WebSocket sessions served by /v1/realtime. New connections beyond this cap are accepted, sent an error{code:too_many_sessions} frame, and closed. Default 32.",
    ] = 32
    preferred_sampling_params: A[
        Optional[str],
        Arg(
            help="json-formatted sampling settings that will be returned in /get_model_info",
            type_parser=json.loads,
        ),
    ] = None
    allow_auto_truncate: A[
        bool,
        "Allow automatically truncating requests that exceed the maximum input length instead of returning an error.",
    ] = False

    # -------------------------------------------------------------------------
    # Prefill delayer
    # -------------------------------------------------------------------------
    enable_prefill_delayer: A[
        bool, "Enable prefill delayer for DP attention to reduce idle time."
    ] = False
    prefill_delayer_max_delay_passes: A[
        int, "Maximum forward passes to delay prefill."
    ] = 30
    prefill_delayer_token_usage_low_watermark: A[
        Optional[float], "Token usage low watermark for prefill delayer."
    ] = None
    prefill_delayer_forward_passes_buckets: A[
        Optional[List[float]],
        "Custom buckets for prefill delayer forward passes histogram. 0 and max_delay_passes-1 will be auto-added.",
    ] = None
    prefill_delayer_wait_seconds_buckets: A[
        Optional[List[float]],
        "Custom buckets for prefill delayer wait seconds histogram. 0 will be auto-added.",
    ] = None
    prefill_delayer_queue_min_ratio: A[
        Optional[float],
        (
            "Opt-in to the adaptive queue-based delay trigger (independent of the "
            "slot-based one). Delays prefill until the waiting queue reaches "
            "min(running_req * ratio, max_prefill_bs) so small fragments batch "
            "into a larger prefill. Unset (default) keeps the original slot-only "
            "behavior. Typical: 0.1 ~ 0.5."
        ),
    ] = None
    prefill_delayer_max_delay_ms: A[
        Optional[float],
        (
            "Wall-clock cap (ms) on a single queue-trigger delay; once exceeded, "
            "prefill is force-released to bound worst-case TTFT. Only consulted "
            "when --prefill-delayer-queue-min-ratio is set. Typical: 1000 ~ "
            "5000; defaults to 5000 if unset."
        ),
    ] = None

    # -------------------------------------------------------------------------
    # Min free slots delay (prefill refill batching)
    # -------------------------------------------------------------------------
    min_free_slots_delay: A[
        Optional[int],
        (
            "Hold new prefills until at least N running-request slots have freed "
            "up, so they are admitted in one batch instead of one at a time. "
            "Useful when each admission is disproportionately expensive, e.g. "
            "speculative decoding with a separate draft prefill pass. Capped to "
            "the DFlash formula (disabled when max-running-requests < 8; "
            "min(4, max(2, (max-run + 5) // 6))). DFlash workloads auto-enable "
            "this with the formula when unset; other workloads stay disabled."
        ),
    ] = None

    # -------------------------------------------------------------------------
    # LoRA
    # -------------------------------------------------------------------------
    enable_lora: A[
        Optional[bool],
        "Enable LoRA support for the model. This argument is automatically set to True if `--lora-paths` is provided for backward compatibility.",
    ] = None
    enable_lora_overlap_loading: A[
        Optional[bool],
        "Enable asynchronous LoRA weight loading in order to overlap H2D transfers with GPU compute. This should be enabled if you find that your LoRA workloads are bottlenecked by adapter weight loading, for example when frequently loading large LoRA adapters.",
    ] = None
    max_lora_rank: A[
        Optional[int],
        "The maximum rank of LoRA adapters. If not specified, it will be automatically inferred from the adapters provided in --lora-paths.",
    ] = None
    lora_target_modules: A[
        Optional[Union[set[str], List[str]]],
        Arg(
            help="The union set of all target modules where LoRA should be applied. If not specified, it will be automatically inferred from the adapters provided in --lora-paths. If 'all' is specified, all supported modules will be targeted.",
            nargs="*",
            choices=SUPPORTED_LORA_TARGET_MODULES + [LORA_TARGET_ALL_MODULES],
        ),
    ] = None
    lora_paths: A[
        Optional[Union[dict[str, str], List[dict[str, str]], List[str], List[LoRARef]]],
        Arg(
            help='The list of LoRA adapters to load. Each adapter must be specified in one of the following formats: <PATH> | <NAME>=<PATH> | JSON with schema {"lora_name":str,"lora_path":str,"pinned":bool}',
            action=LoRAPathAction,
            action_kwargs={"type": str, "nargs": "*"},
        ),
    ] = None
    max_loaded_loras: A[
        Optional[int],
        "If specified, it limits the maximum number of LoRA adapters loaded in CPU memory at a time. The value must be greater than or equal to `--max-loras-per-batch`.",
    ] = None
    max_loras_per_batch: A[
        int,
        "Maximum number of adapters for a running batch, include base-only request.",
    ] = 8
    lora_eviction_policy: A[
        str,
        Arg(
            help="LoRA adapter eviction policy when memory pool is full. 'lru': Least Recently Used (default, better cache efficiency). 'fifo': First-In-First-Out.",
            choices=["lru", "fifo"],
        ),
    ] = "lru"
    lora_backend: A[
        str,
        Arg(
            help="Choose the kernel backend for multi-LoRA serving.",
            choices=LORA_BACKEND_CHOICES,
        ),
    ] = "csgmv"
    max_lora_chunk_size: A[
        Optional[int],
        Arg(
            help="Maximum chunk size for the ChunkedSGMV LoRA backend. Only used when --lora-backend is 'csgmv'. Choosing a larger value might improve performance.",
            choices=[16, 32, 64, 128],
        ),
    ] = 16
    experts_shared_outer_loras: A[
        Optional[bool],
        Arg(
            help="Force shared outer LoRA mode for MoE models. When set, w1/w3 lora_A and w2 lora_B are shared across experts (expert_dim=1). Use --no-experts-shared-outer-loras to force disable. By default this is auto-detected from adapter weights.",
            action=argparse.BooleanOptionalAction,
        ),
    ] = None
    lora_use_virtual_experts: A[
        bool,
        "Enable virtual expert computation for MoE models. When set, the model will use virtual expert computation.",
    ] = False
    lora_strict_loading: A[
        bool,
        Arg(
            help="Enable strict loading for LoRA adapters. When set, mismatched or missing keys in the adapter weights will raise an error.",
            action=argparse.BooleanOptionalAction,
        ),
    ] = False
    lora_drain_wait_threshold: A[
        float,
        "When any LoRA adapter request waits longer than this threshold (in seconds), the scheduler will selectively drain one running adapter to make room. This mitigates extreme tail latency under high or skewed workloads by preventing a small set of adapters from monopolizing batch slots. Set to 0 to disable draining (default).",
    ] = 0.0

    # -------------------------------------------------------------------------
    # Kernel backend
    # -------------------------------------------------------------------------
    attention_backend: A[
        Optional[str],
        Arg(
            help="Choose the kernels for attention layers.",
            choices=ATTENTION_BACKEND_CHOICES,
        ),
    ] = None
    decode_attention_backend: A[
        Optional[str],
        Arg(
            help="Choose the kernels for decode attention layers (have priority over --attention-backend).",
            choices=ATTENTION_BACKEND_CHOICES,
        ),
    ] = None
    prefill_attention_backend: A[
        Optional[str],
        Arg(
            help="Choose the kernels for prefill attention layers (have priority over --attention-backend).",
            choices=ATTENTION_BACKEND_CHOICES,
        ),
    ] = None
    sampling_backend: A[
        Optional[str],
        Arg(
            help="Choose the kernels for sampling layers.",
            choices=SAMPLING_BACKEND_CHOICES,
        ),
    ] = None
    grammar_backend: A[
        Optional[str],
        Arg(
            help="Choose the backend for grammar-guided decoding.",
            choices=GRAMMAR_BACKEND_CHOICES,
        ),
    ] = None
    radix_cache_backend: A[
        Optional[str],
        "Name of a radix-cache backend previously registered via register_radix_cache_backend. Omit this flag to use the built-in default cache selection chain.",
    ] = None
    mm_attention_backend: A[
        Optional[str],
        Arg(
            help="Set multimodal attention backend.",
            choices=[
                "sdpa",
                "fa3",
                "fa4",
                "triton_attn",
                "ascend_attn",
                "aiter_attn",
                "flashinfer_cudnn",
                "amx_attn",
                "xpu_attn",
            ],
        ),
    ] = None
    fp8_gemm_runner_backend: A[
        str,
        Arg(
            help="Choose the runner backend for Blockwise FP8 GEMM operations. Options: 'auto' (default, auto-selects based on hardware), 'deep_gemm' (JIT-compiled; enabled by default on NVIDIA Hopper (SM90) and Blackwell (SM100) when DeepGEMM is installed), 'flashinfer_trtllm' (optimal for Blackwell and low-latency), 'flashinfer_cutlass' (FlashInfer CUTLASS groupwise FP8 GEMM), 'flashinfer_deepgemm' (Hopper SM90 only; uses swapAB optimization for small M dimensions in decoding), 'cutlass' (optimal for Hopper/Blackwell GPUs and high-throughput), 'triton' (fallback, widely compatible), 'aiter' (ROCm only). ",
            cli_name="--fp8-gemm-backend",
            choices=FP8_GEMM_RUNNER_BACKEND_CHOICES,
        ),
    ] = "auto"
    fp4_gemm_runner_backend: A[
        str,
        Arg(
            help="Choose the runner backend for NVFP4 GEMM operations. Options: 'auto' (default; selects flashinfer_cutedsl on SM100, marlin on SM80-SM90, flashinfer_cutlass otherwise (including SM120)), 'cutlass' (SGLang CUTLASS kernel), 'flashinfer_cutlass' (FlashInfer CUTLASS backend), 'flashinfer_cudnn' (FlashInfer cuDNN backend, optimal on CUDA 13+ with cuDNN 9.15+), 'flashinfer_cutedsl' (FlashInfer CuTe DSL backend), 'flashinfer_trtllm' (FlashInfer TensorRT-LLM backend, requires different weight preparation with shuffling), 'marlin' (weight-only W4A16 fallback for SM80+). ",
            cli_name="--fp4-gemm-backend",
            choices=FP4_GEMM_RUNNER_BACKEND_CHOICES,
        ),
    ] = "auto"
    dsa_prefill_backend: A[
        Optional[str],
        Arg(
            help="DSA (DeepSeek Sparse Attention) prefill backend. If not specified, auto-detects based on hardware and kv_cache_dtype.",
            choices=DSA_CHOICES,
        ),
    ] = None
    dsa_decode_backend: A[
        Optional[str],
        Arg(
            help="DSA (DeepSeek Sparse Attention) decode backend. If not specified, auto-detects based on hardware and kv_cache_dtype.",
            choices=DSA_CHOICES,
        ),
    ] = None
    dsa_topk_backend: A[
        str,
        Arg(
            help="DSA indexer top-k backend. Options: 'sgl-kernel', 'torch', 'flashinfer'. The 'torch' backend currently requires SGLANG_DSA_FUSE_TOPK=false.",
            choices=DSA_TOPK_BACKEND_CHOICES,
        ),
    ] = "sgl-kernel"
    disable_flashinfer_autotune: A[bool, "Disable FlashInfer autotuning."] = False
    mamba_backend: A[
        str,
        Arg(
            help="Choose the kernel backend for Mamba SSM operations. Default is 'triton'. Options: 'triton' (default), 'flashinfer' (requires FlashInfer with Mamba support).",
            choices=MAMBA_BACKEND_CHOICES,
        ),
    ] = "triton"

    # -------------------------------------------------------------------------
    # Speculative decoding
    # -------------------------------------------------------------------------
    speculative_algorithm: A[
        Optional[str],
        "Speculative algorithm. Builtins: EAGLE, EAGLE3, NEXTN, STANDALONE, NGRAM, DFLASH. Or any name registered via `SpeculativeAlgorithm.register`.",
    ] = None
    speculative_draft_model_path: A[
        Optional[str],
        Arg(
            help="The path of the draft model weights. This can be a local folder or a Hugging Face repo ID.",
            aliases=["--speculative-draft-model"],
        ),
    ] = None
    speculative_draft_model_revision: A[
        Optional[str],
        "The specific draft model version to use. It can be a branch name, a tag name, or a commit id. If unspecified, will use the default version.",
    ] = None
    speculative_draft_load_format: A[
        Optional[str],
        Arg(
            help="The format of the draft model weights to load. If not specified, will use the same format as --load-format. Use 'dummy' to initialize draft model weights with random values for profiling.",
            choices=LOAD_FORMAT_CHOICES,
        ),
    ] = None
    speculative_num_steps: A[
        Optional[int],
        "The number of steps sampled from draft model in Speculative Decoding.",
    ] = None
    speculative_eagle_topk: A[
        Optional[int],
        "The number of tokens sampled from the draft model in eagle2 each step.",
    ] = None
    speculative_num_draft_tokens: A[
        Optional[int],
        "The number of tokens sampled from the draft model in Speculative Decoding.",
    ] = None
    speculative_dflash_block_size: A[
        Optional[int],
        "DFLASH only. Block size (verify window length). Alias of --speculative-num-draft-tokens for DFLASH.",
    ] = None
    speculative_accept_threshold_single: A[
        float,
        "Accept a draft token if its probability in the target model is greater than this threshold.",
    ] = 1.0
    speculative_accept_threshold_acc: A[
        float,
        "The accept probability of a draft token is raised from its target probability p to min(1, p / threshold_acc).",
    ] = 1.0
    speculative_use_rejection_sampling: A[
        bool,
        "Use rejection sampling for speculative decoding (requires topk=1).",
    ] = False
    speculative_token_map: A[
        Optional[str],
        "The path of the draft model's small vocab table.",
    ] = None
    speculative_attention_mode: A[
        str,
        Arg(
            help="Attention backend for speculative decoding operations (both target verify and draft extend). Can be one of 'prefill' (default) or 'decode'.",
            choices=["prefill", "decode"],
        ),
    ] = "prefill"
    speculative_draft_attention_backend: A[
        Optional[str],
        "Attention backend for speculative decoding drafting.",
    ] = None
    speculative_draft_window_size: A[
        Optional[int],
        "Sliding window size for the draft model. Honored by Llama EAGLE-3 (`LlamaForCausalLMEagle3`) and DFLASH only; other EAGLE-3 backends (e.g. MLA-based drafters) silently ignore it. For Llama EAGLE-3, the drafter only attends to the most recent N keys (verifier hidden states + its own outputs); the verifier is unaffected. For DFLASH, the draft worker keeps a recent target-token window in its local KV cache (paged backends may retain up to one extra page on the left for alignment). Default is full attention/context.",
    ] = None
    speculative_moe_runner_backend: A[
        Optional[str],
        Arg(
            help="Choose the runner backend for MoE in speculative decoding.",
            choices=MOE_RUNNER_BACKEND_CHOICES,
        ),
    ] = None
    speculative_moe_a2a_backend: A[
        Optional[str],
        Arg(
            help="Choose the backend for MoE A2A in speculative decoding",
            choices=MOE_A2A_BACKEND_CHOICES,
        ),
    ] = None
    speculative_draft_model_quantization: A[
        Optional[str],
        Arg(
            help="The quantization method for speculative model.",
            choices=SPECULATIVE_DRAFT_MODEL_QUANTIZATION_CHOICES,
        ),
    ] = None
    speculative_skip_dp_mlp_sync: A[
        bool,
        "Skip the extra MLP sync that the scheduler performs before merging a new batch when speculative decoding + DP attention are both enabled.",
    ] = False
    enable_multi_layer_eagle: A[
        bool,
        "Enable multi-layer Eagle speculative decoding.",
    ] = False
    speculative_adaptive: A[
        bool,
        "Enable adaptive speculative decoding that dynamically adjusts num_steps based on acceptance rate.",
    ] = False
    speculative_adaptive_config: A[
        Optional[str],
        "Path to a JSON config file for adaptive speculative decoding tuning knobs.",
    ] = None

    # Decoupled speculative decoding: draft and verify run as
    # separate engines, currently connected by a ZMQ IPC mesh.
    decoupled_spec_bind_endpoint: A[
        Optional[str],
        "ZMQ endpoint this engine binds for its inbound channel in decoupled "
        "speculative decoding (verifier: result PULL; drafter: control PULL).",
    ] = None
    decoupled_spec_connect_endpoints: A[
        Optional[List[str]],
        Arg(
            help="Peer inbound (bind) endpoints to connect to, ordered by peer "
            "rank, for decoupled speculative decoding.",
            type_parser=json_list_type,
        ),
    ] = None
    decoupled_spec_rank: A[
        Optional[int],
        "This engine's rank within its own role space (verifier-rank or "
        "drafter-rank) for decoupled speculative decoding.",
    ] = None
    decoupled_spec_role: A[
        Literal["null", "verifier", "drafter"],
        "Role in decoupled speculative decoding: 'null' disables it, 'verifier' "
        "runs the target/verify half, 'drafter' runs the draft half.",
    ] = "null"
    spec_trace_dir: A[
        Optional[str],
        "Directory to write decoupled speculative decoding trace files.",
    ] = None

    # Speculative decoding (ngram)
    # -------------------------------------------------------------------------
    speculative_ngram_min_bfs_breadth: A[
        int,
        "The minimum breadth for BFS (Breadth-First Search) in ngram speculative decoding.",
    ] = 1
    speculative_ngram_max_bfs_breadth: A[
        int,
        "The maximum breadth for BFS (Breadth-First Search) in ngram speculative decoding.",
    ] = 10
    speculative_ngram_match_type: A[
        Literal["BFS", "PROB"],
        "The match type for cache tree.",
    ] = "BFS"
    speculative_ngram_max_trie_depth: A[
        int,
        "The max trie depth for ngram speculative decoding.",
    ] = 18
    speculative_ngram_capacity: A[
        int,
        "The cache capacity for ngram speculative decoding.",
    ] = (
        10 * 1000 * 1000
    )
    speculative_ngram_external_corpus_path: A[
        Optional[str],
        "Path to an external JSONL corpus to pre-load into SAM at startup. Additional corpora can be added at runtime via POST /add_external_corpus.",
    ] = None
    speculative_ngram_external_sam_budget: A[
        int,
        "Number of draft nodes reserved for the external SAM subtree in ngram speculative decoding.",
    ] = 0
    speculative_ngram_external_corpus_max_tokens: A[
        int,
        "Fail startup if the tokenized external ngram corpus exceeds this many tokens. Tune this based on your CPU memory budget.",
    ] = 10000000

    # -------------------------------------------------------------------------
    # Expert parallelism
    # -------------------------------------------------------------------------
    ep_size: A[
        int,
        Arg(
            help="The expert parallelism size.",
            aliases=["--expert-parallel-size", "--ep"],
        ),
    ] = 1
    moe_a2a_backend: A[
        Literal[
            "none",
            "deepep",
            "mooncake",
            "nixl",
            "mori",
            "ascend_fuseep",
            "flashinfer",
            "megamoe",
        ],
        Arg(
            help="Choose the backend for MoE A2A.",
            choices=MOE_A2A_BACKEND_CHOICES,
        ),
    ] = "none"
    moe_runner_backend: A[
        str,
        Arg(
            help="Choose the runner backend for MoE.",
            choices=MOE_RUNNER_BACKEND_CHOICES,
        ),
    ] = "auto"
    flashinfer_mxfp4_moe_precision: A[
        Literal["default", "bf16"],
        "Choose the computation precision of flashinfer mxfp4 moe",
    ] = "default"
    deepep_mode: A[
        Literal["auto", "normal", "low_latency"],
        "Select the mode when enable DeepEP or MoriEP MoE, could be `normal`, `low_latency` or `auto`. Default is `auto`, which means `low_latency` for decode batch and `normal` for prefill batch.",
    ] = "auto"
    deepep_dispatcher_output_dtype: A[
        Literal["auto", "bf16", "fp8", "int8", "nvfp4"],
        "Select DeepEP dispatcher output dtype",
    ] = "auto"
    ep_num_redundant_experts: A[
        int,
        "Allocate this number of redundant experts in expert parallel.",
    ] = 0
    ep_dispatch_algorithm: A[
        Optional[Literal["static", "dynamic", "fake", "lp"]],
        "The algorithm to choose ranks for redundant experts in expert parallel.",
    ] = None
    init_expert_location: A[str, "Initial location of EP experts."] = "trivial"
    enable_eplb: A[bool, "Enable EPLB algorithm"] = False
    eplb_algorithm: A[str, "Chosen EPLB algorithm"] = "auto"
    eplb_rebalance_num_iterations: A[
        int,
        "Number of iterations to automatically trigger a EPLB re-balance.",
    ] = 1000
    eplb_rebalance_layers_per_chunk: A[
        Optional[int],
        "Number of layers to rebalance per forward pass.",
    ] = None
    eplb_min_rebalancing_utilization_threshold: A[
        float,
        "Minimum threshold for GPU average utilization to trigger EPLB rebalancing. Must be in the range [0.0, 1.0].",
    ] = 1.0
    expert_distribution_recorder_mode: A[
        Optional[Literal["stat", "stat_approx", "per_pass", "per_token"]],
        "Mode of expert distribution recorder.",
    ] = None
    expert_distribution_recorder_buffer_size: A[
        Optional[int],
        "Circular buffer size of expert distribution recorder. Set to -1 to denote infinite buffer.",
    ] = None
    enable_expert_distribution_metrics: A[
        bool,
        "Enable logging metrics for expert balancedness",
    ] = False
    deepep_config: A[
        Optional[str],
        "Tuned DeepEP config suitable for your own cluster. It can be either a string with JSON content or a file path.",
    ] = None
    moe_dense_tp_size: A[
        Optional[int],
        "TP size for MoE dense MLP layers. This flag is useful when, with large TP size, there are errors caused by weights in MLP layers having dimension smaller than the min dimension GEMM supports.",
    ] = None
    elastic_ep_backend: A[
        Literal[None, "mooncake", "nixl"],
        Arg(
            help="Specify the collective communication backend for elastic EP. Supports 'mooncake' and 'nixl'.",
            choices=["none", "mooncake", "nixl"],
        ),
    ] = None
    enable_elastic_expert_backup: A[bool, "Enable elastic expert backup feature."] = (
        False
    )
    mooncake_ib_device: A[
        Optional[str],
        "The InfiniBand devices for Mooncake Backend transfer, accepts multiple comma-separated devices (e.g., --mooncake-ib-device mlx5_0,mlx5_1). Default is None, which triggers automatic device detection when Mooncake Backend is enabled.",
    ] = None
    enable_deepep_waterfill: A[
        bool,
        "Enable DeepEP Waterfill: dispatch the shared expert as the 9th routed expert to the least-loaded EP rank. Automatically sets --moe-a2a-backend deepep, implicitly enables shared-expert fusion, and supports --deepep-mode auto, normal, or low_latency. Use auto or low_latency for production decode so CUDA graph remains enabled. Supported on DeepSeek-V3/R1 with EP >= 2.",
    ] = False
    elastic_ep_rejoin: A[
        bool,
        "Indicates that this process is a relaunched elastic EP rank that should rejoin an existing process group.",
    ] = False
    disable_flashinfer_cutlass_moe_fp4_allgather: A[
        bool,
        "Disables quantize before all-gather for flashinfer cutlass moe.",
    ] = False
    disable_shared_experts_fusion: A[
        bool,
        "Disable the built-in shared experts fusion optimization for DeepSeek V3/R1. Note: DeepEP Waterfill (--enable-deepep-waterfill) still routes shared expert through DeepEP as an extra MoE slot, so shared expert is not separated from the MoE path when Waterfill is enabled.",
    ] = False
    enforce_shared_experts_fusion: A[
        bool,
        "Enforce shared experts fusion even when it would normally be disabled (e.g. under DeepEP). Mutually exclusive with --disable-shared-experts-fusion.",
    ] = False

    # -------------------------------------------------------------------------
    # Mamba cache and linear attn
    # -------------------------------------------------------------------------
    max_mamba_cache_size: A[Optional[int], "The maximum size of the mamba cache."] = (
        None
    )
    mamba_ssm_dtype: A[
        Optional[str],
        Arg(
            help="The data type of the SSM states in mamba cache. If not set, will be read from model config (mamba_ssm_dtype).",
            choices=["float32", "bfloat16", "float16"],
        ),
    ] = None
    enable_mamba_cache_stochastic_rounding: A[
        bool,
        "Enable stochastic rounding when writing FP16 Mamba SSM cache states. Requires --mamba-ssm-dtype float16 and CUDA. With --mamba-backend triton, requires SM100.",
    ] = False
    mamba_cache_philox_rounds: A[
        int,
        "Number of Philox rounds to use for stochastic rounding of FP16 Mamba SSM cache writes. Triton uses the Triton default when set to 0; FlashInfer uses 10 rounds when set to 0.",
    ] = 0
    mamba_full_memory_ratio: A[
        float,
        "The ratio of mamba state memory to full kv cache memory.",
    ] = 0.9
    mamba_radix_cache_strategy: A[
        str,
        Arg(
            help="The strategy to use for mamba radix cache.",
            choices=MAMBA_RADIX_CACHE_STRATEGY_CHOICES,
        ),
    ] = "auto"
    mamba_track_interval: A[
        int,
        "The interval to track the mamba state during decode.",
    ] = 256
    enable_int8_mamba_checkpoint: A[
        bool,
        "Store radix-cached linear-attn (mamba) states in int8 (separate checkpoint pool) for ~2x cached-prefix capacity at fixed memory.",
    ] = False
    int8_mamba_ckpt_size: A[
        Optional[int],
        "Number of int8 mamba checkpoint slots (default: 2x the active mamba pool size).",
    ] = None
    linear_attn_backend: A[
        str,
        Arg(
            help="The default kernel backend for linear attention (GDN/KDA). Can be overridden per-mode by --linear-attn-decode-backend and --linear-attn-prefill-backend.",
            choices=LINEAR_ATTN_KERNEL_BACKEND_CHOICES,
        ),
    ] = "triton"
    linear_attn_decode_backend: A[
        Optional[str],
        Arg(
            help="Override the kernel backend for linear attention decode. If not set, uses --linear-attn-backend.",
            choices=LINEAR_ATTN_KERNEL_BACKEND_CHOICES,
        ),
    ] = None
    linear_attn_prefill_backend: A[
        Optional[str],
        Arg(
            help="Override the kernel backend for linear attention prefill/extend. If not set, uses --linear-attn-backend.",
            choices=LINEAR_ATTN_KERNEL_BACKEND_CHOICES,
        ),
    ] = None
    # ReplaySSM buffered output-only linear-attn decode (GDN + KDA): per-slot
    # ring + periodic flush to cut per-step HBM state traffic.
    enable_linear_replayssm: A[
        bool,
        "Enable the ReplaySSM buffered output-only linear-attn decode kernel. "
        "Primarily a GDN (scalar-gate) decode-bandwidth optimization (~1.2-1.5x "
        "at batch >= 64). The unified kernel also supports KDA (per-K gate) and "
        "is numerically correct, but KDA decode is SLOWER than the packed "
        "baseline (the per-K g_cache is K x larger and the reconstruction "
        "refolds the per-K decay every step), so it is not recommended for KDA "
        "models. Requires the Triton linear-attn decode backend and "
        "--mamba-scheduler-strategy no_buffer (the default).",
    ] = False
    linear_replayssm_cache_len: A[
        int,
        "Ring-buffer length L for ReplaySSM linear-attn decode. The full recurrent state is flushed to HBM every L decode steps.",
    ] = 16

    # -------------------------------------------------------------------------
    # Hierarchical cache
    # -------------------------------------------------------------------------
    enable_hierarchical_cache: A[bool, "Enable hierarchical cache"] = False
    hicache_ratio: A[
        float,
        "The ratio of the size of host KV cache memory pool to the size of device pool.",
    ] = 2.0
    hicache_size: A[
        int,
        "The size of host KV cache memory pool in gigabytes, which will override the hicache_ratio if set.",
    ] = 0
    hicache_write_policy: A[
        str,
        Arg(
            help="The write policy of hierarchical cache.",
            choices=["write_back", "write_through", "write_through_selective"],
        ),
    ] = "write_through"
    hicache_io_backend: A[
        str,
        Arg(
            help="The IO backend for KV cache transfer between CPU and GPU",
            choices=["direct", "kernel", "kernel_ascend"],
        ),
    ] = "kernel"
    hicache_mem_layout: A[
        str,
        Arg(
            help="The layout of host memory pool for hierarchical cache.",
            choices=[
                "layer_first",
                "page_first",
                "page_first_direct",
                "page_first_kv_split",
                "page_head",
            ],
        ),
    ] = "page_first"
    hicache_storage_backend: A[
        Optional[str],
        Arg(
            help="The storage backend for hierarchical KV cache. Built-in backends: file, mooncake, hf3fs, nixl, aibrix. For dynamic backend, use --hicache-storage-backend-extra-config to specify: backend_name (custom name), module_path (Python module path), class_name (backend class name).",
            choices=[
                "file",
                "mooncake",
                "hf3fs",
                "nixl",
                "aibrix",
                "dynamic",
                "eic",
                "simm",
            ],
        ),
    ] = None
    hicache_storage_prefetch_policy: A[
        str,
        Arg(
            help="Control when prefetching from the storage backend should stop.",
            choices=["best_effort", "wait_complete", "timeout"],
        ),
    ] = "timeout"
    hicache_storage_backend_extra_config: A[
        Optional[str],
        "A dictionary in JSON string format, or a string starting with a leading '@' and a config file in JSON/YAML/TOML format, containing extra configuration for the storage backend.",
    ] = None

    # -------------------------------------------------------------------------
    # Hierarchical sparse attention
    # -------------------------------------------------------------------------
    enable_hisparse: A[bool, "Enable hierarchical sparse attention"] = False
    hisparse_config: A[
        Optional[str],
        Arg(
            help='A dictionary in JSON string format for hierarchical sparse attention configuration. Example: \'{"top_k": 2048, "device_buffer_size": 4096, "host_to_device_ratio": 2}\'',
            aliases=["--hierarchical-sparse-attention-extra-config"],
        ),
    ] = None

    # -------------------------------------------------------------------------
    # LMCache
    # -------------------------------------------------------------------------
    enable_lmcache: A[
        bool,
        "Using LMCache as an alternative hierarchical cache solution",
    ] = False
    lmcache_config_file: A[
        Optional[str],
        "Path to the LMCache YAML configuration file",
    ] = None

    # -------------------------------------------------------------------------
    # Ktransformers/AMX expert parallelism
    # -------------------------------------------------------------------------
    kt_weight_path: A[
        Optional[str],
        "[ktransformers parameter] The path of the quantized expert weights for amx kernel. A local folder.",
    ] = None
    kt_method: A[
        str,
        "[ktransformers parameter] Quantization formats for CPU execution.",
    ] = "AMXINT4"
    kt_cpuinfer: A[
        Optional[int],
        "[ktransformers parameter] The number of CPUInfer threads.",
    ] = None
    kt_threadpool_count: A[
        int,
        "[ktransformers parameter] One-to-one with the number of NUMA nodes (one thread pool per NUMA).",
    ] = 2
    kt_num_gpu_experts: A[
        Optional[int],
        "[ktransformers parameter] The number of GPU experts.",
    ] = None
    kt_max_deferred_experts_per_token: A[
        Optional[int],
        "[ktransformers parameter] Maximum number of experts deferred to CPU per token. All MoE layers except the final one use this value; the final layer always uses 0.",
    ] = None

    # -------------------------------------------------------------------------
    # Diffusion LLM
    # -------------------------------------------------------------------------
    dllm_algorithm: A[
        Optional[str],
        "The diffusion LLM algorithm, such as LowConfidence.",
    ] = None
    dllm_algorithm_config: A[
        Optional[str],
        "The diffusion LLM algorithm configurations. Must be a YAML file.",
    ] = None

    # -------------------------------------------------------------------------
    # Offloading
    # -------------------------------------------------------------------------
    cpu_offload_gb: A[int, "How many GBs of RAM to reserve for CPU offloading."] = 0
    offload_group_size: A[int, "Number of layers per group in offloading."] = -1
    offload_num_in_group: A[
        int,
        "Number of layers to be offloaded within a group.",
    ] = 1
    offload_prefetch_step: A[int, "Steps to prefetch in offloading."] = 1
    offload_mode: A[str, "Mode of offloading."] = "cpu"

    # -------------------------------------------------------------------------
    # Cuda graphs
    # -------------------------------------------------------------------------
    cuda_graph_config: A[
        Optional[CudaGraphConfig],
        Arg(
            help='Per-phase CUDA graph settings as JSON, e.g. \'{"decode":{"backend":"full","max_bs":256},"prefill":{"backend":"tc_piecewise","tc_compiler":"eager"}}\'. Allowed backends per phase: full, breakable, tc_piecewise, disabled (full is decode-only). JSON wins over the per-phase --cuda-graph-* convenience flags and over legacy flags.',
            type_parser=parse_cuda_graph_config_arg,
        ),
    ] = None
    cuda_graph_backend_decode: A[
        Optional[Literal["full", "breakable", "tc_piecewise", "disabled"]],
        Arg(
            help="Backend for the decode phase. Folds into cuda_graph_config[decode].backend.",
            choices=Backend.ALL,
        ),
    ] = None
    cuda_graph_backend_prefill: A[
        Optional[Literal["breakable", "tc_piecewise", "disabled"]],
        Arg(
            help="Backend for the prefill phase. Folds into cuda_graph_config[prefill].backend.",
            choices=Backend.ALL,
        ),
    ] = None
    cuda_graph_max_bs_decode: A[
        Optional[int],
        "Maximum batch size captured for the decode cuda graph.",
    ] = None
    cuda_graph_max_bs_prefill: A[
        Optional[int],
        "Maximum batch size captured for the prefill cuda graph.",
    ] = None
    cuda_graph_bs_decode: A[
        Optional[List[int]],
        "Explicit list of batch sizes to capture for the decode cuda graph.",
    ] = None
    cuda_graph_bs_prefill: A[
        Optional[List[int]],
        "Explicit list of batch sizes to capture for the prefill cuda graph.",
    ] = None
    cuda_graph_tc_compiler: A[
        Optional[Literal["eager", "inductor"]],
        "Compiler used by the tc_piecewise backend (currently only the prefill phase consumes it).",
    ] = None
    disable_prefill_cuda_graph: A[
        bool,
        "Disable the prefill-phase CUDA graph. Convenience for --cuda-graph-backend-prefill=disabled.",
    ] = False
    disable_decode_cuda_graph: A[
        bool,
        "Disable the decode-phase CUDA graph. Convenience for --cuda-graph-backend-decode=disabled.",
    ] = False
    disable_cuda_graph: A[bool, Arg(no_cli=True)] = False
    disable_cuda_graph_padding: A[
        bool,
        "Disable cuda graph when padding is needed. Still uses cuda graph when padding is not needed.",
    ] = False
    enable_profile_cuda_graph: A[bool, "Enable profiling of cuda graph capture."] = (
        False
    )
    enable_cudagraph_gc: A[
        bool,
        "Enable garbage collection during CUDA graph capture. If disabled (default), GC is frozen during capture to speed up the process.",
    ] = False
    debug_cuda_graph: A[
        bool,
        "Enable debug/eager mode for CUDA graph using breakable CUDA graph. When enabled, graph breaks are inserted so every operation runs eagerly while still going through the CUDA graph capture / replay path. Useful for debugging CUDA graph capture / replay issues.",
    ] = False

    # -------------------------------------------------------------------------
    # Communication and kernels
    # -------------------------------------------------------------------------
    enable_layerwise_nvtx_marker: A[
        bool,
        "Enable layerwise NVTX profiling annotations for the model.",
    ] = False
    enable_nccl_nvls: A[
        bool,
        "Enable NCCL NVLS for prefill heavy requests when available.",
    ] = False
    enable_symm_mem: A[
        bool,
        "Enable NCCL symmetric memory for fast collectives.",
    ] = False
    triton_attention_reduce_in_fp32: A[
        bool,
        "Cast the intermediate attention results to fp32 to avoid possible crashes related to fp16."
        "This only affects Triton attention kernels.",
    ] = False
    triton_attention_num_kv_splits: A[
        int,
        "The number of KV splits in flash decoding Triton kernel. Larger value is better in longer context scenarios. The default value is 8.",
    ] = 8
    triton_attention_split_tile_size: A[
        Optional[int],
        "The size of split KV tile in flash decoding Triton kernel. Used for deterministic inference.",
    ] = None
    flashinfer_mla_disable_ragged: A[
        bool,
        "Not using ragged prefill wrapper when running flashinfer mla",
    ] = False
    enable_fused_qk_norm_rope: A[
        bool,
        "Enable fused qk normalization and rope rotary embedding.",
    ] = False
    enable_precise_embedding_interpolation: A[
        bool,
        "Enable corner alignment for resize of embeddings grid to ensure more accurate(but slower) evaluation of interpolated embedding values.",
    ] = False
    enable_fused_moe_sum_all_reduce: A[
        bool,
        "Enable fused moe triton and sum all reduce.",
    ] = False
    enable_deepseek_v4_fp4_indexer: A[
        bool,
        "Enable the experimental FP4 C4 indexer path for DeepSeek V4. Default keeps the existing indexer implementation.",
    ] = False
    disable_custom_all_reduce: A[
        bool,
        "Disable the custom all-reduce kernel and fall back to NCCL.",
    ] = False
    enable_mscclpp: A[
        bool,
        "Enable using mscclpp for small messages for all-reduce kernel and fall back to NCCL.",
    ] = False
    enable_torch_symm_mem: A[
        bool,
        "Enable using torch symm mem for all-reduce kernel and fall back to NCCL. Only supports CUDA device SM90 and above. SM90 supports world size 4, 6, 8. SM100 supports world size 6, 8.",
    ] = False
    pre_warm_nccl: A[
        bool,
        "Pre-warm NCCL/RCCL communicators during startup to reduce P99 TTFT cold-start latency. Default: enabled for AMD/HIP (RCCL), disabled for NVIDIA/CUDA (NCCL).",
    ] = False
    enable_quant_communications: A[
        Optional[bool],
        "Enable INT8 quantization of TP communications (limited support).",
    ] = False
    enable_flashinfer_allreduce_fusion: A[bool, Arg(no_cli=True)] = False
    enforce_disable_flashinfer_allreduce_fusion: A[
        bool,
        "Enforce disable FlashInfer allreduce fusion.",
    ] = False
    flashinfer_allreduce_fusion_backend: A[
        Optional[Literal["auto", "trtllm", "mnnvl"]],
        Arg(
            help=(
                "Enable FlashInfer allreduce fusion and choose backend. "
                "Requires SM90 or SM10X NVIDIA GPUs. "
                "Defaults to auto. "
                "'auto': choose mnnvl on Blackwell (SM100/SM103) systems "
                "(single- and multi-node) and trtllm on SM90 single-node systems. "
                "'trtllm': available on single-node systems only. "
                "'mnnvl': available on SM90 single-node systems and SM100/SM103 "
                "single-node or multi-node systems via MNNVL fabric. "
                "Fuses allreduce with Residual + RMSNorm for supported MoE models."
            ),
        ),
    ] = None
    enable_aiter_allreduce_fusion: A[bool, "Enable Aiter AllReduce Fusion."] = False

    # -------------------------------------------------------------------------
    # Two batch overlap
    # -------------------------------------------------------------------------
    enable_two_batch_overlap: A[bool, "Enabling two micro batches to overlap."] = False
    enable_single_batch_overlap: A[
        bool,
        "Let computation and communication overlap within one micro batch.",
    ] = False
    tbo_token_distribution_threshold: A[
        float,
        "The threshold of token distribution between two batches in micro-batch-overlap, determines whether to two-batch-overlap or two-chunk-overlap. Set to 0 denote disable two-chunk-overlap.",
    ] = 0.48

    # -------------------------------------------------------------------------
    # Torch compile and torchao
    # -------------------------------------------------------------------------
    enable_torch_compile: A[
        bool,
        "Optimize the model with torch.compile. Experimental feature.",
    ] = False
    enable_torch_compile_debug_mode: A[bool, "Enable debug mode for torch compile"] = (
        False
    )
    torch_compile_max_bs: A[
        int,
        "Set the maximum batch size when using torch compile.",
    ] = 32
    torchao_config: A[
        str,
        "Optimize the model with torchao. Experimental feature. Current choices are: int8dq, int8wo, int4wo-<group_size>, fp8wo, fp8dq-per_tensor, fp8dq-per_row",
    ] = ""

    # -------------------------------------------------------------------------
    # Misc runtime features
    # -------------------------------------------------------------------------
    enable_memory_saver: A[
        bool,
        "Allow saving memory using release_memory_occupation and resume_memory_occupation",
    ] = False
    enable_weights_cpu_backup: A[
        bool,
        "Save model weights (both main model and draft model, if any) to CPU memory during release_weights_occupation and resume_weights_occupation",
    ] = False
    enable_draft_weights_cpu_backup: A[
        bool,
        "Save draft model weights to CPU memory during release_weights_occupation and resume_weights_occupation",
    ] = False
    enable_custom_logit_processor: A[
        bool,
        "Enable users to pass custom logit processors to the server (disabled by default for security)",
    ] = False
    enable_return_hidden_states: A[
        bool,
        "Enable returning hidden states with responses.",
    ] = False
    enable_return_routed_experts: A[
        bool,
        "Enable returning routed experts of each layer with responses.",
    ] = False
    enable_return_indexer_topk: A[
        bool,
        "Enable returning indexer topk indices of layers with indexer with responses.",
    ] = False
    disable_outlines_disk_cache: A[
        bool,
        "Disable disk cache of outlines to avoid possible crashes related to file system or high concurrency.",
    ] = False
    enable_mis: A[
        bool,
        "Enable Multi-Item Scoring optimization. Combines query and multiple items into a single sequence for efficient batch processing. Requires --attention-backend flashinfer; auto-disables CUDA graph, radix cache, and chunked prefill.",
    ] = False

    # -------------------------------------------------------------------------
    # Deterministic inference
    # -------------------------------------------------------------------------
    enable_deterministic_inference: A[
        bool,
        "Enable deterministic inference mode with batch invariant ops.",
    ] = False
    rl_on_policy_target: A[
        Optional[str],
        Arg(
            help="The training system that SGLang needs to match for true on-policy.",
            choices=RL_ON_POLICY_TARGET_CHOICES,
        ),
    ] = None

    # -------------------------------------------------------------------------
    # KV canary
    # -------------------------------------------------------------------------
    kv_canary: A[
        str,
        Arg(
            help="KV cache canary mode. 'none' disables the canary (default). 'log' prints them while the server keeps running (production-safe). 'raise' fails the server on the first detected mismatch (CI lane).",
            choices=["none", "log", "raise"],
        ),
    ] = "none"
    kv_canary_real_data: str = "none"
    kv_canary_sweep_interval: A[
        int,
        "Every N forward steps, run a full-pool sweep.",
    ] = 0

    # -------------------------------------------------------------------------
    # Dynamic batch tokenizer
    # -------------------------------------------------------------------------
    enable_dynamic_batch_tokenizer: A[
        bool,
        "Enable async dynamic batch tokenizer for improved performance when multiple requests arrive concurrently.",
    ] = False
    dynamic_batch_tokenizer_batch_size: A[
        int,
        "[Only used if --enable-dynamic-batch-tokenizer is set] Maximum batch size for dynamic batch tokenizer.",
    ] = 32
    dynamic_batch_tokenizer_batch_timeout: A[
        float,
        "[Only used if --enable-dynamic-batch-tokenizer is set] Timeout in seconds for batching tokenization requests.",
    ] = 0.002
    enable_tokenizer_batch_encode: A[
        bool,
        "Enable batch tokenization for improved performance when processing multiple text inputs. Do not use with image inputs, pre-tokenized input_ids, or input_embeds.",
    ] = False
    disable_tokenizer_batch_decode: A[
        bool,
        "Disable batch decoding when decoding multiple completions.",
    ] = False

    # -------------------------------------------------------------------------
    # Debug tensor dumps
    # -------------------------------------------------------------------------
    debug_tensor_dump_output_folder: A[
        Optional[str],
        "The output folder for dumping tensors. In Eagle mode, tensor outputs from draft and target models are stored in separate subdirectories ('draft' and 'target').",
    ] = None
    # None means dump all layers.
    debug_tensor_dump_layers: A[
        Optional[List[int]],
        "The layer ids to dump. Dump all layers if not specified.",
    ] = None
    # TODO(guoyuhong): clean the old dumper code.
    debug_tensor_dump_input_file: A[
        Optional[str],
        "The input filename for dumping tensors",
    ] = None

    # -------------------------------------------------------------------------
    # PD disaggregation
    # -------------------------------------------------------------------------
    disaggregation_mode: A[
        Literal["null", "prefill", "decode"],
        'Only used for PD disaggregation. "prefill" for prefill-only server, and "decode" for decode-only server. If not specified, it is not PD disaggregated',
    ] = "null"
    disaggregation_transfer_backend: A[
        str,
        Arg(
            help="The backend for disaggregation transfer. Default is mooncake.",
            choices=DISAGG_TRANSFER_BACKEND_CHOICES,
        ),
    ] = "mooncake"
    disaggregation_bootstrap_port: A[
        int,
        "Bootstrap server port on the prefill server. Default is 8998.",
    ] = 8998
    disaggregation_ib_device: A[
        Optional[str],
        'The InfiniBand devices for disaggregation transfer. Supports a single device (e.g., --disaggregation-ib-device mlx5_0), a shared comma-separated list (e.g., --disaggregation-ib-device mlx5_0,mlx5_1), a per-GPU JSON mapping (e.g., --disaggregation-ib-device \'{"0": "mlx5_0,mlx5_1", "1": "mlx5_2"}\'), or a path to a JSON file containing that mapping. Default is None, which triggers automatic device detection when mooncake backend is enabled.',
    ] = None
    disaggregation_decode_enable_radix_cache: A[
        bool,
        "Enable radix cache on decode server (PD mode). Caches KV prefixes to avoid redundant transfers. Incompatible with --enable-hisparse, speculative decoding, and --disaggregation-transfer-backend fake.",
    ] = False
    disaggregation_decode_enable_offload_kvcache: A[
        bool,
        "Enable async KV cache offloading on decode server (PD mode).",
    ] = False
    num_reserved_decode_tokens: A[
        int,
        "Number of decode tokens that will have memory reserved when adding new request to the running batch.",
    ] = 512
    disaggregation_decode_extra_slots: A[
        Optional[int],
        "Number of extra decode req_to_token slots pre-allocated for in-transfer requests (PD mode). If unset, defaults to 0 (or 2x the per-worker running batch for small batches).",
    ] = None
    disaggregation_decode_polling_interval: A[
        int,
        "The interval to poll requests in decode server. Can be set to >1 to reduce the overhead of this.",
    ] = 1
    optimistic_prefill_retries: A[
        int,
        "Number of optimistic prefill retries that will skip the bootstrap wait. ",
    ] = 0

    # -------------------------------------------------------------------------
    # Encode prefill disaggregation
    # -------------------------------------------------------------------------
    encoder_only: A[
        bool,
        "For MLLM with an encoder, launch an encoder-only server",
    ] = False
    language_only: A[
        bool,
        "For VLM, load weights for the language model only.",
    ] = False
    encoder_transfer_backend: A[
        str,
        Arg(
            help="The backend for encoder disaggregation transfer. Default is zmq_to_scheduler.",
            choices=ENCODER_TRANSFER_BACKEND_CHOICES,
        ),
    ] = ENCODER_TRANSFER_BACKEND_CHOICES[0]
    encoder_urls: A[
        List[str],
        "List of encoder server urls.",
    ] = dataclasses.field(default_factory=list)
    encoder_bootstrap_port: A[
        int,
        "Port for the EncoderBootstrapServer that runs in the language-only tokenizer manager process. Encoders register here, and language-only receivers fetch the current URL list from here.",
    ] = 8997
    encoder_register_urls: A[
        List[str],
        "One or more EncoderBootstrapServer URLs to register this encoder with on startup, for dynamic encoder discovery. Example: --encoder-register-urls http://prefill0:8997 http://prefill1:8997. Used with --encoder-only servers.",
    ] = dataclasses.field(default_factory=list)
    enable_adaptive_dispatch_to_encoder: A[
        bool,
        "When enabled, adaptively dispatch: multi-image requests go to encoder in language_only epd mode, single-image requests are processed locally.",
    ] = False

    # -------------------------------------------------------------------------
    # PD-Multiplexing
    # -------------------------------------------------------------------------
    enable_pdmux: A[
        bool,
        "Enable PD-Multiplexing, PD running on greenctx stream.",
    ] = False
    pdmux_config_path: A[
        Optional[str],
        "The path of the PD-Multiplexing config file.",
    ] = None
    sm_group_num: A[int, "Number of sm partition groups."] = 8

    # -------------------------------------------------------------------------
    # Model weight update and weight loading
    # -------------------------------------------------------------------------
    custom_weight_loader: A[
        Optional[List[str]],
        Arg(
            help="The custom dataloader which used to update the model. Should be set with a valid import path, such as my_package.weight_load_func",
            nargs="*",
        ),
    ] = None
    weight_loader_disable_mmap: A[
        bool,
        "Disable mmap while loading weight using safetensors.",
    ] = False
    weight_loader_prefetch_checkpoints: A[
        bool,
        "Prefetch checkpoint files into OS page cache before loading. Each rank prefetches a fraction of the shards, reducing total network I/O on shared filesystems (NFS/Lustre) from N*checkpoint to 1*checkpoint. Recommended for models on network storage.",
    ] = False
    weight_loader_prefetch_num_threads: A[
        int,
        "Number of threads per rank for checkpoint prefetching (default: 4).",
    ] = 4
    weight_loader_drop_cache_after_load: A[
        bool,
        "Call posix_fadvise(DONTNEED) on each safetensors shard after loading it.",
    ] = False
    remote_instance_weight_loader_seed_instance_ip: A[
        Optional[str],
        "The ip of the seed instance for loading weights from remote instance.",
    ] = None
    remote_instance_weight_loader_seed_instance_service_port: A[
        Optional[int],
        "The service port of the seed instance for loading weights from remote instance.",
    ] = None
    remote_instance_weight_loader_send_weights_group_ports: A[
        Optional[List[int]],
        Arg(
            help="The communication group ports for loading weights from remote instance.",
            type_parser=json_list_type,
        ),
    ] = None
    remote_instance_weight_loader_backend: A[
        Literal["transfer_engine", "nccl", "modelexpress"],
        "The backend for loading weights from remote instance. Can be 'transfer_engine', 'nccl', or 'modelexpress'. Default is 'nccl'.",
    ] = "nccl"
    remote_instance_weight_loader_start_seed_via_transfer_engine: A[
        bool,
        "Start seed server via transfer engine backend for remote instance weight loader.",
    ] = False
    engine_info_bootstrap_port: A[
        int,
        "Port for the engine info bootstrap server. Default is 6789. Must be set explicitly when running multiple instances on the same node.",
    ] = 6789
    modelexpress_config: A[
        Optional[str],
        'JSON config for ModelExpress P2P weight loading. Keys: "url" (optional gRPC host:port override), "transport" ("nixl" or "transfer_engine"). Example: \'{"url": "localhost:8001", "transport": "nixl"}\'',
    ] = None
    download_dir: A[Optional[str], "Model download directory for huggingface."] = None
    model_checksum: A[
        Optional[str],
        Arg(
            help="Model file integrity verification. If provided without value, uses model-path as HF repo ID. Otherwise, provide checksums JSON file path or HuggingFace repo ID.",
            nargs="?",
            const="",
        ),
    ] = None
    delete_ckpt_after_loading: A[
        bool,
        "Delete the model checkpoint after loading the model.",
    ] = False
    # Checkpoint decryption
    decrypted_config_file: A[
        Optional[str],
        "The path of the decrypted config file.",
    ] = None
    decrypted_draft_config_file: A[
        Optional[str],
        "The path of the decrypted draft config file.",
    ] = None
    checkpoint_engine_wait_weights_before_ready: A[
        bool,
        "If set, the server will wait for initial weights to be loaded via checkpoint-engine or other update methods before serving inference requests.",
    ] = False

    # -------------------------------------------------------------------------
    # Multi-modal optimization configs
    # -------------------------------------------------------------------------
    enable_broadcast_mm_inputs_process: A[
        bool,
        "Enable broadcast mm-inputs process in scheduler.",
    ] = False
    enable_prefix_mm_cache: A[
        bool,
        "Enable prefix multimodal cache. Currently only supports mm-only.",
    ] = False
    mm_enable_dp_encoder: A[
        bool,
        "Enabling data parallelism for mm encoder. The dp size will be set to the tp size automatically.",
    ] = False
    mm_process_config: A[
        Optional[Dict[str, Any]],
        Arg(
            help="Multimodal preprocessing config, a json config contains keys: `image`, `video`, `audio`",
            type_parser=json.loads,
        ),
    ] = None
    limit_mm_data_per_request: A[
        Optional[Union[str, Dict[str, int]]],
        Arg(
            help='Limit the number of multimodal inputs per request. e.g. \'{"image": 1, "video": 1, "audio": 1}\'',
            type_parser=json.loads,
        ),
    ] = None
    enable_mm_global_cache: A[
        bool,
        "Enable global multimodal embedding cache to skip redundant ViT inference.",
    ] = False
    disable_fast_image_processor: A[
        bool,
        "Adopt base image processor instead of fast image processor.",
    ] = False
    keep_mm_feature_on_device: A[
        bool,
        "Keep multimodal feature tensors on device after processing to save D2H copy.",
    ] = False

    # -------------------------------------------------------------------------
    # Custom hooks, probe, and plugins
    # -------------------------------------------------------------------------
    forward_hooks: A[
        Optional[List[dict[str, Any]]],
        Arg(
            help="JSON-formatted forward hook specifications to attach to the model.",
            type_parser=json_list_type,
        ),
    ] = None
    msprobe_dump_config: A[
        Optional[str],
        "The path of the JSON configuration file for msProbe. If specified, enables msProbe dump.",
    ] = None

    def __post_init__(self):
        """
        Orchestrates the handling of various server arguments, ensuring proper configuration and validation.
        """

        self._maybe_download_model_for_runai()

        # Normalize load balancing defaults early (before dummy-model short-circuit).
        self._handle_load_balance_method()

        # Validate mm_process_config before dummy-model early return.
        self._handle_multimodal()
        # Validate SSL arguments early (before dummy-model short-circuit).
        self._handle_ssl_validation()
        # Validate transcription/ASR-specific server args (model-independent).
        self._handle_asr_validation()

        # Validate PD disaggregation flags early (before dummy-model short-circuit).
        from sglang.srt.arg_groups.pd_disaggregation_hook import (
            handle_pd_disaggregation,
        )

        handle_pd_disaggregation(self)
        if self.enable_session_radix_cache and self.radix_eviction_policy != "priority":
            raise ValueError(
                "--enable-session-radix-cache requires --radix-eviction-policy priority"
            )

        # Normalize deprecated CP aliases before validations or model-specific
        # defaults inspect enable_prefill_cp/cp_strategy.
        self._handle_legacy_cp_arguments()
        self._validate_prefill_only_disable_kv_cache_args()
        self._handle_dcp_validation()

        if self.model_path.lower() in ["none", "dummy"]:
            # Skip for dummy models
            return

        # Handle deprecated arguments.
        self._handle_deprecated_args()

        # Handle deprecated environment variables for prefill delayer.
        self._handle_prefill_delayer_env_compat()

        # Resolve --quantization unquant: explicitly opt out of quantization.
        # Convert to None now (before model config validation), but record
        # the intent so auto-detection in _handle_model_specific_adjustments
        # does not override it.
        if self.quantization == "unquant":
            self.quantization = None
            self._quantization_explicitly_unset = True
        else:
            self._quantization_explicitly_unset = False

        # Set missing default values.
        self._handle_missing_default_values()

        self._handle_cuda_graph_config()

        # Handle device-specific backends.
        self._handle_hpu_backends()
        self._handle_cpu_backends()
        self._handle_npu_backends()
        self._handle_mps_backends()
        self._handle_xpu_backends()

        current_platform.apply_server_args_defaults(self)

        # Get GPU memory capacity, which is a common dependency for several configuration steps.
        gpu_mem = get_device_memory_capacity(self.device)

        # Handle memory-related, chunked prefill, and CUDA graph batch size configurations.
        self._handle_gpu_memory_settings(gpu_mem)

        # enforce_disable_flashinfer_allreduce_fusion must be set before
        # _handle_model_specific_adjustments, which auto-enables the fusion
        # for several SM90/SM100 MoE arches.
        if self.enable_deterministic_inference:
            self.enforce_disable_flashinfer_allreduce_fusion = True

        # Apply model-specific adjustments.
        self._handle_model_specific_adjustments()

        # Set kernel backends.
        self._handle_sampling_backend()
        # Must run before _handle_attention_backend_compatibility so the
        # deterministic backend is set before auto-detection fills it in.
        self._handle_deterministic_inference()
        self._handle_attention_backend_compatibility()
        # Must run after the attention backend is resolved so the trtllm_mla
        # default (auto-selected for DeepseekV3ForCausalLM on sm100) is visible.
        self._disable_prefill_cuda_graph_for_deepseek_trtllm_mla()
        self._handle_mamba_backend()
        self._handle_int8_mamba_checkpoint()
        self._handle_linear_attn_backend()
        self._handle_kv4_compatibility()
        self._handle_page_size()
        self._handle_amd_specifics()
        self._handle_nccl_pre_warm()
        self._handle_grammar_backend()

        # Handle multi-item scoring constraints. Must run after the above so
        # the final attention backend and chunked_prefill_size are in effect.
        self._handle_multi_item_scoring()

        # Backend-dependent half of --prefill-only-disable-kv-cache validation.
        # Must stay after _handle_attention_backend_compatibility() (above) and
        # _handle_multi_item_scoring() so the resolved prefill backend is final;
        # the flag/precondition half runs earlier in
        # _validate_prefill_only_disable_kv_cache_args().
        self._handle_prefill_only_disable_kv_cache()

        # Handle Hicache settings.
        self._handle_hicache()

        # Handle data parallelism.
        self._handle_data_parallelism()

        # Re-apply after model-specific defaults resolve attention_backend so
        # canonical CP mirrors to the right legacy runtime aliases.
        self._handle_legacy_cp_arguments()

        # Handle context parallelism.
        self._handle_context_parallelism()

        # Handle MoE configurations.
        self._handle_moe_kernel_config()
        self._handle_a2a_moe()
        self._handle_eplb_and_dispatch()
        self._handle_expert_distribution_metrics()
        self._handle_elastic_ep()

        # Handle pipeline parallelism.
        self._handle_pipeline_parallelism()

        # Handle speculative decoding logic.
        from sglang.srt.arg_groups.speculative_hook import handle_speculative_decoding

        handle_speculative_decoding(self)

        # Validate the CuteDSL A2A token budget now that num_tokens_per_bs is final.
        self._validate_cutedsl_a2a_token_budget()

        # Handle model loading format.
        self._handle_load_format()

        # Handle Encoder disaggregation.
        self._handle_encoder_disaggregation()

        # Validate tokenizer settings.
        self._handle_tokenizer_batching()

        # Propagate environment variables.
        self._handle_environment_variables()

        # Validate cache settings.
        self._handle_cache_compatibility()

        self._handle_page_major_kv_layout()

        # Handle diffusion LLM inference.
        self._handle_dllm_inference()

        # Handle crash dump environment variables (must run before CUDA init).
        self._handle_crash_dump_env()

        # Handle debug utilities.
        self._handle_debug_utils()

        # Handle any other necessary validations.
        self._handle_other_validations()

    def _maybe_download_model_for_runai(self):
        if is_runai_obj_uri(self.model_path):
            ObjectStorageModel.download_and_get_path(self.model_path)

        if (
            self.tokenizer_path is not None
            and is_runai_obj_uri(self.tokenizer_path)
            and self.tokenizer_path != self.model_path
        ):
            ObjectStorageModel.download_and_get_path(self.tokenizer_path)

    def _handle_dcp_validation(self):
        # Decode context parallel (DCP) is currently implemented and validated
        # only on AMD HIP/ROCm. Reject invalid or unverified configurations
        # early instead of letting them fail deeper in model initialization.
        if self.dcp_size < 1:
            raise ValueError(
                "Decode context parallel size (--dcp-size / "
                "--decode-context-parallel-size) must be >= 1, but got "
                f"dcp_size={self.dcp_size}."
            )
        if not self.dcp_size > 1:
            return
        if is_hip():
            return
        elif is_cuda():
            if self.speculative_algorithm is not None:
                raise ValueError(
                    "Decode context parallel (--dcp-size / "
                    "--decode-context-parallel-size > 1) on CUDA platform "
                    "does not support any speculative algorithm, but got "
                    f"dcp_size={self.dcp_size} on a CUDA platform with "
                    "speculative decoding enabled."
                )
        else:
            raise ValueError(
                "Decode context parallel (--dcp-size / "
                "--decode-context-parallel-size > 1) is currently only "
                f"supported on the AMD HIP platform, but got dcp_size="
                f"{self.dcp_size} on a non-HIP platform."
            )

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

    def _handle_ssl_validation(self):
        """Ensure SSL arguments are consistent and referenced files exist."""
        if self.ssl_keyfile and not self.ssl_certfile:
            raise ValueError(
                "--ssl-keyfile requires --ssl-certfile to be specified as well."
            )
        if self.ssl_certfile and not self.ssl_keyfile:
            raise ValueError(
                "--ssl-certfile requires --ssl-keyfile to be specified as well."
            )
        if not self.ssl_certfile and not self.ssl_keyfile:
            if self.ssl_ca_certs:
                raise ValueError(
                    "--ssl-ca-certs has no effect without --ssl-certfile and --ssl-keyfile."
                )
            if self.ssl_keyfile_password:
                raise ValueError(
                    "--ssl-keyfile-password has no effect without --ssl-certfile and --ssl-keyfile."
                )
        # Validate files exist early to avoid late failures after model loading.
        if self.ssl_keyfile and not os.path.isfile(self.ssl_keyfile):
            raise ValueError(
                f"SSL key file not found: '{self.ssl_keyfile}'. "
                f"Please check the --ssl-keyfile path."
            )
        if self.ssl_certfile and not os.path.isfile(self.ssl_certfile):
            raise ValueError(
                f"SSL certificate file not found: '{self.ssl_certfile}'. "
                f"Please check the --ssl-certfile path."
            )
        if self.ssl_ca_certs and not os.path.isfile(self.ssl_ca_certs):
            raise ValueError(
                f"SSL CA certificates file not found: '{self.ssl_ca_certs}'. "
                f"Please check the --ssl-ca-certs path."
            )
        if self.enable_ssl_refresh and not (self.ssl_certfile and self.ssl_keyfile):
            raise ValueError(
                "--enable-ssl-refresh requires --ssl-certfile and --ssl-keyfile "
                "to be specified."
            )

        if self.enable_http2:
            try:
                import granian  # noqa: F401
            except ImportError:
                raise ValueError(
                    "--enable-http2 requires the 'granian' package. "
                    'Install it with: pip install "sglang[http2]"'
                )

            if self.enable_ssl_refresh:
                raise ValueError(
                    "--enable-ssl-refresh is not supported with --enable-http2. "
                    "Granian does not support SSL certificate hot-reloading. "
                    "Use Uvicorn (the default) or handle certificate rotation externally."
                )

    def _handle_multimodal(self):
        """Validate mm_process_config structure before model loading."""
        if self.mm_process_config is not None:
            if not isinstance(self.mm_process_config, dict):
                raise TypeError(
                    f"mm_process_config must be a dict, "
                    f"but got {type(self.mm_process_config)}"
                )
            for key in ("image", "video", "audio"):
                if key in self.mm_process_config and not isinstance(
                    self.mm_process_config[key], dict
                ):
                    raise TypeError(
                        f"mm_process_config['{key}'] must be a dict, "
                        f"but got {type(self.mm_process_config[key])}"
                    )

    def _handle_deprecated_args(self):
        # Handle deprecated tool call parsers
        deprecated_tool_call_parsers = {"qwen25": "qwen", "glm45": "glm"}
        if self.tool_call_parser in deprecated_tool_call_parsers:
            logger.warning(
                f"The tool_call_parser '{self.tool_call_parser}' is deprecated. Please use '{deprecated_tool_call_parsers[self.tool_call_parser]}' instead."
            )
            self.tool_call_parser = deprecated_tool_call_parsers[self.tool_call_parser]

        # When user passes --enable-flashinfer-allreduce-fusion, enable with auto backend
        if (
            self.enable_flashinfer_allreduce_fusion
            and self.flashinfer_allreduce_fusion_backend is None
        ):
            logger.warning(
                "--enable-flashinfer-allreduce-fusion is deprecated. "
                "Please use --flashinfer-allreduce-fusion-backend=auto instead."
            )
            self.flashinfer_allreduce_fusion_backend = "auto"
        self.enable_flashinfer_allreduce_fusion = False
        # Deprecated attention-backend alias: "compressed" -> "dsv4".
        for attr in (
            "attention_backend",
            "decode_attention_backend",
            "prefill_attention_backend",
            "speculative_draft_attention_backend",
        ):
            if getattr(self, attr, None) == "compressed":
                logger.warning(
                    "--%s=compressed is deprecated; use 'dsv4' instead.",
                    attr.replace("_", "-"),
                )
                setattr(self, attr, "dsv4")

        # Native gRPC flags — env-only for now, not exposed as CLI args.
        # Set as instance attributes (not dataclass fields) to avoid
        # argparse namespace lookup in from_cli_args.
        self.enable_grpc = envs.SGLANG_ENABLE_GRPC.get()

        grpc_port_env = envs.SGLANG_GRPC_PORT.get()
        self.grpc_port = (
            grpc_port_env if grpc_port_env is not None else self.port + 10000
        )

        if not (1 <= self.grpc_port <= 65535):
            raise ValueError(
                f"SGLANG_GRPC_PORT ({self.grpc_port}) must be between 1 and 65535"
            )

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
        # strip device index from user if any (e.g. "cuda:0" -> "cuda")
        self.device = self.device.split(":")[0]
        if self.random_seed is None:
            self.random_seed = random.randint(0, 1 << 30)
        if self.mm_process_config is None:
            self.mm_process_config = {}

        # Handle ModelScope model downloads
        if envs.SGLANG_USE_MODELSCOPE.get():
            self._handle_modelscope_paths()

        # In speculative scenario:
        # - If `speculative_draft_model_quantization` is specified, the draft model uses this quantization method.
        # - Otherwise, the draft model defaults to the same quantization as the target model.
        if self.speculative_draft_model_quantization is None:
            self.speculative_draft_model_quantization = self.quantization
        elif self.speculative_draft_model_quantization == "unquant":
            self.speculative_draft_model_quantization = None

    def _handle_modelscope_paths(self):
        """Resolve model / tokenizer / speculative-draft paths from the local
        ModelScope cache when possible, falling back to snapshot_download
        for any path that is not already present on disk.

        Note: speculative_token_map is intentionally NOT handled here
        because its value uses repo_id/filename semantics rather than a
        plain repo ID.  That resolution lives in
        :func:`sglang.srt.speculative.spec_utils.load_token_map`.
        """

        ms_root = None
        ms_snapshot_download = None

        def _resolve_or_download(
            path: Optional[str],
            ignore_patterns: Optional[list] = None,
            revision: Optional[str] = None,
        ) -> Optional[str]:
            nonlocal ms_root, ms_snapshot_download
            if path is None:
                return None
            if not path or os.path.exists(path):
                return path

            if ms_snapshot_download is None:
                from modelscope.hub.snapshot_download import (
                    snapshot_download as _ms_snapshot_download,
                )
                from modelscope.utils.file_utils import get_model_cache_root

                ms_snapshot_download = _ms_snapshot_download
                ms_root = get_model_cache_root()

            # Check ModelScope default cache
            cached = os.path.join(ms_root, path)
            if os.path.exists(cached):
                return cached
            # Check user-specified download dir
            if self.download_dir:
                alt = os.path.join(self.download_dir, path)
                if os.path.exists(alt):
                    return alt

            # Cache miss — download from ModelScope hub
            return ms_snapshot_download(
                path,
                cache_dir=self.download_dir,
                revision=revision,
                **({"ignore_patterns": ignore_patterns} if ignore_patterns else {}),
            )

        self.model_path = _resolve_or_download(self.model_path, revision=self.revision)
        self.tokenizer_path = _resolve_or_download(
            self.tokenizer_path,
            ignore_patterns=["*.bin", "*.safetensors"],
            revision=self.revision,
        )
        if self.speculative_draft_model_path:
            self.speculative_draft_model_path = _resolve_or_download(
                self.speculative_draft_model_path,
                revision=self.speculative_draft_model_revision or "main",
            )

    def _handle_hpu_backends(self):
        if self.device == "hpu":
            self.attention_backend = "torch_native"
            self.sampling_backend = "pytorch"

    def _handle_cpu_backends(self):
        if self.device == "cpu":
            if self.attention_backend is None:
                self.attention_backend = (
                    "torch_native" if is_host_cpu_arm64() else "intel_amx"
                )
            self.sampling_backend = "pytorch"

    def _handle_npu_backends(self):
        if self.device == "npu":
            from sglang.srt.hardware_backend.npu.utils import set_default_server_args

            set_default_server_args(self)

            current = self.cuda_graph_config.prefill.tc_compiler
            if current is not None and current != "eager":
                logger.warning(
                    "At this moment Ascend platform only support prefill graph compilation with "
                    "cuda_graph_config[prefill].tc_compiler='eager'."
                )
                self.cuda_graph_config.prefill.tc_compiler = "eager"

    def _handle_mps_backends(self):
        if self.device == "mps":
            if not use_mlx():
                self.disable_overlap_schedule = True

    def _handle_xpu_backends(self):
        if self.device == "xpu":
            if self.cuda_graph_config.prefill.backend != Backend.DISABLED:
                logger.warning(
                    "XPU platform does not support piecewise CUDA graph, "
                    "disabling prefill cuda graph."
                )
            self.cuda_graph_config.prefill.backend = Backend.DISABLED

    # ------------------------------------------------------------------
    # CUDA graph configuration resolution
    # ------------------------------------------------------------------
    # TODO: add unit tests in test/srt/test_server_args.py covering the
    # precedence cascade + auto-disable matrix (follow-up PR).
    def _handle_cuda_graph_config(self):
        self._parse_cuda_graph_config()
        self._apply_cuda_graph_compatibility()
        self._validate_cuda_graph_config()

    def _parse_cuda_graph_config(self):
        """Resolve cuda_graph_config from explicit JSON, per-phase
        convenience flags, legacy global flags, and defaults.
        Precedence (highest first): explicit JSON > convenience > legacy > defaults.
        Also populates self._cuda_graph_config_locked — the set of
        (phase, key) tuples that came from non-default sources; the
        auto-disable cascade respects this lock (the old
        --enforce-piecewise-cuda-graph semantics generalized).
        """
        raw_input = self.cuda_graph_config
        if isinstance(raw_input, CudaGraphConfig):
            explicit_input = raw_input.to_dict()
        else:
            explicit_input = raw_input or {}
        config = default_cuda_graph_config()
        locked: set = set()

        def _set(phase: str, key: str, value: Any) -> None:
            setattr(getattr(config, phase), key, value)
            locked.add((phase, key))

        # ---- Legacy global flags (lowest precedence above defaults) ----
        if self.disable_cuda_graph:
            _set(Phase.DECODE, "backend", Backend.DISABLED)
            _set(Phase.PREFILL, "backend", Backend.DISABLED)

        # ---- Boolean per-phase off-switches ----
        # Below the explicit backend selectors so --cuda-graph-backend-*
        # wins if both are given.
        if self.disable_prefill_cuda_graph:
            _set(Phase.PREFILL, "backend", Backend.DISABLED)
        if self.disable_decode_cuda_graph:
            _set(Phase.DECODE, "backend", Backend.DISABLED)

        # ---- Per-phase convenience flags ----
        if self.cuda_graph_backend_decode is not None:
            _set(Phase.DECODE, "backend", self.cuda_graph_backend_decode)
        if self.cuda_graph_backend_prefill is not None:
            _set(Phase.PREFILL, "backend", self.cuda_graph_backend_prefill)
        if self.cuda_graph_max_bs_decode is not None:
            _set(Phase.DECODE, "max_bs", self.cuda_graph_max_bs_decode)
        if self.cuda_graph_max_bs_prefill is not None:
            _set(Phase.PREFILL, "max_bs", self.cuda_graph_max_bs_prefill)
        if self.cuda_graph_bs_decode is not None:
            _set(Phase.DECODE, "bs", self.cuda_graph_bs_decode)
        if self.cuda_graph_bs_prefill is not None:
            _set(Phase.PREFILL, "bs", self.cuda_graph_bs_prefill)
        if self.cuda_graph_tc_compiler is not None:
            # Written to both phases so the value is in place when TC_PIECEWISE
            # decode is implemented; today decode ignores it.
            _set(Phase.DECODE, "tc_compiler", self.cuda_graph_tc_compiler)
            _set(Phase.PREFILL, "tc_compiler", self.cuda_graph_tc_compiler)

        # ---- Explicit JSON config (highest precedence) ----
        for phase, phase_config in explicit_input.items():
            if not isinstance(phase_config, dict):
                continue
            for key, value in phase_config.items():
                _set(phase, key, value)

        self.cuda_graph_config = config
        self._cuda_graph_config_locked = locked

    def _apply_cuda_graph_compatibility(self):
        """Auto-disable prefill cuda graph for incompatible configs.
        Rules are split per backend — TcPiecewise and Breakable have
        different constraints. Skipped when the user explicitly set the
        prefill backend (this folds in the old
        --enforce-piecewise-cuda-graph contract).
        """
        if (Phase.PREFILL, "backend") in self._cuda_graph_config_locked:
            return
        if self.cuda_graph_config.prefill.backend == Backend.TC_PIECEWISE:
            self._disable_tc_piecewise_cudagraph_if_incompatible()
        elif self.cuda_graph_config.prefill.backend == Backend.BREAKABLE:
            self._disable_breakable_cudagraph_if_incompatible()

    def _disable_tc_piecewise_cudagraph_if_incompatible(self):
        """TcPiecewise (torch.compile + piecewise) is incompatible with
        these configurations. Most are torch.compile / dynamo limitations.
        """

        rules = [
            (
                "model-arch blacklist",
                lambda: self.get_model_config().is_piecewise_cuda_graph_disabled_model,
            ),
            ("DP attention", lambda: self.enable_dp_attention),
            ("full torch.compile mode", lambda: self.enable_torch_compile),
            ("pipeline parallelism (pp_size > 1)", lambda: self.pp_size > 1),
            (
                "non-CUDA hardware (HIP/NPU/CPU/MPS/XPU/MLU)",
                lambda: is_hip()
                or is_npu()
                or is_cpu()
                or is_mps()
                or is_xpu()
                or current_platform.is_mlu(),
            ),
            (
                "OOT platform without piecewise support",
                lambda: current_platform.is_out_of_tree()
                and not current_platform.support_piecewise_cuda_graph(),
            ),
            ("MoE A2A backend", lambda: self.moe_a2a_backend != "none"),
            ("LoRA", lambda: bool(self.lora_paths) or self.enable_lora),
            (
                "multimodal model",
                lambda: self.get_model_config().is_multimodal
                and not self.get_model_config().is_multimodal_piecewise_cuda_graph_supported,
            ),
            (
                "GGUF quantization",
                lambda: self.load_format == "gguf"
                or self.quantization == "gguf"
                or check_gguf_file(self.model_path),
            ),
            ("DLLM (diffusion LLM)", lambda: self.dllm_algorithm is not None),
            (
                "CPU offload / hierarchical cache",
                lambda: self.cpu_offload_gb > 0 or self.enable_hierarchical_cache,
            ),
            (
                "deterministic inference",
                lambda: self.enable_deterministic_inference,
            ),
            ("PD disaggregation", lambda: self.disaggregation_mode != "null"),
            ("symmetric memory", lambda: self.enable_symm_mem),
            (
                "expert distribution recorder",
                lambda: self.enable_eplb
                or self.expert_distribution_recorder_mode is not None,
            ),
            ("context parallel (attn_cp_size > 1)", lambda: self.attn_cp_size > 1),
            ("CUDA graph debug mode", lambda: self.debug_cuda_graph),
            (
                "DSA prefill context parallelism",
                lambda: self.enable_dsa_prefill_context_parallel,
            ),
        ]
        for _name, predicate in rules:
            if predicate():
                self.cuda_graph_config.prefill.backend = Backend.DISABLED

    def _disable_breakable_cudagraph_if_incompatible(self):
        """Breakable (segmented capture, no torch.compile). Breakable enforces
        memory-saver rejection in its own __init__; config-time rules can be
        added here as they're discovered.
        """
        rules = [
            # MLA prefill takes a different attn-forward path under BCG (no
            # tc_piecewise gate), causing q.view shape mismatches. Disable
            # until the MLA prefill path is BCG-aware.
            ("MLA attention", lambda: self.use_mla_backend()),
        ]
        for name, predicate in rules:
            if predicate():
                logger.warning(
                    "Breakable CUDA graph is incompatible with %s; "
                    "disabling prefill CUDA graph.",
                    name,
                )
                self.cuda_graph_config.prefill.backend = Backend.DISABLED
                return

    def _disable_prefill_cuda_graph_for_deepseek_trtllm_mla(self):
        """Disable prefill CUDA graph for dsr1 by default when using the trtllm_mla
        attention backend. Under any captured prefill CUDA graph (tc_piecewise or
        breakable) trtllm_mla falls back to FlashAttention for prefill and regresses
        performance, so disable whichever prefill graph backend is in effect.
        """

        if (Phase.PREFILL, "backend") in self._cuda_graph_config_locked:
            return
        if self.cuda_graph_config.prefill.backend == Backend.DISABLED:
            return
        if (
            "DeepseekV3ForCausalLM"
            not in self.get_model_config().hf_config.architectures
        ):
            return
        prefill_attention_backend, _ = self.get_attention_backends()
        if prefill_attention_backend != "trtllm_mla":
            return
        logger.warning(
            "Disabling prefill CUDA graph (%s) by default for the DeepSeek-V3 arch on "
            "the trtllm_mla attention backend (a captured prefill graph forces a "
            "FlashAttention fallback that regresses prefill). Set the prefill cuda graph "
            "backend explicitly (e.g. --cuda-graph-backend-prefill tc_piecewise) to override.",
            self.cuda_graph_config.prefill.backend,
        )
        self.cuda_graph_config.prefill.backend = Backend.DISABLED

    def _validate_cuda_graph_config(self):
        if self.cuda_graph_config is None:
            return
        for phase in Phase.ALL:
            backend = getattr(self.cuda_graph_config, phase).backend
            if backend not in ALLOWED_BACKENDS_PER_PHASE[phase]:
                raise ValueError(
                    f"--cuda-graph-config[{phase}].backend={backend!r} not allowed; "
                    f"allowed: {ALLOWED_BACKENDS_PER_PHASE[phase]}"
                )

    def _handle_multi_item_scoring(self):
        """Setup and validate multi-item scoring constraints.

        Auto-disables settings incompatible with MIS mechanics (CUDA graph,
        radix cache, chunked prefill). Asserts on attention backend since
        changing it silently could surprise users who intentionally picked
        a non-flashinfer backend.
        """
        if not self.enable_mis:
            return

        if self.cuda_graph_config.decode.backend != Backend.DISABLED:
            logger.warning("CUDA graph is disabled because --enable-mis is set.")
        self.cuda_graph_config.decode.backend = Backend.DISABLED
        self.cuda_graph_config.prefill.backend = Backend.DISABLED

        if not self.disable_radix_cache:
            logger.warning("Radix cache is disabled because --enable-mis is set.")
            self.disable_radix_cache = True

        if self.chunked_prefill_size != -1:
            logger.warning("Chunked prefill is disabled because --enable-mis is set.")
            self.chunked_prefill_size = -1

        prefill_backend, decode_backend = self.get_attention_backends()
        assert prefill_backend == "flashinfer" and decode_backend == "flashinfer", (
            "Multi-item scoring requires flashinfer attention backend for custom attention mask support. "
            f"Please set --attention-backend flashinfer when using --enable-mis. "
            f"Current backends: prefill={prefill_backend}, decode={decode_backend}"
        )

    def _handle_gpu_memory_settings(self, gpu_mem):
        """
        Configure GPU memory-dependent settings including
        chunked_prefill_size, cuda_graph_config[decode].max_bs, and mem_fraction_static.

        Here are our heuristics:
        - Set chunked_prefill_size and cuda_graph_config[decode].max_bs based on the GPU memory capacity.
          This is because GPUs with more memory are generally more powerful, we need to use a larger
          chunked_prefill_size and a larger decode max_bs to fully utilize the GPU.
        - Then set mem_fraction_static based on chunked_prefill_size and decode max_bs.

          GPU memory capacity = model weights + KV cache pool + activations + cuda graph buffers

          The argument mem_fraction_static is defined as (model weights + KV cache pool) / GPU memory capacity,
          or equivalently, mem_fraction_static = (GPU memory capacity - activations - cuda graph buffers) / GPU memory capacity.

          In order to compute mem_fraction_static, we need to estimate the size of activations and cuda graph buffers.
          The activation memory is proportional to the chunked_prefill_size.
          The cuda graph memory is proportional to the decode max_bs.
          We use reserved_mem = chunked_prefill_size * 1.5 + max_bs * 2 to estimate the size of activations and cuda graph buffers in GB,
          and set mem_fraction_static = (GPU memory capacity - reserved_mem) / GPU memory capacity.

          The coefficient 1.5 is a heuristic value, in the future, we can do better estimation by looking at the model types, hidden sizes or even do a dummy run.
        """
        decode_cuda_graph_config = self.cuda_graph_config.decode
        prefill_cuda_graph_config = self.cuda_graph_config.prefill

        if gpu_mem is not None:
            if gpu_mem < 20 * 1024:
                # T4, 4080
                # (chunked_prefill_size 2k, max_bs 8)
                if self.chunked_prefill_size is None:
                    self.chunked_prefill_size = 2048
                if decode_cuda_graph_config.max_bs is None:
                    decode_cuda_graph_config.max_bs = 8
            elif gpu_mem < 35 * 1024:
                # A10, 4090, 5090
                # (chunked_prefill_size 2k, max_bs 24 if tp < 4 else 80)
                if self.chunked_prefill_size is None:
                    self.chunked_prefill_size = 2048
                if decode_cuda_graph_config.max_bs is None:
                    if self.tp_size < 4:
                        decode_cuda_graph_config.max_bs = 24
                    else:
                        decode_cuda_graph_config.max_bs = 80
            elif gpu_mem < 60 * 1024:
                # A100 (40GB), L40,
                # (chunked_prefill_size 4k, max_bs 32 if tp < 4 else 160)
                if self.chunked_prefill_size is None:
                    self.chunked_prefill_size = 4096
                if decode_cuda_graph_config.max_bs is None:
                    if self.tp_size < 4:
                        decode_cuda_graph_config.max_bs = 32
                    else:
                        decode_cuda_graph_config.max_bs = 160
            elif gpu_mem < 90 * 1024:
                # H100, A100
                # (chunked_prefill_size 8k, max_bs 256 if tp < 4 else 512)
                if self.chunked_prefill_size is None:
                    self.chunked_prefill_size = 8192
                if decode_cuda_graph_config.max_bs is None:
                    if self.tp_size < 4:
                        decode_cuda_graph_config.max_bs = 256
                    else:
                        decode_cuda_graph_config.max_bs = 512
            elif gpu_mem < 160 * 1024:
                # H20, H200
                # (chunked_prefill_size 8k, max_bs 256 if tp < 4 else 512)
                if self.chunked_prefill_size is None:
                    self.chunked_prefill_size = 8192
                if decode_cuda_graph_config.max_bs is None:
                    if self.tp_size < 4:
                        decode_cuda_graph_config.max_bs = 256
                    else:
                        decode_cuda_graph_config.max_bs = 512
            else:
                # B200, MI300
                # (chunked_prefill_size 16k, max_bs 512)
                if self.chunked_prefill_size is None:
                    self.chunked_prefill_size = 16384
                if decode_cuda_graph_config.max_bs is None:
                    decode_cuda_graph_config.max_bs = 512
        else:
            # Fallback defaults when gpu_mem is None
            if self.chunked_prefill_size is None:
                self.chunked_prefill_size = 4096
            if decode_cuda_graph_config.max_bs is None:
                decode_cuda_graph_config.max_bs = 160

        # Set cuda graph batch sizes
        if self.device != "cpu":
            if decode_cuda_graph_config.bs is None:
                decode_cuda_graph_config.bs = (
                    self._generate_decode_cuda_graph_batch_sizes(
                        decode_cuda_graph_config.max_bs
                    )
                )
            else:
                decode_cuda_graph_config.max_bs = max(decode_cuda_graph_config.bs)
        else:
            # Reuse decode_cuda_graph_config.bs for cpu graph and use torch_compile_max_bs for cpu graph batch size limit,
            # as cpu graph is based on torch.compile
            if decode_cuda_graph_config.bs is not None:
                self.torch_compile_max_bs = max(decode_cuda_graph_config.bs)
            else:
                # If decode_cuda_graph_config.bs is not set, we will preferentially use torch_compile_max_bs
                # to generate decode_cuda_graph_config.bs
                self.torch_compile_max_bs = (
                    self.torch_compile_max_bs or decode_cuda_graph_config.max_bs
                )
                decode_cuda_graph_config.bs = self._generate_cpu_graph_batch_sizes()

            assert (
                self.torch_compile_max_bs > 0
            ), "cuda_graph_config[decode].bs should contain positive batch sizes"
            decode_cuda_graph_config.max_bs = self.torch_compile_max_bs

        if prefill_cuda_graph_config.max_bs is None:
            # Refer to pr #15927, by default we set the prefill max_bs to the chunked prefill size.
            # For MLA backend, the introduction of piecewise cuda graph will influence the kernel dispatch difference compared to the original mode.
            # To avoid the performance regression, we set max_bs to 2048 by default.
            if not self.use_mla_backend():
                prefill_cuda_graph_config.max_bs = self.chunked_prefill_size
            else:
                prefill_cuda_graph_config.max_bs = 2048

            # If max_total_tokens is set, cap prefill max_bs to not exceed max_total_tokens.
            if self.max_total_tokens is not None:
                prefill_cuda_graph_config.max_bs = min(
                    prefill_cuda_graph_config.max_bs, self.max_total_tokens
                )

            # For Llama2 series models, max_bs is limited to 4096.
            # TODO(yuwei): remove this after the issue is fixed
            if "llama-2" in self.model_path.lower():
                prefill_cuda_graph_config.max_bs = min(
                    prefill_cuda_graph_config.max_bs, 4096
                )

        # Clamp to context_length if explicitly set — prevents prefill CG
        # warmup from compiling graphs with more tokens than the model
        # buffers can hold, which causes illegal memory access (#21112).
        if self.context_length is not None:
            prefill_cuda_graph_config.max_bs = min(
                prefill_cuda_graph_config.max_bs, self.context_length
            )

        if prefill_cuda_graph_config.bs is None:
            prefill_cuda_graph_config.bs = (
                self._generate_prefill_cuda_graph_batch_sizes(
                    prefill_cuda_graph_config.max_bs
                )
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
            reserved_mem += decode_cuda_graph_config.max_bs * 2
            # Some adjustments for large parallel size
            reserved_mem += self.tp_size * self.pp_size / 8 * 1024

            if self.enable_dp_attention:
                # DP attention needs more padding for some operations
                reserved_mem += decode_cuda_graph_config.max_bs * self.dp_size * 3

                # DP attention uses much more memory for large cuda graph max bs,
                # likely due to some inefficiencies in torch allocator or our implementation.
                # So we need to reserve more memory.
                if decode_cuda_graph_config.max_bs > 300:
                    reserved_mem += decode_cuda_graph_config.max_bs * self.dp_size * 1.5

            # For piecewise cuda graphs
            if prefill_cuda_graph_config.backend != Backend.DISABLED:
                if not self.use_mla_backend():
                    # Only calculate the memory overhead for Non-Torch Memory use since the Torch Memory can be reused with Cuda Graph Capture
                    reserved_mem += len(prefill_cuda_graph_config.bs) * 8
                else:
                    # For MLA backend the memory overhead is much higher than expected with fa3
                    reserved_mem += 1.5 * 1024

            if gpu_mem is not None and gpu_mem > 60 * 1024:
                reserved_mem = max(reserved_mem, 10 * 1024)

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
        if self.enable_symm_mem and not envs.SGLANG_SYMM_MEM_PREALLOC_GB_SIZE.is_set():
            envs.SGLANG_SYMM_MEM_PREALLOC_GB_SIZE.set(4)
            logger.warning(
                "Symmetric memory is enabled, setting symmetric memory prealloc size to 4GB as default."
                "Use environment variable SGLANG_SYMM_MEM_PREALLOC_GB_SIZE to change the prealloc size."
            )

    def _generate_decode_cuda_graph_batch_sizes(self, max_bs: int):
        """
        Generate the list of batch sizes for CUDA graph capture based on max_bs.
        This integrates the logic from cuda_graph_runner.py.
        """
        # Handle disable_cuda_graph_padding as the first condition for both spec and non-spec
        if self.disable_cuda_graph_padding:
            capture_bs = list(range(1, max_bs + 1))
        elif self.speculative_algorithm is None:
            # Normal case:
            capture_bs = (
                [1, 2, 4, 8, 12]
                + list(range(16, 257, 8))
                + list(range(272, 512, 16))
                + list(range(512, max_bs + 1, 32))
            )
        else:
            # Spec decoding case: less padding for smaller batch sizes
            capture_bs = (
                list(range(1, 9, 1))
                + list(range(10, 33, 2))
                + list(range(40, 65, 4))
                + list(range(72, 257, 8))
                + list(range(272, max_bs + 1, 16))
            )

        capture_bs = [bs for bs in capture_bs if bs <= max_bs]

        if max_bs not in capture_bs:
            capture_bs.append(max_bs)

        return capture_bs

    def _generate_cpu_graph_batch_sizes(self):
        """
        Generate the list of batch sizes for CPU graph capture based on torch_compile_max_bs.
        """
        if self.disable_cuda_graph_padding:
            capture_bs = list(range(1, self.torch_compile_max_bs + 1))
        else:
            capture_bs = sorted(
                set().union(
                    range(1, 17),
                    range(18, 31, 2),
                    range(32, 81, 4),
                    range(84, self.torch_compile_max_bs + 1, 8),
                    {self.torch_compile_max_bs},
                )
            )
        capture_bs = [bs for bs in capture_bs if bs <= self.torch_compile_max_bs]

        return capture_bs

    def _generate_prefill_cuda_graph_batch_sizes(self, max_bs: int):
        """
        Generate the list of batch sizes for prefill CUDA graph capture
        based on max_bs. For tc_piecewise prefill, bs carries the
        captured token count (one shape knob per phase).
        """
        capture_sizes = (
            list(range(4, 33, 4))
            + list(range(48, 257, 16))
            + list(range(288, 513, 32))
            + list(range(576, 1024 + 1, 64))
            + list(range(1280, 4096 + 1, 256))
            + list(range(4608, max_bs + 1, 512))
        )

        capture_sizes = [s for s in capture_sizes if s <= max_bs]

        return capture_sizes

    def _set_default_dsa_kv_cache_dtype(self, major: int, quantization: str) -> str:
        user_set_prefill = self.dsa_prefill_backend is not None
        user_set_decode = self.dsa_decode_backend is not None

        # If user specified a backend but didn't explicitly set kv_cache_dtype,
        # suggest them to be explicit about kv_cache_dtype to avoid surprises
        if (user_set_prefill or user_set_decode) and self.kv_cache_dtype == "auto":
            logger.warning(
                "When specifying --dsa-prefill-backend or --dsa-decode-backend, "
                "you should also explicitly set --kv-cache-dtype (e.g., 'fp8_e4m3' or 'bfloat16'). "
                "DeepSeek V3.2 defaults to FP8 KV cache which may not be compatible with all backends."
            )

        if self.kv_cache_dtype == "auto":
            if major >= 10:
                self.kv_cache_dtype = "fp8_e4m3"
            else:
                self.kv_cache_dtype = "bfloat16"
            logger.warning(
                f"Setting KV cache dtype to {self.kv_cache_dtype} for DeepSeek DSA on SM{major} device."
            )
        if self.kv_cache_dtype == "bf16":
            self.kv_cache_dtype = "bfloat16"
        assert self.kv_cache_dtype in [
            "bfloat16",
            "fp8_e4m3",
        ], "DeepSeek DSA only supports bf16/bfloat16 or fp8_e4m3 kv_cache_dtype"

    def _set_default_dsa_backends(self, kv_cache_dtype: str, major: int) -> str:
        from sglang.srt.arg_groups.hisparse_hook import (
            apply_hisparse_dsa_backend_defaults,
        )

        user_set_prefill = self.dsa_prefill_backend is not None
        user_set_decode = self.dsa_decode_backend is not None

        if apply_hisparse_dsa_backend_defaults(
            self, user_set_prefill, user_set_decode, kv_cache_dtype
        ):
            return

        if not user_set_prefill and not user_set_decode and is_hip():
            self.dsa_prefill_backend = "tilelang"
            self.dsa_decode_backend = "tilelang"
        elif kv_cache_dtype == "fp8_e4m3":
            if major >= 10:
                if not user_set_prefill:
                    self.dsa_prefill_backend = "trtllm"
                if not user_set_decode:
                    self.dsa_decode_backend = "trtllm"
            else:
                # Hopper FP8 defaults to flashmla_kv for both prefill and decode.
                if not user_set_prefill:
                    self.dsa_prefill_backend = "flashmla_kv"
                if not user_set_decode:
                    self.dsa_decode_backend = "flashmla_kv"
        else:
            # set prefill/decode backends based on hardware architecture.
            if major >= 10:
                if not user_set_prefill:
                    self.dsa_prefill_backend = "flashmla_sparse"
                if not user_set_decode:
                    self.dsa_decode_backend = "trtllm"
            else:
                # Hopper defaults for bfloat16
                if not user_set_prefill:
                    self.dsa_prefill_backend = "flashmla_sparse"
                if not user_set_decode:
                    self.dsa_decode_backend = "fa3"

        logger.warning(
            f"Set DSA backends for {self.kv_cache_dtype} KV Cache: prefill={self.dsa_prefill_backend}, decode={self.dsa_decode_backend}."
        )

    def _validate_hisparse_dsa_backend(self, attr: str, label: str):
        from sglang.srt.arg_groups.hisparse_hook import validate_hisparse_dsa_backend

        validate_hisparse_dsa_backend(self, attr, label)

    def _validate_hisparse_kv_cache_dtype(self):
        from sglang.srt.arg_groups.hisparse_hook import validate_hisparse_kv_cache_dtype

        validate_hisparse_kv_cache_dtype(self)

    def _handle_model_specific_adjustments(self):
        from sglang.srt.configs.model_config import (
            get_mimo_v2_fused_qkv_expected_tp_size,
            is_deepseek_dsa,
        )

        self.uses_mamba_radix_cache = False
        if parse_connector_type(self.model_path) == ConnectorType.INSTANCE:
            return

        hf_config = self.get_model_config().hf_config
        model_arch = hf_config.architectures[0]

        _hybrid_spec = get_linear_attn_spec_by_arch(model_arch)
        if _hybrid_spec is not None and _hybrid_spec.uses_mamba_radix_cache:
            self._handle_mamba_radix_cache(model_arch=model_arch)

        if model_arch in [
            "MistralLarge3ForCausalLM",
            "PixtralForConditionalGeneration",
        ]:
            self.dtype = "bfloat16"

        if model_arch in [
            "DeepseekV4ForCausalLM",
        ]:
            from sglang.srt.arg_groups.deepseek_v4_hook import (
                apply_deepseek_v4_defaults,
            )

            apply_deepseek_v4_defaults(self, model_arch)

        if model_arch in [
            "DeepseekV3ForCausalLM",
            "DeepseekV32ForCausalLM",
            "KimiK25ForConditionalGeneration",
            "MistralLarge3ForCausalLM",
            "PixtralForConditionalGeneration",
            "GlmMoeDsaForCausalLM",
        ]:
            # Set attention backend for DeepSeek
            if is_deepseek_dsa(hf_config):  # DeepSeek 3.2/GLM 5
                if envs.SGLANG_DSA_PREFILL_DENSE_ATTN_KV_LEN_THRESHOLD.is_set():
                    logger.warning(
                        f"Dense attention kv len threshold is manually set to {envs.SGLANG_DSA_PREFILL_DENSE_ATTN_KV_LEN_THRESHOLD.get()} for DSA. Caution: This may cause performance regression if the threshold is larger than the index topk of model."
                    )
                else:
                    # When threshold is not manually set, set it to the index topk of model
                    from sglang.srt.configs.model_config import get_dsa_index_topk

                    envs.SGLANG_DSA_PREFILL_DENSE_ATTN_KV_LEN_THRESHOLD.set(
                        get_dsa_index_topk(hf_config)
                    )
                    logger.warning(
                        f"Set dense attention kv len threshold to model index_topk={envs.SGLANG_DSA_PREFILL_DENSE_ATTN_KV_LEN_THRESHOLD.get()} for DeepSeek with DSA."
                    )
                if self.is_attention_backend_not_set():
                    self.attention_backend = "dsa"
                    logger.info("Use dsa attention backend for DeepSeek with DSA.")

                index_topk_freq = getattr(hf_config, "index_topk_freq", 1) or 1
                index_topk_pattern = getattr(hf_config, "index_topk_pattern", None)
                if self.enable_two_batch_overlap and (
                    index_topk_freq > 1
                    or (index_topk_pattern is not None and "S" in index_topk_pattern)
                ):
                    raise ValueError(
                        "--enable-two-batch-overlap is not supported with DSA "
                        "index-topk sharing (index_topk_freq > 1 or an "
                        "index_topk_pattern containing shared layers): the TBO op "
                        "path does not propagate topk indices across layers, so "
                        "shared layers would run sparse attention without indices."
                    )

                if not is_npu() and not is_xpu():  # CUDA or ROCm GPU
                    if self.enable_prefill_cp:
                        logger.warning(
                            "Context parallel feature is still under experiment. It has only been verified on Hopper platform."
                        )
                        self.enable_dp_attention = True
                        self.moe_dense_tp_size = 1
                        if self.cp_strategy == "zigzag":
                            self.moe_a2a_backend = "deepep"
                            self.ep_size = self.tp_size
                            logger.warning(
                                "zigzag DSA CP requires moe_dense_tp_size=1, "
                                "moe_a2a_backend=deepep, ep_size=tp_size, batch_size=1."
                            )
                        else:
                            assert (
                                self.dp_size == 1
                            ), "interleave DSA CP does not support DP attention."
                        assert (
                            self.tp_size <= 8
                        ), "Context parallel only supports single machine (tp_size <= 8). Cross-machine CP has precision issues."
                        # Note(kpham-sgl): Keep attn_tp_size == 1 under DSA CP.
                        # DSACPLayerCommunicator does not all-reduce attention-TP
                        # partial o_proj outputs before replicated dense FFNs.
                        self.attn_cp_size = self.tp_size // self.dp_size
                        self.cuda_graph_config.prefill.backend = Backend.DISABLED
                        logger.warning(
                            "Enabled DSA context parallel: "
                            f"strategy={self.cp_strategy}, dp_size={self.dp_size}, "
                            f"moe_dense_tp_size={self.moe_dense_tp_size}, "
                            f"ep_size={self.ep_size}, tp_size={self.tp_size}, "
                            f"attn_cp_size={self.attn_cp_size}, "
                            f"kv_cache_dtype={self.kv_cache_dtype}, "
                            f"moe_a2a_backend={self.moe_a2a_backend}, "
                            f"cuda_graph_config[prefill].backend=disabled"
                        )
                    else:
                        # Pure TP and partial DP Attention mode is active for DSA, logging a warning
                        if self.dp_size < self.tp_size:
                            logger.warning(
                                f"DSA with TP mode is active, dp_size={self.dp_size}, tp_size={self.tp_size}, "
                                f"attn_tp_size={self.tp_size}, attention weights will be sharded across {self.tp_size} ranks."
                            )

                    # Deferred import to avoid a circular import at module-load
                    # time (dsa.utils imports get_global_server_args).
                    from sglang.srt.layers.attention.dsa.utils import (
                        aiter_can_use_preshuffle_paged_mqa,
                    )

                    if is_hip() and not aiter_can_use_preshuffle_paged_mqa():
                        # Legacy ROCm DSA path: aiter's gluon paged-MQA kernel is
                        # unavailable (Triton<3.5 and AITER_ENABLE_AOT_GLUON_PA_MQA_LOGITS
                        # not set, or SGLANG_DSA_HIP_DISABLE_PRESHUFFLE=1 / SGLANG_USE_AITER=0).
                        self.page_size = 1
                        logger.warning(
                            "Setting page size to 1 for DeepSeek DSA on ROCm "
                            "(aiter preshuffle paged-MQA path unavailable: "
                            "needs Triton>=3.5.0 or AITER_ENABLE_AOT_GLUON_PA_MQA_LOGITS=1)."
                        )
                    else:
                        self.page_size = 64
                        logger.warning("Setting page size to 64 for DeepSeek DSA.")

                    import torch

                    major, _ = torch.cuda.get_device_capability()
                    self._set_default_dsa_kv_cache_dtype(major, self.quantization)
                    self._set_default_dsa_backends(self.kv_cache_dtype, major)

                if self.enable_prefill_cp:
                    assert (
                        self.disaggregation_mode != "decode"
                    ), "CP is only supported for prefill when PD disaggregation, please remove --enable-prefill-cp."

            else:
                # DeepSeek V3/R1/V3.1
                if self.cuda_graph_config.prefill.backend != Backend.DISABLED:
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

                # MLA prefill CP auto-config. Mirrors the NSA CP block above
                # (minus the in-seq/round-robin mode split, which MLA CP does not support)
                if self.enable_prefill_cp and self.use_mla_backend():
                    logger.warning(
                        "MLA prefill context parallel is still experimental. "
                        "Verified on Hopper with the fa3 backend."
                    )
                    self.enable_dp_attention = True
                    # TODO(kpham-sgl) Supports moe_dense_tp_size != 1.
                    self.moe_dense_tp_size = 1
                    self.moe_a2a_backend = "deepep"
                    self.ep_size = self.tp_size
                    logger.warning(
                        "For MLA CP, we have the following restrictions: moe_dense_tp_size == 1, moe_a2a_backend == deepep, ep_size == tp_size, batch_size == 1"
                    )
                    # FIXME(kpham-sgl): Keep attn_tp_size == 1 under MLA CP.
                    # DSACPLayerCommunicator does not all-reduce attention-TP
                    # partial o_proj outputs before replicated dense FFNs.
                    self.attn_cp_size = self.tp_size // self.dp_size
                    self.cuda_graph_config.prefill.backend = Backend.DISABLED
                    logger.warning(
                        f"Enable Context Parallel opt for MLA, "
                        f"Setting dp_size == {self.dp_size} and "
                        f"attn_cp_size == {self.attn_cp_size}, "
                        f"moe_dense_tp_size == {self.moe_dense_tp_size}, "
                        f"ep_size == {self.ep_size}, "
                        f"tp_size == {self.tp_size}, "
                        f"moe_a2a_backend {self.moe_a2a_backend}, "
                        f"cuda_graph_config[prefill].backend=disabled"
                    )

            # Set moe backend for DeepSeek
            if is_sm100_supported():
                quant_method = get_quantization_config(hf_config)
                quant_cfg = getattr(hf_config, "quantization_config", None) or {}
                config_groups = quant_cfg.get("config_groups", {})
                group0 = config_groups.get("group_0", {})
                weights_cfg = group0.get("weights", {})
                # this also apply to kimi k2.5
                # since it follow the compressed tensor int4 recipe
                # but not kimi k2 instruct or 0905 instruct.
                is_kimi_k2_k25_thinking_int4 = (
                    quant_method == "compressed-tensors"
                    and weights_cfg.get("num_bits") == 4
                    and weights_cfg.get("group_size") == 32
                    and weights_cfg.get("strategy") == "group"
                    and weights_cfg.get("type") == "int"
                )
                if (
                    self.quantization is None
                    and not self._quantization_explicitly_unset
                ):
                    # DeepSeek V3/R1 uses native FP8 MoE experts without
                    # declaring it in quantization_config.  However, other
                    # models that share the same architecture class (e.g.
                    # Moonlight-16B-A3B) are purely BF16.  Check the actual
                    # safetensors header instead of assuming FP8 by arch name.
                    if quant_method is None and model_arch in ["DeepseekV3ForCausalLM"]:
                        if has_fp8_weights_in_checkpoint(self.model_path):
                            self.quantization = "fp8"
                            logger.info(
                                "Detected FP8 expert weights in checkpoint, "
                                "default to fp8 for DeepSeek on sm100"
                            )
                        else:
                            logger.info(
                                "No FP8 expert weights found in checkpoint, "
                                "keeping bf16 for DeepSeek-arch model on sm100"
                            )
                    else:
                        self.quantization = quant_method
                if (
                    self.moe_a2a_backend == "none"
                    and self.moe_runner_backend == "auto"
                    and (
                        self.quantization
                        in ["fp8", "modelopt_fp8", "modelopt_fp4", "modelopt_mixed"]
                        or is_kimi_k2_k25_thinking_int4
                        or self.quantization is None
                    )
                ):
                    self.moe_runner_backend = "flashinfer_trtllm"
                    if is_kimi_k2_k25_thinking_int4:
                        logger.info(
                            "Use flashinfer_trtllm as MoE runner backend on Blackwell for Kimi K2 / K2.5 thinking int4"
                        )
                    else:
                        logger.info(
                            "Use flashinfer_trtllm as MoE runner backend on sm100 for DeepseekV3ForCausalLM"
                        )
            elif is_hip():
                if not self.enable_dp_attention and self.nnodes == 1:
                    # TODO (Hubert): Put this back later
                    # self.enable_aiter_allreduce_fusion = True
                    logger.info(
                        "Enable Aiter AllReduce Fusion for DeepseekV3ForCausalLM"
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
                        # Validate usage of ep
                        if self.ep_size == 1:
                            raise ValueError(
                                "Invalid configuration: 'deep_gemm' speculative MoE runner backend with "
                                "'deepep' a2a backend requires expert parallelism (ep_size > 1). "
                                f"Current ep_size is {self.ep_size}. "
                                "Please set --ep-size > 1 (e.g., --ep-size 8) to use this configuration, "
                                "or change --speculative-moe-a2a-backend to 'none' if expert parallelism is not available."
                            )
                    else:
                        self.speculative_moe_runner_backend = "triton"
                        self.speculative_moe_a2a_backend = "none"
                        logger.info(
                            "Use triton fused moe by default for bf16 nextn layer in deepseek fp4 checkpoint."
                        )

        elif model_arch in [
            "DeepseekV4ForCausalLM",
        ]:
            from sglang.srt.arg_groups.deepseek_v4_hook import validate_deepseek_v4_cp

            validate_deepseek_v4_cp(self)

            if is_sm120_supported():
                if self.moe_runner_backend == "auto":
                    self.moe_runner_backend = "marlin"
                    logger.info(
                        "Use marlin as MoE runner backend on SM120 for DeepseekV4"
                    )
                # SM120 lacks tcgen05/TMEM: disable features that depend on
                # DeepGEMM or require >99KB SMEM (topk_v2).
                envs.SGLANG_OPT_FP8_WO_A_GEMM.set(False)
                envs.SGLANG_OPT_USE_TOPK_V2.set(False)
                envs.SGLANG_OPT_USE_TILELANG_MHC_PRE.set(False)
                envs.SGLANG_OPT_DEEPGEMM_HC_PRENORM.set(False)
                envs.SGLANG_FP8_PAGED_MQA_LOGITS_TORCH.set(True)
            elif is_hip():
                envs.SGLANG_OPT_DEEPGEMM_HC_PRENORM.set(False)
                envs.SGLANG_OPT_USE_FUSED_COMPRESS.set(True)
                envs.SGLANG_OPT_FP8_WO_A_GEMM.set(False)
                envs.SGLANG_OPT_USE_JIT_INDEXER_METADATA.set(False)
                envs.SGLANG_OPT_USE_TOPK_V2.set(False)
                envs.SGLANG_OPT_USE_AITER_INDEXER.set(True)
                envs.SGLANG_OPT_USE_TILELANG_MHC_PRE.set(False)
                envs.SGLANG_OPT_USE_TILELANG_MHC_POST.set(False)
                envs.SGLANG_FP8_PAGED_MQA_LOGITS_TORCH.set(True)
                envs.SGLANG_OPT_USE_MULTI_STREAM_OVERLAP.set(False)
                envs.SGLANG_EAGER_INPUT_NO_COPY.set(True)

        elif model_arch in ["GptOssForCausalLM"]:
            # Set attention backend for GPT-OSS
            if self.is_attention_backend_not_set():
                if is_sm100_supported():
                    self.attention_backend = "trtllm_mha"
                elif is_sm90_supported():
                    self.attention_backend = "fa3"
                elif is_cpu() and cpu_has_amx_support():
                    self.attention_backend = "intel_amx"
                elif is_xpu():
                    self.attention_backend = "intel_xpu"
                elif is_hip():
                    self.attention_backend = "aiter"
                else:
                    self.attention_backend = "triton"

            if is_xpu():
                # Check for bf16 dtype on Intel XPU
                if self.dtype == "auto":
                    logger.warning(
                        "GptOssForCausalLM on Intel XPU currently supports bfloat16 dtype only"
                    )
                elif self.dtype not in ["bfloat16"]:
                    raise NotImplementedError(
                        f"GptOssForCausalLM on Intel XPU only supports bfloat16 dtype, "
                        f"but got '{self.dtype}'. Please use --dtype bfloat16 or remove --dtype to use auto."
                    )

            supported_backends = [
                "triton",
                "trtllm_mha",
                "fa3",
                "fa4",
                "ascend",
                "intel_amx",
                "intel_xpu",
                "aiter",
            ]
            prefill_attn_backend, decode_attn_backend = self.get_attention_backends()
            assert (
                prefill_attn_backend in supported_backends
                and decode_attn_backend in supported_backends
            ), (
                f"GptOssForCausalLM requires one of {supported_backends} attention backend, but got the following backends\n"
                f"- Prefill: {prefill_attn_backend}\n"
                f"- Decode: {decode_attn_backend}\n"
            )

            quant_method = get_quantization_config(hf_config)
            is_mxfp4_quant_format = quant_method == "mxfp4"
            if not self.enable_dp_attention and self.nnodes == 1 and is_hip():
                # TODO (Hubert): Put this back later
                # self.enable_aiter_allreduce_fusion = True
                logger.info("Enable Aiter AllReduce Fusion for GptOssForCausalLM")
            quantization_config = getattr(hf_config, "quantization_config", None)
            is_mxfp4_quant_format = (
                quantization_config is not None
                and quantization_config.get("quant_method") == "mxfp4"
            )
            if is_mxfp4_quant_format:
                # use bf16 for mxfp4 triton kernels
                self.dtype = "bfloat16"

            if self.moe_runner_backend == "auto":
                if is_sm100_supported() and is_mxfp4_quant_format:
                    self.moe_runner_backend = "flashinfer_mxfp4"
                    logger.warning(
                        "Detected SM100 and MXFP4 quantization format for GPT-OSS model, enabling FlashInfer MXFP4 MOE kernel."
                    )
                elif is_sm120_supported() and is_mxfp4_quant_format:
                    # trtllm-gen only supports SM100
                    self.moe_runner_backend = "marlin"
                    logger.warning(
                        "Detected SM120 and MXFP4 quantization format for GPT-OSS model, enabling Marlin MOE kernel."
                    )
                elif (
                    is_hip() and envs.SGLANG_USE_AITER.get()
                ) and is_mxfp4_quant_format:
                    self.moe_runner_backend = "auto"
                    logger.warning(
                        "Detected ROCm and MXFP4 quantization format for GPT-OSS model, enabling aiter MXFP4 MOE kernel."
                    )
                    ## The AITER MXFP4 fused-MoE path for GPT-OSS expects the
                    ## SEPARATED gate/up tile layout (matches the
                    ## `gptoss_fp4_tuned_fmoe.csv` flydsl entries and the
                    ## Mxfp4MoEMethod weight shuffle). Other AITER MXFP4
                    ## callers default to INTERLEAVE; opt this path out
                    ## unless the user explicitly overrode it.
                    # envs.SGLANG_USE_AITER_MOE_GU_ITLV.set(False)
                elif is_hip() and envs.SGLANG_USE_AITER.get():
                    # For GPT-OSS bf16 on ROCm with aiter, use triton backend
                    # because aiter CK kernel doesn't support all GEMM dimensions
                    self.moe_runner_backend = "triton"
                    logger.warning(
                        "Detected ROCm with SGLANG_USE_AITER for GPT-OSS bf16 model, using triton MOE kernel."
                    )
                elif is_musa() and envs.SGLANG_DEEPEP_BF16_DISPATCH.get():
                    self.moe_runner_backend = "deep_gemm"
                    logger.warning(
                        "Detected MUSA with SGLANG_DEEPEP_BF16_DISPATCH for bf16 model, using deep_gemm kernel."
                    )
                elif (
                    self.ep_size == 1
                    and is_triton_kernels_available()
                    and self.quantization is None
                    and not (is_cpu() and cpu_has_amx_support())
                ):
                    # The triton_kernels package segfaults on Blackwell (B200)
                    # with NVIDIA driver >= 595. Fall back to triton backend.
                    if is_blackwell_supported() and get_nvidia_driver_version() >= (
                        595,
                    ):
                        self.moe_runner_backend = "triton"
                        logger.warning(
                            "Detected GPT-OSS model on Blackwell with driver >= 595, "
                            "using triton MOE kernel to avoid triton_kernels SIGSEGV."
                        )
                    else:
                        self.moe_runner_backend = "triton_kernel"
                        logger.warning(
                            "Detected GPT-OSS model, enabling triton_kernels MOE kernel."
                        )

            if self.moe_runner_backend == "triton_kernel":
                assert (
                    self.ep_size == 1
                ), "Triton kernel MoE is only supported when ep_size == 1"

        elif model_arch in MIMO_V2_MODEL_ARCHS:
            if model_arch == "MiMoV2ForCausalLM" and not self.encoder_only:
                expected_attn_tp_size = get_mimo_v2_fused_qkv_expected_tp_size(
                    hf_config
                )
                attn_dp_size = self.dp_size if self.enable_dp_attention else 1
                effective_attn_tp_size = (
                    self.tp_size // attn_dp_size // self.attn_cp_size
                )
                if (
                    expected_attn_tp_size is not None
                    and effective_attn_tp_size != expected_attn_tp_size
                ):
                    raise ValueError(
                        "MiMoV2ForCausalLM requires effective attention TP "
                        f"size {expected_attn_tp_size} because its fused "
                        "qkv_proj weights are "
                        f"TP={expected_attn_tp_size}-interleaved; got "
                        f"{effective_attn_tp_size} "
                        f"(tp_size={self.tp_size}, dp_size={self.dp_size}, "
                        f"enable_dp_attention={self.enable_dp_attention}, "
                        f"attn_cp_size={self.attn_cp_size}). "
                        "Set --tp, --dp, --enable-dp-attention, and "
                        "--attention-context-parallel-size so the effective "
                        f"attention TP size is {expected_attn_tp_size}."
                    )

            if self.speculative_algorithm == "EAGLE":
                self.enable_multi_layer_eagle = True
                logger.info(
                    "Enable multi-layer EAGLE speculative decoding for MiMoV2 model."
                )

            if self.enable_hierarchical_cache:
                if not envs.SGLANG_ENABLE_UNIFIED_RADIX_TREE.get():
                    raise ValueError(
                        "Hierarchical cache for MiMoV2 requires the unified "
                        "radix tree. Set SGLANG_ENABLE_UNIFIED_RADIX_TREE=1 "
                        "to enable --enable-hierarchical-cache for this model."
                    )

                # MiMoV2 has head_dim != v_head_dim, so the host KV pool uses
                # asymmetric K/V allocation. Both kernel/page_first and
                # direct/page_first_direct have split K/V transfer paths.
        elif (
            "Step3p5ForCausalLM" in model_arch
            or "Step3p7ForConditionalGeneration" in model_arch
        ):
            if self.is_attention_backend_not_set():
                if is_blackwell_supported():
                    self.attention_backend = "fa4"
                    logger.info(
                        "Auto-select fa4 attention backend for Step3p7 on Blackwell."
                    )
                elif is_sm90_supported():
                    self.attention_backend = "fa3"
                    logger.info(
                        "Auto-select fa3 attention backend for Step3p7 on Hopper."
                    )
            if self.speculative_algorithm == "EAGLE":
                self.enable_multi_layer_eagle = True
                logger.info(
                    "Enable multi-layer EAGLE speculative decoding for Step3p5ForCausalLM model."
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
        elif model_arch in LLAMA4_MODEL_ARCHS and self.device != "cpu":
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
                "ascend",
                "trtllm_mha",
                "intel_xpu",
            }, f"fa3, aiter, triton, ascend, trtllm_mha or intel_xpu is required for Llama4 model but got {self.attention_backend}"
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
        elif model_arch in (
            "Gemma4ForConditionalGeneration",
            "Gemma4ForCausalLM",
            "Gemma4UnifiedForConditionalGeneration",
        ):
            default_attention_backend = (
                "trtllm_mha" if is_sm100_supported() else "triton"
            )
            if self.is_attention_backend_not_set():
                self.attention_backend = default_attention_backend
                logger.info(
                    f"Use {self.attention_backend} as default attention backend for Gemma4"
                )
            else:
                # If only one split backend is set, keep the other side on a
                # Gemma4-compatible fallback instead of letting generic backend
                # selection choose an unsupported backend later.
                if self.attention_backend is None:
                    self.attention_backend = default_attention_backend

            prefill_backend, decode_backend = self.get_attention_backends()
            accepted_backends = ("trtllm_mha", "triton", "ascend", "intel_xpu")
            assert (
                prefill_backend in accepted_backends
                and decode_backend in accepted_backends
            ), (
                "Gemma4 only supports trtllm_mha, triton, or intel_xpu attention backend, "
                f"got prefill={prefill_backend}, decode={decode_backend}"
            )

            if is_sm100_supported() and self.moe_runner_backend == "auto":
                if self.get_model_config().quantization == "modelopt_fp4":
                    self.quantization = "modelopt_fp4"
                    self.moe_runner_backend = "flashinfer_trtllm"
                    logger.info(
                        "Use flashinfer_trtllm as MoE runner backend on "
                        "SM100 for Gemma-4 (modelopt_fp4)"
                    )
        elif model_arch == "MossVLForConditionalGeneration":
            if self.is_attention_backend_not_set():
                self.prefill_attention_backend = "flashinfer"
                logger.info(
                    "Use flashinfer as default prefill attention backend for Moss-VL"
                )
            prefill_backend, _ = self.get_attention_backends()
            assert prefill_backend == "flashinfer", (
                "MossVLForConditionalGeneration requires flashinfer prefill "
                "attention backend for cross-attention custom mask support."
            )
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
            self._handle_mamba_radix_cache(model_arch=model_arch)
        elif model_arch in ["BailingMoeV2_5ForCausalLM"]:
            self._handle_mamba_radix_cache(model_arch=model_arch)
        elif model_arch in ["NemotronHForCausalLM", "NemotronHPuzzleForCausalLM"]:
            from sglang.srt.arg_groups.nemotron_h_hook import (
                apply_nemotron_h_defaults,
            )

            apply_nemotron_h_defaults(self, model_arch)
        elif model_arch in [
            "Qwen3MoeForCausalLM",
            "Qwen3VLMoeForConditionalGeneration",
            "Qwen3NextForCausalLM",
            "Qwen3_5MoeForConditionalGeneration",
            "InternS2PreviewForConditionalGeneration",
            "Qwen3_5ForConditionalGeneration",
        ]:
            if is_sm100_supported():
                quant_method = get_quantization_config(hf_config)
                if (
                    self.quantization is None
                    and not self._quantization_explicitly_unset
                    and quant_method is not None
                ):
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

            if model_arch in [
                "Qwen3NextForCausalLM",
                "Qwen3_5MoeForConditionalGeneration",
                "InternS2PreviewForConditionalGeneration",
                "Qwen3_5ForConditionalGeneration",
            ]:
                sm100_default_attn_backend = "triton"
                if is_sm100_supported():
                    # trtllm_mha requires speculative_eagle_topk == 1 and page_size > 1.
                    # _get_default_attn_backend handles the eagle_topk check.
                    # There is only one case where page_size=1 is required,
                    # which is when radix cache is enabled and both extra_buffer
                    # and spec decoding are disabled.
                    default_attn_backend = self._get_default_attn_backend(
                        use_mla_backend=self.use_mla_backend(),
                        model_config=self.get_model_config(),
                    )
                    if default_attn_backend == "trtllm_mha" and not (
                        not self.enable_mamba_extra_buffer()
                        and not self.disable_radix_cache
                        and self.speculative_algorithm is None
                    ):
                        sm100_default_attn_backend = "trtllm_mha"

                    if self.attention_backend is None:
                        self.attention_backend = sm100_default_attn_backend
                        self.page_size = (
                            64 if sm100_default_attn_backend == "trtllm_mha" else 1
                        )

                self._handle_mamba_radix_cache(model_arch=model_arch)

        elif model_arch == "MiniCPMV4_6ForConditionalGeneration":
            # 4.6 wraps a Qwen3.5 hybrid GDN backbone, so it needs the same
            # mamba radix cache handling as Qwen3_5ForConditionalGeneration.
            if is_sm100_supported() and self.attention_backend is None:
                self.attention_backend = "triton"
            self._handle_mamba_radix_cache(model_arch=model_arch)

        elif model_arch in ["Glm4MoeForCausalLM"]:
            if is_sm100_supported():
                quantization_config = getattr(hf_config, "quantization_config", None)
                quant_method = (
                    quantization_config.get("quant_method")
                    if quantization_config is not None
                    else None
                )
                if (
                    self.quantization is None
                    and not self._quantization_explicitly_unset
                    and quant_method is not None
                ):
                    self.quantization = quant_method
                if (
                    self.quantization in {"modelopt_fp4", None}
                    and self.moe_a2a_backend == "none"
                    and self.moe_runner_backend == "auto"
                ):
                    self.moe_runner_backend = "flashinfer_trtllm"
                    logger.info(
                        "Use flashinfer_trtllm as MoE runner backend on sm100 for Glm4MoeForCausalLM"
                    )
            self.enable_tf32_matmul = True
            logger.info(
                "Enable TF32 matmul for Glm4MoeForCausalLM model to improve gate gemm performance."
            )

        elif model_arch in [
            "FalconH1ForCausalLM",
            "JetNemotronForCausalLM",
            "JetVLMForConditionalGeneration",
        ]:
            if is_sm100_supported() and self.attention_backend is None:
                self.attention_backend = "triton"
            self._handle_mamba_radix_cache(model_arch=model_arch)

        elif model_arch == "GraniteMoeHybridForCausalLM":
            hf_config = self.get_model_config().hf_config
            has_mamba = any(
                layer_type == "mamba"
                for layer_type in getattr(hf_config, "layer_types", [])
            )
            if has_mamba:
                if is_sm100_supported() and self.attention_backend is None:
                    self.attention_backend = "flashinfer"
                self._handle_mamba_radix_cache(model_arch=model_arch)

        elif model_arch in ["Lfm2ForCausalLM"]:
            if is_sm100_supported() and self.attention_backend is None:
                self.attention_backend = "flashinfer"
            self._handle_mamba_radix_cache(model_arch=model_arch)
            assert self.attention_backend != "triton", (
                f"{model_arch} does not support triton attention backend, "
                "as the first layer might not be an attention layer"
            )

        elif model_arch in ["ZayaForCausalLM"]:
            self._handle_mamba_radix_cache(model_arch=model_arch)

        elif model_arch in ["MiniMaxM2ForCausalLM"]:
            self.enable_tf32_matmul = True
            logger.info(
                "Enable TF32 matmul for MiniMaxM2ForCausalLM model to improve gate gemm performance."
            )

        if (
            model_arch in ["Qwen3VLForConditionalGeneration"]
            and is_hip()
            and envs.SGLANG_USE_AITER_UNIFIED_ATTN.get()
            and self.page_size is None
        ):
            self.page_size = 16
            logger.info(
                "Setting page_size=16 for aiter unified attention on Qwen3VLForConditionalGeneration."
            )

        if envs.SGLANG_EMBEDDINGS_SPARSE_HEAD.is_set():
            self.disable_overlap_schedule = True
            logger.warning(
                "Overlap scheduler is disabled when using sparse head for embedding model."
            )

        # Auto-enable FlashInfer AllReduce Fusion on SM90/SM100, for models with
        # explicit support (DeepseekV3, GptOss, Glm4Moe, MistralLarge3,
        # Qwen3/Qwen3-VL/Qwen3Next/Qwen3.5 MoE families). auto resolves to mnnvl on
        # Blackwell (single- and multi-node) and trtllm on SM90 single-node systems.
        if (
            self.flashinfer_allreduce_fusion_backend is None
            and model_arch
            in [
                "DeepseekV3ForCausalLM",
                "DeepseekV32ForCausalLM",
                "GptOssForCausalLM",
                "GlmMoeDsaForCausalLM",
                "Glm4MoeForCausalLM",
                "Glm4MoeLiteForCausalLM",
                "MistralLarge3ForCausalLM",
                "Qwen3MoeForCausalLM",
                "Qwen3VLMoeForConditionalGeneration",
                "Qwen3NextForCausalLM",
                "KimiK25ForConditionalGeneration",
                "Qwen3_5MoeForConditionalGeneration",
                "InternS2PreviewForConditionalGeneration",
                "Qwen3_5ForConditionalGeneration",
                "NemotronHForCausalLM",
                "NemotronHPuzzleForCausalLM",
            ]
            and (is_sm90_supported() or is_sm100_supported())
            and self.tp_size > 1
            and not self.enable_dp_attention
            and (self.nnodes == 1 or is_sm100_supported())
            and self.moe_a2a_backend == "none"
        ):
            self.flashinfer_allreduce_fusion_backend = "auto"
            logger.info(
                f"Auto-enabling FlashInfer AllReduce Fusion on SM90/SM10X for {model_arch}"
            )

        # Apply enforce_disable_flashinfer_allreduce_fusion after all model-specific adjustments
        if self.enforce_disable_flashinfer_allreduce_fusion:
            self.flashinfer_allreduce_fusion_backend = None
            logger.info(
                "FlashInfer allreduce fusion is forcibly disabled "
                "via --enforce-disable-flashinfer-allreduce-fusion."
            )

    def _support_mamba_cache_extra_buffer(self, model_arch: str):
        if model_arch in [
            "Qwen3_5ForConditionalGeneration",
            "Qwen3_5MoeForConditionalGeneration",
            "Qwen3NextForCausalLM",
            "InternS2PreviewForConditionalGeneration",
            "MiniCPMV4_6ForConditionalGeneration",
            "BailingMoeV2_5ForCausalLM",
            "FalconH1ForCausalLM",
            "GraniteMoeHybridForCausalLM",
            "NemotronHForCausalLM",
            "NemotronHPuzzleForCausalLM",
        ]:
            return self.linear_attn_backend == "triton"

        return False

    def _validate_mamba_no_buffer(self, model_arch: str):
        assert self.page_size in (1, None), "no_buffer only supports page_size=1."
        assert (
            self.disable_overlap_schedule
        ), "no_buffer do not support overlap schedule. Try to set disable_overlap_schedule=True."
        assert (
            self.attention_backend != "trtllm_mha"
        ), "no_buffer do not support trtllm_mha attention backend."

    def _validate_mamba_extra_buffer(self, model_arch: str):
        assert self._support_mamba_cache_extra_buffer(
            model_arch
        ), f"extra_buffer is not supported for {model_arch}; use no_buffer."
        assert (
            is_cuda() or is_musa() or is_npu()
        ), "extra_buffer needs CUDA/MUSA/NPU (FLA)."
        if self.speculative_num_draft_tokens is not None:
            assert (
                not self.enable_mamba_extra_buffer_lazy()
            ), "extra_buffer_lazy unsupported with spec."
            assert self.mamba_track_interval >= self.speculative_num_draft_tokens
        if self.page_size is not None:
            assert self.mamba_track_interval % self.page_size == 0
            assert self.mamba_cache_chunk_size is not None

    def _handle_mamba_radix_cache(self, model_arch: str):
        if self.disable_radix_cache:
            return

        self.uses_mamba_radix_cache = True
        if self.mamba_radix_cache_strategy == "auto":
            wants_overlap = not self.disable_overlap_schedule
            wants_paging = self.page_size is not None and self.page_size > 1
            if (
                wants_overlap or wants_paging
            ) and self._support_mamba_cache_extra_buffer(model_arch):
                self.mamba_radix_cache_strategy = "extra_buffer"
            else:
                self.mamba_radix_cache_strategy = "no_buffer"
                self.disable_overlap_schedule = True

        if self.enable_mamba_extra_buffer():
            self._validate_mamba_extra_buffer(model_arch)
        else:
            self._validate_mamba_no_buffer(model_arch)

    def _handle_sampling_backend(self):
        if self.sampling_backend is None:
            self.sampling_backend = (
                "flashinfer" if is_flashinfer_available() else "pytorch"
            )

    def _get_default_attn_backend(self, use_mla_backend: bool, model_config):
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
        # Non-CUDA platforms can provide their own default attention backend.
        if current_platform.is_out_of_tree() or current_platform.is_mlu():
            return current_platform.get_default_attention_backend()

        # Whisper requires flashinfer for cross-attention CUDA graph support.
        if "WhisperForConditionalGeneration" in (
            model_config.hf_config.architectures or []
        ):
            return "flashinfer"

        if not use_mla_backend:
            # MHA architecture
            if is_hopper_with_cuda_12_3() and is_no_spec_infer_or_topk_one(self):
                # Note: flashinfer 0.6.1 caused performance regression on Hopper attention kernel
                # Before the kernel is fixed, we choose fa3 as the default backend on Hopper MHA
                # ref: https://github.com/sgl-project/sglang/issues/17411
                return "fa3"
            elif (
                is_sm100_supported()
                and is_no_spec_infer_or_topk_one(self)
                and (
                    self.speculative_algorithm is None
                    or self.speculative_eagle_topk is not None
                )
            ):
                return "trtllm_mha"
            elif is_hip():
                return "aiter"
            elif is_mps():
                return "torch_native"
            else:
                # FlashInfer does not support attention sinks.
                if is_flashinfer_available() and not model_config.has_attention_sinks:
                    return "flashinfer"
                return "triton"
        else:
            # MLA architecture
            if is_hopper_with_cuda_12_3():
                return "fa3"
            elif is_sm100_supported():
                return "flashinfer"
            elif is_hip():
                head_num = model_config.get_num_kv_heads(self.tp_size)
                # TODO current aiter only support head number 16 or 128 head number
                if head_num == 128 or head_num == 16:
                    return "aiter"
                else:
                    return "triton"
            elif is_mps():
                return "torch_native"
            else:
                return "triton"

    def _handle_attention_backend_compatibility(self):
        model_config = self.get_model_config()
        use_mla_backend = self.use_mla_backend()

        if self.prefill_attention_backend is not None and (
            self.prefill_attention_backend == self.decode_attention_backend
        ):  # override the default attention backend
            self.attention_backend = self.prefill_attention_backend

        # Pick the default attention backend if not specified
        if self.attention_backend is None:
            self.attention_backend = self._get_default_attn_backend(
                use_mla_backend, model_config
            )

            logger.info(
                f"Attention backend not specified. Use {self.attention_backend} backend by default."
            )

        # Torch native and flex attention backends
        if self.attention_backend == "torch_native":
            logger.warning(
                "Cuda graph is disabled because of using torch native attention backend"
            )
            self.cuda_graph_config.decode.backend = Backend.DISABLED
            self.cuda_graph_config.prefill.backend = Backend.DISABLED

        if self.attention_backend == "flex_attention":
            logger.warning(
                "Cuda graph is disabled because of using torch Flex Attention backend"
            )
            self.cuda_graph_config.decode.backend = Backend.DISABLED
            self.cuda_graph_config.prefill.backend = Backend.DISABLED
            assert (
                self.speculative_algorithm is None
            ), "Speculative decoding is currently not supported with Flex Attention backend"

        # Whisper's encoder token padding conflicts with prefix caching.
        # Only disable for Whisper; other encoder-decoder models (e.g., mllama) use radix cache.
        if (
            model_config.is_encoder_decoder
            and not self.disable_radix_cache
            and "WhisperForConditionalGeneration"
            in (model_config.hf_config.architectures or [])
        ):
            logger.info("Radix cache is disabled for Whisper")
            self.disable_radix_cache = True

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
                    "TRTLLM MLA backend is only supported on Blackwell GPUs (SM100/SM12x). Please use a different backend."
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
            self.attention_backend == "tokenspeed_mla"
            or self.decode_attention_backend == "tokenspeed_mla"
        ):
            if not is_blackwell_supported():
                raise ValueError(
                    "tokenspeed_mla backend is only supported on Blackwell GPUs (SM100/SM12x)."
                )
            if self.page_size not in [32, 64]:
                logger.warning(
                    f"tokenspeed_mla only supports page_size of 32 or 64, changing page_size from {self.page_size} to 64."
                )
                self.page_size = 64
            if self.kv_cache_dtype not in ["fp8_e4m3"]:
                raise ValueError(
                    "tokenspeed_mla backend requires kv-cache-dtype=fp8_e4m3, "
                    f"got {self.kv_cache_dtype}."
                )

        if (
            self.attention_backend == "cutedsl_mla"
            or self.decode_attention_backend == "cutedsl_mla"
            or self.prefill_attention_backend == "cutedsl_mla"
        ):
            assert (
                self.prefill_attention_backend != "cutedsl_mla"
            ), "CuteDSL MLA only supports decoding for now"
            if not is_sm100_supported():
                raise ValueError(
                    "CuteDSL MLA backend is only supported on Blackwell GPUs (SM100). Please use a different backend."
                )
            if self.page_size not in [32, 64]:
                logger.warning(
                    f"CuteDSL MLA only supports page_size of 32 or 64, changing page_size from {self.page_size} to 64."
                )
                self.page_size = 64
            if self.kv_cache_dtype not in [
                "fp8_e4m3",
                "bf16",
                "bfloat16",
                "auto",
            ]:
                raise ValueError(
                    "CuteDSL MLA backend only supports kv-cache-dtype of fp8_e4m3, bf16, or auto."
                )
            if self.prefill_attention_backend is None:
                self.prefill_attention_backend = "trtllm_mla"

        if (
            self.attention_backend == "trtllm_mha"
            or self.decode_attention_backend == "trtllm_mha"
            or self.prefill_attention_backend == "trtllm_mha"
        ):
            # Check prefill backend
            prefill_backend = (
                self.prefill_attention_backend
                if self.prefill_attention_backend is not None
                else self.attention_backend
            )
            if prefill_backend == "trtllm_mha" and not is_sm100_supported():
                raise ValueError(
                    "TRTLLM MHA backend for prefill is only supported on Blackwell GPUs (SM100). Please use a different prefill backend."
                )

            # Check decode backend
            decode_backend = (
                self.decode_attention_backend
                if self.decode_attention_backend is not None
                else self.attention_backend
            )
            if decode_backend == "trtllm_mha" and not (
                is_sm90_supported() or is_sm100_supported() or is_sm120_supported()
            ):
                raise ValueError(
                    "TRTLLM MHA backend for decode is only supported on Hopper (SM90), Blackwell (SM100) and (SM120) GPUs. Please use a different decode backend."
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

        if (
            (
                self.attention_backend == "fa4"
                or self.decode_attention_backend == "fa4"
                or self.prefill_attention_backend == "fa4"
            )
            and not self.use_mla_backend()
            and is_sm100_supported()
            # EAGLE topk>1 spec runs the two-pass page-tree cascade, which the FA4
            # CUTLASS kernel aborts on at page_size>1. That path only works at
            # page_size==1, so skip the 128 auto-force for it and keep the default.
            and (self.speculative_eagle_topk or 0) <= 1
        ):
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

        prefill_backend, decode_backend = self.get_attention_backends()
        if self.use_mla_backend() and prefill_backend == "intel_xpu":
            raise ValueError(
                "intel_xpu backend is only supported on decode for MLA models, please set --decode-attention-backend to intel_xpu and do not set --attention-backend or --prefill-attention-backend to intel_xpu for prefill instead use triton."
            )

        if decode_backend == "intel_xpu":
            if self.use_mla_backend():
                supported_page_sizes = [16, 32, 64, 128]
                msg = "Intel XPU attention backend for MLA Decode"
            else:
                supported_page_sizes = [64, 128]
                msg = "Intel XPU attention backend"

            if self.page_size not in supported_page_sizes:
                logger.warning(
                    f"{msg} only supports page_sizes of {supported_page_sizes}, changing page_size from {self.page_size} to 128."
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
            # SHUFFLE 5D vectorized KV layout (aiter backend + pa_decode_gluon)
            # is tuned for and prefers page_size=64 — making it the default
            # when the layout flag is set avoids users having to pass
            # --page-size 64 explicitly. The env var is only consumed by the
            # ROCm AITER backend, so the auto-bump is gated on HIP; on other
            # platforms the SHUFFLE 5D pool has no consumer kernels and the
            # env var is silently ignored (see MHATokenToKVPool).
            if (
                is_hip()
                and envs.SGLANG_AITER_KV_CACHE_LAYOUT.get().lower() == "vectorized_5d"
            ):
                self.page_size = 64
                logger.info(
                    "Setting page_size=64 as default for "
                    "SGLANG_AITER_KV_CACHE_LAYOUT=vectorized_5d."
                )
            elif not is_musa():
                self.page_size = 1
            else:
                self.page_size = 64

    def _handle_amd_specifics(self):
        if is_hip():
            self.triton_attention_num_kv_splits = 16

    def _handle_nccl_pre_warm(self):
        # pre_warm_nccl is only used with CUDA or HIP hardware
        if self.pre_warm_nccl and not (is_cuda() or is_hip()):
            logger.warning(
                "pre_warm_nccl is only applicable for CUDA or HIP hardware. "
                "Ignoring pre_warm_nccl setting on current hardware."
            )
            self.pre_warm_nccl = False

    def _handle_grammar_backend(self):
        if self.grammar_backend is None:
            self.grammar_backend = "xgrammar"

    def _handle_mamba_backend(self):
        if self.mamba_cache_philox_rounds < 0:
            raise ValueError("--mamba-cache-philox-rounds must be non-negative.")

        if self.enable_mamba_cache_stochastic_rounding:
            if self.mamba_ssm_dtype != "float16":
                raise ValueError(
                    "Stochastic rounding for the Mamba SSM cache requires "
                    f"--mamba-ssm-dtype float16, got {self.mamba_ssm_dtype!r}. "
                    "Run with --mamba-ssm-dtype float16 or disable "
                    "--enable-mamba-cache-stochastic-rounding."
                )
            if not is_cuda():
                raise ValueError(
                    "Stochastic rounding for the Mamba SSM cache is only "
                    "supported on NVIDIA CUDA platforms. Disable "
                    "--enable-mamba-cache-stochastic-rounding on this platform."
                )
            if self.mamba_backend == "triton" and not is_sm100_supported():
                raise ValueError(
                    "Stochastic rounding for the Mamba SSM cache with "
                    "--mamba-backend triton requires SM100 with CUDA >= 12.8 "
                    "because it uses the cvt.rs.f16x2.f32 PTX instruction. On "
                    "H100/SM90, run with --mamba-backend flashinfer "
                    "--mamba-ssm-dtype float16, or disable "
                    "--enable-mamba-cache-stochastic-rounding."
                )

        if self.mamba_backend == "flashinfer":
            flashinfer_error = (
                "FlashInfer mamba module not available, please check the "
                "FlashInfer installation."
            )
            if self.enable_mamba_cache_stochastic_rounding:
                flashinfer_error += (
                    " Stochastic rounding with --mamba-backend flashinfer "
                    "requires FlashInfer Mamba and --mamba-ssm-dtype float16."
                )
            if is_flashinfer_available():
                try:
                    import flashinfer.mamba  # noqa: F401

                    logger.info("Successfully imported FlashInfer mamba module")
                except (ImportError, AttributeError):
                    raise ValueError(flashinfer_error)
            else:
                raise ValueError(flashinfer_error)

    def _handle_int8_mamba_checkpoint(self):
        # The int8 mamba checkpoint pool is only wired into the built-in
        # MambaRadixCache. The host-offload variant (HiMambaRadixCache, enabled by
        # --enable-hierarchical-cache) and custom radix-cache backends are NOT
        # int8-aware: they would read int8 checkpoint slots as bf16 active slots
        # (wrong pool / out-of-range). Reject the combination up front rather than
        # silently corrupting state.
        if not self.enable_int8_mamba_checkpoint:
            return
        if self.enable_hierarchical_cache:
            raise ValueError(
                "--enable-int8-mamba-checkpoint is not supported together with "
                "--enable-hierarchical-cache: the host-offload path "
                "(HiMambaRadixCache) is not int8-aware. Disable one of them."
            )
        if self.radix_cache_backend is not None:
            raise ValueError(
                "--enable-int8-mamba-checkpoint only supports the built-in mamba "
                f"radix cache; --radix-cache-backend={self.radix_cache_backend!r} "
                "is not int8-aware. Omit --radix-cache-backend."
            )

    def _handle_linear_attn_backend(self):
        import torch

        # SM100+: default to FlashInfer GDN decode (and MTP verify, via pool API)
        # when the user hasn't explicitly chosen a decode backend and
        # mamba-ssm-dtype is bf16 (required by FlashInfer GDN on SM100+).
        # Fixed in FlashInfer v0.6.7: flashinfer-ai/flashinfer#2810
        if (
            self.linear_attn_decode_backend is None
            and is_sm100_supported()
            and self.mamba_ssm_dtype == "bfloat16"
        ):
            self.linear_attn_decode_backend = "flashinfer"
            logger.info(
                "SM100+ detected with mamba-ssm-dtype=bfloat16, "
                "defaulting --linear-attn-decode-backend to flashinfer."
            )

        # SM100+ FlashInfer GDN decode requires bf16 state; SM90 uses float32.
        decode = self.linear_attn_decode_backend or self.linear_attn_backend
        if (
            decode == "flashinfer"
            and self.mamba_ssm_dtype != "bfloat16"
            and is_cuda()
            and torch.cuda.get_device_capability()[0] >= 10
        ):
            raise ValueError(
                "--linear-attn-decode-backend flashinfer on SM100+ requires "
                "--mamba-ssm-dtype bfloat16, "
                f"got {self.mamba_ssm_dtype!r}"
            )

        # SM100+ FlashInfer GDN prefill requires CUDA 13+ (CuTe DSL kernel)
        # for correctness and best performance.
        prefill = self.linear_attn_prefill_backend or self.linear_attn_backend
        cuda_version = torch.version.cuda
        cuda_major = int(cuda_version.split(".")[0]) if cuda_version is not None else 0
        if (
            prefill == "flashinfer"
            and is_cuda()
            and torch.cuda.get_device_capability()[0] >= 10
            and cuda_major < 13
        ):
            raise ValueError(
                "--linear-attn-prefill-backend flashinfer on SM100+ requires CUDA 13+, "
                f"got CUDA {cuda_version or 'unknown'}"
            )

        # GDN ReplaySSM buffered decode guards. Runs on the Triton GDN decode
        # backend. cuda-graph is supported (slice 1b: CUDA-graph-safe static
        # write-cursor buffers). The RADIX prefix cache is now supported (slice
        # 2b: the decode kernel force-flushes the ring into temporal[slot] on
        # the radix track boundary `seq_lens % mamba_track_interval == 0`, and
        # the COW copy-into-slot path resets the ring cursor) -- so the
        # --disable-radix-cache requirement is dropped.
        #
        # Slice 2b only wires the no_buffer mamba scheduler strategy (the
        # default). The extra_buffer strategy donates the track snapshot via
        # `donate_mamba_ping_pong_slot` with a separate ping-pong slot swap that
        # does NOT route through MambaPool.copy_from, so the ReplaySSM ring
        # cursor of the donated/kept slot would not be reset there. Handling
        # that donation path is a follow-up; for now require no_buffer.
        if self.enable_linear_replayssm:
            if decode != "triton":
                raise ValueError(
                    "--enable-linear-replayssm requires the Triton "
                    "linear-attn decode backend, got "
                    f"--linear-attn-decode-backend={decode!r}."
                )
            if self.enable_mamba_extra_buffer():
                raise ValueError(
                    "--enable-linear-replayssm requires --mamba-scheduler-strategy "
                    "no_buffer (the default); the extra_buffer ping-pong "
                    "donation path is not yet supported (follow-up). Got "
                    f"--mamba-scheduler-strategy={self.mamba_scheduler_strategy!r}."
                )
            if self.disaggregation_mode != "null":
                # The disaggregated decode pool (HybridMambaDecodeReqToTokenPool)
                # is not wired for the ReplaySSM ring, so the flag would silently
                # no-op there; disagg also runs a different cache/coordination
                # flow that is not yet validated for ReplaySSM (follow-up).
                raise ValueError(
                    "--enable-linear-replayssm is not supported under PD "
                    "disaggregation yet (follow-up). Got "
                    f"--disaggregation-mode={self.disaggregation_mode!r}."
                )
            if self.linear_replayssm_cache_len < 1:
                raise ValueError(
                    "--linear-replayssm-cache-len must be >= 1, got "
                    f"{self.linear_replayssm_cache_len}."
                )

    def _handle_legacy_cp_arguments(self):
        legacy_mode_to_strategy = {
            "in-seq-split": "zigzag",
            "round-robin-split": "interleave",
        }
        strategy_to_legacy_mode = {
            "zigzag": "in-seq-split",
            "interleave": "round-robin-split",
        }

        if (
            self.enable_prefill_context_parallel
            or self.enable_dsa_prefill_context_parallel
        ):
            self.enable_prefill_cp = True

        if self.enable_prefill_context_parallel and self.cp_strategy is None:
            self.cp_strategy = legacy_mode_to_strategy[self.prefill_cp_mode]
        if self.enable_dsa_prefill_context_parallel and self.cp_strategy is None:
            self.cp_strategy = legacy_mode_to_strategy[self.dsa_prefill_cp_mode]

        if (
            self.enable_prefill_context_parallel
            and self.enable_dsa_prefill_context_parallel
        ):
            return

        if not self.enable_prefill_cp or self.cp_strategy is None:
            return

        mode = strategy_to_legacy_mode[self.cp_strategy]
        use_dsa_legacy_aliases = self.enable_dsa_prefill_context_parallel or getattr(
            self, "attention_backend", None
        ) in ("dsa", "dsv4")
        if use_dsa_legacy_aliases:
            self.enable_dsa_prefill_context_parallel = True
            self.enable_prefill_context_parallel = False
        else:
            self.enable_prefill_context_parallel = True
        self.dsa_prefill_cp_mode = mode
        self.prefill_cp_mode = mode

    def _handle_context_parallelism(self):
        if parse_connector_type(self.model_path) != ConnectorType.INSTANCE:
            from sglang.srt.layers.cp.utils import CP_V2_DEFAULT_MODEL_CLASSES

            model_config = self.get_model_config()
            model_arch = model_config.hf_config.architectures[0]
            if (
                model_arch in CP_V2_DEFAULT_MODEL_CLASSES
                and not envs.SGLANG_ENABLE_CP_V2.is_set()
            ):
                envs.SGLANG_ENABLE_CP_V2.set(True)

        if self.enable_prefill_cp and self.cp_strategy is None:
            raise ValueError(
                "--cp-strategy must be set when --enable-prefill-cp is enabled."
            )

        if (
            self.enable_prefill_context_parallel
            and self.enable_dsa_prefill_context_parallel
        ):
            raise ValueError(
                "--enable-prefill-context-parallel and "
                "--enable-nsa-prefill-context-parallel are mutually "
                "exclusive. Use --enable-nsa-prefill-context-parallel for "
                "DeepSeek V3.2 (NSA) models and "
                "--enable-prefill-context-parallel for MLA-based models "
                "(DeepSeek V3/R1, Kimi K2.5) or MHA/GQA-based models."
            )

        if self.attn_cp_size > 1:
            # The tp_size is the world size, not the real tensor parallel size
            assert (
                self.tp_size % self.attn_cp_size == 0
            ), "tp_size must be divisible by attn_cp_size"
            assert (
                self.tp_size % (self.dp_size * self.attn_cp_size) == 0
            ), "tp_size must be divisible by dp_size * attn_cp_size"

            assert (
                not self.enable_aiter_allreduce_fusion
            ), "Aiter allreduce fusion is not supported with context parallelism"

        if self.moe_dp_size > 1:
            # The tp_size is the world size, not the real tensor parallel size
            assert (
                self.tp_size % self.moe_dp_size == 0
            ), "tp_size must be divisible by moe_dp_size"
            assert (
                self.ep_size * self.moe_dp_size <= self.tp_size
            ), "ep_size * moe_dp_size must be less than or equal to tp_size"
            assert self.pp_size == 1, "PP is not supported with context parallelism"

            if self.ep_size > 1:
                assert (
                    self.ep_size * self.moe_dp_size == self.tp_size
                ), "ep_size * moe_dp_size must be equal to tp_size"

            assert (
                not self.enable_aiter_allreduce_fusion
            ), "Aiter allreduce fusion is not supported with context parallelism"

        if self.attn_cp_size != self.moe_dp_size:
            assert (
                self.moe_dp_size == 1
            ), "attn_cp_size != moe_dp_size is only supported when moe_dp_size == 1"

        from sglang.srt.layers.cp.base import init_cp_strategy

        init_cp_strategy(self)

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
        if self.quantization == "nvfp4_online":
            if not is_sm100_supported():
                raise ValueError(
                    "--quantization nvfp4_online is supported only on "
                    "NVIDIA Blackwell SM100/SM103 GPUs."
                )
            if self.moe_runner_backend == "auto":
                self.moe_runner_backend = "flashinfer_trtllm"
            elif self.moe_runner_backend not in [
                "flashinfer_trtllm",
                "flashinfer_trtllm_routed",
            ]:
                raise ValueError(
                    "--quantization nvfp4_online supports only "
                    "--moe-runner-backend flashinfer_trtllm or "
                    "flashinfer_trtllm_routed."
                )
        if self.quantization == "mxfp8":
            if self.moe_runner_backend == "auto":
                self.moe_runner_backend = "flashinfer_trtllm"
            elif self.moe_runner_backend not in [
                "cutlass",
                "flashinfer_trtllm",
                "flashinfer_trtllm_routed",
            ]:
                logger.warning(
                    "mxfp8 quantization supports only cutlass, flashinfer_trtllm, "
                    "or flashinfer_trtllm_routed backends. "
                    f"Overriding {self.moe_runner_backend!r}."
                )
                self.moe_runner_backend = "flashinfer_trtllm"

        if (
            self.moe_runner_backend == "auto"
            and self.quantization == "modelopt_fp4"
            and is_sm120_supported()
        ):
            self.moe_runner_backend = "flashinfer_cutlass"
            logger.info(
                "Use flashinfer_cutlass as MoE runner backend on SM120 for "
                "modelopt_fp4 (trtllm-gen MoE kernels are SM100-only)"
            )

        if self.moe_runner_backend == "flashinfer_cutlass":
            assert self.quantization in [
                "modelopt_fp4",
                "modelopt_fp8",
                "modelopt_mixed",
                None,
            ], f"Invalid quantization '{self.quantization}'. \nFlashInfer Cutlass MOE supports only: 'modelopt_fp4', 'modelopt_fp8', 'modelopt_mixed', or bfloat16 (None)."
            assert self.ep_size in [
                1,
                self.tp_size,
            ], "The expert parallel size must be 1 or the same as the tensor parallel size"

        if self.moe_runner_backend == "flashinfer_cutedsl":
            assert (
                self.quantization in ["modelopt_fp4"]
                or self.get_model_config().nvfp4_moe_meta is not None
            ), f"Invalid quantization '{self.quantization}'. \nFlashInfer CuteDSL MOE currently supports only: 'modelopt_fp4' or hybrid NVFP4 models."
            assert self.ep_size in [
                1,
                self.tp_size,
            ], "The expert parallel size must be 1 or the same as the tensor parallel size"
            assert self.moe_a2a_backend in [
                "none",
                "deepep",
                "flashinfer",
            ], (
                f"flashinfer_cutedsl supports moe_a2a_backend='none', 'deepep', or 'flashinfer', "
                f"got '{self.moe_a2a_backend}'."
            )
            self.disable_shared_experts_fusion = True
            logger.warning(
                "FlashInfer CuteDSL MoE is enabled. --disable-shared-experts-fusion is automatically set."
            )

        if self.moe_runner_backend in ["flashinfer_trtllm", "experimental_sgl_trtllm"]:
            assert self.quantization in [
                "modelopt_fp4",
                "nvfp4_online",
                "fp8",
                "mxfp8",
                "modelopt_fp8",
                "modelopt_mixed",
                "compressed-tensors",
                None,
            ], f"Invalid quantization '{self.quantization}'. \nFlashInfer TRTLLM MOE supports only: 'modelopt_fp4', 'nvfp4_online', 'fp8', 'modelopt_fp8', 'modelopt_mixed', 'compressed-tensors', or bfloat16 (None)."
            self.disable_shared_experts_fusion = True
            logger.warning(
                "FlashInfer TRTLLM MoE is enabled. --disable-shared-experts-fusion is automatically set."
            )

        if self.moe_runner_backend == "flashinfer_trtllm_routed":
            assert self.quantization in [
                "fp8",
                "mxfp8",
                "modelopt_fp4",
                "nvfp4_online",
                None,
            ], f"Invalid quantization '{self.quantization}'. \nFlashInfer TRTLLM routed MOE supports only: 'fp8', 'mxfp8', 'modelopt_fp4', 'nvfp4_online', or bfloat16 (None)."
            self.disable_shared_experts_fusion = True
            logger.warning(
                "FlashInfer TRTLLM routed MoE is enabled. --disable-shared-experts-fusion is automatically set."
            )

        if envs.SGLANG_CUTLASS_MOE.get():
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

    def cutedsl_moe_max_num_tokens(self) -> int:
        """Largest number of tokens a single forward routes through a CuteDSL
        MoE layer on one (DP) rank. Single source of truth for both the
        standard-allgather wrapper buffers and the FlashInfer A2A dispatcher
        budget. Max over the prefill (max_prefill_tokens), piecewise-prefill
        capture, and decode/verify bounds; num_tokens_per_bs is
        speculative_num_draft_tokens under speculative decoding, else 1.
        """
        if self.speculative_algorithm:
            num_tokens_per_bs = self.speculative_num_draft_tokens or 1
        else:
            num_tokens_per_bs = 1
        prefill_tokens = self.max_prefill_tokens
        cg_config = self.cuda_graph_config
        if cg_config is not None and cg_config.prefill.backend == Backend.TC_PIECEWISE:
            prefill_tokens = max(prefill_tokens, cg_config.prefill.max_bs or 0)
        decode_max_bs = (cg_config.decode.max_bs if cg_config is not None else 0) or 0
        decode_tokens = decode_max_bs * num_tokens_per_bs
        return max(prefill_tokens, decode_tokens)

    def max_prefill_buffer_tokens(self) -> int:
        """Prefill-buffer ceiling: chunked_prefill_size, except PP dynamic
        chunking can grow chunks toward max_prefill_tokens and probe at 1.25x."""
        chunked = (
            self.chunked_prefill_size
            if self.chunked_prefill_size and self.chunked_prefill_size > 0
            else 0
        )
        tokens = chunked
        if self.enable_dynamic_chunking and self.pp_size > 1 and chunked:
            tokens = max(
                tokens, self.max_prefill_tokens or 0, math.ceil(chunked * 1.25)
            )
        return tokens

    def _validate_cutedsl_a2a_token_budget(self):
        """Fail fast if the FlashInfer A2A dispatcher workspace cannot cover the
        largest CuteDSL MoE forward. Runs after speculative decoding is resolved
        so cutedsl_moe_max_num_tokens() sees the final num_tokens_per_bs."""
        if not (
            self.moe_a2a_backend == "flashinfer"
            and self.moe_runner_backend == "flashinfer_cutedsl"
            and self.max_prefill_tokens > 0
            and self.disaggregation_mode != "decode"
        ):
            return
        required_tokens = self.cutedsl_moe_max_num_tokens()
        max_dispatch_tokens_per_rank = get_int_env_var(
            "SGLANG_FLASHINFER_NUM_MAX_DISPATCH_TOKENS_PER_RANK", 1024
        )
        max_cutedsl_tokens = max_dispatch_tokens_per_rank * self.ep_size
        if max_cutedsl_tokens < required_tokens:
            required_per_rank = (required_tokens + self.ep_size - 1) // self.ep_size
            raise ValueError(
                "FlashInfer MoE A2A with flashinfer_cutedsl requires "
                "SGLANG_FLASHINFER_NUM_MAX_DISPATCH_TOKENS_PER_RANK * "
                "ep_size to cover the largest CuteDSL MoE forward "
                f"({required_tokens} tokens). Otherwise the FlashInfer "
                "dispatcher can crash at runtime with "
                "`ValueError: num_tokens (...) exceeds max_num_tokens (...)`. "
                "Current values: "
                f"SGLANG_FLASHINFER_NUM_MAX_DISPATCH_TOKENS_PER_RANK="
                f"{max_dispatch_tokens_per_rank}, ep_size={self.ep_size}, "
                f"capacity={max_cutedsl_tokens}, required={required_tokens}. "
                f"Set `export "
                f"SGLANG_FLASHINFER_NUM_MAX_DISPATCH_TOKENS_PER_RANK="
                f"{required_per_rank}` or lower the relevant limit "
                f"(e.g. --max-prefill-tokens) to <= {max_cutedsl_tokens}."
            )

    def _handle_a2a_moe(self):
        if self.enable_deepep_waterfill and self.moe_a2a_backend != "deepep":
            logger.warning(
                "moe_a2a_backend is overridden to 'deepep' because DeepEP "
                "Waterfill requires the DeepEP backend."
            )
            self.moe_a2a_backend = "deepep"

        if (
            envs.SGLANG_OPT_USE_DEEPGEMM_MEGA_MOE.get()
            and self.moe_a2a_backend != "megamoe"
        ):
            self.moe_a2a_backend = "megamoe"
            logger.info(
                "SGLANG_OPT_USE_DEEPGEMM_MEGA_MOE is set, "
                "auto-configuring --moe-a2a-backend megamoe."
            )

        if self.moe_a2a_backend == "megamoe":
            self.ep_size = self.tp_size
            if not envs.SGLANG_OPT_FIX_MEGA_MOE_MEMORY.is_set():
                envs.SGLANG_OPT_FIX_MEGA_MOE_MEMORY.set(True)
            logger.info(
                f"Mega MoE is enabled. The expert parallel size is adjusted "
                f"to be the same as the tensor parallel size[{self.tp_size}]."
            )

        if self.moe_a2a_backend == "deepep":
            if self.deepep_mode == "normal":
                logger.warning("Cuda graph is disabled because deepep_mode=`normal`")
                self.cuda_graph_config.decode.backend = Backend.DISABLED
                self.cuda_graph_config.prefill.backend = Backend.DISABLED
            self.ep_size = self.tp_size
            logger.warning(
                f"DeepEP MoE is enabled. The expert parallel size is adjusted to be the same as the tensor parallel size[{self.tp_size}]."
            )
            if self.enable_deepep_waterfill:
                if self.disable_shared_experts_fusion:
                    logger.warning(
                        "disable_shared_experts_fusion is overridden to False because DeepEP Waterfill requires shared expert fusion."
                    )
                    self.disable_shared_experts_fusion = False
                self.enforce_shared_experts_fusion = True
                logger.info(
                    "DeepEP Waterfill is enabled. Shared expert will be dispatched through DeepEP for load balancing."
                )

        if self.moe_a2a_backend == "mooncake":
            self.ep_size = self.tp_size
            logger.warning(
                f"Mooncake MoE is enabled. The expert parallel size is adjusted to be the same as the tensor parallel size[{self.tp_size}]."
            )

        if self.moe_a2a_backend == "nixl":
            self.ep_size = self.tp_size
            logger.warning(
                f"Nixl MoE is enabled. The expert parallel size is adjusted to be the same as the tensor parallel size[{self.tp_size}]."
            )

        if self.moe_a2a_backend == "ascend_fuseep":
            self.ep_size = self.tp_size
            logger.warning(
                f"Ascend fused EP MoE is enabled. The expert parallel size is adjusted to be the same as the tensor parallel size[{self.tp_size}]."
            )
            fuse_mode = envs.SGLANG_NPU_FUSED_MOE_MODE.get()
            if fuse_mode not in [1, 2]:
                raise ValueError(
                    f"Wrong value of {fuse_mode=}, the NPU only support 1 or 2."
                )
            elif fuse_mode == 2:
                assert (
                    self.quantization == "modelslim"
                ), "When fuse_mode is set to 2, the NPU supports only ModelSlim quantization."
        if self.moe_a2a_backend == "flashinfer":
            assert (
                self.enable_dp_attention and self.dp_size == self.tp_size
            ), "Flashinfer MoE A2A is only supported with dp_size == tp_size and --enable-dp-attention"
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
            if not envs.SGLANG_MOE_NVFP4_DISPATCH.is_set() and (
                self.quantization == "modelopt_fp4"
                or self.get_model_config().nvfp4_moe_meta is not None
            ):
                envs.SGLANG_MOE_NVFP4_DISPATCH.set(True)
                logger.warning(
                    "SGLANG_MOE_NVFP4_DISPATCH is set to True for Flashinfer MoE A2A"
                )
            assert self.moe_runner_backend in [
                "flashinfer_cutlass",
                "flashinfer_cutedsl",
                "flashinfer_trtllm_routed",
            ], "Flashinfer MoE A2A is only supported with flashinfer_cutlass, flashinfer_cutedsl or flashinfer_trtllm_routed moe runner backend"

        if self.moe_a2a_backend == "mori":
            self.ep_size = self.tp_size
            if self.deepep_mode == "auto":
                self.deepep_mode = "normal"
                logger.warning("auto set deepep_mode=`normal` for MORI EP")
            logger.warning(
                f"MoRI MoE is enabled. The expert parallel size is adjusted to be the same as the tensor parallel size[{self.tp_size}]."
            )

            # Check chunked prefill for mori
            # Skip validation if chunked prefill is disabled (i.e., size <= 0).
            # Skip validation if disaggregation mode is decode.
            if self.chunked_prefill_size > 0 and self.disaggregation_mode != "decode":
                assert (
                    self._required_mori_dispatch_tokens_per_rank()
                ) <= envs.SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK.get(), (
                    "SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK (default 4096) "
                    "must be >= the per-rank MoRI dispatch tokens "
                    "(chunked_prefill_size by default)"
                )

    def _required_mori_dispatch_tokens_per_rank(self) -> int:
        """Max tokens a single rank dispatches through MoRI in one forward."""
        return self.chunked_prefill_size

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
                assert self.eplb_algorithm in [
                    "elasticity_aware",
                    "elasticity_aware_hierarchical",
                ], "Elastic EP requires eplb_algorithm to be set to 'auto' or 'elasticity_aware(_hierarchical)'."

            assert self.pp_size == 1, "PP size should be set to 1 under elastic EP"

            if self.elastic_ep_backend == "mooncake":
                self.mooncake_ib_device = self._validate_ib_devices(
                    self.mooncake_ib_device
                )
        if self.elastic_ep_rejoin:
            assert (
                self.elastic_ep_backend is not None
            ), "Elastic EP rejoin requires elastic_ep_backend to be set."

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

    def _validate_prefill_only_disable_kv_cache_args(self):
        """Validate --prefill-only-disable-kv-cache flag/precondition constraints.

        Runs before the dummy-model short-circuit so misuse is rejected even
        for dummy models. Backend resolution is checked separately by
        _handle_prefill_only_disable_kv_cache after backends settle.
        """
        if not self.prefill_only_disable_kv_cache:
            return

        # This flag is intentionally scoped to embedding mode for now. Other
        # prefill-only paths (for example scoring and MIS) can benefit from
        # the same idea later, but some of them still stage K/V through the
        # paged cache today.
        if not self.is_embedding:
            raise ValueError(
                "--prefill-only-disable-kv-cache currently requires --is-embedding. "
                "Other prefill-only workloads may be supported in a future change once "
                "their attention paths stop reading or writing the paged KV cache."
            )
        if self.kv_cache_dtype == "fp4_e2m1":
            raise ValueError(
                "--prefill-only-disable-kv-cache does not currently support "
                "--kv-cache-dtype=fp4_e2m1 because the FP4 pool uses a separate "
                "allocation path."
            )

        # Structural preconditions for the FA backend's fa_skip_kv_cache path,
        # which is the only embedding path that doesn't read or write the pool:
        # - chunked_prefill_size == -1 keeps a request in a single forward,
        #   so K/V never has to be reused across prefill chunks.
        # - disable_radix_cache stops the prefix cache from indexing pool
        #   slots that no longer hold real data.
        if self.chunked_prefill_size != -1:
            raise ValueError(
                "--prefill-only-disable-kv-cache requires --chunked-prefill-size=-1 so the FA "
                "backend takes the fa_skip_kv_cache path; otherwise the pool would be touched "
                "between prefill chunks."
            )
        if not self.disable_radix_cache:
            raise ValueError(
                "--prefill-only-disable-kv-cache requires --disable-radix-cache because the "
                "radix cache indexes KV pool slots that no longer hold real data."
            )

        # Context-parallel prefill stages K/V through cp_allgather_and_save_kv_cache,
        # which writes to the pool via set_kv_buffer. NoOpMHATokenToKVPool intentionally
        # raises on writes, so the engine would boot fine but fail on the first request.
        if self.attn_cp_size > 1:
            raise ValueError(
                "--prefill-only-disable-kv-cache is incompatible with --attn-cp-size > 1: "
                "the context-parallel attention path writes K/V to the pool via set_kv_buffer, "
                "which the no-op pool intentionally rejects."
            )
        if self.enable_prefill_cp:
            raise ValueError(
                "--prefill-only-disable-kv-cache is incompatible with "
                "--enable-prefill-cp: the prefill-CP path stages K/V through "
                "the paged cache, which the no-op pool does not support."
            )

        # HiSparse selects a different pool class (HiSparseDSATokenToKVPool /
        # HiSparseTokenToKVPoolAllocator) that is not the no-op pool.
        if self.enable_hisparse:
            raise ValueError(
                "--prefill-only-disable-kv-cache is incompatible with --enable-hisparse: "
                "HiSparse uses a dedicated pool family that is not the no-op MHA pool."
            )

    def _handle_prefill_only_disable_kv_cache(self):
        """Validate --prefill-only-disable-kv-cache backend constraint.

        Must run after _handle_attention_backend_compatibility() (which fills
        the default attention_backend if unset) and _handle_multi_item_scoring()
        (which may further mutate it). The assertion below guards against
        accidental call-site reordering: if attention_backend is still None,
        backends haven't settled yet and get_attention_backends() would return
        a stale (None, None).
        """
        if not self.prefill_only_disable_kv_cache:
            return

        assert self.attention_backend is not None, (
            "_handle_prefill_only_disable_kv_cache must run after "
            "_handle_attention_backend_compatibility() so the prefill backend is resolved."
        )

        prefill_backend, _ = self.get_attention_backends()
        if prefill_backend not in ("fa3", "fa4"):
            raise ValueError(
                "--prefill-only-disable-kv-cache currently requires the FA prefill backend "
                f"(fa3/fa4), but got prefill backend {prefill_backend!r}. Other prefill-only "
                "workloads and backends may be supported in a future change."
            )

    def _handle_hicache(self):
        """Normalize hicache-related knobs into a valid runtime configuration.

        Resolution order:
        1) Layout <-> I/O compatibility for direct conflicts.
        2) Storage <-> layout compatibility (may rewrite layout).
        """
        # Skip all normalization when neither hicache nor decode-offload path is active.
        if not (
            self.enable_hierarchical_cache
            or self.disaggregation_decode_enable_offload_kvcache
        ):
            return

        # Step 1: Initial layout-io compatibility normalization.
        self._resolve_layout_io_compatibility()

        # Step 2: Storage-layout normalization without changing io backend.
        self._resolve_storage_layout_compatibility()

        # Step 3: HiCache is not yet supported with the DeepSeek-V4 hip unified_kv
        # layout, so fall back to the default tilelang FlashMLA backend.
        self._resolve_unified_kv_hicache_compatibility()

    def _resolve_unified_kv_hicache_compatibility(self):
        # The DeepSeek-V4 unified_kv layout (SGLANG_HACK_FLASHMLA_BACKEND=
        # unified_kv_triton) keeps swa/c4/c128 in a single per-layer buffer and
        # has no HiCache host-pool support yet, so reset the backend to the
        # default (tilelang) so the server still starts.
        if not self.enable_hierarchical_cache:
            return

        if envs.SGLANG_HACK_FLASHMLA_BACKEND.get() == "unified_kv_triton":
            envs.SGLANG_HACK_FLASHMLA_BACKEND.set("tilelang")
            logger.warning(
                "SGLANG_HACK_FLASHMLA_BACKEND=unified_kv_triton is not yet "
                "compatible with --enable-hierarchical-cache; falling back to "
                "SGLANG_HACK_FLASHMLA_BACKEND=tilelang."
            )

    def _resolve_layout_io_compatibility(self):
        if (
            self.hicache_mem_layout == "page_first_direct"
            and self.hicache_io_backend == "kernel"
        ):
            self.hicache_io_backend = "direct"
            logger.warning(
                "Kernel io backend does not support page first direct layout, switching to direct io backend"
            )

        if (
            self.hicache_mem_layout == "page_first"
            and self.hicache_io_backend == "direct"
        ):
            self.hicache_mem_layout = "page_first_direct"
            logger.warning(
                "Page first layout is not supported with direct IO backend, switching to page first direct layout"
            )

        # The page_first kernel write-back relies on the CUDA-only JIT staged
        # kernel. On ROCm it falls back to a kernel that requires CUDA index
        # tensors and crashes on host write-back, so use layer_first there.
        if (
            self.hicache_mem_layout == "page_first"
            and self.hicache_io_backend == "kernel"
            and is_hip()
        ):
            self.hicache_mem_layout = "layer_first"
            logger.warning(
                "page_first kernel write-back requires the CUDA JIT kernel; "
                "falling back to layer_first layout on ROCm."
            )

    def _resolve_storage_layout_compatibility(self):
        if (
            self.hicache_storage_backend != "mooncake"
            or self.hicache_mem_layout != "layer_first"
        ):
            return

        if self.hicache_io_backend == "direct":
            new_layout = "page_first_direct"
        elif self.hicache_io_backend == "kernel":
            new_layout = "page_first"
        else:
            # Keep current behavior for unknown backends (e.g., kernel_ascend).
            new_layout = self.hicache_mem_layout

        self.hicache_mem_layout = new_layout
        logger.warning(
            f"Mooncake storage backend does not support layer_first layout, "
            f"switching to {new_layout} layout for {self.hicache_io_backend} io backend"
        )

    def _handle_load_format(self):
        if (
            self.load_format == "auto" or self.load_format == "gguf"
        ) and check_gguf_file(self.model_path):
            self.quantization = self.load_format = "gguf"

        if self.load_format == "auto" and self._is_mistral_native_format():
            self.load_format = "mistral"
            logger.info(
                "Detected Mistral native format checkpoint, setting load_format='mistral'"
            )

        if is_runai_obj_uri(self.model_path):
            self.load_format = "runai_streamer"
        elif is_remote_url(self.model_path):
            self.load_format = "remote"

        if self.custom_weight_loader is None:
            self.custom_weight_loader = []

        if self.load_format == "remote_instance":
            if self.remote_instance_weight_loader_backend != "modelexpress" and (
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
                self.remote_instance_weight_loader_backend == "transfer_engine"
                and not self.validate_transfer_engine()
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

    def _is_mistral_native_format(self) -> bool:
        """True iff the checkpoint requires load_format=mistral.

        Looks for consolidated*.safetensors with no competing
        model-*.safetensors; when both weight formats ship in the
        same checkpoint (e.g. Mistral-7B-Instruct-v0.3) the HF path is
        preferred to avoid loading Mistral-named weights into an
        HF-named architecture.

        Name override: mistral-large-3 / mistral-small-4 /
        leanstral always treat as Mistral-native when params.json
        is present -- those families need Mistral weight loading
        regardless of which weight files happen to be present.
        """
        _MISTRAL_NATIVE_PATTERNS = (
            "mistral-large-3",
            "mistral-small-4",
            "leanstral",
        )
        name_matches = any(
            p in str(self.model_path).lower() for p in _MISTRAL_NATIVE_PATTERNS
        )

        def _check_format(has_params, has_consolidated, has_hf_weights) -> bool:
            if has_params and name_matches:
                return True
            return has_consolidated and not has_hf_weights

        if os.path.isdir(self.model_path):
            return _check_format(
                has_params=os.path.exists(os.path.join(self.model_path, "params.json")),
                has_consolidated=bool(
                    glob.glob(
                        os.path.join(self.model_path, "consolidated*.safetensors")
                    )
                ),
                has_hf_weights=bool(
                    glob.glob(os.path.join(self.model_path, "model-*.safetensors"))
                ),
            )

        try:
            from huggingface_hub import HfApi

            files = {s.rfilename for s in HfApi().model_info(self.model_path).siblings}
            return _check_format(
                has_params="params.json" in files,
                has_consolidated=any(
                    f.startswith("consolidated") and f.endswith(".safetensors")
                    for f in files
                ),
                has_hf_weights=any(
                    f.startswith("model-") and f.endswith(".safetensors") for f in files
                ),
            )
        except Exception:
            return False

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
            logger.info(
                "--language-only is set without --encoder-urls. Encoders are "
                "expected to register dynamically via the "
                "EncoderBootstrapServer."
            )

        # Validate IB devices when mooncake backend is used
        if (
            self.disaggregation_transfer_backend == "mooncake"
            and self.disaggregation_mode in ("prefill", "decode")
        ) or self.encoder_transfer_backend == "mooncake":
            self.disaggregation_ib_device = self._validate_ib_devices(
                self.disaggregation_ib_device
            )

        # Validate model type for encoder disaggregation
        hf_config = self.get_model_config().hf_config
        model_arch = hf_config.architectures[0]
        if (self.encoder_only or self.language_only) and model_arch not in [
            "Qwen2VLForConditionalGeneration",
            "Qwen3VLForConditionalGeneration",
            "Qwen2_5_VLForConditionalGeneration",
            "Qwen3VLMoeForConditionalGeneration",
            "Qwen3_5ForConditionalGeneration",
            "Qwen3_5MoeForConditionalGeneration",
            "InternS2PreviewForConditionalGeneration",
            "Qwen3OmniMoeForConditionalGeneration",
            "Qwen2AudioForConditionalGeneration",
            "Qwen2_5OmniForConditionalGeneration",
            "KimiVLForConditionalGeneration",
            "KimiK25ForConditionalGeneration",
            "MiMoV2ForCausalLM",
        ]:
            raise ValueError(
                f"Model type {model_arch} is not supported for encoder disaggregation. "
                f"Supported architectures: Qwen2VL, Qwen3VL, Qwen3.5, InternS2, Qwen2Audio, Qwen2.5Omni, Kimi, MiMoV2."
            )

    def _validate_ib_devices(self, device_str: Optional[str]) -> Optional[str]:
        """
        Validate IB devices before passing to mooncake.

        Args:
            device_str: Comma-separated IB device names, a per-GPU JSON mapping,
                or a path to a JSON file containing that mapping.

        Returns:
            A normalized comma-separated string or per-GPU JSON mapping string, or None if input is None.
        """
        if device_str is None:
            logger.warning(
                "No IB devices specified for Mooncake backend, falling back to auto discovery."
            )
            return None

        def _normalize_device_group(raw_value: str, context: str) -> str:
            if not isinstance(raw_value, str):
                raise ValueError(
                    f"Invalid IB device format for {context}: expected a string. "
                    f"Got {type(raw_value)}"
                )
            devices = [d.strip() for d in raw_value.split(",") if d.strip()]
            if not devices:
                raise ValueError(f"No valid IB devices specified for {context}")
            unique_devices = list(dict.fromkeys(devices))
            if len(unique_devices) != len(devices):
                logger.warning(
                    "Duplicate IB devices specified for %s: %s. Deduplicating to: %s",
                    context,
                    raw_value,
                    ",".join(unique_devices),
                )
            invalid_devices = [d for d in unique_devices if d not in available_devices]
            if len(invalid_devices) != 0:
                raise ValueError(
                    f"Invalid IB devices specified for {context}: {invalid_devices}. "
                    f"Available devices: {sorted(available_devices)}"
                )
            return ",".join(unique_devices)

        normalized_input = device_str.strip()
        if not normalized_input:
            raise ValueError("No valid IB devices specified")

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

        parsed_config = parse_ib_device_config(normalized_input)
        if isinstance(parsed_config, str):
            return _normalize_device_group(normalized_input, "all GPUs")
        assert parsed_config is not None

        normalized_mapping: Dict[str, str] = {}
        for gpu_key, gpu_devices in parsed_config.items():
            normalized_key = str(gpu_key)
            normalized_mapping[normalized_key] = _normalize_device_group(
                gpu_devices, f"GPU {normalized_key}"
            )

        if not normalized_mapping:
            raise ValueError("No valid GPU mappings found in IB device JSON")

        return json.dumps(normalized_mapping, separators=(",", ":"))

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
            if self.detokenizer_worker_num != 1:
                logger.warning(
                    "skip_tokenizer_init=True disables detokenizer workers; forcing detokenizer_worker_num=1 "
                    f"(requested {self.detokenizer_worker_num})."
                )
                self.detokenizer_worker_num = 1

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

            logger.info(
                "skip_tokenizer_init=True: string-based stop conditions (stop, stop_regex) "
                "and min_new_tokens are unavailable."
            )

    def _handle_environment_variables(self):
        envs.SGLANG_ENABLE_TORCH_COMPILE.set("1" if self.enable_torch_compile else "0")
        if self.mamba_ssm_dtype is not None:
            envs.SGLANG_MAMBA_SSM_DTYPE.set(self.mamba_ssm_dtype)
        envs.SGLANG_DISABLE_OUTLINES_DISK_CACHE.set(
            "1" if self.disable_outlines_disk_cache else "0"
        )
        envs.SGLANG_ENABLE_DETERMINISTIC_INFERENCE.set(
            "1" if self.enable_deterministic_inference else "0"
        )
        # Custom all-reduce v2 uses IPC handles and is intra-node only. Force-disable
        # on multi-node so the dispatch falls back to the legacy CustomAllreduce path.
        if self.nnodes > 1 and envs.SGLANG_OPT_USE_CUSTOM_ALL_REDUCE_V2.get():
            if envs.SGLANG_OPT_USE_CUSTOM_ALL_REDUCE_V2.is_set():
                logger.warning(
                    "Disabling SGLANG_OPT_USE_CUSTOM_ALL_REDUCE_V2 because nnodes=%d "
                    "(custom all-reduce v2 is intra-node only).",
                    self.nnodes,
                )
            envs.SGLANG_OPT_USE_CUSTOM_ALL_REDUCE_V2.set("0")
        if self.debug_cuda_graph:
            if not (is_cuda() or is_hip()):
                logger.warning(
                    "--debug-cuda-graph is not supported on non CUDA/HIP devices. "
                    "Disabling breakable CUDA graph."
                )
                self.debug_cuda_graph = False
            else:
                envs.SGLANG_USE_BREAKABLE_CUDA_GRAPH.set("1")
                logger.warning(
                    "Debug mode for CUDA graph is enabled via breakable CUDA graph. "
                    "All operations will run eagerly through the graph capture/replay path."
                )
        if self.enable_deepseek_v4_fp4_indexer and not is_sm100_supported():
            raise ValueError(
                "--enable-deepseek-v4-fp4-indexer requires SM100 GPUs with "
                "DeepGEMM FP4 indexer support."
            )
        # FP8 W_o GEMM requires Blackwell (sm100+). Auto-disable on Hopper.
        if is_cuda() and envs.SGLANG_OPT_FP8_WO_A_GEMM.get() and get_device_sm() < 100:
            if envs.SGLANG_OPT_FP8_WO_A_GEMM.is_set():
                logger.warning(
                    "Disabling SGLANG_OPT_FP8_WO_A_GEMM: requires sm100+ (Blackwell), "
                    "detected sm%d.",
                    get_device_sm(),
                )
            envs.SGLANG_OPT_FP8_WO_A_GEMM.set(False)

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
            if self.hicache_storage_backend is None:
                raise ValueError(
                    "The argument disaggregation-decode-enable-offload-kvcache is only supported when hicache-storage-backend is provided."
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
            envs.SGLANG_VLM_CACHE_SIZE_MB.set(0)
            # TODO remove this environment variable as a whole
            envs.SGLANG_ENABLE_DETERMINISTIC_INFERENCE.set(True)

        if self.enable_deterministic_inference:
            if self.enable_aiter_allreduce_fusion:
                logger.warning(
                    "Disable --enable-aiter-allreduce-fusion because deterministic inference is enabled."
                )
                self.enable_aiter_allreduce_fusion = False

            if self.flashinfer_allreduce_fusion_backend is not None:
                logger.warning(
                    "Disable --flashinfer-allreduce-fusion-backend because deterministic inference is enabled."
                )
                self.flashinfer_allreduce_fusion_backend = None

            # Check sampling backend
            if self.sampling_backend != "ascend":
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
                        "GlmMoeDsaForCausalLM",
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

    def _handle_page_major_kv_layout(self):
        if not self.enable_page_major_kv_layout:
            return
        # Only the Triton attention kernels read the strided 4-D envelope K/V
        # views; FA3 / FlashInfer do not.
        backends = {
            self.attention_backend,
            self.prefill_attention_backend,
            self.decode_attention_backend,
        }
        backends.discard(None)
        assert backends <= {"triton"}, (
            "--enable-page-major-kv-layout requires the Triton attention backend "
            f"for the full-attention layers; got {sorted(backends)}. Pass "
            "--attention-backend triton."
        )
        # The Mamba state is stored in envelope-strided views; only the
        # stride-aware Triton causal-conv / SSM kernels read them correctly.
        linear_backends = {
            self.linear_attn_backend,
            self.linear_attn_decode_backend,
            self.linear_attn_prefill_backend,
            self.mamba_backend,
        }
        linear_backends.discard(None)
        assert linear_backends <= {"triton"}, (
            "--enable-page-major-kv-layout requires the Triton linear-attention / "
            f"Mamba kernels for the strided conv/SSM state; got "
            f"{sorted(linear_backends)}. Pass --linear-attn-backend triton and "
            "--mamba-backend triton."
        )

    def _handle_dllm_inference(self):
        if self.dllm_algorithm is None:
            return
        # On AMD/HIP, disable cuda graph for DLLM and use triton backend
        if is_hip():
            if (
                self.cuda_graph_config.decode.backend != Backend.DISABLED
                or self.cuda_graph_config.prefill.backend != Backend.DISABLED
            ):
                logger.warning(
                    "Cuda graph is disabled for diffusion LLM inference on AMD GPUs"
                )
                self.cuda_graph_config.decode.backend = Backend.DISABLED
                self.cuda_graph_config.prefill.backend = Backend.DISABLED
            if self.attention_backend not in ["triton", "aiter"]:
                logger.warning(
                    "Attention backend is set to triton for diffusion LLM inference on AMD GPUs"
                )
                self.attention_backend = "triton"
        elif is_npu():
            if self.attention_backend != "ascend":
                logger.warning(
                    "Attention backend is overridden to 'ascend' when running on NPU for diffusion LLM inference."
                )
                self.attention_backend = "ascend"
        elif self.cuda_graph_config.decode.backend != Backend.DISABLED:
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
            from sglang.srt.dllm.config import DllmConfig

            config = DllmConfig.from_server_args(self)
            if self.page_size % config.block_size != 0:
                logger.warning(
                    f"Setting page size to {config.block_size} for diffusion LLM inference"
                )
                self.page_size = config.block_size
            if self.enable_hierarchical_cache:
                logger.warning(
                    "Hierarchical cache is disabled because of using diffusion LLM inference"
                )
                self.enable_hierarchical_cache = False
            if self.enable_lmcache:
                logger.warning(
                    "LMCache is disabled because of using diffusion LLM inference"
                )
                self.enable_lmcache = False

        if self.pp_size > 1:
            logger.warning(
                "Pipeline parallelism is disabled because of using diffusion LLM inference"
            )
            self.pp_size = 1

        if self.enable_lora:
            logger.warning(
                "Currently LoRA is not supported by diffusion LLM inference."
            )
            self.enable_lora = False

        if self.disaggregation_mode != "null":
            logger.warning(
                "Currently disaggregation is not supported by diffusion LLM inference."
            )
            self.disaggregation_mode = "null"

        if self.enable_mixed_chunk:
            logger.warning(
                "Mixed chunked prefill is disabled because of using diffusion LLM inference."
            )
            self.enable_mixed_chunk = False

    def _handle_asr_validation(self):
        """Validate transcription/ASR-specific server args."""
        if self.asr_max_buffer_seconds <= 0:
            raise ValueError(
                f"--asr-max-buffer-seconds must be positive "
                f"(got {self.asr_max_buffer_seconds})."
            )
        if self.asr_max_concurrent_sessions <= 0:
            raise ValueError(
                f"--asr-max-concurrent-sessions must be positive "
                f"(got {self.asr_max_concurrent_sessions})."
            )

    def _handle_other_validations(self):
        # Handle optimistic prefill validation
        if (
            self.optimistic_prefill_retries > 0
            and self.disaggregation_mode == "prefill"
        ):
            if self.pp_size > 1:
                logger.warning("Optimistic prefill does not support pp_size > 1")
                self.optimistic_prefill_retries = 0
            elif self.enable_hierarchical_cache:
                logger.warning("Optimistic prefill does not support hierarchical cache")
                self.optimistic_prefill_retries = 0
            elif getattr(self, "uses_mamba_radix_cache", False):
                logger.warning(
                    "Optimistic prefill does not support models that use "
                    "mamba radix cache."
                )
                self.optimistic_prefill_retries = 0

        # Handle model inference tensor dump.
        if self.debug_tensor_dump_output_folder is not None:
            logger.warning(
                "Cuda graph and server warmup are disabled because of using tensor dump mode"
            )
            self.cuda_graph_config.decode.backend = Backend.DISABLED
            self.cuda_graph_config.prefill.backend = Backend.DISABLED
            self.skip_server_warmup = True

        if self.msprobe_dump_config is not None:
            logger.warning(
                "When msProbe is enabled, "
                "cuda graph is disabled because msProbe only supports dump in eager mode, "
                "warmup is disabled(skip_server_warmup=True) because there is no need to dump data for this stage."
            )
            self.cuda_graph_config.decode.backend = Backend.DISABLED
            self.cuda_graph_config.prefill.backend = Backend.DISABLED
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

            # Validate preferred_sampling_params doesn't use tokenizer-dependent features
            if self.skip_tokenizer_init:
                from sglang.srt.sampling.sampling_params import SamplingParams

                test_params = SamplingParams(**self.preferred_sampling_params)
                # raises if tokenizer-dependent features used
                test_params.normalize(None)

    def _handle_crash_dump_env(self):
        if not self.crash_dump_folder:
            return
        _CUDA_COREDUMP_DEFAULTS = {
            "CUDA_ENABLE_COREDUMP_ON_EXCEPTION": "1",
            "CUDA_ENABLE_USER_TRIGGERED_COREDUMP": "1",
            "CUDA_COREDUMP_SHOW_PROGRESS": "1",
            "CUDA_COREDUMP_GENERATION_FLAGS": (
                "skip_nonrelocated_elf_images,skip_global_memory,"
                "skip_shared_memory,skip_local_memory,skip_constbank_memory"
            ),
            "CUDA_COREDUMP_FILE": f"{self.crash_dump_folder}/%h/core.cuda.%t.%p",
            "CUDA_COREDUMP_PIPE": "/tmp/corepipe.cuda.%h.%p",
        }
        for key, value in _CUDA_COREDUMP_DEFAULTS.items():
            if key not in os.environ:
                os.environ[key] = value
                logger.info("Auto-set %s=%s (from --crash-dump-folder)", key, value)

                if key == "CUDA_COREDUMP_FILE":
                    # cuda curedump cannot write to a folder that does not exist,
                    # so we have to create the folder first.
                    hostname = socket.gethostname()
                    os.makedirs(
                        os.path.join(self.crash_dump_folder, hostname),
                        exist_ok=True,
                    )

    def _handle_debug_utils(self):
        if is_in_ci() and self.soft_watchdog_timeout is None:
            logger.info("Set soft_watchdog_timeout since in CI")
            self.soft_watchdog_timeout = 300

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser):

        # Auto-derived from Annotated[..., Arg(...)] field metadata.
        add_cli_args_from_dataclass(parser, ServerArgs)

        # --- Fields with dynamic choices (computed at add_cli_args time) ---
        reasoning_parser_choices = list(ReasoningParser.DetectorMap.keys())
        parser.add_argument(
            "--reasoning-parser",
            type=str,
            choices=["auto"] + reasoning_parser_choices,
            default=ServerArgs.reasoning_parser,
            help=f"Specify the parser for reasoning models. "
            f"Use 'auto' to detect from chat template. "
            f"Options include: {reasoning_parser_choices}.",
        )
        tool_call_parser_choices = list(FunctionCallParser.ToolCallParserEnum.keys())
        parser.add_argument(
            "--tool-call-parser",
            type=str,
            choices=["auto"] + tool_call_parser_choices,
            default=ServerArgs.tool_call_parser,
            help=f"Specify the parser for handling tool-call interactions. "
            f"Use 'auto' to detect from chat template. "
            f"Options include: {tool_call_parser_choices}.",
        )
        parser.add_argument(
            "--kv-canary-real-data",
            type=str,
            default=ServerArgs.kv_canary_real_data,
            choices=[m.name.lower() for m in RealKvHashMode],
            help=(
                "Check the real KV-cache in the canary. "
                "'none' (default) disables the feature. "
                "'partial' checks the first 16 bytes of each real-KV slot. "
                "'all' checks the full real-KV slot."
            ),
        )

        # --- Configuration file support ---
        parser.add_argument(
            "--config",
            type=str,
            help="Read CLI options from a config file. Must be a YAML file with configuration options.",
        )

        # --- Deprecated argument registrations ---
        parser.add_argument(
            "--stream-output",
            action=DeprecatedStoreTrueAction,
            dest="incremental_streaming_output",
            new_flag="--incremental-streaming-output",
            help="[Deprecated] Use --incremental-streaming-output instead.",
        )
        parser.add_argument(
            "--prefill-round-robin-balance",
            action=DeprecatedAction,
            help="Note: --prefill-round-robin-balance is deprecated now.",
        )
        parser.add_argument(
            "--collect-tokens-histogram",
            action=DeprecatedAction,
            help="Deprecated. Token histograms are now automatically collected when --enable-metrics is set.",
        )
        parser.add_argument(
            "--nsa-prefill-backend",
            dest="dsa_prefill_backend",
            action=DeprecatedAliasStoreAction,
            new_flag="--dsa-prefill-backend",
            default=argparse.SUPPRESS,
            type=str,
            choices=DSA_CHOICES,
            help="[Deprecated] Use --dsa-prefill-backend instead.",
        )
        parser.add_argument(
            "--nsa-decode-backend",
            dest="dsa_decode_backend",
            action=DeprecatedAliasStoreAction,
            new_flag="--dsa-decode-backend",
            default=argparse.SUPPRESS,
            type=str,
            choices=DSA_CHOICES,
            help="[Deprecated] Use --dsa-decode-backend instead.",
        )
        parser.add_argument(
            "--speculative-dflash-draft-window-size",
            type=int,
            dest="speculative_draft_window_size",
            action=DeprecatedAliasStoreAction,
            new_flag="--speculative-draft-window-size",
            help=argparse.SUPPRESS,
        )
        parser.add_argument(
            "--mamba-scheduler-strategy",
            dest="mamba_radix_cache_strategy",
            type=str,
            action=DeprecatedAliasStoreAction,
            new_flag="--mamba-radix-cache-strategy",
            default=ServerArgs.mamba_radix_cache_strategy,
            help="Deprecated alias for --mamba-radix-cache-strategy.",
        )
        parser.add_argument(
            "--cuda-graph-max-bs",
            type=int,
            action=DeprecatedAliasStoreAction,
            new_flag="--cuda-graph-max-bs-decode",
            dest="cuda_graph_max_bs_decode",
            help="Deprecated alias for --cuda-graph-max-bs-decode.",
        )
        parser.add_argument(
            "--cuda-graph-bs",
            type=int,
            nargs="+",
            action=DeprecatedAliasStoreAction,
            new_flag="--cuda-graph-bs-decode",
            dest="cuda_graph_bs_decode",
            help="Deprecated alias for --cuda-graph-bs-decode.",
        )
        parser.add_argument(
            "--disable-cuda-graph",
            action=DeprecatedStoreTrueAction,
            new_flag="--cuda-graph-backend-{decode,prefill}=disabled",
            help="Deprecated. Use --cuda-graph-backend-{decode,prefill}=disabled instead.",
        )
        parser.add_argument(
            "--enable-breakable-cuda-graph",
            action=DeprecatedStoreConstAction,
            dest="cuda_graph_backend_prefill",
            const_value=Backend.BREAKABLE,
            new_flag="--cuda-graph-backend-prefill=breakable",
            help="Deprecated alias for --cuda-graph-backend-prefill=breakable.",
        )
        parser.add_argument(
            "--disable-piecewise-cuda-graph",
            action=DeprecatedStoreConstAction,
            dest="cuda_graph_backend_prefill",
            const_value=Backend.DISABLED,
            new_flag="--cuda-graph-backend-prefill=disabled",
            help="Deprecated alias for --cuda-graph-backend-prefill=disabled.",
        )
        parser.add_argument(
            "--enforce-piecewise-cuda-graph",
            action=DeprecatedStoreConstAction,
            dest="cuda_graph_backend_prefill",
            const_value=Backend.TC_PIECEWISE,
            new_flag="--cuda-graph-backend-prefill=tc_piecewise",
            help="Deprecated alias for --cuda-graph-backend-prefill=tc_piecewise. "
            "Explicitly setting the prefill backend now skips the auto-disable "
            "cascade automatically.",
        )
        parser.add_argument(
            "--piecewise-cuda-graph-tokens",
            type=int,
            nargs="+",
            action=DeprecatedAliasStoreAction,
            new_flag="--cuda-graph-bs-prefill",
            dest="cuda_graph_bs_prefill",
            help="Deprecated alias for --cuda-graph-bs-prefill.",
        )
        parser.add_argument(
            "--piecewise-cuda-graph-compiler",
            type=str,
            choices=["eager", "inductor"],
            action=DeprecatedAliasStoreAction,
            new_flag="--cuda-graph-tc-compiler",
            dest="cuda_graph_tc_compiler",
            help="Deprecated alias for --cuda-graph-tc-compiler.",
        )
        parser.add_argument(
            "--piecewise-cuda-graph-max-tokens",
            type=int,
            action=DeprecatedAliasStoreAction,
            new_flag="--cuda-graph-max-bs-prefill",
            dest="cuda_graph_max_bs_prefill",
            help="Deprecated alias for --cuda-graph-max-bs-prefill.",
        )
        parser.add_argument(
            "--enable-dsa-prefill-context-parallel",
            dest="enable_dsa_prefill_context_parallel",
            action=DeprecatedStoreTrueAction,
            new_flag="--enable-prefill-cp",
            help="[Deprecated] Use --enable-prefill-cp instead.",
        )
        parser.add_argument(
            "--enable-nsa-prefill-context-parallel",
            dest="enable_dsa_prefill_context_parallel",
            action=DeprecatedStoreTrueAction,
            new_flag="--enable-prefill-cp",
            help="[Deprecated] Use --enable-prefill-cp instead.",
        )
        parser.add_argument(
            "--enable-prefill-context-parallel",
            dest="enable_prefill_context_parallel",
            action=DeprecatedStoreTrueAction,
            new_flag="--enable-prefill-cp",
            help="[Deprecated] Use --enable-prefill-cp instead.",
        )
        parser.add_argument(
            "--dsa-prefill-cp-mode",
            dest="dsa_prefill_cp_mode",
            action=DeprecatedAliasStoreAction,
            new_flag="--cp-strategy",
            type=str,
            default=ServerArgs.dsa_prefill_cp_mode,
            choices=DSA_PREFILL_CP_SPLIT_CHOICES,
            help=(
                "[Deprecated] Use --cp-strategy {zigzag,interleave} instead. "
                "'in-seq-split' maps to 'zigzag'; 'round-robin-split' maps to "
                "'interleave'."
            ),
        )
        parser.add_argument(
            "--nsa-prefill-cp-mode",
            dest="dsa_prefill_cp_mode",
            action=DeprecatedAliasStoreAction,
            new_flag="--cp-strategy",
            type=str,
            default=argparse.SUPPRESS,
            choices=DSA_PREFILL_CP_SPLIT_CHOICES,
            help="[Deprecated] Use --cp-strategy instead.",
        )
        parser.add_argument(
            "--prefill-cp-mode",
            dest="prefill_cp_mode",
            action=DeprecatedAliasStoreAction,
            new_flag="--cp-strategy",
            type=str,
            default=ServerArgs.prefill_cp_mode,
            choices=PREFILL_CP_SPLIT_CHOICES,
            help=(
                "[Deprecated] Use --cp-strategy {zigzag,interleave} instead. "
                "'in-seq-split' maps to 'zigzag'."
            ),
        )
        parser.add_argument(
            "--enable-flashinfer-allreduce-fusion",
            action="store_true",
            help="(Deprecated: use --flashinfer-allreduce-fusion-backend=auto) "
            "Enable FlashInfer allreduce fusion with Residual RMSNorm.",
        )

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace):
        # Some dataclass fields (e.g. stat_loggers) intentionally have no CLI
        # surface and won't appear on the argparse Namespace. Skip them so the
        # dataclass default applies.
        attrs = [
            attr.name for attr in dataclasses.fields(cls) if hasattr(args, attr.name)
        ]
        return cls(**{attr: getattr(args, attr) for attr in attrs})

    def url(self, port: Optional[int] = None):
        scheme = "https" if self.ssl_certfile else "http"
        # When binding to all interfaces, use loopback for internal requests.
        host = self.host
        if not host or host == "0.0.0.0":
            host = "127.0.0.1"
        elif host == "::":
            host = "::1"
        return NetworkAddress(host, port if port is not None else self.port).to_url(
            scheme
        )

    @property
    def engine_info_bootstrap_url(self):
        return self.url(port=self.engine_info_bootstrap_port)

    def ssl_verify(self):
        """Return the value for the requests library's verify= parameter.

        When SSL is configured:
          - If a CA certificate file is provided, return its path so requests
            validates the server certificate against that CA.
          - Otherwise, return False to disable certificate verification
            (suitable for self-signed certificates in development/testing).
            A warning is logged once when this happens.
        When SSL is not configured, return True to use the system's default
        CA bundle.
        """
        if self.ssl_ca_certs:
            return self.ssl_ca_certs
        if self.ssl_certfile:
            if not getattr(self, "_ssl_verify_warned", False):
                logger.warning(
                    "SSL is enabled but --ssl-ca-certs was not provided. "
                    "Certificate verification is DISABLED for internal "
                    "health checks. For production deployments, provide "
                    "--ssl-ca-certs or use CA-signed certificates."
                )
                self._ssl_verify_warned = True
            return False
        return True

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
        return (
            self.disable_radix_cache is False
            and self.mamba_radix_cache_strategy in ("extra_buffer", "extra_buffer_lazy")
        )

    def enable_mamba_extra_buffer_lazy(self) -> bool:
        return (
            self.disable_radix_cache is False
            and self.mamba_radix_cache_strategy == "extra_buffer_lazy"
        )

    @cached_property
    def max_speculative_num_draft_tokens(self) -> Optional[int]:
        """Return the maximum draft-token count speculative decoding may use."""
        if self.speculative_num_draft_tokens is None:
            return None
        if not self.speculative_adaptive:
            return self.speculative_num_draft_tokens

        from sglang.srt.speculative.adaptive_spec_params import (
            resolve_candidate_steps_from_config,
        )

        candidate_steps = resolve_candidate_steps_from_config(
            cfg_path=self.speculative_adaptive_config,
        )
        # TODO: adaptive spec currently requires topk=1, so each runtime state
        # needs steps + 1 draft-token slots. Revisit this if topk>1 is supported.
        return max(candidate_steps) + 1

    @property
    def mamba_cache_chunk_size(self) -> int:
        # For mamba cache with extra buffer, the chunk size is the max of FLA_CHUNK_SIZE
        # (or mamba_chunk_size if it is defined in the model's config) and page_size.
        # It is used to determine the caching point in a sequence during prefill.
        if not hasattr(self, "_mamba_cache_chunk_size"):
            hf_config = self.get_model_config().hf_config
            chunk_size = getattr(hf_config, "mamba_chunk_size", FLA_CHUNK_SIZE)
            assert (
                max(chunk_size, self.page_size) % min(chunk_size, self.page_size) == 0
            ), f"For SSM models, either chunk_size or page_size must be divisible by the other, got {chunk_size=}, {self.page_size=}"
            self._mamba_cache_chunk_size = max(chunk_size, self.page_size)
        return self._mamba_cache_chunk_size

    def check_server_args(self):
        # Check parallel size constraints
        assert (
            self.tp_size * self.pp_size
        ) % self.nnodes == 0, "tp_size must be divisible by number of nodes"

        assert (
            self.pp_max_micro_batch_size is None or self.pp_max_micro_batch_size >= 1
        ), (
            "pp_max_micro_batch_size must be a positive integer or None (for auto-compute). "
            f"Got: {self.pp_max_micro_batch_size}"
        )

        assert not (self.disable_cuda_graph_padding and self.enable_torch_compile), (
            "--disable-cuda-graph-padding is incompatible with --enable-torch-compile. "
            "With padding disabled, every distinct batch size gets its own torch.compile + "
            "Triton autotune cycle (O(max_batch_size) compilations) instead of the small fixed "
            "set of padded bucket sizes, causing engine initialisation to stall for many minutes. "
            "Remove --disable-cuda-graph-padding or --enable-torch-compile."
        )

        if self.pp_size > 1:
            assert (
                self.disable_overlap_schedule and self.speculative_algorithm is None
            ), "Pipeline parallelism is not compatible with overlap schedule, speculative decoding"

        assert not (
            self.dp_size > 1 and self.nnodes != 1 and not self.enable_dp_attention
        ), "multi-node data parallel is not supported unless dp attention!"

        assert self.base_gpu_id >= 0, "base_gpu_id must be non-negative"
        assert self.gpu_id_step >= 1, "gpu_id_step must be positive"

        assert self.moe_dense_tp_size in (
            None,
            1,
            self.tp_size,
        ), "moe_dense_tp_size only supports None, 1, or tp_size currently"

        # Check served model name to not have colon as it is reserved for LoRA adapter syntax
        if not is_runai_obj_uri(self.served_model_name):
            assert ":" not in self.served_model_name, (
                "served_model_name cannot contain a colon (':') character. "
                "The colon is reserved for the 'model:adapter' syntax used in LoRA adapter specification. "
                f"Invalid value: '{self.served_model_name}'"
            )

        # Check LoRA
        self.check_lora_server_args()

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
        assert self.detokenizer_worker_num > 0, "Detokenizer worker num must >= 1"
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
            if self.default_priority_value is None:
                logger.warning(
                    "--default-priority-value is not set while --enable-priority-scheduling is enabled. "
                    "Requests without explicit priority will have priority=None, "
                    "resulting in priority='None' string labels in Prometheus metrics."
                )
        else:
            if self.disable_priority_preemption:
                logger.warning(
                    "--disable-priority-preemption has no effect without --enable-priority-scheduling"
                )
            if self.default_priority_value is not None:
                logger.warning(
                    "--default-priority-value has no effect without --enable-priority-scheduling"
                )

        # Check hisparse
        from sglang.srt.arg_groups.hisparse_hook import validate_hisparse

        validate_hisparse(self)

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

        # Check communications compression
        if self.enable_quant_communications and self.tp_size == 1:
            raise ValueError(
                "Communications quantization is only used with tp_size != 1"
            )

        if self.enable_quant_communications and self.device != "npu":
            raise ValueError(
                "Communications quantization is only supported for NPU device"
            )

        if (
            self.enable_grpc
            and self.grpc_port is not None
            and self.grpc_port == self.port
        ):
            raise ValueError(
                f"SGLANG_GRPC_PORT ({self.grpc_port}) must differ from --port ({self.port})"
            )

        # TODO: Also validate grpc_port != metrics_http_port and grpc_port != nccl_port
        # to avoid opaque bind errors at runtime. Deferred because metrics_http_port
        # and nccl_port have dynamic defaults that may not be resolved yet here.

        if self.gc_threshold:
            if not (1 <= len(self.gc_threshold) <= 3):
                raise ValueError(
                    "When setting gc_threshold, it must contain 1 to 3 integers."
                )

        if self.kv_canary_sweep_interval > 0 and self.kv_canary == "none":
            raise ValueError(
                "--kv-canary-sweep-interval requires --kv-canary in {log, raise}"
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
                                lora_id=LoRARef.deterministic_id(name, path),
                                lora_name=name,
                                lora_path=path,
                                pinned=False,
                            )
                        else:
                            lora_ref = LoRARef(
                                lora_id=LoRARef.deterministic_id(lora_path, lora_path),
                                lora_name=lora_path,
                                lora_path=lora_path,
                                pinned=False,
                            )
                    elif isinstance(lora_path, dict):
                        assert (
                            "lora_name" in lora_path and "lora_path" in lora_path
                        ), f"When providing LoRA paths as a list of dict, each dict should contain 'lora_name' and 'lora_path' keys. Got: {lora_path}"
                        lora_ref = LoRARef(
                            lora_id=LoRARef.deterministic_id(
                                lora_path["lora_name"], lora_path["lora_path"]
                            ),
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
                    LoRARef(
                        lora_id=LoRARef.deterministic_id(k, v),
                        lora_name=k,
                        lora_path=v,
                        pinned=False,
                    )
                    for k, v in self.lora_paths.items()
                ]
            elif self.lora_paths is None:
                self.lora_paths = []
            else:
                raise ValueError(
                    f"Invalid type for --lora-paths: {type(self.lora_paths)}. "
                    "Expected a list or a dictionary."
                )

            # Normalize target modules to a set; keep {"all"} as a sentinel
            # that gets resolved model-awarely in lora_manager.init_lora_shapes().
            if self.lora_target_modules:
                self.lora_target_modules = set(self.lora_target_modules)
                if "all" in self.lora_target_modules:
                    assert (
                        len(self.lora_target_modules) == 1
                    ), "If 'all' is specified in --lora-target-modules, it should be the only module specified."

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

            if self.lora_use_virtual_experts:
                logger.info("Virtual expert computation enabled.")

            assert (
                self.lora_drain_wait_threshold >= 0.0
            ), "--lora-drain-wait-threshold must be non-negative."

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
        try:
            mooncake_available = importlib.util.find_spec("mooncake.engine") is not None
        except (ModuleNotFoundError, ValueError):
            mooncake_available = False
        if not mooncake_available:
            logger.warning(
                "Failed to import mooncake.engine. Does not support using TransferEngine as remote instance weight loader backend."
            )
            return False
        elif self.enable_memory_saver:
            logger.warning(
                "Memory saver is enabled, which is not compatible with TransferEngine. Does not support using TransferEngine as remote instance weight loader backend."
            )
            return False
        else:
            return True

    @property
    def _parsed_modelexpress_config(self) -> dict:
        cache = getattr(self, "_mx_config_cache", None)
        if cache is not None:
            return cache
        if self.modelexpress_config is None:
            result = {}
        elif isinstance(self.modelexpress_config, str):
            result = json.loads(self.modelexpress_config)
        else:
            result = self.modelexpress_config
        object.__setattr__(self, "_mx_config_cache", result)
        return result

    @property
    def modelexpress_url(self) -> Optional[str]:
        return self._parsed_modelexpress_config.get("url")

    @property
    def modelexpress_transport(self) -> str:
        """Transport backend for modelexpress."""
        return self._parsed_modelexpress_config.get("transport", "nixl")

    def remote_instance_weight_loader_use_transfer_engine(self):
        # Use TransferEngine as seed backend.
        if self.remote_instance_weight_loader_start_seed_via_transfer_engine:
            return True
        # Use TransferEngine as client backend.
        if self.load_format == "remote_instance" and (
            self.remote_instance_weight_loader_backend == "transfer_engine"
            or (
                self.remote_instance_weight_loader_backend == "modelexpress"
                and self.modelexpress_transport == "transfer_engine"
            )
        ):
            return True
        else:
            return False

    def describe_kv_events_publisher(self) -> Optional[dict]:
        """Return a structured description of this server's KV-event
        publisher, or `None` if publishing is disabled / misconfigured.

        This is the wire contract surfaced under the `kv_events` key on
        `/server_info` so KV-aware routers (e.g. the SGLang model
        gateway) can subscribe per-worker without operator-supplied port
        coordination. The router constructs the per-DP-rank SUB endpoint
        as tcp://<worker_host>:<endpoint_port_base + dp_rank> for
        every rank reported in dp_size.

        Returned descriptor shape:

            {
                "publisher": "zmq",
                "endpoint_host": "*",             # may be a ZMQ wildcard
                                                  # ("*", "0.0.0.0", "::");
                                                  # subscribers MUST substitute
                                                  # the worker URL's host when
                                                  # dialing
                "endpoint_port_base": 5557,       # base TCP port; per-rank
                                                  # port = base + dp_rank
                "topic": "",                      # ZMQ topic prefix on the
                                                  # SUB filter (empty =
                                                  # subscribe-all)
                "block_size": <page_size>,        # subscribers MUST hash
                                                  # prompts at this size
                "dp_size": <dp_size>,             # number of SUB sockets
                                                  # to open
            }

        Returns None (i.e. "no publisher to describe") when any of:

        * --kv-events-config is unset / empty / malformed JSON,
        * the configured publisher is "null",
        * page_size is missing or non-positive (a placeholder
          block_size would cause silent KV-cache misses by hashing
          prompts at the wrong granularity on the router side),
        * the endpoint is not a routable TCP address (inproc:// /
          ipc://, missing port, non-integer port, or port outside
          1..65535).

        Reuses KVEventsConfig.from_cli for JSON parsing; the inline
        rfind(":") endpoint split mirrors
        ZmqEventPublisher.offset_endpoint_port rather than adding a
        new module-level helper.
        """
        # Lazy import so loading server_args doesn't pull in
        # disaggregation / msgspec / zmq at module top level.
        from sglang.srt.disaggregation.kv_events import KVEventsConfig

        raw = self.kv_events_config
        page_size = self.page_size
        if not raw or page_size is None or page_size <= 0:
            return None
        try:
            cfg = KVEventsConfig.from_cli(raw)
        except Exception:
            # Malformed JSON / schema mismatch. The publisher would
            # have failed at server startup; /server_info must
            # keep working, so just report "no publisher" to consumers.
            return None
        if cfg.publisher == "null" or not cfg.endpoint:
            return None
        if not cfg.endpoint.startswith("tcp://"):
            return None
        body = cfg.endpoint[len("tcp://") :]
        last_colon = body.rfind(":")
        if last_colon < 0:
            return None
        host = body[:last_colon]
        try:
            port = int(body[last_colon + 1 :])
        except ValueError:
            return None
        if not host or not (0 < port < 65536):
            return None
        return {
            "publisher": cfg.publisher,
            "endpoint_host": host,
            "endpoint_port_base": port,
            "topic": cfg.topic,
            "block_size": page_size,
            "dp_size": self.dp_size,
        }


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
    parser = argparse.ArgumentParser(prog="sglang serve")
    ServerArgs.add_cli_args(parser)

    # Check for config file and merge arguments if present
    if "--config" in argv:
        # Import here to avoid circular imports
        from sglang.srt.server_args_config_parser import ConfigArgumentMerger

        # Extract boolean actions from the parser to handle them correctly
        config_merger = ConfigArgumentMerger(parser)
        argv = config_merger.merge_config_with_args(argv)

    raw_args = parser.parse_args(argv)

    # Set up basic logging before ServerArgs.__post_init__ so that
    # logger.info / logger.warning calls there are properly formatted.
    logging.basicConfig(
        level=getattr(logging, raw_args.log_level.upper()),
        format="[%(asctime)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )

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

    # The ipc filename for MultiTokenizerRouter to receive inputs from TokenizerWorker processes (zmq)
    tokenizer_worker_ipc_name: Optional[str]

    # The ipc endpoints between verifier scheduler and drafter scheduler
    decoupled_spec_ipc_config: Optional[DecoupledSpecIpcConfig]

    # zmq address for load snapshot PUSH/PULL (dp-attention TCP mode only;
    # empty when IPC mode derives the address from instance_id).
    load_collector_ipc_name: str = ""

    # Stable token shared by all processes in one server instance, used to
    # derive the /dev/shm path for load snapshots.
    instance_id: str = ""

    @staticmethod
    def init_new(
        server_args: ServerArgs,
        dp_rank: Optional[int] = None,
        worker_ports: Optional[List[int]] = None,
    ) -> PortArgs:
        if server_args.nccl_port is None:
            nccl_port = get_free_port()
        else:
            nccl_port = server_args.nccl_port

        if server_args.tokenizer_worker_num == 1:
            tokenizer_worker_ipc_name = None
        else:
            tokenizer_worker_ipc_name = (
                f"ipc://{tempfile.NamedTemporaryFile(delete=False).name}"
            )

        instance_id = uuid.uuid4().hex[:12]

        decoupled_spec_ipc_config = None
        if server_args.decoupled_spec_role != "null":
            if (
                server_args.decoupled_spec_bind_endpoint is None
                or server_args.decoupled_spec_connect_endpoints is None
                or server_args.decoupled_spec_rank is None
            ):
                raise ValueError(
                    "--decoupled-spec-bind-endpoint, "
                    "--decoupled-spec-connect-endpoints, and "
                    "--decoupled-spec-rank are required for decoupled speculative decoding."
                )
            decoupled_spec_ipc_config = DecoupledSpecIpcConfig(
                bind_endpoint=server_args.decoupled_spec_bind_endpoint,
                connect_endpoints=tuple(server_args.decoupled_spec_connect_endpoints),
                rank=int(server_args.decoupled_spec_rank),
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
                decoupled_spec_ipc_config=decoupled_spec_ipc_config,
                instance_id=instance_id,
            )
        else:
            # DP attention. Use TCP + port to handle both single-node and multi-node.
            if server_args.nnodes == 1 and server_args.dist_init_addr is None:
                derived_port = server_args.port + ZMQ_TCP_PORT_DELTA
                if derived_port > 65535:
                    derived_port = server_args.port - ZMQ_TCP_PORT_DELTA
                na = NetworkAddress("127.0.0.1", derived_port)
            else:
                na = NetworkAddress.parse(server_args.dist_init_addr)

            dist_init_host = na.host
            dist_init_port = na.port

            # We need 5 consecutive ports from port_base for:
            # port_base, detokenizer, rpc, metrics, scheduler.
            # In multi-node, all nodes derive ports independently from
            # dist_init_port, so the derivation must be deterministic
            # (no availability-based search). If incrementing would
            # overflow the valid TCP range, decrement instead.
            NUM_DERIVED_PORTS = 5
            if dist_init_port + NUM_DERIVED_PORTS > 65535:
                port_base = dist_init_port - NUM_DERIVED_PORTS - 1
            else:
                port_base = dist_init_port + 1

            detokenizer_port = port_base + 1
            rpc_port = port_base + 2
            metrics_port = port_base + 3
            load_collector_port = port_base + 5
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
                    wait_port_available(metrics_port, "metrics_port")
                    if server_args.nnodes > 1:
                        wait_port_available(load_collector_port, "load_collector_port")
                # Check scheduler_input_port only for dp.
                # Skip check when using worker_ports since the port is already bound by our ZMQ socket
                if dp_rank is None or worker_ports is None:
                    wait_port_available(scheduler_input_port, "scheduler_input_port")
            except ValueError:
                logger.exception(
                    f"Port is already in use. {dist_init_port=} {port_base=} {detokenizer_port=} {nccl_port=} {scheduler_input_port=}"
                )
                raise

            return PortArgs(
                tokenizer_ipc_name=NetworkAddress(dist_init_host, port_base).to_tcp(),
                scheduler_input_ipc_name=NetworkAddress(
                    dist_init_host, scheduler_input_port
                ).to_tcp(),
                detokenizer_ipc_name=NetworkAddress(
                    dist_init_host, detokenizer_port
                ).to_tcp(),
                nccl_port=nccl_port,
                rpc_ipc_name=NetworkAddress(dist_init_host, rpc_port).to_tcp(),
                metrics_ipc_name=NetworkAddress(dist_init_host, metrics_port).to_tcp(),
                tokenizer_worker_ipc_name=tokenizer_worker_ipc_name,
                decoupled_spec_ipc_config=decoupled_spec_ipc_config,
                load_collector_ipc_name=NetworkAddress(
                    dist_init_host, load_collector_port
                ).to_tcp(),
                instance_id=instance_id,
            )
