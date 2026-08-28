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
"""Server argument declarations, resolution, and CLI registration.

Keep this file in the following top-level order:

1. Imports and the module logger.
2. Public extension-point choice lists, with each legacy ``add_*`` alias
   immediately below the choice list it extends.
3. Shared (non-extensible) choice lists, scalar defaults, and deprecated
   aliases. A choice list used by only one field belongs inline in that field.
4. ``ServerArgs``: fields first, then resolution/validation helpers, then CLI
   registration and small query helpers. New resolution steps are appended at
   the end of ``_run_resolution_pipeline``, immediately before resolution is
   marked complete, unless an earlier dependency is documented explicitly.
5. Module-level ``ServerArgs`` construction/runtime shims.
6. Networking constants and ``PortArgs``.

Model- or vendor-specific utilities belong in ``sglang.srt.arg_groups`` (or
their owning subsystem), not before ``ServerArgs`` in this module.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import functools
import glob
import importlib
import importlib.util
import json
import logging
import math
import os
import tempfile
import uuid
from typing import Any, Callable, Dict, List, Literal, Optional, Union

from sglang.kernels.ops.kv_canary.consts import RealKvHashMode
from sglang.srt.arg_groups.arg_utils import NS, A, Arg, add_cli_args_from_dataclass
from sglang.srt.arg_groups.argparse_actions import (
    DeprecatedAction,
    DeprecatedAliasStoreAction,
    DeprecatedStoreConstAction,
    DeprecatedStoreTrueAction,
    LoRAPathAction,
)
from sglang.srt.arg_groups.overrides import (
    attention_backends_of,
    declare_direct_writes,
    mamba_extra_buffer_lazy_of,
    mamba_extra_buffer_of,
    remote_instance_transfer_engine_of,
    resolved_view,
    resolving_view,
)
from sglang.srt.distributed.device_communicators.mooncake_transfer_engine import (
    parse_ib_device_config,
)
from sglang.srt.environ import envs
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.srt.hardware_backend.mlx.runtime import use_mlx
from sglang.srt.lora.lora_registry import LoRARef
from sglang.srt.model_executor.cuda_graph_config import (
    ALLOWED_BACKENDS_PER_PHASE,
    Backend,
    CudaGraphConfig,
    Phase,
    parse_cuda_graph_config_arg,
    with_phase,
)
from sglang.srt.parser.reasoning_parser import ReasoningParser
from sglang.srt.platforms import current_platform
from sglang.srt.speculative.decoupled_spec_io import DecoupledSpecIpcConfig
from sglang.srt.utils.common import (
    LORA_TARGET_ALL_MODULES,
    SUPPORTED_LORA_TARGET_MODULES,
    get_device_memory_capacity,
    human_readable_int,
    is_cuda,
    is_flashinfer_available,
    is_hip,
    is_hopper_with_cuda_12_3,
    is_mps,
    is_musa,
    is_no_spec_infer_or_topk_one,
    is_npu,
    is_sm100_supported,
    is_xpu,
    json_list_type,
    nullable_str,
    torch_release,
)
from sglang.srt.utils.network import NetworkAddress, get_free_port, wait_port_available
from sglang.srt.utils.runai_utils import is_runai_obj_uri

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Extension points: out-of-tree platforms and plugins extend these lists
# before ServerArgs is constructed. Each list owns its adder on the line
# below it. A list with no adder is not an extension point -- inline it into
# the field's Arg(choices=...) instead of hoisting it here.
# --------------------------------------------------------------------------

# --- Model loading and quantization ---

LOAD_FORMAT_CHOICES = [
    "auto",
    "pt",
    "safetensors",
    "npcache",
    "dummy",
    "sharded_state",
    "presharded",
    "gguf",
    # Experimental and intentionally narrow: expert_pack is validated only for
    # DeepSeek-V4-Flash-0731 MXFP4 GGUF (MXFP4 experts, FP8 dense weights)
    # and KIMI-K3-MXP4-DERISKED-Q2_K-*.gguf (Q2_K gate/up, Q3_K down weights):
    # https://huggingface.co/unsloth/DeepSeek-V4-Flash-0731-GGUF
    # https://huggingface.co/Blackfrost-AI/KIMI-K3-Q2_K-GGUF-ABLITERATED
    "expert_pack",
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
add_load_format_choices = LOAD_FORMAT_CHOICES.extend
# NOTE: LoadFormat.IPC_CACHE intentionally has no public --load-format choice.
# It is an internal dispatch format set automatically by ModelRunner when the
# weight cache is enabled (weight_cache_mode != "off"). Exposing it as a CLI
# choice let users create contradictory combos (see _handle_load_format).

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
    "w4afp8",
    "mxfp4",  # MOE-only.
    "auto-round",
    "auto-round-int8",
    "compressed-tensors",  # for Ktransformers
    "modelslim",  # for NPU
    "mxfp_w4a8",  # for NPU W4A8 (MXFP4 weights + MXFP8 activations)
    "quark",  # AMD Quark quantizer (FP8 / MXFP4 / Int4FP8 etc.)
    "quark_int4fp8_moe",
    "quark_mxfp4",  # Online MOE + linear quantization (incl. NVFP4 -> MXFP4 requantization).
    # Apple Silicon MLX backend — on-the-fly quantization of fp16 weights at load
    # time via mlx.nn.quantize. Only takes effect when SGLANG_USE_MLX=1.
    "mlx_q4",  # 4 bits, group_size=64 (mlx-community default)
    "mlx_q8",  # 8 bits, group_size=64
    "unquant",
    "humming",
]
add_quantization_method_choices = QUANTIZATION_CHOICES.extend

# --- Attention backends ---

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
    "hpc_ops",  # HPC-Ops (https://github.com/Tencent/hpc-ops), Hopper (SM90) only, requires --page-size 64
    "minicpm_flashattn",
    "minicpm_flashinfer",
    # AMD specific
    "aiter",
    "wave",
    # Other platforms
    "intel_amx",
    "ascend",
    "intel_xpu",
]
add_attention_backend_choices = ATTENTION_BACKEND_CHOICES.extend

# trtllm_mha is valid for decode-only dense-MQA drafts. DFLASH rejects it
# earlier when its per-layer attention requirements are not met.
DRAFT_ATTENTION_BACKEND_CHOICES = [
    "flashinfer",
    "fa3",
    "fa4",
    "triton",
    "ascend",
    "trtllm_mha",
]
add_draft_attention_backend_choices = DRAFT_ATTENTION_BACKEND_CHOICES.extend

# Attention backends whose kernels read the chunked prefix-cache layout.
# Out-of-tree platforms may extend this list (via
# add_chunked_prefix_cache_attention_backend) before ServerArgs construction;
# the chunked-prefix gate is evaluated during resolution.
CHUNKED_PREFIX_CACHE_SUPPORTED_ATTENTION_BACKENDS = [
    "flashinfer",
    "fa3",
    "fa4",
    "flashmla",
    "cutedsl_mla",
    "cutlass_mla",
    "trtllm_mla",
    "tokenspeed_mla",
]
add_chunked_prefix_cache_attention_backend = (
    CHUNKED_PREFIX_CACHE_SUPPORTED_ATTENTION_BACKENDS.append
)

DETERMINISTIC_ATTENTION_BACKEND_CHOICES = [
    "ascend",
    "fa3",
    "fa4",
    "flashinfer",
    "intel_xpu",
    "triton",
]
add_deterministic_attention_backend_choices = (
    DETERMINISTIC_ATTENTION_BACKEND_CHOICES.extend
)

RADIX_SUPPORTED_DETERMINISTIC_ATTENTION_BACKEND = ["ascend", "fa3", "fa4", "triton"]
add_radix_supported_deterministic_attention_backend_choices = (
    RADIX_SUPPORTED_DETERMINISTIC_ATTENTION_BACKEND.extend
)

# --- Transport ---

DISAGG_TRANSFER_BACKEND_CHOICES = [
    "mooncake",
    "nixl",
    "ascend",
    "fake",
    "mori",
    "mooncake_tcp",
]
add_disagg_transfer_backend_choices = DISAGG_TRANSFER_BACKEND_CHOICES.extend

# --- Sampling and grammar ---

GRAMMAR_BACKEND_CHOICES = ["xgrammar", "outlines", "llguidance", "none"]
add_grammar_backend_choices = GRAMMAR_BACKEND_CHOICES.extend

SAMPLING_BACKEND_CHOICES = {"flashinfer", "pytorch", "ascend"}

# --- MoE and GEMM runners ---

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
    "humming",
    "experimental_sgl_marlin",
    "hpc_ops",  # HPC-Ops (https://github.com/Tencent/hpc-ops), FP8 MoE on Hopper (SM90) only
    "megamoe",
    "intel_xpu",
]
add_moe_runner_backend_choices = MOE_RUNNER_BACKEND_CHOICES.extend

MXFP8_MOE_RUNNER_BACKEND_CHOICES = [
    "cutlass",
    "deep_gemm",
    "flashinfer_trtllm",
    "flashinfer_trtllm_routed",
]
add_mxfp8_moe_runner_backend_choices = MXFP8_MOE_RUNNER_BACKEND_CHOICES.extend

FP8_GEMM_RUNNER_BACKEND_CHOICES = [
    "auto",
    "deep_gemm",
    "flashinfer_trtllm",
    "flashinfer_cutlass",
    "flashinfer_deepgemm",
    "flashinfer_cutedsl",
    "cutlass",
    "triton",
    "aiter",
]
add_fp8_gemm_runner_backend_choices = FP8_GEMM_RUNNER_BACKEND_CHOICES.extend

FP4_GEMM_RUNNER_BACKEND_CHOICES = [
    "auto",
    "flashinfer_cudnn",
    "flashinfer_cutedsl",
    "flashinfer_cutlass",
    "flashinfer_trtllm",
    "marlin",
]
add_fp4_gemm_runner_backend_choices = FP4_GEMM_RUNNER_BACKEND_CHOICES.extend

# --- Cache and scheduling policy ---

RADIX_EVICTION_POLICY_CHOICES = ["lru", "lfu", "slru", "priority"]
add_radix_eviction_policy_choices = RADIX_EVICTION_POLICY_CHOICES.extend

# --- Reinforcement learning ---

RL_ON_POLICY_TARGET_CHOICES = ["fsdp"]
add_rl_on_policy_target_choices = RL_ON_POLICY_TARGET_CHOICES.extend

# --- Linear attention ---

LINEAR_ATTN_KERNEL_BACKEND_CHOICES = [
    "triton",
    "cutedsl",
    "flashinfer",
    "flashkda",
    "nvidia_kda",
    "ptx_kda",
    "helion",
    "intel_xpu",
]
add_linear_attn_kernel_backend_choices = LINEAR_ATTN_KERNEL_BACKEND_CHOICES.extend

# --------------------------------------------------------------------------
# Add new extension points at the end of the matching group above. A new
# choice list is inlined into its field by default; hoisting one here makes
# it public API for out-of-tree code and is a deliberate decision.
# --------------------------------------------------------------------------


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
        NS("model"),
    ]
    tokenizer_path: A[Optional[str], "The path of the tokenizer.", NS("serving")] = None
    tokenizer_mode: A[
        str,
        Arg(
            help="Tokenizer mode. 'auto' will use the fast tokenizer if available, "
            "and 'slow' will always use the slow tokenizer.",
            choices=["auto", "slow"],
        ),
        NS("serving"),
    ] = "auto"
    tokenizer_backend: A[
        str,
        Arg(
            help="Tokenizer backend. 'huggingface' uses the default HuggingFace "
            "tokenizers library, and 'fastokens' uses the fastokens library "
            "for faster tokenization. Requires the fastokens package to be installed.",
            choices=["huggingface", "fastokens"],
        ),
        NS("serving"),
    ] = "huggingface"
    tokenizer_worker_num: A[
        int, "The worker num of the tokenizer manager.", NS("serving")
    ] = 1
    detokenizer_worker_num: A[
        int, "The worker num of the detokenizer manager.", NS("serving")
    ] = 1
    skip_tokenizer_init: A[
        bool,
        "If set, skip init tokenizer and pass input_ids in generate request.",
        NS("serving"),
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
            '"expert_pack" is experimental and loads only the validated '
            "DeepSeek-V4-Flash-0731 MXFP4 or text-only Kimi-K3 Q2_K GGUF "
            "model with routed experts stored in an SSD expert pack. "
            '"bitsandbytes" will load the weights using bitsandbytes '
            "quantization."
            '"layered" loads weights layer by layer so that one can quantize a '
            "layer before loading another to make the peak memory envelope "
            "smaller."
            '"presharded" performs a normal first-time load (with quantization), '
            "then dumps a per-rank/per-tensor sharded checkpoint with content "
            "deduplication into "
            "<model_path>/presharded/<parallelism+quant subfolder>/. "
            "Subsequent runs with the same parallelism+quantization config "
            "load directly from this presharded checkpoint and skip "
            "re-quantization. "
            "The dump directory must be on a shared filesystem across all "
            "ranks/nodes. Optional model_loader_extra_config roots: "
            "presharded_path (target) and draft_presharded_path (speculative "
            "draft); each replaces <model_path>/presharded and still gets a "
            "config subfolder appended. Use a writable path when model_path "
            "is read-only (e.g. HF cache mounts).",
            choices=LOAD_FORMAT_CHOICES,
        ),
        NS("model"),
    ] = "auto"
    model_loader_extra_config: A[
        str,
        "Extra config for model loader. This will be passed to the model loader "
        "corresponding to the chosen load_format. For load_format=presharded, "
        "JSON may include presharded_path (target cache root), "
        "draft_presharded_path (draft cache root), max_file_bytes, "
        "hash_num_threads, and verify_on_load.",
        NS("model"),
    ] = "{}"
    trust_remote_code: A[
        bool,
        "Whether or not to allow for custom models defined on the Hub in their own modeling files.",
        NS("model"),
    ] = False
    context_length: A[
        Optional[int],
        Arg(
            help="The model's maximum context length. Defaults to None (will use the value from the model's config.json instead)."
            f"\n\n{human_readable_int.__doc__}",
            type_parser=human_readable_int,
        ),
        NS("model"),
    ] = None
    is_embedding: A[
        bool, "Whether to use a CausalLM as an embedding model.", NS("model")
    ] = False
    enable_multimodal: A[
        Optional[bool],
        "Enable the multimodal functionality for the served model. If the model being served is not multimodal, nothing will happen",
        NS("mm"),
    ] = None
    revision: A[
        Optional[str],
        "The specific model version to use. It can be a branch name, a tag name, or a commit id. If unspecified, will use the default version.",
        NS("model"),
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
        NS("model"),
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
        NS("model"),
    ] = "auto"
    json_model_override_args: A[
        str,
        "A dictionary in JSON string format used to override default model configurations.",
        NS("model"),
    ] = "{}"

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
            resolvable=True,
        ),
        NS("model"),
    ] = "auto"
    quantization: A[
        Optional[str],
        Arg(
            help="The quantization method.",
            choices=QUANTIZATION_CHOICES,
            resolvable=True,
        ),
        NS("model"),
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
        NS("model"),
    ] = None
    kv_cache_dtype: A[
        str,
        Arg(
            help=(
                'Data type for kv cache storage. "auto" will use model data type. '
                '"bf16" or "bfloat16" for BF16 KV cache. "fp8_e5m2" and '
                '"fp8_e4m3" are supported for CUDA 11.8+. "mxfp8" is supported '
                'by the FA4 backend. "nvfp4" selects '
                'the NVFP4 FP4 E2M1 KV cache recipe; "fp4_mx_block16" '
                "selects the MX-style block-size-16 FP4 E2M1 KV cache "
                "recipe. Both require CUDA 12.8+ and PyTorch 2.8.0+"
            ),
            choices=[
                "auto",
                "fp8_e5m2",
                "fp8_e4m3",
                "mxfp8",
                "bf16",
                "bfloat16",
                "nvfp4",
                "fp4_mx_block16",
                "fp4_e2m1",
            ],
            resolvable=True,
        ),
        NS("model"),
    ] = "auto"
    enable_fp32_lm_head: A[
        bool, "If set, the LM head outputs (logits) are in FP32.", NS("exec.features")
    ] = False
    modelopt_quant: A[
        Optional[Union[str, Dict]],
        (
            "The ModelOpt quantization configuration. Supported values: 'fp8', "
            "'int4_awq', 'w4a8_awq', 'nvfp4', 'nvfp4_awq'. This requires the "
            "NVIDIA Model Optimizer library to be installed: pip install "
            "nvidia-modelopt"
        ),
        NS("model"),
    ] = None
    modelopt_checkpoint_restore_path: A[
        Optional[str],
        (
            "Path to restore a previously saved ModelOpt quantized checkpoint. "
            "If provided, the quantization process will be skipped and the model "
            "will be loaded from this checkpoint."
        ),
        NS("model"),
    ] = None
    modelopt_checkpoint_save_path: A[
        Optional[str],
        (
            "Path to save the ModelOpt quantized checkpoint after quantization. "
            "This allows reusing the quantized model in future runs."
        ),
        NS("model"),
    ] = None
    modelopt_export_path: A[
        Optional[str],
        (
            "Path to export the quantized model in HuggingFace format after "
            "ModelOpt quantization. The exported model can then be used directly "
            "with SGLang for inference. If not provided, the model will not be "
            "exported."
        ),
        NS("model"),
    ] = None
    quantize_and_serve: A[
        bool,
        (
            "Quantize the model with ModelOpt and immediately serve it without "
            "exporting. This is useful for development and prototyping. For "
            "production, it's recommended to use separate quantization and "
            "deployment steps."
        ),
        NS("model"),
    ] = False
    rl_quant_profile: A[
        Optional[str],
        "Path to the FlashRL quantization profile. Required when using --load-format flash_rl.",
        NS("model"),
    ] = None  # For flash_rl load format
    enable_tf32_matmul: A[
        bool,
        Arg(
            help="Enable float32 matmuls to use TensorFloat32 precision for better performance (via torch.set_float32_matmul_precision). CUDA only.",
            resolvable=True,
        ),
        NS("exec.features"),
    ] = False

    # -------------------------------------------------------------------------
    # Memory and scheduling
    # -------------------------------------------------------------------------
    mem_fraction_static: A[
        Optional[float],
        "The fraction of the memory used for static allocation (model weights and KV cache memory pool). Use a smaller value if you see out-of-memory errors.",
        NS("schedule"),
    ] = None
    max_running_requests: A[
        Optional[int], "The maximum number of running requests.", NS("schedule")
    ] = None
    max_queued_requests: A[
        Optional[int],
        "The maximum number of queued requests. This option is ignored when using disaggregation-mode.",
        NS("schedule"),
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
        NS("schedule"),
    ] = None
    chunked_prefill_size: A[
        Optional[int],
        "The maximum number of tokens in a chunk for the chunked prefill. Setting this to -1 means disabling chunked prefill.",
        NS("schedule"),
    ] = None
    prefill_decode_interval: A[
        int,
        "The number of decode rounds to run after a prefill batch before scheduling the next prefill. In data-parallel attention mode, the interval is synchronized across all DP ranks. Set to 0 to disable.",
        NS("schedule"),
    ] = 0
    enable_dynamic_chunking: A[
        bool,
        "Enable dynamic chunk size adjustment for pipeline parallelism. When enabled, chunk sizes are dynamically calculated based on fitted function to maintain consistent execution time across chunks.",
        NS("schedule"),
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
        NS("schedule"),
    ] = 16384
    prefill_max_requests: A[
        Optional[int],
        "The maximum number of requests in a prefill batch. If not specified, there is no limit.",
        NS("schedule"),
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
        NS("schedule"),
    ] = "fcfs"
    enable_priority_scheduling: A[
        bool,
        "Enable priority scheduling. Requests with higher priority integer values will be scheduled first by default.",
        NS("schedule"),
    ] = False
    disable_priority_preemption: A[
        bool, "Disable priority scheduling preemption.", NS("schedule")
    ] = False
    default_priority_value: A[
        Optional[int],
        "Default priority for requests without explicit priority.",
        NS("schedule"),
    ] = None
    abort_on_priority_when_disabled: A[
        bool,
        "If set, abort requests that specify a priority when priority scheduling is disabled.",
        NS("schedule"),
    ] = False
    schedule_low_priority_values_first: A[
        bool,
        "If specified with --enable-priority-scheduling, the scheduler will schedule requests with lower priority integer values first.",
        NS("schedule"),
    ] = False
    priority_scheduling_preemption_threshold: A[
        int,
        "Minimum difference in priorities for an incoming request to have to preempt running request(s).",
        NS("schedule"),
    ] = 10
    retraction_policy: A[
        str,
        Arg(
            help=(
                "The decode retraction policy to use when the KV cache is full. "
                "'length' preserves the existing behavior and retracts short-output, "
                "long-input requests first. 'priority' retracts lower-priority "
                "requests first, using the same priority direction as priority "
                "scheduling."
            ),
            choices=["length", "priority"],
        ),
        NS("schedule"),
    ] = "length"
    schedule_conservativeness: A[
        float,
        "How conservative the schedule policy is. A larger value means more conservative scheduling. Use a larger value if you see requests being retracted frequently.",
        NS("schedule"),
    ] = 1.0
    page_size: A[
        Optional[int],
        Arg(help="The number of tokens in a page.", resolvable=True),
        NS("schedule"),
    ] = None
    c128_page_size: A[
        int,
        "The physical page size of the NPU DSV4 C128 KV cache. Must be a positive multiple of 16.",
        NS("schedule"),
    ] = 16
    swa_full_tokens_ratio: A[
        float,
        Arg(
            help=(
                "The ratio of SWA layer KV tokens / full layer KV tokens, regardless "
                "of the number of swa:full layers. It should be between 0 and 1. "
                "E.g. 0.5 means if each swa layer has 50 tokens, then each full "
                "layer has 100 tokens."
            ),
            resolvable=True,
        ),
        NS("schedule"),
    ] = 0.8
    disable_hybrid_swa_memory: A[
        bool,
        Arg(help="Disable the hybrid SWA memory pool.", resolvable=True),
        NS("schedule"),
    ] = False
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
        NS("memory"),
    ] = "lru"
    prefill_only_disable_kv_cache: A[
        bool,
        "Skip the physical KV cache allocation for embedding-mode prefill-only workloads. Currently only valid with --is-embedding, --chunked-prefill-size=-1, --disable-radix-cache, an FA prefill backend, and non-FP4 KV cache so the fa_skip_kv_cache path is active (no layer reads or writes the cache). Other prefill-only workloads such as scoring/MIS may benefit from this later once their attention paths stop using paged KV. Scheduler admission accounting is unchanged; per-layer K/V tensors are sized to (page_size, head_num, head_dim) placeholders so GPU memory is not wasted.",
        NS("schedule"),
    ] = False
    disable_radix_cache: A[
        bool,
        Arg(
            help="Disable RadixAttention for prefix caching.",
            resolvable=True,
        ),
        NS("memory"),
    ] = False
    enable_page_major_kv_layout: A[
        bool,
        "Enable the page-major KV layout: lay out the Mamba state and full/SWA "
        "KV caches in a page-granularity envelope (page is the outermost axis, "
        "layer-major within a page) instead of the default per-layer "
        "(layer-major) layout. Requires the Triton attention / linear-attn / "
        "Mamba backends.",
        NS("memory"),
    ] = False
    enable_unified_memory: A[
        bool,
        "Replace the statically-partitioned hybrid-model pools (full-attn KV + "
        "SWA/Mamba state) with one byte buffer split dynamically between "
        "sub-pools. Requires the Triton attention / linear-attn / Mamba "
        "backends; not yet compatible with PD disaggregation or speculative "
        "decoding.",
        NS("memory"),
    ] = False
    disable_chunked_prefix_cache: A[
        bool,
        "Disable chunked prefix cache feature for deepseek, which should save overhead for short sequences.",
        NS("schedule"),
    ] = False
    disable_overlap_schedule: A[
        bool,
        Arg(
            help="Disable the overlap scheduler, which overlaps the CPU scheduler with GPU model worker.",
            resolvable=True,
        ),
        NS("schedule"),
    ] = False
    num_continuous_decode_steps: A[
        int,
        "Run multiple continuous decoding steps to reduce scheduling overhead. This can potentially increase throughput but may also increase time-to-first-token latency. The default value is 1, meaning only run one decoding step at a time.",
        NS("schedule"),
    ] = 1
    scheduler_recv_interval: A[
        int,
        "The interval to poll requests in scheduler. Can be set to >1 to reduce the overhead of this.",
        NS("schedule"),
    ] = 1
    enable_mixed_chunk: A[
        bool,
        "Enabling mixing prefill and decode in a batch when using chunked prefill.",
        NS("schedule"),
    ] = False

    # -------------------------------------------------------------------------
    # Distributed topology and parallelism (TP, PP, DP, CP)
    # -------------------------------------------------------------------------
    nccl_port: A[
        Optional[int],
        "The port for NCCL distributed environment setup. Defaults to a random port.",
        NS("parallel"),
    ] = None
    dist_timeout: A[
        Optional[int],
        "Set timeout for torch.distributed initialization.",
        NS("parallel"),
    ] = None
    dist_init_addr: A[
        Optional[str],
        Arg(
            help="The host address for initializing distributed backend (e.g., `192.168.0.2:25000`).",
            aliases=["--nccl-init-addr"],
        ),
        NS("parallel"),
    ] = None
    gated_launch_port: A[
        Optional[int],
        "The port of the gated launch control server. When set, every rank blocks right after the distributed environment is initialized, before any sizable GPU allocation, until `POST /gate/activate` is sent to this port on the host of the first rank. This lets an external orchestrator defer the memory hungry part of startup to a safe window. Defaults to None, which disables the gate.",
        NS("parallel"),
    ] = None
    nnodes: A[int, "The number of nodes.", NS("parallel")] = 1
    node_rank: A[int, "The node rank.", NS("parallel")] = 0
    tp_size: A[
        int,
        Arg(
            help="The tensor parallelism size.",
            aliases=["--tensor-parallel-size"],
        ),
        NS("parallel"),
    ] = 1
    dcp_size: A[
        int,
        Arg(
            help="The decode context parallelism size.",
            aliases=["--decode-context-parallel-size"],
        ),
        NS("parallel"),
    ] = 1
    pp_size: A[
        int,
        Arg(
            help="The pipeline parallelism size.",
            aliases=["--pipeline-parallel-size"],
        ),
        NS("parallel"),
    ] = 1
    pp_max_micro_batch_size: A[
        Optional[int],
        "The maximum micro batch size in pipeline parallelism.",
        NS("parallel"),
    ] = None
    pp_async_batch_depth: A[
        int, "The async batch depth of pipeline parallelism.", NS("parallel")
    ] = 0
    dp_size: A[
        int,
        Arg(
            help="The data parallelism size.",
            aliases=["--data-parallel-size"],
        ),
        NS("parallel"),
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
        NS("parallel"),
    ] = "auto"
    attn_cp_size: A[
        int,
        Arg(
            help="The attention context parallelism size.",
            aliases=["--attention-context-parallel-size"],
            resolvable=True,
        ),
        NS("parallel"),
    ] = 1
    moe_dp_size: A[
        int,
        Arg(
            help="The moe data parallelism size.",
            aliases=["--moe-data-parallel-size"],
        ),
        NS("parallel"),
    ] = 1
    dwdp_size: A[
        int,
        Arg(
            help="DWDP (Distributed Weight Data Parallelism) group size. "
            "When > 1, MoE prefill uses weight prefetch instead of token all-to-all. "
            "Must equal tp_size. Only supported with --disaggregation-mode null or prefill.",
        ),
        NS("parallel"),
    ] = 1
    dcp_comm_backend: A[
        str,
        Arg(
            help="Communication backend for the decode context-parallel (DCP) "
            "attention reduction: 'ag_rs' (AllGather + ReduceScatter), 'a2a' "
            "(fused NCCL All-to-All exchange of output+LSE + local Triton LSE "
            "combine), or 'fi_a2a' (FlashInfer MNNVL All-to-All kernel; requires "
            "SM90+ and MNNVL fabric memory, e.g. GB200 NVL72).",
            choices=["ag_rs", "a2a", "fi_a2a"],
            resolvable=True,
        ),
        NS("parallel"),
    ] = "ag_rs"
    dcp_replicate_q_proj: A[
        Optional[bool],
        Arg(
            help="For MLA decode context parallelism with the a2a/fi_a2a "
            "backend: replicate the Q projection so each DCP rank computes the "
            "full-head query locally (redundant projection compute), eliminating "
            "the per-layer head-dim all-gather of Q. Trades a small amount of "
            "extra GEMM for one fewer collective per layer. Use "
            "--no-dcp-replicate-q-proj to disable the model-specific default.",
            action=argparse.BooleanOptionalAction,
            resolvable=True,
        ),
        NS("parallel"),
    ] = None
    enable_prefill_cp: A[
        bool,
        "Enable context parallelism for the prefill phase. Select the layout with --cp-strategy.",
        NS("parallel"),
    ] = False
    cp_strategy: A[
        Optional[str],
        Arg(
            help="Sharding strategy for prefill CP. 'zigzag' is the former in-seq-split mode; 'interleave' is the former round-robin-split mode.",
            choices=("zigzag", "interleave"),
        ),
        NS("parallel"),
    ] = None
    # Split DSA GPU KV/indexer cache layers across CP ranks.
    enable_dsa_cache_layer_split: A[
        bool,
        "Split DSA (DeepSeek Sparse Attention) GPU KV/indexer cache layers across context-parallel ranks to reduce per-rank KV memory. Currently only supported with the mooncake transfer backend (mooncake / mooncake_tcp); mori/nixl support will be added later by the community.",
        NS("parallel"),
    ] = False
    enable_dsa_prefill_context_parallel: A[bool, Arg(no_cli=True), NS("parallel")] = (
        False
    )
    dsa_prefill_cp_mode: A[str, Arg(no_cli=True), NS("parallel")] = "round-robin-split"
    enable_prefill_context_parallel: A[bool, Arg(no_cli=True), NS("parallel")] = False
    prefill_cp_mode: A[str, Arg(no_cli=True), NS("parallel")] = "in-seq-split"
    enable_cp_decode_attn_tp: A[
        bool,
        "Enable attention tensor-parallel weight slicing during decode under context parallel (cp_size>1). Slices the replicated attention linears to the local CP partition, eliminating redundant decode GEMMs.",
        NS("parallel"),
    ] = False
    # DP attention
    enable_dp_attention: A[
        bool,
        Arg(
            help="Enabling data parallelism for attention and tensor parallelism for FFN. The dp size should be equal to the tp size. Currently DeepSeek-V2 and Qwen 2/3 MoE models are supported.",
            resolvable=True,
        ),
        NS("parallel"),
    ] = False
    enable_dp_attention_local_control_broadcast: A[
        bool,
        "With DP-attention, send control messages to every DP group leader and broadcast within attn_tp_group instead of the full tp_group. Eliminates a costly all-ranks gloo sync on every scheduler iteration.",
        NS("parallel"),
    ] = False
    enable_dp_lm_head: A[
        bool,
        Arg(
            help="Enable vocabulary parallel across the attention TP group to avoid all-gather across DP groups, optimizing performance under DP attention.",
            resolvable=True,
        ),
        NS("parallel"),
    ] = False
    enable_tp_lm_head_all_to_all: A[
        Optional[bool],
        Arg(
            help="Use all-to-all instead of TP all-gather followed by DP scatter "
            "for the TP-sharded LM head under DP attention. By default this is "
            "enabled only on decode-only PD nodes with pure DP attention "
            "(tp_size == dp_size > 1 and attn_cp_size == 1), and disabled on "
            "prefill-only and colocated nodes. Pass "
            "--no-enable-tp-lm-head-all-to-all to opt out. The path is "
            "incompatible with --enable-dp-lm-head; batches without an equal "
            "padded row count fall back to the existing all-gather path.",
            action=argparse.BooleanOptionalAction,
            resolvable=True,
        ),
        NS("parallel"),
    ] = None
    enable_attn_tp_input_scattered: A[
        bool,
        "Allow input of attention to be scattered when only using tensor parallelism, to reduce the computational load of operations such as qkv latent.",
        NS("parallel"),
    ] = False
    enable_shared_experts_attn_tp: A[
        bool,
        "Shard shared expert weights across the attention TP group when using an expert-parallel all-to-all backend.",
        NS("parallel"),
    ] = False
    enable_dense_mlp_attn_tp: A[
        bool,
        "Shard dense MLP weights across the attention TP group under DP attention.",
        NS("parallel"),
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
        NS("parallel"),
    ] = False
    enable_p2p_check: A[
        bool,
        "Enable P2P check for GPU access, otherwise the p2p access is allowed by default.",
        NS("parallel"),
    ] = False

    # -------------------------------------------------------------------------
    # Device info and server timeout
    # -------------------------------------------------------------------------
    device: A[
        Optional[str],
        "The device to use ('cuda', 'xpu', 'hpu', 'npu', 'cpu', 'musa'). Defaults to auto-detection if not specified.",
        NS("device"),
    ] = None
    base_gpu_id: A[
        int,
        "The base GPU ID to start allocating GPUs from. Useful when running multiple instances on the same machine.",
        NS("device"),
    ] = 0
    gpu_id_step: A[
        int,
        "The delta between consecutive GPU IDs that are used. For example, setting it to 2 will use GPU 0,2,4,...",
        NS("device"),
    ] = 1
    random_seed: A[Optional[int], "The random seed.", NS("device")] = None
    mlx_enable_sampling: A[
        bool,
        (
            "MLX backend only: sample decode tokens (temperature / top-k / "
            "top-p / min-p) instead of greedy argmax. Sampling runs inside "
            "the lazy MLX graph, so it works with the overlap scheduler; "
            "first tokens from prefill/extend are sampled too. Greedy "
            "requests keep exact argmax behavior. Also enables on the MLX "
            "path: grammar vocab masks and custom logit processors (these "
            "break decode chaining per step; custom processors run on "
            "pure-decode steps only), logit_bias, output logprobs (sampled "
            "token / top-k / token_ids; prompt input logprobs are not "
            "computed), NaN sanitization (SGLANG_SANITIZE_NAN_LOGITS), and "
            "per-request sampling_seed under "
            "--enable-deterministic-inference (deterministic within MLX "
            "only). Penalties are not applied."
        ),
        NS("device"),
    ] = False
    watchdog_timeout: A[
        float,
        "Set watchdog timeout in seconds. If a forward batch takes longer than this, the server will crash to prevent hanging.",
        NS("device"),
    ] = 300
    soft_watchdog_timeout: A[
        Optional[float],
        "Set soft watchdog timeout in seconds. If a forward batch takes longer than this, the server will dump information for debugging.",
        NS("device"),
    ] = None
    sleep_on_idle: A[bool, "Reduce CPU usage when sglang is idle.", NS("device")] = (
        False
    )
    use_ray: A[
        bool, "Use Ray actors for scheduler process management.", NS("device")
    ] = False
    custom_sigquit_handler: A[Optional[Callable], NS("device")] = None
    numa_node: A[
        Optional[List[int]],
        "Sets the numa node for the subprocesses. i-th element corresponds to i-th subprocess. If unset, will be automatically detected on NUMA systems.",
        NS("device"),
    ] = None
    gc_threshold: A[
        Optional[List[int]],
        "Set the garbage collection thresholds (the collection frequency). Accepts 1 to 3 integers.",
        NS("device"),
    ] = None

    # -------------------------------------------------------------------------
    # HTTP server
    # -------------------------------------------------------------------------
    host: A[str, "The host of the HTTP server.", NS("serving")] = "127.0.0.1"
    port: A[int, "The port of the HTTP server.", NS("serving")] = 30000
    fastapi_root_path: A[
        str, "App is behind a path based routing proxy.", NS("serving")
    ] = ""
    smg_grpc_mode: A[
        bool,
        "Use the legacy SMG gRPC server (smg-grpc-servicer) instead of the HTTP "
        "server. Replaces the deprecated --grpc-mode.",
        NS("serving"),
    ] = False
    grpc_mode: A[
        bool,
        "(Deprecated, use --smg-grpc-mode) Legacy SMG gRPC server selector.",
        NS("serving"),
    ] = False
    grpc_port: A[
        Optional[int],
        "Port for the native gRPC server, started alongside HTTP. Setting this "
        "(or SGLANG_GRPC_PORT) enables the native gRPC server; it is off by "
        "default. In legacy --smg-grpc-mode this is the SMG server port and "
        "defaults to --port + 10000.",
        NS("serving"),
    ] = None
    # Env-only (SGLANG_GRPC_WORKER_THREADS); a field so the projection sees it.
    grpc_worker_threads: A[Optional[int], Arg(no_cli=True), NS("serving")] = None
    sidecar: A[
        Optional[str],
        "Start a locally managed sidecar against the native gRPC server. "
        "The selected module must expose main(argv) and read the resolved "
        "native gRPC endpoint from SGLANG_GRPC_ENDPOINT. Requires --grpc-port "
        "or SGLANG_GRPC_PORT.",
        NS("serving"),
    ] = None
    sidecar_args: A[
        Optional[List[str]],
        Arg(
            help="JSON array passed to the selected sidecar module's "
            "main(argv) function. --sidecar-shutdown-timeout SECONDS is "
            "consumed by SGLang.",
            type_parser=json_list_type,
        ),
        NS("serving"),
    ] = None
    skip_server_warmup: A[bool, "If set, skip warmup.", NS("serving")] = False
    warmups: A[
        Optional[str],
        "Specify custom warmup functions (csv) to run before server starts eg. --warmups=warmup_name1,warmup_name2 will run the functions `warmup_name1` and `warmup_name2` specified in warmup.py before the server starts listening for requests",
        NS("serving"),
    ] = None
    enable_http2: A[
        bool,
        "Use Granian instead of Uvicorn as the ASGI server, enabling HTTP/1.1 and HTTP/2 auto-negotiation. Clients may use h2c (cleartext HTTP/2) or plain HTTP/1.1. Requires 'pip install sglang[http2]'.",
        NS("serving"),
    ] = False
    http2_max_concurrent_streams: A[
        int,
        "Maximum number of concurrent streams advertised on each HTTP/2 "
        "connection (1 to 2^32 - 1). Only applies with --enable-http2.",
        NS("serving"),
    ] = 200

    # -------------------------------------------------------------------------
    # SSL/TLS
    # -------------------------------------------------------------------------
    ssl_keyfile: A[
        Optional[str], "The file path to the SSL key file.", NS("serving")
    ] = None
    ssl_certfile: A[
        Optional[str], "The file path to the SSL certificate file.", NS("serving")
    ] = None
    ssl_ca_certs: A[Optional[str], "The CA certificates file.", NS("serving")] = None
    ssl_keyfile_password: A[
        Optional[str], "The password to decrypt the SSL keyfile.", NS("serving")
    ] = None
    enable_ssl_refresh: A[
        bool,
        "Enable automatic SSL certificate hot-reloading when cert/key files change on disk. Requires --ssl-certfile and --ssl-keyfile.",
        NS("serving"),
    ] = False

    # -------------------------------------------------------------------------
    # API related
    # -------------------------------------------------------------------------
    api_key: A[
        Optional[str],
        "Set API key of the server. It is also used in the OpenAI API compatible server.",
        NS("serving"),
    ] = None
    admin_api_key: A[
        Optional[str],
        "Set admin API key for sensitive management endpoints (e.g. /clear_hicache_storage_backend). When set, admin endpoints require this key and do NOT accept --api-key.",
        NS("serving"),
    ] = None
    served_model_name: A[
        Optional[str],
        "Override the model name returned by the v1/models endpoint in OpenAI API server.",
        NS("serving"),
    ] = None
    weight_version: A[
        str,
        "Version identifier for the model weights. Defaults to 'default' if not specified.",
        NS("serving"),
    ] = "default"
    chat_template: A[
        Optional[str],
        "The buliltin chat template name or the path of the chat template file. This is only used for OpenAI-compatible API server.",
        NS("serving"),
    ] = None
    hf_chat_template_name: A[
        Optional[str],
        "When the HuggingFace tokenizer has multiple chat templates (e.g., 'default', 'tool_use', 'rag'), specify which named template to use. If not set, the first available template is used.",
        NS("serving"),
    ] = None
    completion_template: A[
        Optional[str],
        "The buliltin completion template name or the path of the completion template file. This is only used for OpenAI-compatible API server. only for code completion currently.",
        NS("serving"),
    ] = None
    file_storage_path: A[
        str, "The path of the file storage in backend.", NS("serving")
    ] = "sglang_storage"
    enable_cache_report: A[
        bool,
        "Return number of cached tokens in usage.prompt_tokens_details for each openai request.",
        NS("serving"),
    ] = False
    reasoning_parser: A[Optional[str], NS("serving")] = None
    default_chat_template_kwargs: A[
        Optional[Dict[str, Any]],
        Arg(
            help="Default chat template kwargs applied to every request when not "
            "overridden per-request. Keys must match what the model's chat template "
            "expects (e.g. enable_thinking, thinking, reasoning_effort). Per-request "
            "chat_template_kwargs takes precedence.",
            type_parser=json.loads,
        ),
        NS("serving"),
    ] = None
    strip_thinking_cache: A[
        bool,
        "Skip caching reasoning-model output (thinking + answer) in the radix tree on finish; keep only the prompt prefix. Opt-in: changes cache contents.",
        NS("serving"),
    ] = False
    enable_strict_thinking: A[
        bool,
        "Enable strict token filtering during the thinking phase. Blocks model-specific excluded tokens (e.g., tool call markers) during reasoning. Requires a grammar backend that supports token filtering.",
        NS("serving"),
    ] = False
    tool_call_parser: A[Optional[str], NS("serving")] = None
    tool_server: A[
        Optional[str],
        "Either 'demo' or a comma-separated list of tool server urls to use for the model. If not specified, no tool server will be used.",
        NS("serving"),
    ] = None
    sampling_defaults: A[
        str,
        Arg(
            help="Where to get default sampling parameters. 'openai' uses SGLang/OpenAI defaults (temperature=1.0, top_p=1.0, etc.). 'model' uses the model's generation_config.json to get the recommended sampling parameters if available. Default is 'model'.",
            choices=["openai", "model"],
        ),
        NS("serving"),
    ] = "model"
    asr_max_buffer_seconds: A[
        int,
        "Maximum seconds of PCM audio the streaming ASR WebSocket handler will accumulate before closing the session with a buffer_overflow error. Guards against OOM when a client streams audio faster than inference can consume it. Default 60s.",
        NS("serving"),
    ] = 60
    asr_max_concurrent_sessions: A[
        int,
        "Maximum number of concurrent realtime ASR WebSocket sessions served by /v1/realtime. New connections beyond this cap are accepted, sent an error{code:too_many_sessions} frame, and closed. Default 32.",
        NS("serving"),
    ] = 32
    preferred_sampling_params: A[
        Optional[str],
        Arg(
            help="json-formatted sampling settings that will be returned in /get_model_info",
            type_parser=json.loads,
        ),
        NS("serving"),
    ] = None
    allow_auto_truncate: A[
        bool,
        "Allow automatically truncating requests that exceed the maximum input length instead of returning an error.",
        NS("serving"),
    ] = False

    # -------------------------------------------------------------------------
    # Streaming
    # -------------------------------------------------------------------------
    stream_interval: A[
        int,
        "The interval (or buffer size) for streaming in terms of the token length. A smaller value makes streaming smoother, while a larger value makes the throughput higher",
        NS("serving"),
    ] = 1
    batch_notify_size: A[
        int,
        "Number of streaming notifications to batch before yielding to the event loop. Reduces asyncio wakeup overhead under high concurrency.",
        NS("serving"),
    ] = 16
    stream_response_default_include_usage: A[
        bool,
        "Include usage in every streaming response (even when stream_options is not specified).",
        NS("serving"),
    ] = False
    incremental_streaming_output: A[
        bool, "Whether to output as a sequence of disjoint segments.", NS("serving")
    ] = False
    enable_streaming_session: A[
        bool,
        "Enable streaming session mode and StreamingSession wrapper.",
        NS("serving"),
    ] = False
    enable_session_radix_cache: A[
        bool,
        "Track per-session references on UnifiedRadixCache KV: eviction consumes unreferenced entries before referenced ones, and closing a session only dereferences its KV.",
        NS("memory"),
    ] = False

    # -------------------------------------------------------------------------
    # Logging, metrics, and tracing
    # -------------------------------------------------------------------------
    log_level: A[str, "The logging level of all loggers.", NS("observability")] = "info"
    log_level_http: A[
        Optional[str],
        "The logging level of HTTP server. If not set, reuse --log-level by default.",
        NS("observability"),
    ] = None
    log_requests: A[
        bool,
        "Log metadata, inputs, outputs of all requests. The verbosity is decided by --log-requests-level",
        NS("observability"),
    ] = False
    log_requests_level: A[
        int,
        Arg(
            help="0: Log metadata (no sampling parameters). 1: Log metadata and sampling parameters. 2: Log metadata, sampling parameters and partial input/output. 3: Log every input/output.",
            choices=[0, 1, 2, 3],
        ),
        NS("observability"),
    ] = 2
    log_requests_format: A[
        str,
        Arg(
            help="Format for request logging: 'text' (human-readable) or 'json' (structured)",
            choices=["text", "json"],
        ),
        NS("observability"),
    ] = "text"
    log_requests_target: A[
        Optional[List[str]],
        "Target(s) for request logging: 'stdout' and/or directory path(s) for file output. Can specify multiple targets, e.g., '--log-requests-target stdout /my/path'. ",
        NS("observability"),
    ] = None
    uvicorn_access_log_exclude_prefixes: A[
        List[str],
        Arg(
            help="Exclude uvicorn access logs whose request path starts with any of these prefixes. Defaults to empty (disabled). Example: --uvicorn-access-log-exclude-prefixes /metrics /health",
            nargs="*",
        ),
        NS("observability"),
    ] = dataclasses.field(default_factory=list)
    crash_dump_folder: A[
        Optional[str],
        "Folder path to dump requests from the last 5 min before a crash (if any). If not specified, crash dumping is disabled.",
        NS("observability"),
    ] = None
    show_time_cost: A[bool, "Show time cost of custom marks.", NS("observability")] = (
        False
    )
    enable_metrics: A[bool, "Enable log prometheus metrics.", NS("observability")] = (
        False
    )
    smg_http_sidecar_port: A[
        Optional[int],
        Arg(
            help="Port for the HTTP sidecar server in legacy SMG gRPC mode (--smg-grpc-mode). Serves Prometheus metrics and profiling endpoints. Defaults to --port + 1. Not used in HTTP mode.",
            aliases=["--grpc-http-sidecar-port"],
        ),
        NS("observability"),
    ] = None
    enable_mfu_metrics: A[
        bool, "Enable estimated MFU-related prometheus metrics.", NS("observability")
    ] = False
    enable_metrics_for_all_schedulers: A[
        bool,
        "Enable --enable-metrics-for-all-schedulers when you want schedulers on all TP ranks (not just TP 0) to record request metrics separately. This is especially useful when dp_attention is enabled, as otherwise all metrics appear to come from TP 0.",
        NS("observability"),
    ] = False
    load_snapshot_publish_interval: A[
        int,
        "Publish load snapshot to shared memory every N decode iterations. Prefill and idle always publish immediately.",
        NS("observability"),
    ] = 15
    tokenizer_metrics_custom_labels_header: A[
        str,
        "Specify the HTTP header for passing custom labels for tokenizer metrics.",
        NS("observability"),
    ] = "x-custom-labels"
    tokenizer_metrics_allowed_custom_labels: A[
        Optional[List[str]],
        "The custom labels allowed for tokenizer metrics. The labels are specified via a dict in '--tokenizer-metrics-custom-labels-header' field in HTTP requests, e.g., {'label1': 'value1', 'label2': 'value2'} is allowed if '--tokenizer-metrics-allowed-custom-labels label1 label2' is set.",
        NS("observability"),
    ] = None
    extra_metric_labels: A[
        Optional[Dict[str, str]],
        Arg(
            help='The custom labels for metrics. e.g. \'{"label1": "value1", "label2": "value2"}\'',
            type_parser=json.loads,
        ),
        NS("observability"),
    ] = None
    bucket_time_to_first_token: A[
        Optional[List[float]],
        "The buckets of time to first token, specified as a list of floats.",
        NS("observability"),
    ] = None
    bucket_inter_token_latency: A[
        Optional[List[float]],
        "The buckets of inter-token latency, specified as a list of floats.",
        NS("observability"),
    ] = None
    bucket_e2e_request_latency: A[
        Optional[List[float]],
        "The buckets of end-to-end request latency, specified as a list of floats.",
        NS("observability"),
    ] = None
    prompt_tokens_buckets: A[
        Optional[List[str]],
        "The buckets rule of prompt tokens. "
        "Supports 3 rule types: 'default' uses predefined buckets; 'tse <middle> <base> <count>' "
        "generates two sides exponential distributed buckets (e.g., 'tse 1000 2 8' generates buckets "
        "[984.0, 992.0, 996.0, 998.0, 1000.0, 1002.0, 1004.0, 1008.0, 1016.0]).); 'custom <value1> "
        "<value2> ...' uses custom bucket values (e.g., 'custom 10 50 100 500').",
        NS("observability"),
    ] = None
    generation_tokens_buckets: A[
        Optional[List[str]],
        "The buckets rule for generation tokens histogram. "
        "Supports 3 rule types: 'default' uses predefined buckets; 'tse <middle> <base> <count>' "
        "generates two sides exponential distributed buckets (e.g., 'tse 1000 2 8' generates buckets "
        "[984.0, 992.0, 996.0, 998.0, 1000.0, 1002.0, 1004.0, 1008.0, 1016.0]).); 'custom <value1> "
        "<value2> ...' uses custom bucket values (e.g., 'custom 10 50 100 500').",
        NS("observability"),
    ] = None
    gc_warning_threshold_secs: A[
        float,
        "The threshold for long GC warning. If a GC takes longer than this, a warning will be logged. Set to 0 to disable.",
        NS("observability"),
    ] = 0.0
    decode_log_interval: A[
        int,
        "The log and metrics reporting interval (in decode iterations) for decode batches.",
        NS("observability"),
    ] = 40
    enable_request_time_stats_logging: A[
        bool, "Enable per request time stats logging", NS("observability")
    ] = False
    kv_events_config: A[
        Optional[str],
        "Config in json format for NVIDIA dynamo KV event publishing. Publishing will be enabled if this flag is used. Runtime-load publishing for load-aware routers is a separate opt-in; see --load-publish-endpoint.",
        NS("observability"),
    ] = None
    load_publish_endpoint: A[
        Optional[str],
        "Opt in to the runtime-load PUB socket that load-aware routers subscribe to. Off by default (unset or 'off'). Use 'auto' to reserve the dp_size ports packed after the --kv-events-config range, or a wildcard-host TCP address (e.g. tcp://*:6000) to place it explicitly; rank r binds port+r and /server_info advertises the base under the kv_events block. Requires --kv-events-config to describe a publisher (routers discover the base through /server_info); startup fails if this is set without one, is not bindable, or overlaps the KV range. Note: 'auto' reserves 2*dp_size ports from the KV base — space co-hosted engines accordingly. The router-facing update cadence follows --load-snapshot-publish-interval (shared to avoid double-collecting the snapshot), so a large value there also staleness-caps this feed.",
        NS("observability"),
    ] = None
    enable_forward_pass_metrics: A[
        bool,
        "Enable per-iteration forward pass metrics via ZMQ IPC. External consumers (e.g. Dynamo planner) subscribe to the IPC endpoint exposed in server_args.forward_pass_metrics_ipc_name.",
        NS("observability"),
    ] = False
    forward_pass_metrics_worker_id: A[
        str, Arg(help=argparse.SUPPRESS), NS("observability")
    ] = ""
    forward_pass_metrics_ipc_name: A[
        Optional[str], Arg(help=argparse.SUPPRESS), NS("observability")
    ] = None
    enable_trace: A[bool, "Enable opentelemetry trace", NS("observability")] = False
    trace_modules: A[
        str,
        "Select the components to trace. Available options are 'request' and 'mooncake'. Format: <module1 name>,<module2 name>,...",
        NS("observability"),
    ] = "request"
    otlp_traces_endpoint: A[
        str,
        "Config opentelemetry collector endpoint if --enable-trace is set. format: <ip>:<port>",
        NS("observability"),
    ] = "localhost:4317"
    # RequestMetricsExporter configuration
    export_metrics_to_file: A[
        bool,
        "Export performance metrics for each request to local file (e.g. for forwarding to external systems).",
        NS("observability"),
    ] = False
    export_metrics_to_file_dir: A[
        Optional[str],
        "Directory path for writing performance metrics files (required when --export-metrics-to-file is enabled).",
        NS("observability"),
    ] = None
    # Class-level DI for the five *MetricsCollector classes. Maps collector role
    # (one of: "scheduler", "tokenizer", "storage", "radix_cache", "expert_dispatch")
    # to a subclass of the matching base collector. The five instantiation sites
    # read from this map and fall back to the base class. Class-object only (no
    # CLI surface) since this exists for embedded use cases that pass a Python
    # class directly. Default None preserves existing behavior.
    stat_loggers: A[Optional[Dict[str, type]], NS("observability")] = None

    # -------------------------------------------------------------------------
    # Constrained decoding
    # -------------------------------------------------------------------------
    constrained_json_whitespace_pattern: A[
        Optional[str],
        "(outlines and llguidance backends only) Regex pattern for syntactic whitespaces allowed in JSON constrained output. For example, to allow the model generate consecutive whitespaces, set the pattern to [\n\t ]*",
        NS("serving"),
    ] = None
    constrained_json_disable_any_whitespace: A[
        bool,
        "(xgrammar and llguidance backends only) Enforce compact representation in JSON constrained output.",
        NS("serving"),
    ] = False

    # -------------------------------------------------------------------------
    # Kernel backend
    # -------------------------------------------------------------------------
    attention_backend: A[
        Optional[str],
        Arg(
            help="Choose the kernels for attention layers.",
            choices=ATTENTION_BACKEND_CHOICES,
            resolvable=True,
        ),
        NS("exec.kernel"),
    ] = None
    decode_attention_backend: A[
        Optional[str],
        Arg(
            help="Choose the kernels for decode attention layers (have priority over --attention-backend).",
            choices=ATTENTION_BACKEND_CHOICES,
            resolvable=True,
        ),
        NS("exec.kernel"),
    ] = None
    prefill_attention_backend: A[
        Optional[str],
        Arg(
            help="Choose the kernels for prefill attention layers (have priority over --attention-backend).",
            choices=ATTENTION_BACKEND_CHOICES,
            resolvable=True,
        ),
        NS("exec.kernel"),
    ] = None
    sampling_backend: A[
        Optional[str],
        Arg(
            help="Choose the kernels for sampling layers.",
            no_cli=True,
            resolvable=True,
        ),
        NS("exec.kernel"),
    ] = None
    grammar_backend: A[
        Optional[str],
        Arg(
            help="Choose the backend for grammar-guided decoding.",
            choices=GRAMMAR_BACKEND_CHOICES,
        ),
        NS("exec.kernel"),
    ] = None
    radix_cache_backend: A[
        Optional[str],
        "Name of a radix-cache backend previously registered via register_radix_cache_backend. Omit this flag to use the built-in default cache selection chain.",
        NS("memory"),
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
        NS("mm"),
    ] = None
    fp8_gemm_runner_backend: A[
        str,
        Arg(
            help="Choose the runner backend for Blockwise FP8 GEMM operations. Options: 'auto' (default, auto-selects based on hardware; MXFP8 dense picks flashinfer_cutedsl on SM100/SM103 and FlashInfer CUTLASS on other supported Blackwell GPUs), 'deep_gemm' (JIT-compiled; enabled by default on NVIDIA Hopper (SM90) and Blackwell (SM100) when DeepGEMM is installed), 'flashinfer_trtllm' (optimal for Blackwell and low-latency), 'flashinfer_cutlass' (FlashInfer CUTLASS groupwise FP8 GEMM), 'flashinfer_cutedsl' (FlashInfer CuTe DSL MXFP8 GEMM on SM100/SM103), 'flashinfer_deepgemm' (Hopper SM90 only; uses swapAB optimization for small M dimensions in decoding), 'cutlass' (optimal for SM120 GPUs), 'triton' (fallback, widely compatible), 'aiter' (ROCm only). ",
            cli_name="--fp8-gemm-backend",
            choices=FP8_GEMM_RUNNER_BACKEND_CHOICES,
            resolvable=True,
        ),
        NS("exec.kernel"),
    ] = "auto"
    fp4_gemm_runner_backend: A[
        str,
        Arg(
            help="Choose the runner backend for NVFP4 GEMM operations. Options: 'auto' (default; selects flashinfer_cutedsl on SM100, marlin on SM80-SM90, flashinfer_cutlass otherwise (including SM120)), 'flashinfer_cutlass' (FlashInfer CUTLASS backend), 'flashinfer_cudnn' (FlashInfer cuDNN backend, optimal on CUDA 13+ with cuDNN 9.15+), 'flashinfer_cutedsl' (FlashInfer CuTe DSL backend), 'flashinfer_trtllm' (FlashInfer TensorRT-LLM backend, requires different weight preparation with shuffling), 'marlin' (weight-only W4A16 fallback for SM80+). ",
            cli_name="--fp4-gemm-backend",
            choices=FP4_GEMM_RUNNER_BACKEND_CHOICES,
            resolvable=True,
        ),
        NS("exec.kernel"),
    ] = "auto"
    bf16_gemm_backend: A[
        str,
        Arg(
            help="Choose the backend for unquantized BF16 GEMM operations. Options: 'auto' (default; selects 'cutedsl' on SM10x GPUs, except deterministic inference selects 'torch'; otherwise uses cuBLAS via torch.nn.functional.linear), 'cutedsl' (SGLang JIT CuTe DSL TGV BF16 GEMM on SM10x; dispatches between the CuTe DSL kernel and cuBLAS), 'torch' (always uses cuBLAS via torch.nn.functional.linear).",
            cli_name="--bf16-gemm-backend",
            choices=["auto", "cutedsl", "gemv", "torch"],
        ),
        NS("exec.kernel"),
    ] = "auto"
    dsa_prefill_backend: A[
        Optional[str],
        Arg(
            help="DSA (DeepSeek Sparse Attention) prefill backend. If not specified, auto-detects based on hardware and kv_cache_dtype.",
            choices=[
                "flashmla_sparse",
                "flashmla_sparse_q8",
                "flashmla_kv",
                "flashmla_auto",
                "flashinfer_sparse_mla",
                "fa3",
                "tilelang",
                "aiter",
                "trtllm",
            ],
            resolvable=True,
        ),
        NS("exec.kernel"),
    ] = None
    dsv4_prefill_backend: A[
        str,
        Arg(
            help=(
                "DeepSeek-V4 sparse prefill backend. 'auto' and "
                "'flashmla_sparse' use the existing BF16 sparse prefill path; "
                "'flashmla_sparse_q8' enables the Q8KV8 sparse prefill path."
            ),
            choices=["auto", "flashmla_sparse", "flashmla_sparse_q8"],
        ),
        NS("exec.kernel"),
    ] = "auto"
    dsa_decode_backend: A[
        Optional[str],
        Arg(
            help="DSA (DeepSeek Sparse Attention) decode backend. If not specified, auto-detects based on hardware and kv_cache_dtype.",
            choices=[
                "flashmla_sparse",
                "flashmla_sparse_q8",
                "flashmla_kv",
                "flashmla_auto",
                "flashinfer_sparse_mla",
                "fa3",
                "tilelang",
                "aiter",
                "trtllm",
            ],
            resolvable=True,
        ),
        NS("exec.kernel"),
    ] = None
    dsa_paged_mqa_logits_backend: A[
        str,
        Arg(
            help="DSA indexer paged MQA logits kernel backend. Options: 'auto' (default; DeepGEMM on CUDA, aiter on ROCm), 'deepgemm', 'cutedsl' (CuTe DSL kernel, SM 100 (Blackwell) only; wins at low batch size and long context), 'aiter' (ROCm only).",
            choices=["auto", "deepgemm", "cutedsl", "aiter"],
        ),
        NS("exec.kernel"),
    ] = "auto"
    dsa_topk_backend: A[
        str,
        Arg(
            help="DSA indexer top-k backend for the target model. Options: 'sgl-kernel', 'torch', 'flashinfer'. The 'torch' backend currently requires SGLANG_DSA_FUSE_TOPK=false.",
            choices=["sgl-kernel", "torch", "flashinfer"],
        ),
        NS("exec.kernel"),
    ] = "sgl-kernel"
    disable_flashinfer_autotune: A[
        bool, "Disable FlashInfer autotuning.", NS("exec.kernel")
    ] = False
    flashinfer_autotune_skip_ops: A[
        Optional[List[str]],
        Arg(
            help=(
                "FlashInfer custom-op identifiers to skip during autotuning. "
                "Skipped ops use FlashInfer's heuristic fallback. SGLang "
                "temporarily skips mxfp8_gemm by default due to an IMA."
            ),
            nargs="+",
        ),
        NS("exec.kernel"),
    ] = None
    mamba_backend: A[
        str,
        Arg(
            help="Choose the kernel backend for Mamba SSM operations. Default is 'triton'. Options: 'triton' (default), 'flashinfer' (requires FlashInfer with Mamba support).",
            choices=["triton", "flashinfer"],
        ),
        NS("exec.mamba"),
    ] = "triton"

    # -------------------------------------------------------------------------
    # Cuda graphs
    # -------------------------------------------------------------------------
    cuda_graph_config: A[
        Optional[CudaGraphConfig],
        Arg(
            help='Per-phase CUDA graph settings as JSON, e.g. \'{"decode":{"backend":"full","max_bs":256},"prefill":{"backend":"tc_piecewise","tc_compiler":"eager"}}\'. Allowed backends per phase: full, breakable, tc_piecewise, disabled (full is decode-only). JSON wins over the per-phase --cuda-graph-* convenience flags and over legacy flags.',
            type_parser=parse_cuda_graph_config_arg,
        ),
        NS("exec.graph"),
    ] = None
    cuda_graph_backend_decode: A[
        Optional[Literal["full", "breakable", "tc_piecewise", "disabled"]],
        Arg(
            help="Backend for the decode phase. Folds into cuda_graph_config[decode].backend.",
            choices=Backend.ALL,
        ),
        NS("exec.graph"),
    ] = None
    cuda_graph_backend_prefill: A[
        Optional[Literal["full", "breakable", "tc_piecewise", "disabled"]],
        Arg(
            help="Backend for the prefill phase. Folds into cuda_graph_config[prefill].backend.",
            choices=Backend.ALL,
        ),
        NS("exec.graph"),
    ] = None
    cuda_graph_max_bs_decode: A[
        Optional[int],
        "Maximum batch size captured for the decode cuda graph.",
        NS("exec.graph"),
    ] = None
    cuda_graph_max_bs_prefill: A[
        Optional[int],
        "Maximum batch size captured for the prefill cuda graph.",
        NS("exec.graph"),
    ] = None
    cuda_graph_bs_decode: A[
        Optional[List[int]],
        "Explicit list of batch sizes to capture for the decode cuda graph.",
        NS("exec.graph"),
    ] = None
    cuda_graph_bs_prefill: A[
        Optional[List[int]],
        "Explicit list of batch sizes to capture for the prefill cuda graph.",
        NS("exec.graph"),
    ] = None
    cuda_graph_tc_compiler: A[
        Optional[Literal["eager", "inductor"]],
        "Compiler used by the tc_piecewise backend (currently only the prefill phase consumes it).",
        NS("exec.graph"),
    ] = None
    disable_prefill_cuda_graph: A[
        bool,
        "Disable the prefill-phase CUDA graph. Convenience for --cuda-graph-backend-prefill=disabled.",
        NS("exec.graph"),
    ] = False
    disable_decode_cuda_graph: A[
        bool,
        "Disable the decode-phase CUDA graph. Convenience for --cuda-graph-backend-decode=disabled.",
        NS("exec.graph"),
    ] = False
    disable_cuda_graph: A[bool, Arg(no_cli=True), NS("exec.graph")] = False
    disable_cuda_graph_padding: A[
        bool,
        "Disable cuda graph when padding is needed. Still uses cuda graph when padding is not needed.",
        NS("exec.graph"),
    ] = False
    enable_profile_cuda_graph: A[
        bool, "Enable profiling of cuda graph capture.", NS("exec.graph")
    ] = False
    enable_cudagraph_gc: A[
        bool,
        "Enable garbage collection during CUDA graph capture. If disabled (default), GC is frozen during capture to speed up the process.",
        NS("exec.graph"),
    ] = False
    debug_cuda_graph: A[
        bool,
        "Enable debug/eager mode for CUDA graph using breakable CUDA graph. When enabled, graph breaks are inserted so every operation runs eagerly while still going through the CUDA graph capture / replay path. Useful for debugging CUDA graph capture / replay issues.",
        NS("exec.graph"),
    ] = False

    # -------------------------------------------------------------------------
    # Communication and kernels
    # -------------------------------------------------------------------------
    enable_layerwise_nvtx_marker: A[
        bool,
        "Enable layerwise NVTX profiling annotations for the model.",
        NS("exec.comm"),
    ] = False
    enable_nccl_nvls: A[
        bool,
        "Enable NCCL NVLS for prefill heavy requests when available.",
        NS("exec.comm"),
    ] = False
    enable_symm_mem: A[
        bool,
        Arg(
            help="Enable NCCL symmetric memory for fast collectives.",
            resolvable=True,
        ),
        NS("exec.comm"),
    ] = False
    triton_attention_reduce_in_fp32: A[
        bool,
        "Cast the intermediate attention results to fp32 to avoid possible crashes related to fp16."
        "This only affects Triton attention kernels.",
        NS("exec.kernel"),
    ] = False
    triton_attention_num_kv_splits: A[
        int,
        "The number of KV splits in flash decoding Triton kernel. Larger value is better in longer context scenarios. The default value is 8.",
        NS("exec.kernel"),
    ] = 8
    triton_attention_split_tile_size: A[
        Optional[int],
        "The size of split KV tile in flash decoding Triton kernel. Used for deterministic inference.",
        NS("exec.kernel"),
    ] = None
    flashinfer_mla_disable_ragged: A[
        bool,
        "Not using ragged prefill wrapper when running flashinfer mla",
        NS("exec.kernel"),
    ] = False
    enable_fused_qk_norm_rope: A[
        bool,
        "Enable fused qk normalization and rope rotary embedding.",
        NS("exec.kernel"),
    ] = False
    enable_precise_embedding_interpolation: A[
        bool,
        "Enable corner alignment for resize of embeddings grid to ensure more accurate(but slower) evaluation of interpolated embedding values.",
        NS("exec.kernel"),
    ] = False
    enable_fused_moe_sum_all_reduce: A[
        bool, "Enable fused moe triton and sum all reduce.", NS("exec.moe")
    ] = False
    enable_deepseek_v4_fp4_indexer: A[
        bool,
        "Enable the experimental FP4 C4 indexer path for DeepSeek V4. Default keeps the existing indexer implementation.",
        NS("exec.kernel"),
    ] = False
    disable_custom_all_reduce: A[
        bool,
        Arg(
            help="Disable the custom all-reduce kernel and fall back to NCCL.",
            resolvable=True,
        ),
        NS("exec.comm"),
    ] = False
    enable_mscclpp: A[
        bool,
        "Enable using mscclpp for small messages for all-reduce kernel and fall back to NCCL.",
        NS("exec.comm"),
    ] = False
    enable_torch_symm_mem: A[
        bool,
        "Enable using torch symm mem for all-reduce kernel and fall back to NCCL. Only supports CUDA device SM90 and above. SM90 supports world size 4, 6, 8. SM100 supports world size 6, 8.",
        NS("exec.comm"),
    ] = False
    enable_scattered_sconv: A[
        bool,
        "Inkling: replace the attention/MLP output all-reduce with a hidden-dimension reduce-scatter, run the channelwise output short convolution on the [T, H/P] shard, then all-gather before the residual add. This shards the convolution cache across tensor-parallel ranks without changing communication volume.",
        NS("exec.comm"),
    ] = False
    pre_warm_nccl: A[
        bool,
        "Pre-warm NCCL/RCCL communicators during startup to reduce P99 TTFT cold-start latency. Default: enabled for AMD/HIP (RCCL), disabled for NVIDIA/CUDA (NCCL).",
        NS("exec.comm"),
    ] = False
    enable_quant_communications: A[
        Optional[bool],
        "Enable INT8 quantization of TP communications (limited support).",
        NS("exec.comm"),
    ] = False
    enable_flashinfer_allreduce_fusion: A[bool, Arg(no_cli=True), NS("exec.comm")] = (
        False
    )
    enforce_disable_flashinfer_allreduce_fusion: A[
        bool, "Enforce disable FlashInfer allreduce fusion.", NS("exec.comm")
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
            resolvable=True,
        ),
        NS("exec.comm"),
    ] = None
    enable_aiter_allreduce_fusion: A[
        bool,
        Arg(help="Enable Aiter AllReduce Fusion.", resolvable=True),
        NS("exec.comm"),
    ] = False

    # -------------------------------------------------------------------------
    # Torch compile
    # -------------------------------------------------------------------------
    enable_torch_compile: A[
        bool,
        "Optimize the model with torch.compile. Experimental feature.",
        NS("exec.graph"),
    ] = False
    enable_torch_compile_debug_mode: A[
        bool, "Enable debug mode for torch compile", NS("exec.graph")
    ] = False
    torch_compile_max_bs: A[
        int, "Set the maximum batch size when using torch compile.", NS("exec.graph")
    ] = 32
    # -------------------------------------------------------------------------
    # Speculative decoding
    # -------------------------------------------------------------------------
    speculative_algorithm: A[
        Optional[str],
        "Speculative algorithm. Builtins: EAGLE, EAGLE3, NEXTN, STANDALONE, NGRAM, DFLASH, DSPARK. Or any name registered via `SpeculativeAlgorithm.register`.",
        NS("spec"),
    ] = None
    speculative_draft_model_path: A[
        Optional[str],
        Arg(
            help="The path of the draft model weights. This can be a local folder or a Hugging Face repo ID.",
            aliases=["--speculative-draft-model"],
        ),
        NS("spec"),
    ] = None
    speculative_draft_model_revision: A[
        Optional[str],
        "The specific draft model version to use. It can be a branch name, a tag name, or a commit id. If unspecified, will use the default version.",
        NS("spec"),
    ] = None
    speculative_draft_load_format: A[
        Optional[str],
        Arg(
            help="The format of the draft model weights to load. If not specified, will use the same format as --load-format. Use 'dummy' to initialize draft model weights with random values for profiling.",
            choices=LOAD_FORMAT_CHOICES,
        ),
        NS("spec"),
    ] = None
    speculative_num_steps: A[
        Optional[int],
        "The number of steps sampled from draft model in Speculative Decoding.",
        NS("spec"),
    ] = None
    speculative_eagle_topk: A[
        Optional[int],
        "The number of tokens sampled from the draft model in eagle2 each step.",
        NS("spec"),
    ] = None
    speculative_num_draft_tokens: A[
        Optional[int],
        "The number of tokens sampled from the draft model in Speculative Decoding.",
        NS("spec"),
    ] = None
    speculative_dflash_block_size: A[
        Optional[int],
        "DFLASH only. Block size (verify window length). Alias of --speculative-num-draft-tokens for DFLASH.",
        NS("spec"),
    ] = None
    speculative_dspark_block_size: A[
        Optional[int],
        "DSPARK only. Draft block size gamma (number of proposed draft tokens). The verify window is gamma + 1, so this sets --speculative-num-draft-tokens = gamma + 1. Omit to auto-infer gamma from the draft checkpoint block_size.",
        NS("spec"),
    ] = None
    speculative_dspark_sps_table_path: A[
        Optional[str],
        "DSPARK only. Path to a pre-profiled SPS cost table (JSON) built offline with "
        "sglang.benchmark.dspark_sps_profiler, consumed by the ragged-verify "
        "scheduler (cap-accept / compact). Omit for an uninitialized flat "
        "constant-SPS table: the budget degenerates to verify-all (zero throughput "
        "gain by itself).",
        NS("spec"),
    ] = None
    speculative_dspark_confidence_sts_path: A[
        Optional[str],
        "DSPARK only. Optional path to a per-position STS (sequential temperature "
        "scaling) calibration JSON, fit offline with sglang.benchmark.dspark_sts_fit. "
        "Calibrates the confidence-head survival probabilities the ragged-verify "
        "scheduler consumes. Omit to use identity (no calibration); losslessness is "
        "unaffected either way.",
        NS("spec"),
    ] = None
    speculative_dspark_align_verify_tokens_to_graph_tier: A[
        bool,
        "DSPARK compact ragged-verify only. Fill the per-request verify lengths so "
        "the total verify-token count reaches the cuda-graph tier the forward is "
        "already padded to: round the dp-max scheduled total up to the captured "
        "token bucket and let the top-k allocator admit that many real draft tokens "
        "(confidence-ordered). This recovers the padding the forward pays for anyway "
        "-- both the cuda-graph bucket round-up and the dp cross-rank max -- turning "
        "it into extra real verification at the same step time. Off by default; when "
        "off the schedule is byte-for-byte unchanged.",
        NS("spec"),
    ] = False
    speculative_accept_threshold_single: A[
        float,
        "Accept a draft token if its probability in the target model is greater than this threshold.",
        NS("spec"),
    ] = 1.0
    speculative_accept_threshold_acc: A[
        float,
        "The accept probability of a draft token is raised from its target probability p to min(1, p / threshold_acc).",
        NS("spec"),
    ] = 1.0
    speculative_use_rejection_sampling: A[
        bool,
        "Use rejection sampling for speculative decoding (requires topk=1).",
        NS("spec"),
    ] = False
    speculative_token_map: A[
        Optional[str], "The path of the draft model's small vocab table.", NS("spec")
    ] = None
    speculative_attention_mode: A[
        str,
        Arg(
            help="Attention backend for speculative decoding operations (both target verify and draft extend). Can be one of 'prefill' (default) or 'decode'.",
            choices=["prefill", "decode"],
            resolvable=True,
        ),
        NS("spec"),
    ] = "prefill"
    speculative_draft_attention_backend: A[
        Optional[str],
        Arg(
            help="Attention backend for speculative decoding drafting.",
            resolvable=True,
        ),
        NS("spec"),
    ] = None
    speculative_dsa_topk_backend: A[
        str,
        Arg(
            help="DSA indexer top-k backend for speculative draft workers. Options: 'sgl-kernel', 'torch', 'flashinfer'. The 'torch' backend currently requires SGLANG_DSA_FUSE_TOPK=false.",
            choices=["sgl-kernel", "torch", "flashinfer"],
        ),
        NS("spec"),
    ] = "sgl-kernel"
    speculative_draft_kv_cache_dtype: A[
        Optional[str],
        Arg(
            help="KV cache dtype for the speculative draft model only. The draft pool is "
            "allocated with one slot per target token (draft and target share a slot index "
            "space), so for a small draft it can still rival the target pool: a 5-layer "
            "DFLASH draft costs 10240 bytes/token in bf16. Setting fp8_e4m3 halves the draft "
            "pool; the saving shows up as free device memory, so raise "
            "--mem-fraction-static to convert it into KV capacity. Default follows "
            "--kv-cache-dtype.",
            choices=["auto", "fp8_e5m2", "fp8_e4m3", "bf16", "bfloat16"],
        ),
        NS("spec"),
    ] = None
    speculative_draft_window_size: A[
        Optional[int],
        "Sliding window size for the draft model. Honored by Llama EAGLE-3 (`LlamaForCausalLMEagle3`) and DFLASH only; other EAGLE-3 backends (e.g. MLA-based drafters) silently ignore it. For Llama EAGLE-3, the drafter only attends to the most recent N keys (verifier hidden states + its own outputs); the verifier is unaffected. For DFLASH, the draft worker keeps a recent target-token window in its local KV cache (paged backends may retain up to one extra page on the left for alignment). Default is full attention/context.",
        NS("spec"),
    ] = None
    speculative_moe_runner_backend: A[
        Optional[str],
        Arg(
            help="Choose the runner backend for MoE in speculative decoding.",
            choices=MOE_RUNNER_BACKEND_CHOICES,
            resolvable=True,
        ),
        NS("spec"),
    ] = None
    speculative_moe_a2a_backend: A[
        Optional[str],
        Arg(
            help="Choose the backend for MoE A2A in speculative decoding",
            choices=[
                "none",
                "deepep",
                "mooncake",
                "nixl",
                "mori",
                "ascend_fuseep",
                "flashinfer",
                "megamoe",
                "deepep_v2",
                "pplx",
                "ascend_tp",
            ],
            resolvable=True,
        ),
        NS("spec"),
    ] = None
    speculative_draft_model_quantization: A[
        Optional[str],
        Arg(
            help="The quantization method for speculative model.",
            choices=QUANTIZATION_CHOICES,
        ),
        NS("spec"),
    ] = None
    # Internal provenance used after the public draft quantization inherits the
    # target value. It is a dataclass field so ServerArgs round-trips preserve
    # whether the user explicitly set the draft option; it has no CLI surface.
    _speculative_draft_quantization_explicitly_set: A[
        Optional[bool], Arg(no_cli=True), NS("spec")
    ] = None
    speculative_skip_dp_mlp_sync: A[
        bool,
        "Skip the extra MLP sync that the scheduler performs before merging a new batch when speculative decoding + DP attention are both enabled.",
        NS("spec"),
    ] = False
    enable_multi_layer_eagle: A[
        bool,
        Arg(
            help="Enable multi-layer Eagle speculative decoding.",
            resolvable=True,
        ),
        NS("spec"),
    ] = False
    speculative_adaptive: A[
        bool,
        "Enable adaptive speculative decoding that dynamically adjusts num_steps based on acceptance rate.",
        NS("spec"),
    ] = False
    speculative_adaptive_config: A[
        Optional[str],
        "Path to a JSON config file for adaptive speculative decoding tuning knobs.",
        NS("spec"),
    ] = None

    # Decoupled speculative decoding: draft and verify run as
    # separate engines, currently connected by a ZMQ IPC mesh.
    decoupled_spec_bind_endpoint: A[
        Optional[str],
        "ZMQ endpoint this engine binds for its inbound channel in decoupled "
        "speculative decoding (verifier: result PULL; drafter: control PULL).",
        NS("disagg"),
    ] = None
    decoupled_spec_connect_endpoints: A[
        Optional[List[str]],
        Arg(
            help="Peer inbound (bind) endpoints to connect to, ordered by peer "
            "rank, for decoupled speculative decoding.",
            type_parser=json_list_type,
        ),
        NS("disagg"),
    ] = None
    decoupled_spec_rank: A[
        Optional[int],
        "This engine's rank within its own role space (verifier-rank or "
        "drafter-rank) for decoupled speculative decoding.",
        NS("disagg"),
    ] = None
    decoupled_spec_role: A[
        Literal["null", "verifier", "drafter"],
        "Role in decoupled speculative decoding: 'null' disables it, 'verifier' "
        "runs the target/verify half, 'drafter' runs the draft half.",
        NS("disagg"),
    ] = "null"
    spec_trace_dir: A[
        Optional[str],
        "Directory to write decoupled speculative decoding trace files.",
        NS("spec"),
    ] = None

    # -------------------------------------------------------------------------
    # Speculative decoding (ngram)
    # -------------------------------------------------------------------------
    speculative_ngram_min_bfs_breadth: A[
        int,
        "The minimum breadth for BFS (Breadth-First Search) in ngram speculative decoding.",
        NS("spec"),
    ] = 1
    speculative_ngram_max_bfs_breadth: A[
        int,
        "The maximum breadth for BFS (Breadth-First Search) in ngram speculative decoding.",
        NS("spec"),
    ] = 10
    speculative_ngram_match_type: A[
        Literal["BFS", "PROB"], "The match type for cache tree.", NS("spec")
    ] = "BFS"
    speculative_ngram_max_trie_depth: A[
        int, "The max trie depth for ngram speculative decoding.", NS("spec")
    ] = 18
    speculative_ngram_capacity: A[
        int, "The cache capacity for ngram speculative decoding.", NS("spec")
    ] = (10 * 1000 * 1000)
    speculative_ngram_external_corpus_path: A[
        Optional[str],
        "Path to an external JSONL corpus to pre-load into SAM at startup. Additional corpora can be added at runtime via POST /add_external_corpus.",
        NS("spec"),
    ] = None
    speculative_ngram_external_sam_budget: A[
        int,
        "Number of draft nodes reserved for the external SAM subtree in ngram speculative decoding.",
        NS("spec"),
    ] = 0
    speculative_ngram_external_corpus_max_tokens: A[
        int,
        "Fail startup if the tokenized external ngram corpus exceeds this many tokens. Tune this based on your CPU memory budget.",
        NS("spec"),
    ] = 10000000

    # -------------------------------------------------------------------------
    # Expert parallelism
    # -------------------------------------------------------------------------
    ep_size: A[
        int,
        Arg(
            help="The expert parallelism size.",
            aliases=["--expert-parallel-size", "--ep"],
            resolvable=True,
        ),
        NS("parallel"),
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
            "deepep_v2",
            "ascend_tp",
            "pplx",
        ],
        Arg(
            help="Choose the backend for MoE A2A.",
            choices=[
                "none",
                "deepep",
                "mooncake",
                "nixl",
                "mori",
                "ascend_fuseep",
                "flashinfer",
                "megamoe",
                "deepep_v2",
                "pplx",
                "ascend_tp",
            ],
            resolvable=True,
        ),
        NS("exec.moe"),
    ] = "none"
    enable_w4a4_mxfp4_megamoe: A[
        bool,
        "Enable the W4A4 MXFP4 MegaMoE path by setting DeepGEMM's "
        "DG_USE_FP4_ACTS=1 and DG_USE_MXF4_KIND=1. Use with "
        "--moe-a2a-backend megamoe.",
        NS("exec.moe"),
    ] = False
    deepep_v2_mode: A[
        Literal["direct", "hybrid"],
        "DeepEP v2 ElasticBuffer communication topology, fixed at server init: "
        "`direct` (single-node NVLink) or `hybrid` (multi-node scale-out). "
        "Layout/grouped-GEMM and the decode CUDA graph are chosen per batch by "
        "inference phase, independent of this knob; not equivalent to DeepEP v1 "
        "normal/low_latency.",
        NS("exec.moe"),
    ] = "direct"
    moe_runner_backend: A[
        str,
        Arg(
            help="Choose the runner backend for MoE.",
            choices=MOE_RUNNER_BACKEND_CHOICES,
            resolvable=True,
        ),
        NS("exec.moe"),
    ] = "auto"
    flashinfer_mxfp4_moe_precision: A[
        Literal["default", "bf16"],
        "Choose the computation precision of flashinfer mxfp4 moe",
        NS("exec.moe"),
    ] = "default"
    deepep_mode: A[
        Literal["auto", "normal", "low_latency"],
        "Select the mode when enable DeepEP or MoriEP MoE, could be `normal`, `low_latency` or `auto`. Default is `auto`, which means `low_latency` for decode batch and `normal` for prefill batch.",
        NS("exec.moe"),
    ] = "auto"
    fuseep_mode: A[
        Literal[1, 2],
        "Select the mode when enable Ascend FuseEP MoE, 1 -> dispatch_gmm_combine_decode is executed；2 -> dispatch_ffn_combine is executed (support hybrid deployment when 2).",
        NS("exec.moe"),
    ] = 2
    deepep_dispatcher_output_dtype: A[
        Literal["auto", "bf16", "fp8", "int8", "nvfp4"],
        "Select DeepEP dispatcher output dtype",
        NS("exec.moe"),
    ] = "auto"
    ep_num_redundant_experts: A[
        int,
        "Allocate this number of redundant experts in expert parallel.",
        NS("exec.moe"),
    ] = 0
    ep_dispatch_algorithm: A[
        Optional[Literal["static", "dynamic", "fake", "lp"]],
        "The algorithm to choose ranks for redundant experts in expert parallel.",
        NS("exec.moe"),
    ] = None
    init_expert_location: A[str, "Initial location of EP experts.", NS("exec.moe")] = (
        "trivial"
    )
    enable_eplb: A[bool, "Enable EPLB algorithm", NS("exec.moe")] = False
    eplb_algorithm: A[str, "Chosen EPLB algorithm", NS("exec.moe")] = "auto"
    eplb_rebalance_num_iterations: A[
        int,
        "Number of iterations to automatically trigger a EPLB re-balance.",
        NS("exec.moe"),
    ] = 1000
    eplb_rebalance_layers_per_chunk: A[
        Optional[int], "Number of layers to rebalance per forward pass.", NS("exec.moe")
    ] = None
    eplb_min_rebalancing_utilization_threshold: A[
        float,
        "Minimum threshold for GPU average utilization to trigger EPLB rebalancing. Must be in the range [0.0, 1.0].",
        NS("exec.moe"),
    ] = 1.0
    expert_distribution_recorder_mode: A[
        Optional[Literal["stat", "stat_approx", "per_pass", "per_token"]],
        "Mode of expert distribution recorder.",
        NS("exec.moe"),
    ] = None
    expert_distribution_recorder_buffer_size: A[
        Optional[int],
        "Circular buffer size of expert distribution recorder. Set to -1 to denote infinite buffer.",
        NS("exec.moe"),
    ] = None
    expert_balancedness_report_mode: A[
        Literal["off", "server_log", "prometheus", "both"],
        "Where to report expert balancedness. Options: off, server_log, prometheus, both.",
        NS("exec.moe"),
    ] = "off"
    deepep_config: A[
        Optional[str],
        "Tuned DeepEP config suitable for your own cluster. It can be either a string with JSON content or a file path.",
        NS("exec.moe"),
    ] = None
    moe_dense_tp_size: A[
        Optional[int],
        Arg(
            help="TP size for MoE dense MLP layers. This flag is useful when, with large TP size, there are errors caused by weights in MLP layers having dimension smaller than the min dimension GEMM supports.",
            resolvable=True,
        ),
        NS("parallel"),
    ] = None
    elastic_ep_backend: A[
        Literal[None, "mooncake", "nixl"],
        Arg(
            help="Specify the collective communication backend for elastic EP. Supports 'mooncake' and 'nixl'.",
            choices=["none", "mooncake", "nixl"],
        ),
        NS("exec.moe"),
    ] = None
    enable_elastic_expert_backup: A[
        bool, "Enable elastic expert backup feature.", NS("exec.moe")
    ] = False
    mooncake_ib_device: A[
        Optional[str],
        "The InfiniBand devices for Mooncake Backend transfer, accepts multiple comma-separated devices (e.g., --mooncake-ib-device mlx5_0,mlx5_1). Default is None, which triggers automatic device detection when Mooncake Backend is enabled.",
        NS("exec.moe"),
    ] = None
    enable_waterfill: A[
        bool,
        "Enable Waterfill: dispatch the fused shared expert as an extra routed expert slot to the least-loaded EP rank. Supports DeepEP and MegaMOE MoE A2A backends, implicitly enables shared-expert fusion, and supports --deepep-mode auto, normal, or low_latency when used with DeepEP. Use auto or low_latency for production DeepEP decode so CUDA graph remains enabled. Supported on DeepSeek-V3/R1 with EP >= 2.",
        NS("exec.moe"),
    ] = False
    ep_join_mode: A[
        Optional[Literal["scale", "recover"]],
        Arg(
            help="Join mode for elastic EP. 'recover' rejoins an existing slot after a fault. 'scale' joins as a new rank beyond the original group size and requires --node-rank 1.",
            cli_name="--elastic-ep-join-mode",
            choices=["scale", "recover"],
        ),
        NS("exec.moe"),
    ] = None
    ep_join_rank_offset: A[
        int,
        Arg(
            help=(
                "Global rank offset of an elastic EP joining group. Scale "
                "joiners must set this to the current effective EP size."
            ),
            cli_name="--elastic-ep-join-rank-offset",
        ),
        NS("parallel"),
    ] = 0
    elastic_ep_initial_size: A[
        Optional[int],
        "EP size used to define the immutable per-rank expert storage layout. "
        "Scale joiners must use the primary deployment's launch-time EP size.",
        NS("parallel"),
    ] = None
    max_ep_size: A[
        Optional[int],
        "Maximum EP size the server can scale to at runtime. Pre-allocates active-rank state and backend buffers to this size. Defaults to the launch-time world size.",
        NS("parallel"),
    ] = None
    elastic_ep_scale_timeout: A[
        float,
        "Timeout in seconds for a pending elastic EP scale operation.",
        NS("exec.moe"),
    ] = 600
    elastic_ep_rejoin: A[
        bool, "[Deprecated] Alias for --elastic-ep-join-mode recover.", NS("exec.moe")
    ] = False
    disable_flashinfer_cutlass_moe_fp4_allgather: A[
        bool,
        "Disables quantize before all-gather for flashinfer cutlass moe.",
        NS("exec.moe"),
    ] = False
    disable_shared_experts_fusion: A[
        bool,
        Arg(
            help="Disable the built-in shared experts fusion optimization for DeepSeek V3/R1. Note: Waterfill (--enable-waterfill) routes the shared expert as an extra MoE slot, so the shared expert is not separated from the MoE path when Waterfill is enabled.",
            resolvable=True,
        ),
        NS("exec.moe"),
    ] = False
    enforce_shared_experts_fusion: A[
        bool,
        "Enforce shared experts fusion even when it would normally be disabled (e.g. under DeepEP). Mutually exclusive with --disable-shared-experts-fusion.",
        NS("exec.moe"),
    ] = False

    # -------------------------------------------------------------------------
    # Mamba cache and linear attn
    # -------------------------------------------------------------------------
    max_mamba_cache_size: A[
        Optional[int], "The maximum size of the mamba cache.", NS("schedule")
    ] = None
    mamba_ssm_dtype: A[
        Optional[str],
        Arg(
            help="The data type of the SSM states in mamba cache. If not set, will be read from model config (mamba_ssm_dtype).",
            choices=["float32", "bfloat16", "float16"],
        ),
        NS("exec.mamba"),
    ] = None
    mamba_max_states_per_path: A[
        int,
        "Maximum number of cached Mamba states retained per root-to-tail path "
        "(-1 means unlimited). When enabled, after each insert the shallowest eligible "
        "interior states beyond the cap are removed while their full KV remains. "
        "Tail, fork, and locked nodes are preserved. Must be -1 or a positive integer.",
        NS("exec.mamba"),
    ] = -1
    enable_mamba_cache_stochastic_rounding: A[
        bool,
        "Enable stochastic rounding when writing FP16 Mamba SSM cache states. Requires --mamba-ssm-dtype float16 and CUDA. With --mamba-backend triton, requires SM100.",
        NS("exec.mamba"),
    ] = False
    mamba_cache_philox_rounds: A[
        int,
        "Number of Philox rounds to use for stochastic rounding of FP16 Mamba SSM cache writes. Triton uses the Triton default when set to 0; FlashInfer uses 10 rounds when set to 0.",
        NS("exec.mamba"),
    ] = 0
    mamba_full_memory_ratio: A[
        float,
        Arg(
            help="The ratio of mamba state memory to full kv cache memory.",
            resolvable=True,
        ),
        NS("schedule"),
    ] = 0.9
    mamba_radix_cache_strategy: A[
        str,
        Arg(
            help="The strategy to use for mamba radix cache.",
            choices=["auto", "no_buffer", "extra_buffer", "extra_buffer_lazy"],
            resolvable=True,
        ),
        NS("exec.mamba"),
    ] = "auto"
    uses_mamba_radix_cache: A[
        bool,
        Arg(
            help="(Derived) whether the model routes through the hybrid-mamba "
            "radix cache handling; resolved from the model architecture, no "
            "CLI surface.",
            no_cli=True,
            resolvable=True,
        ),
        NS("exec.mamba"),
    ] = False
    mamba_track_interval: A[
        int, "The interval to track the mamba state during decode.", NS("exec.mamba")
    ] = 256
    enable_int8_mamba_checkpoint: A[
        bool,
        "Store radix-cached linear-attn (mamba) states in int8 (separate checkpoint pool) for ~2x cached-prefix capacity at fixed memory.",
        NS("exec.mamba"),
    ] = False
    int8_mamba_ckpt_size: A[
        Optional[int],
        "Number of int8 mamba checkpoint slots (default: 2x the active mamba pool size).",
        NS("exec.mamba"),
    ] = None
    linear_attn_backend: A[
        str,
        Arg(
            help="The default kernel backend for linear attention (GDN/KDA). Can be overridden per-mode by --linear-attn-decode-backend and --linear-attn-prefill-backend. The Helion backend is KDA-only.",
            choices=LINEAR_ATTN_KERNEL_BACKEND_CHOICES,
        ),
        NS("exec.mamba"),
    ] = "triton"
    linear_attn_decode_backend: A[
        Optional[str],
        Arg(
            help="Override the kernel backend for linear attention decode. If not set, uses --linear-attn-backend.",
            choices=LINEAR_ATTN_KERNEL_BACKEND_CHOICES,
        ),
        NS("exec.mamba"),
    ] = None
    linear_attn_prefill_backend: A[
        Optional[str],
        Arg(
            help="Override the kernel backend for linear attention prefill/extend. If not set, uses --linear-attn-backend; compatible SM100 GDN models may automatically select FlashInfer.",
            choices=LINEAR_ATTN_KERNEL_BACKEND_CHOICES,
        ),
        NS("exec.mamba"),
    ] = None
    linear_attn_verify_backend: A[
        Optional[str],
        Arg(
            help="Override the kernel backend for linear attention speculative target-verify. If not set, follows the decode backend (flashinfer decode -> flashinfer verify, otherwise triton). KDA supports triton, nv_cutedsl, and flashinfer verify backends.",
            choices=LINEAR_ATTN_KERNEL_BACKEND_CHOICES + ["nv_cutedsl"],
        ),
        NS("exec.mamba"),
    ] = None
    # ReplaySSM buffered output-only linear-attn decode (GDN + KDA): per-slot
    # ring + periodic flush to cut per-step HBM state traffic.
    enable_linear_replayssm: A[
        bool,
        "Enable the ReplaySSM buffered output-only linear-attn decode kernel. "
        "Primarily a GDN (scalar-gate) decode-bandwidth optimization (~1.2-1.5x "
        "at batch >= 64). KDA uses its selected Triton or Helion implementation, "
        "but its per-K gate ring is larger and ReplaySSM is typically slower "
        "than packed KDA decode; benchmark before enabling it. Requires the "
        "Triton linear-attn decode backend, or Helion for KDA, and "
        "--mamba-radix-cache-strategy no_buffer (the default).",
        NS("exec.mamba"),
    ] = False
    linear_replayssm_cache_len: A[
        int,
        "Ring-buffer length L for ReplaySSM linear-attn decode. The full recurrent state is flushed to HBM every L decode steps.",
        NS("exec.mamba"),
    ] = 16
    # ReplaySSM spec-verify (Part B of RFC #28511): linear-attn target-verify via
    # fold-every-commit instead of per-draft full-state snapshots -- the verify
    # stores each draft step's raw inputs into a per-slot window and the commit
    # replays the accepted prefix into the fp32 checkpoint. GDN sizes the window
    # to the draft maximum; KDA folds a (raw v, pre-norm k, gate, beta) ring of
    # length --linear-replayssm-cache-len. Linear-chain (topk <= 1) only.
    enable_linear_replayssm_spec: A[
        bool,
        "Enable the ReplaySSM spec-verify: fold-every-commit -- a per-slot raw-input window replaces the recurrent verify's per-draft full-state snapshots. GDN or KDA hybrid linear-attn models, linear-chain (--speculative-eagle-topk in {None, 1}) only.",
        NS("exec.mamba"),
    ] = False

    # -------------------------------------------------------------------------
    # Hierarchical cache
    # -------------------------------------------------------------------------
    enable_hierarchical_cache: A[bool, "Enable hierarchical cache", NS("memory")] = (
        False
    )
    hicache_host_memory_mode: A[
        str,
        Arg(
            help="Whether host memory is a persistent HiCache tier (cache) or a transient staging buffer between GPU and the storage backend (buffer_only). buffer_only requires --hicache-storage-backend.",
            choices=["cache", "buffer_only"],
        ),
        NS("memory"),
    ] = "cache"
    hicache_ratio: A[
        Optional[float],
        "The ratio of the size of host KV cache memory pool to the size of device pool. Defaults to 2.0 in cache mode, 1.2 in buffer_only mode, or 0.2 for backup-only host-pool decode retraction.",
        NS("memory"),
    ] = None
    hicache_size: A[
        int,
        "The size of host KV cache memory pool in gigabytes. Overrides --hicache-ratio in either host memory mode.",
        NS("memory"),
    ] = 0
    hicache_write_policy: A[
        str,
        Arg(
            help="The write policy of hierarchical cache.",
            choices=["write_back", "write_through", "write_through_selective"],
        ),
        NS("memory"),
    ] = "write_through"
    hicache_io_backend: A[
        str,
        Arg(
            help="The IO backend for KV cache transfer between CPU and GPU",
            choices=["direct", "kernel", "kernel_ascend"],
        ),
        NS("memory"),
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
        NS("memory"),
    ] = "page_first"
    hicache_storage_backend: A[
        Optional[str],
        Arg(
            help="The storage backend for hierarchical KV cache. Built-in backends: file, mooncake, hf3fs, nixl, aibrix. For dynamic backend, use --hicache-storage-backend-extra-config to specify: backend_name (custom name), module_path (Python module path), class_name (backend class name).",
            choices=[
                "file",
                "sim",
                "mooncake",
                "hf3fs",
                "nixl",
                "aibrix",
                "dynamic",
                "eic",
                "simm",
                "mori",
                "shm",
            ],
        ),
        NS("memory"),
    ] = None
    hicache_storage_prefetch_policy: A[
        str,
        Arg(
            help="Control when prefetching from the storage backend should stop.",
            choices=["best_effort", "wait_complete", "timeout"],
        ),
        NS("memory"),
    ] = "timeout"
    hicache_storage_backend_extra_config: A[
        Optional[str],
        "A dictionary in JSON string format, or a string starting with a leading '@' and a config file in JSON/YAML/TOML format, containing extra configuration for the storage backend.",
        NS("memory"),
    ] = None

    # -------------------------------------------------------------------------
    # Hierarchical sparse attention
    # -------------------------------------------------------------------------
    enable_hisparse: A[bool, "Enable hierarchical sparse attention", NS("memory")] = (
        False
    )
    hisparse_config: A[
        Optional[str],
        Arg(
            help='A dictionary in JSON string format for hierarchical sparse attention configuration. Example: \'{"top_k": 2048, "device_buffer_size": 4096, "host_to_device_ratio": 2}\'',
            aliases=["--hierarchical-sparse-attention-extra-config"],
        ),
        NS("memory"),
    ] = None

    # -------------------------------------------------------------------------
    # Multi-modal optimization configs
    # -------------------------------------------------------------------------
    enable_broadcast_mm_inputs_process: A[
        bool, "Enable broadcast mm-inputs process in scheduler.", NS("mm")
    ] = False
    enable_prefix_mm_cache: A[
        bool,
        "Enable prefix multimodal cache. Currently only supports mm-only.",
        NS("mm"),
    ] = False
    mm_enable_dp_encoder: A[
        bool,
        "Enabling data parallelism for mm encoder. The dp size will be set to the tp size automatically.",
        NS("mm"),
    ] = False
    mm_process_config: A[
        Optional[Dict[str, Any]],
        Arg(
            help="Multimodal preprocessing config, a json config contains keys: `image`, `video`, `audio`",
            type_parser=json.loads,
        ),
        NS("mm"),
    ] = None
    mm_processor_worker_num: A[
        int,
        "Number of threads for multimodal processor calls. 0 selects the "
        "model-specific default. Only processors with isolated-worker support "
        "can use more than one thread.",
        NS("mm"),
    ] = 0
    mm_io_worker_num: A[
        int,
        "Number of threads for multimodal data loading and decoding. 0 selects "
        "the model-specific default. SGLANG_IO_WORKERS remains supported as an "
        "environment override when this argument is 0.",
        NS("mm"),
    ] = 0
    allowed_media_domains: A[
        List[str],
        "Restrict client-supplied HTTP(S) image, video, and audio URLs to these "
        "exact hostnames. Redirect destinations are checked against the same "
        "allowlist. When unset, remote media from any domain is allowed.",
        NS("mm"),
    ] = dataclasses.field(default_factory=list)
    media_url_max_file_size_mb: A[
        int,
        "Maximum size in MiB for one client-supplied remote media download. "
        "The limit is enforced while streaming; set to 0 to disable it.",
        NS("mm"),
    ] = 64
    mm_preprocess_cache_size_mb: A[
        Optional[int],
        "CPU memory budget for content-addressed multimodal preprocessing "
        "artifacts. Unset selects a model-specific default (256 MiB for "
        "Kimi-K3); 0 disables the cache. The budget is divided across "
        "tokenizer workers and does not reserve GPU memory.",
        NS("mm"),
    ] = None
    trust_mm_content_hashes: A[
        bool,
        "Trust caller-provided multimodal SHA-256 content hashes. This can "
        "skip reading media on a hot metadata-cache hit; only enable it when "
        "the caller guarantees that hashes identify immutable media bytes.",
        NS("mm"),
    ] = False
    limit_mm_data_per_request: A[
        Optional[Union[str, Dict[str, int]]],
        Arg(
            help='Limit the number of multimodal inputs per request. e.g. \'{"image": 1, "video": 1, "audio": 1}\'',
            type_parser=json.loads,
        ),
        NS("mm"),
    ] = None
    enable_mm_global_cache: A[
        bool,
        "Enable global multimodal embedding cache to skip redundant ViT inference.",
        NS("mm"),
    ] = False
    image_processor_backend: A[
        Literal["auto", "torchvision", "pil"],
        "Image processor backend. 'auto' lets Transformers select the best "
        "available backend.",
        NS("mm"),
    ] = "auto"
    mm_global_cache_backend: A[
        str,
        Arg(
            help="Storage backend for the multimodal global embedding cache. "
            "Used when --enable-mm-global-cache is set.",
            choices=["mooncake"],
        ),
        NS("mm"),
    ] = "mooncake"
    disable_fast_image_processor: A[
        bool,
        "Deprecated. Use --image-processor-backend=pil instead.",
        NS("mm"),
    ] = False
    mm_feature_transport: A[
        Optional[Literal["cpu", "cuda_ipc", "cuda_vmm"]],
        "Transport multimodal features through CPU memory, a bounded CUDA IPC "
        "pool, or a bounded CUDA VMM pool. "
        "Unset uses cpu except for validated multi-node GB200/GB300 MNNVL models, "
        "which use cuda_vmm when an IMEX channel is available. Select cuda_ipc "
        "explicitly for single-node GPU transport. GPU transports reserve "
        "SGLANG_MM_FEATURE_CACHE_MB (default 1024 MiB) on the base GPU and fall "
        "back to CPU transport when the pool is full.",
        NS("mm"),
    ] = None
    keep_mm_feature_on_device: A[
        bool,
        "Deprecated. Use --mm-feature-transport=cuda_ipc for bounded GPU-resident "
        "multimodal feature transport.",
        NS("mm"),
    ] = False

    # -------------------------------------------------------------------------
    # LoRA
    # -------------------------------------------------------------------------
    enable_lora: A[
        Optional[bool],
        "Enable LoRA support for the model. This argument is automatically set to True if `--lora-paths` is provided for backward compatibility.",
        NS("lora"),
    ] = None
    enable_lora_overlap_loading: A[
        Optional[bool],
        "Enable asynchronous LoRA weight loading in order to overlap H2D transfers with GPU compute. This should be enabled if you find that your LoRA workloads are bottlenecked by adapter weight loading, for example when frequently loading large LoRA adapters.",
        NS("lora"),
    ] = None
    max_lora_rank: A[
        Optional[int],
        "The maximum rank of LoRA adapters. If not specified, it will be automatically inferred from the adapters provided in --lora-paths.",
        NS("lora"),
    ] = None
    lora_target_modules: A[
        Optional[Union[set[str], List[str]]],
        Arg(
            help="The union set of all target modules where LoRA should be applied. If not specified, it will be automatically inferred from the adapters provided in --lora-paths. If 'all' is specified, all supported modules will be targeted.",
            nargs="*",
            choices=SUPPORTED_LORA_TARGET_MODULES + [LORA_TARGET_ALL_MODULES],
        ),
        NS("lora"),
    ] = None
    lora_paths: A[
        Optional[Union[dict[str, str], List[dict[str, str]], List[str], List[LoRARef]]],
        Arg(
            help='The list of LoRA adapters to load. Each adapter must be specified in one of the following formats: <PATH> | <NAME>=<PATH> | JSON with schema {"lora_name":str,"lora_path":str,"pinned":bool}',
            action=LoRAPathAction,
            action_kwargs={"type": str, "nargs": "*"},
        ),
        NS("lora"),
    ] = None
    max_loaded_loras: A[
        Optional[int],
        "If specified, it limits the maximum number of LoRA adapters loaded in CPU memory at a time. The value must be greater than or equal to `--max-loras-per-batch`.",
        NS("lora"),
    ] = None
    max_loras_per_batch: A[
        int,
        "Maximum number of adapters for a running batch, include base-only request.",
        NS("lora"),
    ] = 8
    lora_eviction_policy: A[
        str,
        Arg(
            help="LoRA adapter eviction policy when memory pool is full. 'lru': Least Recently Used (default, better cache efficiency). 'fifo': First-In-First-Out.",
            choices=["lru", "fifo"],
        ),
        NS("lora"),
    ] = "lru"
    lora_backend: A[
        str,
        Arg(
            help="Choose the kernel backend for multi-LoRA serving.",
            choices=["triton", "csgmv", "ascend", "torch_native"],
        ),
        NS("lora"),
    ] = "csgmv"
    max_lora_chunk_size: A[
        Optional[int],
        Arg(
            help="Maximum chunk size for the ChunkedSGMV LoRA backend. Only used when --lora-backend is 'csgmv'. Choosing a larger value might improve performance.",
            choices=[16, 32, 64, 128],
        ),
        NS("lora"),
    ] = 16
    experts_shared_outer_loras: A[
        Optional[bool],
        Arg(
            help="Force shared outer LoRA mode for MoE models. When set, w1/w3 lora_A and w2 lora_B are shared across experts (expert_dim=1). Use --no-experts-shared-outer-loras to force disable. By default this is auto-detected from adapter weights.",
            action=argparse.BooleanOptionalAction,
        ),
        NS("lora"),
    ] = None
    lora_use_virtual_experts: A[
        bool,
        "Enable virtual expert computation for MoE models. When set, the model will use virtual expert computation.",
        NS("lora"),
    ] = False
    lora_strict_loading: A[
        bool,
        Arg(
            help="Enable strict loading for LoRA adapters. When set, mismatched or missing keys in the adapter weights will raise an error.",
            action=argparse.BooleanOptionalAction,
        ),
        NS("lora"),
    ] = False
    lora_drain_wait_threshold: A[
        float,
        "When any LoRA adapter request waits longer than this threshold (in seconds), the scheduler will selectively drain one running adapter to make room. This mitigates extreme tail latency under high or skewed workloads by preventing a small set of adapters from monopolizing batch slots. Set to 0 to disable draining (default).",
        NS("lora"),
    ] = 0.0

    # -------------------------------------------------------------------------
    # Two batch overlap
    # -------------------------------------------------------------------------
    enable_two_batch_overlap: A[
        bool, "Enabling two micro batches to overlap.", NS("exec.overlap")
    ] = False
    enable_single_batch_overlap: A[
        bool,
        "Let computation and communication overlap within one micro batch.",
        NS("exec.overlap"),
    ] = False
    tbo_token_distribution_threshold: A[
        float,
        "The threshold of token distribution between two batches in micro-batch-overlap, determines whether to two-batch-overlap or two-chunk-overlap. Set to 0 denote disable two-chunk-overlap.",
        NS("exec.overlap"),
    ] = 0.48

    # -------------------------------------------------------------------------
    # Offloading
    # -------------------------------------------------------------------------
    cpu_offload_gb: A[
        int, "How many GBs of RAM to reserve for CPU offloading.", NS("exec.offload")
    ] = 0
    offload_group_size: A[
        int, "Number of layers per group in offloading.", NS("exec.offload")
    ] = -1
    offload_num_in_group: A[
        int, "Number of layers to be offloaded within a group.", NS("exec.offload")
    ] = 1
    offload_prefetch_step: A[
        int, "Steps to prefetch in offloading.", NS("exec.offload")
    ] = 1
    offload_mode: A[str, "Mode of offloading.", NS("exec.offload")] = "cpu"

    # -------------------------------------------------------------------------
    # LMCache
    # -------------------------------------------------------------------------
    enable_lmcache: A[
        bool,
        "Using LMCache as an alternative hierarchical cache solution",
        NS("memory"),
    ] = False
    lmcache_config_file: A[
        Optional[str], "Path to the LMCache YAML configuration file", NS("memory")
    ] = None

    # -------------------------------------------------------------------------
    # FlexKV
    # -------------------------------------------------------------------------
    enable_flexkv: A[
        bool,
        (
            "Route the default RadixCache through FlexKV's KVManager for "
            "host-tier (CPU / SSD / Remote) KV cache offload. Equivalent "
            "to --radix-cache-backend=flexkv but also participates in the "
            "auto-selection chain alongside --enable-lmcache."
        ),
        NS("memory"),
    ] = False
    flexkv_config_file: A[
        Optional[str],
        (
            "Path to the FlexKV YAML / JSON configuration file. "
            "Equivalent to setting the FLEXKV_CONFIG_PATH environment "
            "variable."
        ),
        NS("memory"),
    ] = None

    # -------------------------------------------------------------------------
    # Ktransformers/AMX expert parallelism
    # -------------------------------------------------------------------------
    kt_weight_path: A[
        Optional[str],
        "[ktransformers parameter] The path of the quantized expert weights for amx kernel. A local folder.",
        NS("exec.moe"),
    ] = None
    kt_method: A[
        str,
        "[ktransformers parameter] Quantization formats for CPU execution.",
        NS("exec.moe"),
    ] = "AMXINT4"
    kt_cpuinfer: A[
        Optional[int],
        "[ktransformers parameter] The number of CPUInfer threads.",
        NS("exec.moe"),
    ] = None
    kt_threadpool_count: A[
        int,
        "[ktransformers parameter] One-to-one with the number of NUMA nodes (one thread pool per NUMA).",
        NS("exec.moe"),
    ] = 2
    kt_num_gpu_experts: A[
        Optional[int],
        "[ktransformers parameter] The number of GPU experts.",
        NS("exec.moe"),
    ] = None
    kt_max_deferred_experts_per_token: A[
        Optional[int],
        "[ktransformers parameter] Maximum number of experts deferred to CPU per token. All MoE layers except the final one use this value; the final layer always uses 0.",
        NS("exec.moe"),
    ] = None

    # -------------------------------------------------------------------------
    # Diffusion LLM
    # -------------------------------------------------------------------------
    dllm_algorithm: A[
        Optional[str],
        "The diffusion LLM algorithm, such as LowConfidence.",
        NS("exec.dllm"),
    ] = None
    dllm_algorithm_config: A[
        Optional[str],
        "The diffusion LLM algorithm configurations. Must be a YAML file.",
        NS("exec.dllm"),
    ] = None
    dllm_fdfo: A[
        bool,
        Arg(
            help="Enable First-Done-First-Out (FDFO) scheduling for diffusion LLM inference. Enabled by default; use --no-dllm-fdfo to fall back to synchronous block scheduling.",
            action=argparse.BooleanOptionalAction,
        ),
        NS("exec.dllm"),
    ] = True

    # -------------------------------------------------------------------------
    # PD disaggregation
    # -------------------------------------------------------------------------
    disaggregation_mode: A[
        Literal["null", "prefill", "decode"],
        'Only used for PD disaggregation. "prefill" for prefill-only server, and "decode" for decode-only server. If not specified, it is not PD disaggregated',
        NS("disagg"),
    ] = "null"
    disaggregation_transfer_backend: A[
        str,
        Arg(
            help="The backend for disaggregation transfer. Default is mooncake.",
            choices=DISAGG_TRANSFER_BACKEND_CHOICES,
        ),
        NS("disagg"),
    ] = "mooncake"
    disaggregation_bootstrap_port: A[
        int,
        "Bootstrap server port on the prefill server. Default is 8998.",
        NS("disagg"),
    ] = 8998
    disaggregation_ib_device: A[
        Optional[str],
        'The InfiniBand devices for disaggregation transfer. Supports a single device (e.g., --disaggregation-ib-device mlx5_0), a shared comma-separated list (e.g., --disaggregation-ib-device mlx5_0,mlx5_1), a per-GPU JSON mapping (e.g., --disaggregation-ib-device \'{"0": "mlx5_0,mlx5_1", "1": "mlx5_2"}\'), or a path to a JSON file containing that mapping. Default is None, which triggers automatic device detection when mooncake backend is enabled.',
        NS("disagg"),
    ] = None
    disaggregation_decode_enable_radix_cache: A[
        bool,
        "Enable radix cache on decode server (PD mode). Caches KV prefixes to avoid redundant transfers. Incompatible with --enable-hisparse, speculative decoding, and --disaggregation-transfer-backend fake.",
        NS("disagg"),
    ] = False
    disaggregation_decode_enable_offload_kvcache: A[
        bool,
        "Enable async KV cache offloading on decode server (PD mode).",
        NS("disagg"),
    ] = False
    disaggregation_decode_retraction_backup: A[
        Optional[str],
        Arg(
            help=(
                "Storage backend for KV preserved across PD decode retraction. "
                "'cpu_tensor' uses per-request CPU tensors. 'host_pool' uses "
                "a reserved HiCache pool and does not fall back on exhaustion. "
                "If omitted, the backend is inferred from the decode KV pool."
            ),
            choices=["cpu_tensor", "host_pool"],
        ),
        NS("disagg"),
    ] = None
    num_reserved_decode_tokens: A[
        int,
        "Number of decode tokens that will have memory reserved when adding new request to the running batch.",
        NS("disagg"),
    ] = 512
    disaggregation_decode_extra_slots: A[
        Optional[int],
        "Number of extra decode req_to_token slots pre-allocated for in-transfer requests (PD mode). If unset, defaults to 0 (or 2x the per-worker running batch for small batches).",
        NS("disagg"),
    ] = None
    disaggregation_decode_polling_interval: A[
        int,
        "The interval to poll requests in decode server. Can be set to >1 to reduce the overhead of this.",
        NS("disagg"),
    ] = 1
    optimistic_prefill_attempts: A[
        int,
        "Number of optimistic prefill forward passes that skip the bootstrap wait.",
        NS("disagg"),
    ] = 0

    # -------------------------------------------------------------------------
    # Encode prefill disaggregation
    # -------------------------------------------------------------------------
    encoder_only: A[
        bool, "For MLLM with an encoder, launch an encoder-only server", NS("disagg")
    ] = False
    language_only: A[
        bool, "For VLM, load weights for the language model only.", NS("disagg")
    ] = False
    language_model_only: A[
        bool,
        "Skip the multimodal encoder entirely: its weights are never loaded and the "
        "tower is never built, freeing that GPU memory for KV cache. Multimodal "
        "requests are rejected. Unlike --language-only this is a standalone mode, "
        "not part of encoder/decoder disaggregation.",
        NS("disagg"),
    ] = False
    encoder_transfer_backend: A[
        str,
        Arg(
            help="The backend for encoder disaggregation transfer. Auto selects a model- and TP-aware backend.",
            choices=["auto", "zmq_to_scheduler", "zmq_to_tokenizer", "mooncake"],
        ),
        NS("disagg"),
    ] = "auto"
    encoder_urls: A[List[str], "List of encoder server urls.", NS("disagg")] = (
        dataclasses.field(default_factory=list)
    )
    encoder_bootstrap_port: A[
        int,
        "Port for the EncoderBootstrapServer that runs in the language-only tokenizer manager process. Encoders register here, and language-only receivers fetch the current URL list from here.",
        NS("disagg"),
    ] = 8997
    encoder_register_urls: A[
        List[str],
        "One or more EncoderBootstrapServer URLs to register this encoder with on startup, for dynamic encoder discovery. Example: --encoder-register-urls http://prefill0:8997 http://prefill1:8997. Used with --encoder-only servers.",
        NS("disagg"),
    ] = dataclasses.field(default_factory=list)
    enable_adaptive_dispatch_to_encoder: A[
        bool,
        "When enabled, adaptively dispatch: multi-image requests go to encoder in language_only epd mode, single-image requests are processed locally.",
        NS("disagg"),
    ] = False

    # -------------------------------------------------------------------------
    # PD-Multiplexing
    # -------------------------------------------------------------------------
    enable_pdmux: A[
        bool, "Enable PD-Multiplexing, PD running on greenctx stream.", NS("disagg")
    ] = False
    pdmux_config_path: A[
        Optional[str], "The path of the PD-Multiplexing config file.", NS("disagg")
    ] = None
    sm_group_num: A[int, "Number of sm partition groups.", NS("disagg")] = 8

    # -------------------------------------------------------------------------
    # Model weight update and weight loading
    # -------------------------------------------------------------------------
    startup_weight_load_mode: A[
        Literal["serial", "overlap"],
        (
            "Control startup weight loading relative to CUDA graph capture. "
            "'serial' preserves the existing startup order; 'overlap' stages "
            "checkpoint files while CUDA graphs are captured and commits the "
            "real weights afterward."
        ),
        NS("model"),
    ] = "serial"
    custom_weight_loader: A[
        Optional[List[str]],
        Arg(
            help="The custom dataloader which used to update the model. Should be set with a valid import path, such as my_package.weight_load_func",
            nargs="*",
        ),
        NS("model"),
    ] = None
    weight_loader_disable_mmap: A[
        bool, "Disable mmap while loading weight using safetensors.", NS("model")
    ] = False
    weight_loader_prefetch_checkpoints: A[
        bool,
        "Prefetch checkpoint files into OS page cache before loading. Each rank prefetches a fraction of the shards, reducing total network I/O on shared filesystems (NFS/Lustre) from N*checkpoint to 1*checkpoint. Recommended for models on network storage. When enabled, multi-threaded safetensors loading is disabled by default to avoid I/O oversubscription with the prefetch threads; set enable_multithread_load=true in --model-loader-extra-config to keep multi-threaded loading (e.g. on local NVMe where prefetch is a no-op).",
        NS("model"),
    ] = False
    weight_loader_prefetch_num_threads: A[
        int,
        "Number of threads per rank for checkpoint prefetching (default: 4).",
        NS("model"),
    ] = 4
    weight_loader_drop_cache_after_load: A[
        bool,
        "Call posix_fadvise(DONTNEED) on each safetensors shard after loading it.",
        NS("model"),
    ] = False
    remote_instance_weight_loader_seed_instance_ip: A[
        Optional[str],
        "The ip of the seed instance for loading weights from remote instance.",
        NS("model"),
    ] = None
    remote_instance_weight_loader_seed_instance_service_port: A[
        Optional[int],
        "The service port of the seed instance for loading weights from remote instance.",
        NS("model"),
    ] = None
    remote_instance_weight_loader_send_weights_group_ports: A[
        Optional[List[int]],
        Arg(
            help="The communication group ports for loading weights from remote instance.",
            type_parser=json_list_type,
        ),
        NS("model"),
    ] = None
    remote_instance_weight_loader_backend: A[
        Literal["transfer_engine", "nccl", "modelexpress"],
        "The backend for loading weights from remote instance. Can be 'transfer_engine', 'nccl', or 'modelexpress'. Default is 'nccl'.",
        NS("model"),
    ] = "nccl"
    remote_instance_weight_loader_start_seed_via_transfer_engine: A[
        bool,
        "Start seed server via transfer engine backend for remote instance weight loader.",
        NS("model"),
    ] = False
    engine_info_bootstrap_port: A[
        int,
        "Port for the engine info bootstrap server. Default is 6789. Must be set explicitly when running multiple instances on the same node.",
        NS("model"),
    ] = 6789
    modelexpress_config: A[
        Optional[str],
        'JSON config for ModelExpress P2P weight loading. Keys: "url" (optional gRPC host:port override), "transport" ("nixl" or "transfer_engine"). Example: \'{"url": "localhost:8001", "transport": "nixl"}\'',
        NS("model"),
    ] = None
    download_dir: A[
        Optional[str], "Model download directory for huggingface.", NS("model")
    ] = None
    model_checksum: A[
        Optional[str],
        Arg(
            help="Model file integrity verification. If provided without value, uses model-path as HF repo ID. Otherwise, provide checksums JSON file path or HuggingFace repo ID.",
            nargs="?",
            const="",
        ),
        NS("model"),
    ] = None
    delete_ckpt_after_loading: A[
        bool, "Delete the model checkpoint after loading the model.", NS("model")
    ] = False
    # Checkpoint decryption
    decrypted_config_file: A[
        Optional[str], "The path of the decrypted config file.", NS("model")
    ] = None
    decrypted_draft_config_file: A[
        Optional[str], "The path of the decrypted draft config file.", NS("model")
    ] = None
    checkpoint_engine_wait_weights_before_ready: A[
        bool,
        "If set, the server will wait for initial weights to be loaded via checkpoint-engine or other update methods before serving inference requests.",
        NS("model"),
    ] = False

    # -------------------------------------------------------------------------
    # Prefill delayer
    # -------------------------------------------------------------------------
    enable_prefill_delayer: A[
        bool,
        "Enable prefill delayer for DP attention to reduce idle time.",
        NS("schedule"),
    ] = False
    prefill_delayer_max_delay_passes: A[
        int, "Maximum forward passes to delay prefill.", NS("schedule")
    ] = 30
    prefill_delayer_token_usage_low_watermark: A[
        Optional[float],
        "Token usage low watermark for prefill delayer.",
        NS("schedule"),
    ] = None
    prefill_delayer_forward_passes_buckets: A[
        Optional[List[float]],
        "Custom buckets for prefill delayer forward passes histogram. 0 and max_delay_passes-1 will be auto-added.",
        NS("schedule"),
    ] = None
    prefill_delayer_wait_seconds_buckets: A[
        Optional[List[float]],
        "Custom buckets for prefill delayer wait seconds histogram. 0 will be auto-added.",
        NS("schedule"),
    ] = None
    prefill_delayer_queue_min_ratio: A[
        Optional[float],
        (
            "Opt-in to the adaptive queue-based delay trigger (independent of the "
            "slot-based one). Delays prefill until the waiting queue reaches "
            "min(running_req * ratio, prefill_max_requests), falling back to the "
            "observed max_prefill_bs when no request limit is set. Unset (default) "
            "keeps the original slot-only behavior. Typical: 0.1 ~ 0.5."
        ),
        NS("schedule"),
    ] = None
    prefill_delayer_max_delay_ms: A[
        Optional[float],
        (
            "Wall-clock cap (ms) on a single queue-trigger delay; once exceeded, "
            "prefill is force-released to bound worst-case TTFT. Only consulted "
            "when --prefill-delayer-queue-min-ratio is set. Typical: 1000 ~ "
            "5000; defaults to 5000 if unset."
        ),
        NS("schedule"),
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
            "speculative decoding with a separate draft prefill pass. An "
            "explicit value always wins, capped by max-running-requests "
            "(1 disables). When unset, DFlash workloads auto-enable the "
            "formula; other workloads stay disabled. Not supported with "
            "pipeline parallelism."
        ),
        NS("schedule"),
    ] = None

    # -------------------------------------------------------------------------
    # Deterministic inference
    # -------------------------------------------------------------------------
    enable_deterministic_inference: A[
        bool,
        "Enable deterministic inference mode with batch invariant ops.",
        NS("exec.deterministic"),
    ] = False
    rl_on_policy_target: A[
        Optional[str],
        Arg(
            help="The training system that SGLang needs to match for true on-policy.",
            choices=RL_ON_POLICY_TARGET_CHOICES,
        ),
        NS("exec.deterministic"),
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
        NS("observability"),
    ] = "none"
    kv_canary_real_data: A[str, NS("observability")] = "none"
    kv_canary_sweep_interval: A[
        int, "Every N forward steps, run a full-pool sweep.", NS("observability")
    ] = 0

    # -------------------------------------------------------------------------
    # Dynamic batch tokenizer
    # -------------------------------------------------------------------------
    enable_dynamic_batch_tokenizer: A[
        bool,
        "Enable async dynamic batch tokenizer for improved performance when multiple requests arrive concurrently.",
        NS("serving"),
    ] = False
    dynamic_batch_tokenizer_batch_size: A[
        int,
        "[Only used if --enable-dynamic-batch-tokenizer is set] Maximum batch size for dynamic batch tokenizer.",
        NS("serving"),
    ] = 32
    dynamic_batch_tokenizer_batch_timeout: A[
        float,
        "[Only used if --enable-dynamic-batch-tokenizer is set] Timeout in seconds for batching tokenization requests.",
        NS("serving"),
    ] = 0.002
    enable_tokenizer_batch_encode: A[
        bool,
        "Enable batch tokenization for improved performance when processing multiple text inputs. Do not use with image inputs, pre-tokenized input_ids, or input_embeds.",
        NS("serving"),
    ] = False
    disable_tokenizer_batch_decode: A[
        bool,
        "Disable batch decoding when decoding multiple completions.",
        NS("serving"),
    ] = False

    # -------------------------------------------------------------------------
    # Debug tensor dumps
    # -------------------------------------------------------------------------
    debug_tensor_dump_output_folder: A[
        Optional[str],
        "The output folder for dumping tensors. In Eagle mode, tensor outputs from draft and target models are stored in separate subdirectories ('draft' and 'target').",
        NS("observability"),
    ] = None
    # None means dump all layers.
    debug_tensor_dump_layers: A[
        Optional[List[int]],
        "The layer ids to dump. Dump all layers if not specified.",
        NS("observability"),
    ] = None
    # TODO(guoyuhong): clean the old dumper code.
    debug_tensor_dump_input_file: A[
        Optional[str], "The input filename for dumping tensors", NS("observability")
    ] = None

    # -------------------------------------------------------------------------
    # Misc runtime features
    # -------------------------------------------------------------------------
    enable_memory_saver: A[
        bool,
        "Allow saving memory using release_memory_occupation and resume_memory_occupation",
        NS("exec.features"),
    ] = False
    enable_weights_cpu_backup: A[
        bool,
        "Save model weights (both main model and draft model, if any) to CPU memory during release_weights_occupation and resume_weights_occupation",
        NS("exec.features"),
    ] = False
    enable_draft_weights_cpu_backup: A[
        bool,
        "Save draft model weights to CPU memory during release_weights_occupation and resume_weights_occupation",
        NS("exec.features"),
    ] = False
    enable_custom_logit_processor: A[
        bool,
        "Enable users to pass custom logit processors to the server (disabled by default for security)",
        NS("exec.features"),
    ] = False
    enable_return_hidden_states: A[
        bool,
        "Enable returning full hidden states with responses. Equivalent to "
        "`--return-hidden-states-mode full`.",
        NS("exec.features"),
    ] = False
    return_hidden_states_mode: A[
        Optional[str],
        Arg(
            help="Set the maximum hidden-state return mode supported by the "
            "server. `last` allows requests with return_hidden_states=False or "
            "`last`; `full` also allows return_hidden_states=True.",
            choices=["last", "full"],
        ),
        NS("exec.features"),
    ] = None
    enable_return_routed_experts: A[
        bool,
        "Enable returning routed experts of each layer with responses.",
        NS("exec.features"),
    ] = False
    enable_return_indexer_topk: A[
        bool,
        "Enable returning indexer topk indices of layers with indexer with responses.",
        NS("exec.features"),
    ] = False
    disable_outlines_disk_cache: A[
        bool,
        "Disable disk cache of outlines to avoid possible crashes related to file system or high concurrency.",
        NS("exec.features"),
    ] = False
    enable_mis: A[
        bool,
        "Enable Multi-Item Scoring optimization. Combines query and multiple items into a single sequence for efficient batch processing. Requires --attention-backend flashinfer; auto-disables CUDA graph, radix cache, and chunked prefill.",
        NS("exec.features"),
    ] = False

    # -------------------------------------------------------------------------
    # Weight cache
    # -------------------------------------------------------------------------
    weight_cache_mode: A[
        str,
        Arg(
            help="Weight cache mode. 'off': normal disk loading. "
            "'daemon': launch weight cache daemon (holds weights in GPU memory). "
            "Engine-spawned daemons are co-terminal with the engine and do NOT "
            "persist across restarts, so this alone does not speed up restart "
            "(the first start is slower). For fast recovery, run the standalone "
            "daemon (python -m sglang.srt.weight_cache.daemon) and connect with "
            "'client'. 'client': connect to existing daemon and load via IPC.",
            choices=["off", "daemon", "client"],
        ),
        NS("model"),
    ] = "off"
    weight_cache_socket: A[
        Optional[str],
        Arg(
            help="Unix socket path for weight cache daemon (client mode)."
            "If not set, uses /tmp/sglang_weight_cache_rank{global_rank}.sock",
        ),
        NS("model"),
    ] = None
    weight_cache_timeout: A[
        int,
        Arg(
            help="Timeout in seconds for weight cache daemon readiness (default: 1800).",
        ),
        NS("model"),
    ] = 1800

    # -------------------------------------------------------------------------
    # Custom hooks, probe, and plugins
    # -------------------------------------------------------------------------
    forward_hooks: A[
        Optional[List[dict[str, Any]]],
        Arg(
            help="JSON-formatted forward hook specifications to attach to the model.",
            type_parser=json_list_type,
        ),
        NS("observability"),
    ] = None
    msprobe_dump_config: A[
        Optional[str],
        "The path of the JSON configuration file for msProbe. If specified, enables msProbe dump.",
        NS("observability"),
    ] = None

    def __post_init__(self):
        """Construction leaves the record at what the caller asked for.

        Resolution is a separate act, entered through ``resolve_once``: the
        launcher runs it once per engine, and every publishing process asks the
        gate on the way in. A record that is only constructed -- a fixture, a
        config being inspected, one being handed to a subprocess that will
        resolve it itself -- stays raw.
        """

    def resolve_once(self) -> None:
        """Run the resolution pipeline, unless this record has been through it.

        Resolution is a deterministic function of the raw inputs -- two records
        built from the same arguments declare the same things -- but the
        handlers do not survive a second pass over their own output: DP
        attention halves ``chunked_prefill_size`` again on every re-entry.

        The publishing entry of every process calls this. In a child the record
        arrived by pickle and brought its declarations along, so the child has
        nothing left to derive and projects what the parent decided.
        """
        if getattr(self, "_resolution_finished", False):
            return
        if getattr(self, "_resolution_failed", False):
            raise RuntimeError(
                "resolution already failed on this ServerArgs; the handlers that "
                "ran left their writes on the record, and a second pass would "
                "read that partial output as fresh input. Build a new record "
                "from the corrected arguments."
            )
        try:
            self._run_resolution_pipeline()
        except BaseException:
            # The handlers that ran already declared, and they are not
            # idempotent over their own output.
            self._resolution_failed = True
            raise
        # Set here too, because the dummy/absent-model path returns before the
        # end of the pipeline that normally sets it: the gate is about whether
        # the handlers ran, not how far they got.
        self._resolution_finished = True

    def resolved_dict(self) -> Dict[str, Any]:
        """This configuration as a plain dict of resolved field values.

        What the whole-object readbacks report (`/server_info` and its gRPC and
        in-process twins). `dataclasses.asdict(self)` reads the fields, which
        carry the raw input; this reads the declarations, so it answers with what
        resolution decided. Nested dataclass fields are expanded
        the way `asdict` expands them; the private resolution bookkeeping and the
        `model_config` memo are not fields and do not appear.
        """
        from sglang.srt.arg_groups.overrides import resolution_projection

        return resolution_projection(self)

    def replace_resolved(self, source: str, **changes: Any) -> ServerArgs:
        """A copy of this record that stays resolved, and says what it changed.

        `dataclasses.replace` builds a new instance, so the copy carries none of
        what makes a record resolved: no raw snapshot, no declarations, no
        finished flag. The next publish therefore resolves it again, which
        drops every decision the stash held -- the late ones (the auto-detected
        parsers) and the direct ones alike -- and re-runs the device probes in
        whatever process opened the copy. The Ray paths replace
        `dist_init_addr` on a resolved record, which is how they reach this.

        The change is appended to the stash rather than left on the field: the
        projection reads the raw snapshot plus the declarations, so a field the
        copy set on its own would publish the parent's raw value instead.

        The carry is shallow. The containers are copied so the copy's own
        declaration does not travel back into the parent, but everything inside
        them -- the stash entries, the raw-input values, the memoized
        `ModelConfig` -- is shared. That is fine for what this is for: a copy
        that immediately crosses a process boundary (Ray actors, the gateway's
        workers), where pickling severs the sharing. A caller that mutates the
        copy's deep structure in-process mutates the parent's too.
        """
        replacement = dataclasses.replace(self, **changes)
        if not getattr(self, "_resolution_finished", False):
            # Not resolved yet: the copy goes through the gate itself.
            return replacement

        # Everything outside the fields, enumerated from the instance: the raw
        # snapshot, the stash, and what resolution memoized -- including the
        # `get_model_config()` memo, which the copy carries over rather than
        # rebuild.
        field_names = {field.name for field in dataclasses.fields(self)}
        for name, value in vars(self).items():
            if name in field_names or name == "_resolution_finished":
                continue
            if isinstance(value, (dict, list, set)):
                value = copy.copy(value)
            object.__setattr__(replacement, name, value)
        stash = getattr(replacement, "_resolved_overrides", None)
        if stash is None:
            stash = []
            object.__setattr__(replacement, "_resolved_overrides", stash)
        if changes:
            stash.append((source, dict(changes)))
        object.__setattr__(replacement, "_resolution_finished", True)
        return replacement

    def _declare(self, source: str, **fields: Any) -> None:
        """This record's handlers declaring their resolution writes.

        See ``arg_groups.overrides.declare_resolution``, which the hooks these
        handlers call reach directly.
        """
        from sglang.srt.arg_groups.overrides import declare_resolution

        declare_resolution(self, source, **fields)

    def _run_resolution_pipeline(self):
        """
        Orchestrates the handling of various server arguments, ensuring proper configuration and validation.

        Dispatcher style principles:
        1. Keep this method as an ordered dispatcher. Each step should be a
           named self._handle_* call; put imports, conditionals, mutations, and
           raises inside helpers instead of inline here.
        2. Keep the dummy-model boundary as early as correctness allows. Only
           model-independent bootstrap, API/network/protocol validation, and
           errors that should fire for dummy models should run before it.
        3. Order handlers by dependency domains, not by historical insertion:
           internal/bootstrap, API/network/protocol, model source/path
           resolution, hardware/platform, model-specific adjustment,
           parallelism, kernel/attention backend, cuda graph, memory/cache,
           and advanced/debug features.
        4. Hide narrow integrations behind general handler names. The
           dispatcher should say what phase is being handled, not expose a
           vendor-, hook-, or feature-specific implementation detail.
        5. Give each handler one clear contract: what state it expects, what it
           may mutate, and whether it validates only. Long ordering comments
           belong in the helper or signal that the helper should be split.
        """

        # What the caller asked for, before any handler runs; this plus the
        # stash is the resolution result the projection reads.
        self._raw_input = {
            field.name: getattr(self, field.name) for field in dataclasses.fields(self)
        }

        # Declaration stash for the override/post-process passes. Set before any
        # short-circuit (none/dummy model paths) so run_post_process_pass and
        # direct handler invocations can rely on it even when
        # _handle_model_specific_adjustments never runs.
        self._resolved_overrides = []

        cfg = resolving_view(self)

        from sglang.srt.arg_groups.mega_moe_hook import handle_mega_moe

        handle_mega_moe(self)
        self._handle_return_hidden_states_mode()
        self._handle_media_url_security()
        self._handle_hicache_ratio_default()
        self._validate_prefill_decode_interval()

        # Reject an explicitly enabled but incompatible hardware runtime before
        # model path resolution, downloads, or the dummy-model short circuit.
        self._handle_hardware_runtime_validation()
        if cfg.model_path.lower() in ["none", "dummy"]:
            return

        self._handle_model_source_paths()

        # Validate mm_process_config.
        self._handle_multimodal()
        # Validate SSL arguments early.
        self._handle_ssl_validation()
        # Validate transcription/ASR-specific server args.
        self._handle_asr_validation()

        # Handle deprecated arguments.
        self._handle_deprecated_args()

        # Handle deprecated environment variables for prefill delayer.
        self._handle_prefill_delayer_env_compat()

        # Set missing default values.
        self._handle_missing_default_values()

        # expert_pack may replace a raw GGUF input with its generated local
        # model metadata before any model-specific handler calls get_model_config.
        # It also establishes eager-only invariants before CUDA graph parsing.
        self._handle_expert_pack()

        # Validate PD disaggregation flags before CUDA graph config.
        self._handle_pd_disaggregation()

        # Normalize deprecated CP aliases before validations or model-specific
        # defaults inspect enable_prefill_cp/cp_strategy.
        self._handle_legacy_cp_arguments()
        self._validate_prefill_only_disable_kv_cache_args()
        self._handle_dcp_validation()

        # Model-arch prefill CUDA-graph default must land before cuda-graph
        # resolution (the declarative registry materializes too late to affect
        # it). Inkling opts into full-graph prefill capture here.
        self._apply_inkling_prefill_cuda_graph_default()
        self._apply_muse_glimmer_prefill_cuda_graph_max_bs_default()

        # must run before _handle_cuda_graph_config and _handle_data_parallelism
        self._handle_dwdp()

        self._handle_cuda_graph_config()

        # Handle device-specific backends.
        self._handle_hpu_backends()
        self._handle_cpu_backends()
        self._handle_npu_backends()
        self._handle_mps_backends()
        self._handle_xpu_backends()

        # OOT platform plugins set fields directly (an interface this tree
        # does not own); the diff records what they applied.
        declare_direct_writes(
            self,
            f"platform:{current_platform.device_name}",
            current_platform.apply_server_args_defaults,
        )

        gpu_mem = get_device_memory_capacity(cfg.device)

        # Handle memory-related, chunked prefill, and CUDA graph batch size configurations.
        self._handle_gpu_memory_settings(gpu_mem)

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
        self._handle_mxfp8_kv_cache_compatibility()
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

        # Normalize load balancing defaults.
        self._handle_load_balance_method()

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
        self._validate_experimental_sgl_marlin()

        # Handle pipeline parallelism.
        self._handle_pipeline_parallelism()

        # Handle speculative decoding logic.
        from sglang.srt.arg_groups.speculative_hook import handle_speculative_decoding

        handle_speculative_decoding(self)

        # Validate the CuteDSL A2A token budget now that num_tokens_per_req is final.
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

        self._handle_unified_memory_pool()

        # Handle diffusion LLM inference.
        self._handle_dllm_inference()

        # Handle crash dump environment variables (must run before CUDA init).
        self._handle_crash_dump_env()

        # Handle debug utilities.
        self._handle_debug_utils()

        # Handle any other necessary validations.
        self._handle_other_validations()

        # Model-capability adjustments that legacy code applied at model-load
        # time; last declarations of the resolution, mirroring that order.
        self._handle_model_capability_adjustments()

        # Validate after all batch-size declarations are visible.
        self._validate_deepep_v2_speculative_draft()
        self._validate_deepep_v2_dispatch_token_budget()

        self._resolution_finished = True

    def _handle_return_hidden_states_mode(self):
        from sglang.srt.arg_groups.serving_hook import handle_return_hidden_states_mode

        handle_return_hidden_states_mode(self)

    def _handle_model_capability_adjustments(self):
        from sglang.srt.arg_groups.model_hook import handle_model_capability_adjustments

        handle_model_capability_adjustments(self)

    def _handle_model_source_paths(self):
        from sglang.srt.arg_groups.model_path_hook import handle_model_source_paths

        handle_model_source_paths(self)

    def _handle_pd_disaggregation(self):
        from sglang.srt.arg_groups.pd_disaggregation_hook import (
            handle_pd_disaggregation,
        )

        handle_pd_disaggregation(self)

    def _handle_dcp_validation(self):
        from sglang.srt.arg_groups.parallel_hook import handle_dcp_validation

        handle_dcp_validation(self)

    def _handle_load_balance_method(self):
        from sglang.srt.arg_groups.serving_hook import handle_load_balance_method

        handle_load_balance_method(self)

    def _handle_ssl_validation(self):
        from sglang.srt.arg_groups.serving_hook import handle_ssl_validation

        handle_ssl_validation(self)

    def _handle_multimodal(self):
        from sglang.srt.arg_groups.serving_hook import handle_multimodal

        handle_multimodal(self)

    def _handle_media_url_security(self):
        from sglang.srt.arg_groups.serving_hook import handle_media_url_security

        handle_media_url_security(self)

    def _handle_deprecated_args(self):
        from sglang.srt.arg_groups.serving_hook import handle_deprecated_args

        handle_deprecated_args(self)

    def _handle_prefill_delayer_env_compat(self):
        from sglang.srt.arg_groups.serving_hook import handle_prefill_delayer_env_compat

        handle_prefill_delayer_env_compat(self)

    def _handle_missing_default_values(self):
        from sglang.srt.arg_groups.serving_hook import handle_missing_default_values

        handle_missing_default_values(self)

    def _handle_modelscope_paths(self):
        from sglang.srt.arg_groups.model_path_hook import handle_modelscope_paths

        handle_modelscope_paths(self)

    def _handle_hpu_backends(self):
        from sglang.srt.arg_groups.platform_hook import handle_hpu_backends

        handle_hpu_backends(self)

    def _handle_cpu_backends(self):
        from sglang.srt.arg_groups.platform_hook import handle_cpu_backends

        handle_cpu_backends(self)

    def _handle_hardware_runtime_validation(self):
        # This is intentionally independent of self.device: setting
        # SGLANG_USE_MLX opts into the MLX backend and must fail immediately if
        # the environment cannot honor that request. With the flag unset,
        # use_mlx() remains lazy and does not import MLX.
        use_mlx()

    def _handle_npu_backends(self):
        from sglang.srt.arg_groups.platform_hook import handle_npu_backends

        handle_npu_backends(self)

    def _handle_mps_backends(self):
        from sglang.srt.arg_groups.platform_hook import handle_mps_backends

        handle_mps_backends(self)

    def _handle_xpu_backends(self):
        from sglang.srt.arg_groups.platform_hook import handle_xpu_backends

        handle_xpu_backends(self)

    # ------------------------------------------------------------------
    # CUDA graph configuration resolution
    # ------------------------------------------------------------------
    def _apply_inkling_prefill_cuda_graph_default(self):
        from sglang.srt.arg_groups.cuda_graph_hook import (
            apply_inkling_prefill_cuda_graph_default,
        )

        apply_inkling_prefill_cuda_graph_default(self)

    def _apply_muse_glimmer_prefill_cuda_graph_max_bs_default(self):
        from sglang.srt.arg_groups.cuda_graph_hook import (
            apply_muse_glimmer_prefill_cuda_graph_max_bs_default,
        )

        apply_muse_glimmer_prefill_cuda_graph_max_bs_default(self)

    def _handle_cuda_graph_config(self):
        from sglang.srt.arg_groups.cuda_graph_hook import handle_cuda_graph_config

        handle_cuda_graph_config(self)

    def _apply_deepep_adjustments(self):
        from sglang.srt.arg_groups.cuda_graph_hook import apply_deepep_adjustments

        apply_deepep_adjustments(self)

    def _parse_cuda_graph_config(self):
        from sglang.srt.arg_groups.cuda_graph_hook import parse_cuda_graph_config

        parse_cuda_graph_config(self)

    def _apply_cuda_graph_compatibility(self):
        from sglang.srt.arg_groups.cuda_graph_hook import apply_cuda_graph_compatibility

        apply_cuda_graph_compatibility(self)

    def _apply_cuda_graph_disaggregation_roles(self):
        cfg = resolving_view(self)
        if cfg.disaggregation_mode == "prefill":
            if (Phase.DECODE, "backend") not in self._cuda_graph_config_locked:
                self._declare(
                    "_apply_cuda_graph_disaggregation_roles",
                    cuda_graph_config=with_phase(
                        cfg.cuda_graph_config, Phase.DECODE, backend=Backend.DISABLED
                    ),
                )
        elif cfg.disaggregation_mode == "decode":
            if (Phase.PREFILL, "backend") not in self._cuda_graph_config_locked:
                self._declare(
                    "_apply_cuda_graph_disaggregation_roles",
                    cuda_graph_config=with_phase(
                        cfg.cuda_graph_config, Phase.PREFILL, backend=Backend.DISABLED
                    ),
                )

    def _disable_tc_piecewise_cudagraph_if_incompatible(self):
        from sglang.srt.arg_groups.cuda_graph_hook import (
            disable_tc_piecewise_cudagraph_if_incompatible,
        )

        disable_tc_piecewise_cudagraph_if_incompatible(self)

    def _disable_breakable_cudagraph_if_incompatible(self):
        from sglang.srt.arg_groups.cuda_graph_hook import (
            disable_breakable_cudagraph_if_incompatible,
        )

        disable_breakable_cudagraph_if_incompatible(self)

    def _disable_full_prefill_cudagraph_if_incompatible(self):
        from sglang.srt.arg_groups.cuda_graph_hook import (
            disable_full_prefill_cudagraph_if_incompatible,
        )

        disable_full_prefill_cudagraph_if_incompatible(self)

    def _disable_prefill_cuda_graph_for_deepseek_trtllm_mla(self):
        from sglang.srt.arg_groups.cuda_graph_hook import (
            disable_prefill_cuda_graph_for_deepseek_trtllm_mla,
        )

        disable_prefill_cuda_graph_for_deepseek_trtllm_mla(self)

    def _validate_cuda_graph_config(self):
        cfg = resolving_view(self)
        if cfg.cuda_graph_config is None:
            return
        for phase in Phase.ALL:
            backend = getattr(cfg.cuda_graph_config, phase).backend
            if backend not in ALLOWED_BACKENDS_PER_PHASE[phase]:
                raise ValueError(
                    f"--cuda-graph-config[{phase}].backend={backend!r} not allowed; "
                    f"allowed: {ALLOWED_BACKENDS_PER_PHASE[phase]}"
                )

    def _handle_multi_item_scoring(self):
        from sglang.srt.arg_groups.attention_hook import handle_multi_item_scoring

        handle_multi_item_scoring(self)

    def _handle_gpu_memory_settings(self, gpu_mem):
        from sglang.srt.arg_groups.memory_hook import handle_gpu_memory_settings

        handle_gpu_memory_settings(self, gpu_mem)

    def post_capture_kv_sizing_planned(self) -> bool:
        """Whether the mem_fraction heuristic may skip the graph reserve; must be
        False for any config the runtime won't post-capture-size, else it gets an
        under-reserved fraction."""
        cfg = resolving_view(self)
        # use_mla_backend is a method at args time but ModelRunner overwrites it
        # with a bool on global_server_args (see the FIXME there) -- handle both.
        use_mla = self.use_mla_backend
        mla_enabled = use_mla() if callable(use_mla) else use_mla
        if not envs.SGLANG_ENABLE_POST_CAPTURE_KV_SIZING.get():
            return False
        if cfg.device != "cuda":
            return False
        if cfg.dcp_size != 1:
            return False
        if mla_enabled:
            return False
        if cfg.kv_cache_dtype == "fp4_e2m1":
            return False
        if cfg.prefill_only_disable_kv_cache:
            return False
        if cfg.enable_memory_saver:
            return False
        if envs.SGLANG_MOONCAKE_CUSTOM_MEM_POOL.get() is not None:
            return False

        if (
            cfg.disaggregation_mode != "prefill"
            and cfg.cuda_graph_config.decode.backend == Backend.DISABLED
        ):
            return False

        if cfg.disaggregation_mode != "decode":
            prefill_cfg = cfg.cuda_graph_config.prefill
            # We can only skip eager activation headroom when the largest
            # prefill forward batch size is already graph-captured. Otherwise,
            # an eager forward will need more memory and lead to OOM.
            if (
                prefill_cfg.backend == Backend.DISABLED
                or cfg.chunked_prefill_size <= 0
                or self.max_prefill_buffer_tokens() > max(prefill_cfg.bs or (0,))
            ):
                return False

        from sglang.srt.configs.model_config import is_deepseek_v4, is_minimax_sparse

        hf_config = self.get_model_config().hf_config
        if is_deepseek_v4(hf_config) or is_minimax_sparse(hf_config):
            return False

        return True

    def pre_capture_activation_reserve_mb(self, gpu_mem: Optional[float]) -> float:
        # Runtime activation working-set reserve for eager decode above the captured
        # max_bs and transient prefill/logits; also covers fixed state caches.
        cfg = resolving_view(self)
        if cfg.disaggregation_mode == "decode":
            running_requests = (
                cfg.max_running_requests or cfg.cuda_graph_config.decode.max_bs or 1
            )
            activation_tokens = max(
                running_requests * (cfg.speculative_num_draft_tokens or 1), 2048
            )
        elif cfg.chunked_prefill_size > 0:
            activation_tokens = max(cfg.chunked_prefill_size, 2048)
        else:
            activation_tokens = max(cfg.max_prefill_tokens, 2048)
        reserved_mem = (
            512 + activation_tokens * 1.5 + cfg.tp_size * cfg.pp_size / 8 * 1024
        )
        if gpu_mem is not None and gpu_mem > 60 * 1024:
            reserved_mem = max(reserved_mem, 10 * 1024)
        return reserved_mem

    def reserve_for_graph_mb(self) -> float:
        cfg = resolving_view(self)
        decode_cuda_graph_config = cfg.cuda_graph_config.decode
        prefill_cuda_graph_config = cfg.cuda_graph_config.prefill

        reserved_mem = 0.0
        if (
            cfg.disaggregation_mode != "prefill"
            and decode_cuda_graph_config.backend != Backend.DISABLED
        ):
            reserved_mem += decode_cuda_graph_config.max_bs * 2

        if (
            self._resolved().enable_dp_attention
            and cfg.disaggregation_mode != "prefill"
        ):
            # DP attention needs more padding for some operations, and much more for large
            # cuda graph max bs (torch allocator / implementation inefficiencies).
            reserved_mem += decode_cuda_graph_config.max_bs * cfg.dp_size * 3
            if decode_cuda_graph_config.max_bs > 300:
                reserved_mem += decode_cuda_graph_config.max_bs * cfg.dp_size * 1.5

        if (
            cfg.disaggregation_mode != "decode"
            and prefill_cuda_graph_config.backend != Backend.DISABLED
        ):
            if not self.use_mla_backend():
                # Only non-torch memory is counted; torch memory is reused by cuda graph capture.
                reserved_mem += len(prefill_cuda_graph_config.bs) * 8
            else:
                # MLA backend overhead is much higher than expected with fa3.
                reserved_mem += 1.5 * 1024

            if (
                prefill_cuda_graph_config.backend == Backend.BREAKABLE
                and resolved_view(self).moe_a2a_backend == "deepep"
            ):
                # Prefill-BCG DeepEP delta (bridge pool + NVL first-touch
                # during capture); decode-side DeepEP is a baseline cost.
                reserved_mem += 1 * 1024

        return reserved_mem

    def reserve_for_deepep_a2a_mb(self) -> float:
        # DeepEP all-to-all buffers captured in the decode graph are real extra
        # allocations, reserved on top of the floor.

        cfg = resolving_view(self)
        decode_cuda_graph_config = cfg.cuda_graph_config.decode
        if (
            cfg.disaggregation_mode != "prefill"
            and decode_cuda_graph_config.backend != Backend.DISABLED
            and resolved_view(self).moe_a2a_backend == "deepep"
        ):
            return 2 * 1024
        return 0.0

    def _generate_decode_cuda_graph_batch_sizes(self, max_bs: int):
        """
        Generate the list of batch sizes for CUDA graph capture based on max_bs.
        This integrates the logic from cuda_graph_runner.py.
        """
        cfg = resolving_view(self)
        # Handle disable_cuda_graph_padding as the first condition for both spec and non-spec
        if cfg.disable_cuda_graph_padding:
            capture_bs = list(range(1, max_bs + 1))
        elif cfg.speculative_algorithm is None:
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
        cfg = resolving_view(self)
        if cfg.disable_cuda_graph_padding:
            capture_bs = list(range(1, cfg.torch_compile_max_bs + 1))
        else:
            capture_bs = sorted(
                set().union(
                    range(1, 17),
                    range(18, 31, 2),
                    range(32, 81, 4),
                    range(84, cfg.torch_compile_max_bs + 1, 8),
                    {cfg.torch_compile_max_bs},
                )
            )
        capture_bs = [bs for bs in capture_bs if bs <= cfg.torch_compile_max_bs]

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

    def _set_default_dsa_kv_cache_dtype(self, major: int, quantization: str) -> None:
        # Moved to the resolution pipeline (arg_groups/overrides.py:
        # _dsa_kv_cache_dtype_default), invoked here at its legacy slot.
        from sglang.srt.arg_groups.overrides import (
            _dsa_kv_cache_dtype_default,
            run_post_process_pass,
        )

        run_post_process_pass(self, _dsa_kv_cache_dtype_default)

    def _set_default_dsa_backends(self, major: int) -> None:
        # Moved to the resolution pipeline (arg_groups/overrides.py:
        # _dsa_split_backend_resolution), invoked here at its legacy slot.
        from sglang.srt.arg_groups.overrides import (
            _dsa_split_backend_resolution,
            run_post_process_pass,
        )

        run_post_process_pass(self, _dsa_split_backend_resolution)

    def _validate_hisparse_dsa_backend(self, attr: str, label: str):
        from sglang.srt.arg_groups.hisparse_hook import validate_hisparse_dsa_backend

        validate_hisparse_dsa_backend(self, attr, label)

    def _validate_hisparse_kv_cache_dtype(self):
        from sglang.srt.arg_groups.hisparse_hook import validate_hisparse_kv_cache_dtype

        validate_hisparse_kv_cache_dtype(self)

    def _handle_model_specific_adjustments(self):
        from sglang.srt.arg_groups.model_hook import handle_model_specific_adjustments

        handle_model_specific_adjustments(self)

    def _support_mamba_cache_extra_buffer(self, model_arch: str):
        from sglang.srt.arg_groups.overrides import supports_mamba_cache_extra_buffer

        return supports_mamba_cache_extra_buffer(self, model_arch)

    def _validate_mamba_no_buffer(self, view, model_arch: str):
        assert view.page_size in (1, None), "no_buffer only supports page_size=1."
        assert (
            view.disable_overlap_schedule
        ), "no_buffer do not support overlap schedule. Try to set disable_overlap_schedule=True."
        assert (
            view.attention_backend != "trtllm_mha"
        ), "no_buffer do not support trtllm_mha attention backend."

    def _validate_mamba_extra_buffer(self, view, model_arch: str):
        from sglang.srt.arg_groups.overrides import supports_mamba_cache_extra_buffer

        assert supports_mamba_cache_extra_buffer(
            view, model_arch
        ), f"extra_buffer is not supported for {model_arch}; use no_buffer."
        assert (
            is_cuda() or is_musa() or is_npu() or is_hip() or is_xpu()
        ), "extra_buffer needs CUDA/MUSA/NPU/ROCm/XPU (FLA)."
        if view.mamba_radix_cache_strategy == "extra_buffer_lazy":
            # The PD-disagg decode pool is not wired for lazy slots.
            assert view.disaggregation_mode == "null", (
                "extra_buffer_lazy unsupported under PD disaggregation; use "
                "--mamba-radix-cache-strategy extra_buffer."
            )
            # eagle/ngram/dspark/dflash all verify through
            # prepare_mamba_track_for_verify (lazy plan wired); dflash gained
            # the hook in DFlashVerifyInput.prepare_for_verify.
        if view.speculative_num_draft_tokens is not None:
            assert view.mamba_track_interval >= view.speculative_num_draft_tokens
        if view.page_size is not None:
            assert view.mamba_track_interval % view.page_size == 0
            assert self.mamba_cache_chunk_size is not None

            if (
                view.chunked_prefill_size is not None
                and 0 < view.chunked_prefill_size < self.mamba_cache_chunk_size
            ):
                logger.warning(
                    "Mamba radix extra-buffer is enabled with chunked_prefill_size=%s "
                    "smaller than mamba_cache_chunk_size=%s. This can make "
                    "mamba_track_mask false for unfinished chunked-prefill handoff "
                    "and skip Mamba state checkpoints.",
                    view.chunked_prefill_size,
                    self.mamba_cache_chunk_size,
                )

    def _handle_mamba_radix_cache(self, model_arch: str):
        # Resolution moved to the resolution pipeline (arg_groups/overrides.py:
        # _mamba_radix_cache_resolution), invoked here at each legacy call
        # slot; this handler keeps the validation.
        from sglang.srt.arg_groups.model_hook import handle_mamba_radix_cache

        handle_mamba_radix_cache(self, model_arch)

    def _handle_sampling_backend(self):
        # Moved to the resolution pipeline (arg_groups/overrides.py:
        # _sampling_backend_default), invoked here at its legacy slot.
        from sglang.srt.arg_groups.overrides import (
            _sampling_backend_default,
            run_post_process_pass,
        )

        run_post_process_pass(self, _sampling_backend_default)

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
        cfg = resolving_view(self)
        # OOT platforms provide their own default attention backend.
        if current_platform.is_out_of_tree():
            return current_platform.get_default_attention_backend()

        # Whisper requires flashinfer for cross-attention CUDA graph support.
        if "WhisperForConditionalGeneration" in (
            model_config.hf_config.architectures or []
        ):
            return "flashinfer"

        if not use_mla_backend:
            # MHA architecture

            if is_hopper_with_cuda_12_3() and is_no_spec_infer_or_topk_one(
                resolved_view(self)
            ):
                # Note: flashinfer 0.6.1 caused performance regression on Hopper attention kernel
                # Before the kernel is fixed, we choose fa3 as the default backend on Hopper MHA
                # ref: https://github.com/sgl-project/sglang/issues/17411
                return "fa3"
            elif (
                is_sm100_supported()
                and is_no_spec_infer_or_topk_one(resolved_view(self))
                and (
                    cfg.speculative_algorithm is None
                    or cfg.speculative_eagle_topk is not None
                )
            ):
                # trtllm_mha requires equal K/V row widths; fa4 carries
                # v_head_dim through.
                if model_config.has_asymmetric_kv:
                    return "fa4"
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
        from sglang.srt.arg_groups.attention_hook import (
            handle_attention_backend_compatibility,
        )

        handle_attention_backend_compatibility(self)

    def _handle_mxfp8_kv_cache_compatibility(self):
        from sglang.srt.arg_groups.kv_cache_hook import (
            handle_mxfp8_kv_cache_compatibility,
        )

        handle_mxfp8_kv_cache_compatibility(self)

    def _handle_kv4_compatibility(self):
        from sglang.srt.arg_groups.kv_cache_hook import handle_kv4_compatibility

        handle_kv4_compatibility(self)

    def _handle_page_size(self):
        # Moved to the resolution pipeline (arg_groups/overrides.py:
        # _page_size_default), invoked here at its legacy slot.
        from sglang.srt.arg_groups.overrides import (
            _page_size_default,
            run_post_process_pass,
        )

        run_post_process_pass(self, _page_size_default)

    def _handle_amd_specifics(self):
        from sglang.srt.arg_groups.platform_hook import handle_amd_specifics

        handle_amd_specifics(self)

    def _handle_nccl_pre_warm(self):
        from sglang.srt.arg_groups.platform_hook import handle_nccl_pre_warm

        handle_nccl_pre_warm(self)

    def _handle_grammar_backend(self):
        from sglang.srt.arg_groups.serving_hook import handle_grammar_backend

        handle_grammar_backend(self)

    def _handle_mamba_backend(self):
        from sglang.srt.arg_groups.mamba_hook import handle_mamba_backend

        handle_mamba_backend(self)

    def _handle_int8_mamba_checkpoint(self):
        from sglang.srt.arg_groups.mamba_hook import handle_int8_mamba_checkpoint

        handle_int8_mamba_checkpoint(self)

    def _handle_linear_attn_backend(self):
        from sglang.srt.arg_groups.attention_hook import handle_linear_attn_backend

        handle_linear_attn_backend(self)

    def _handle_legacy_cp_arguments(self):
        from sglang.srt.arg_groups.parallel_hook import handle_legacy_cp_arguments

        handle_legacy_cp_arguments(self)

    def _handle_context_parallelism(self):
        from sglang.srt.arg_groups.parallel_hook import handle_context_parallelism

        handle_context_parallelism(self)

    def _handle_dwdp(self):
        from sglang.srt.arg_groups.parallel_hook import handle_dwdp

        handle_dwdp(self)

    def _handle_data_parallelism(self):
        from sglang.srt.arg_groups.parallel_hook import handle_data_parallelism

        handle_data_parallelism(self)

    def _handle_moe_kernel_config(self):
        from sglang.srt.arg_groups.moe_hook import handle_moe_kernel_config

        handle_moe_kernel_config(self)

    def cutedsl_moe_max_num_tokens(self) -> int:
        """Largest number of tokens a single forward routes through a CuteDSL
        MoE layer on one (DP) rank. Single source of truth for both the
        standard-allgather wrapper buffers and the FlashInfer A2A dispatcher
        budget. Max over the prefill (max_prefill_tokens), piecewise-prefill
        capture, and decode/verify bounds; num_tokens_per_req is
        speculative_num_draft_tokens under speculative decoding, else 1.
        """
        cfg = resolving_view(self)
        if cfg.speculative_algorithm:
            num_tokens_per_req = cfg.speculative_num_draft_tokens or 1
        else:
            num_tokens_per_req = 1
        prefill_tokens = cfg.max_prefill_tokens
        cg_config = cfg.cuda_graph_config
        if cg_config is not None and cg_config.prefill.backend == Backend.TC_PIECEWISE:
            prefill_tokens = max(prefill_tokens, cg_config.prefill.max_bs or 0)
        decode_max_bs = (cg_config.decode.max_bs if cg_config is not None else 0) or 0
        decode_tokens = decode_max_bs * num_tokens_per_req
        return max(prefill_tokens, decode_tokens)

    def max_prefill_buffer_tokens(self) -> int:
        """Prefill-buffer ceiling: chunked_prefill_size, except PP dynamic
        chunking can grow chunks toward max_prefill_tokens and probe at 1.25x."""
        cfg = resolving_view(self)
        chunked = (
            cfg.chunked_prefill_size
            if cfg.chunked_prefill_size and cfg.chunked_prefill_size > 0
            else 0
        )
        tokens = chunked
        if cfg.enable_dynamic_chunking and cfg.pp_size > 1 and chunked:
            tokens = max(tokens, cfg.max_prefill_tokens or 0, math.ceil(chunked * 1.25))
        return tokens

    def _validate_cutedsl_a2a_token_budget(self):
        from sglang.srt.arg_groups.moe_hook import validate_cutedsl_a2a_token_budget

        validate_cutedsl_a2a_token_budget(self)

    def _validate_deepep_v2_dispatch_token_budget(self) -> None:
        from sglang.srt.arg_groups.moe_hook import (
            validate_deepep_v2_dispatch_token_budget,
        )

        validate_deepep_v2_dispatch_token_budget(self)

    def _validate_deepep_v2_model_architecture(self) -> None:
        from sglang.srt.arg_groups.moe_hook import validate_deepep_v2_model_architecture

        validate_deepep_v2_model_architecture(self)

    def _validate_deepep_v2_speculative_draft(self) -> None:
        from sglang.srt.arg_groups.moe_hook import validate_deepep_v2_speculative_draft

        validate_deepep_v2_speculative_draft(self)

    def _handle_a2a_moe(self):
        from sglang.srt.arg_groups.moe_hook import handle_a2a_moe

        handle_a2a_moe(self)

    def _required_mori_dispatch_tokens_per_rank(self) -> int:
        """Max tokens a single rank dispatches through MoRI in one forward."""
        cfg = resolving_view(self)
        return cfg.chunked_prefill_size

    def _required_pplx_dispatch_tokens_per_rank(self) -> int:
        """Max tokens a single rank dispatches through pplx in one forward."""
        cfg = resolving_view(self)
        required = cfg.chunked_prefill_size
        if cfg.cuda_graph_max_bs_decode is not None:
            required = max(required, cfg.cuda_graph_max_bs_decode)
        return required

    def _handle_eplb_and_dispatch(self):
        from sglang.srt.arg_groups.parallel_hook import handle_eplb_and_dispatch

        handle_eplb_and_dispatch(self)

    def _handle_elastic_ep(self):
        from sglang.srt.arg_groups.parallel_hook import handle_elastic_ep

        handle_elastic_ep(self)

    def _validate_experimental_sgl_marlin(self):
        view = self._resolved()
        if view.moe_runner_backend != "experimental_sgl_marlin":
            return

        # ===== TO BE REFACTORED ====
        from sglang.srt.lora.marlin_lora_temp.policy import (
            validate_experimental_sgl_marlin_server_args,
        )

        validate_experimental_sgl_marlin_server_args(self, view)
        # ===== END TO BE REFACTORED ====

    def _handle_expert_distribution_metrics(self):
        from sglang.srt.arg_groups.parallel_hook import (
            handle_expert_distribution_metrics,
        )

        handle_expert_distribution_metrics(self)

    def _handle_pipeline_parallelism(self):
        # Moved to the resolution pipeline (arg_groups/overrides.py:
        # _pipeline_parallel_overlap_disable), invoked here at its legacy slot.
        from sglang.srt.arg_groups.overrides import (
            _pipeline_parallel_overlap_disable,
            run_post_process_pass,
        )

        run_post_process_pass(self, _pipeline_parallel_overlap_disable)

    def _validate_prefill_only_disable_kv_cache_args(self):
        """Validate --prefill-only-disable-kv-cache flag/precondition constraints.

        Backend resolution is checked separately by
        _handle_prefill_only_disable_kv_cache after backends settle.
        """
        cfg = resolving_view(self)
        if not cfg.prefill_only_disable_kv_cache:
            return

        # This flag is intentionally scoped to embedding mode for now. Other
        # prefill-only paths (for example scoring and MIS) can benefit from
        # the same idea later, but some of them still stage K/V through the
        # paged cache today.
        if not cfg.is_embedding:
            raise ValueError(
                "--prefill-only-disable-kv-cache currently requires --is-embedding. "
                "Other prefill-only workloads may be supported in a future change once "
                "their attention paths stop reading or writing the paged KV cache."
            )
        if cfg.kv_cache_dtype in ("nvfp4", "fp4_mx_block16"):
            raise ValueError(
                "--prefill-only-disable-kv-cache does not currently support "
                "--kv-cache-dtype=nvfp4 or --kv-cache-dtype=fp4_mx_block16 because "
                "the FP4 pool uses a separate allocation path."
            )
        if cfg.kv_cache_dtype == "mxfp8":
            raise ValueError(
                "--prefill-only-disable-kv-cache does not currently support "
                "--kv-cache-dtype=mxfp8 because the MXFP8 pool stores separate "
                "scale-factor buffers."
            )

        # Structural preconditions for the FA backend's fa_skip_kv_cache path,
        # which is the only embedding path that doesn't read or write the pool:
        # - chunked_prefill_size == -1 keeps a request in a single forward,
        #   so K/V never has to be reused across prefill chunks.
        # - disable_radix_cache stops the prefix cache from indexing pool
        #   slots that no longer hold real data.
        if cfg.chunked_prefill_size != -1:
            raise ValueError(
                "--prefill-only-disable-kv-cache requires --chunked-prefill-size=-1 so the FA "
                "backend takes the fa_skip_kv_cache path; otherwise the pool would be touched "
                "between prefill chunks."
            )
        if not cfg.disable_radix_cache:
            raise ValueError(
                "--prefill-only-disable-kv-cache requires --disable-radix-cache because the "
                "radix cache indexes KV pool slots that no longer hold real data."
            )

        # Context-parallel prefill stages K/V through cp_allgather_and_save_kv_cache,
        # which writes to the pool via set_kv_buffer. NoOpMHATokenToKVPool intentionally
        # raises on writes, so the engine would boot fine but fail on the first request.
        if self._resolved().attn_cp_size > 1:
            raise ValueError(
                "--prefill-only-disable-kv-cache is incompatible with --attn-cp-size > 1: "
                "the context-parallel attention path writes K/V to the pool via set_kv_buffer, "
                "which the no-op pool intentionally rejects."
            )
        if cfg.enable_prefill_cp:
            raise ValueError(
                "--prefill-only-disable-kv-cache is incompatible with "
                "--enable-prefill-cp: the prefill-CP path stages K/V through "
                "the paged cache, which the no-op pool does not support."
            )

        # HiSparse selects a different pool class (HiSparseDSATokenToKVPool /
        # HiSparseTokenToKVPoolAllocator) that is not the no-op pool.
        if cfg.enable_hisparse:
            raise ValueError(
                "--prefill-only-disable-kv-cache is incompatible with --enable-hisparse: "
                "HiSparse uses a dedicated pool family that is not the no-op MHA pool."
            )

    def _handle_prefill_only_disable_kv_cache(self):
        from sglang.srt.arg_groups.kv_cache_hook import (
            handle_prefill_only_disable_kv_cache,
        )

        handle_prefill_only_disable_kv_cache(self)

    def _handle_hicache_ratio_default(self):
        from sglang.srt.arg_groups.hicache_hook import handle_hicache_ratio_default

        handle_hicache_ratio_default(self)

    def _handle_hicache(self):
        from sglang.srt.arg_groups.hicache_hook import handle_hicache

        handle_hicache(self)

    def _validate_hicache_host_memory_mode(self):
        from sglang.srt.arg_groups.hicache_hook import validate_hicache_host_memory_mode

        validate_hicache_host_memory_mode(self)

    def _resolve_hicache_dcp_compatibility(self):
        from sglang.srt.arg_groups.hicache_hook import resolve_hicache_dcp_compatibility

        resolve_hicache_dcp_compatibility(self)

    def _resolve_layout_io_compatibility(self):
        from sglang.srt.arg_groups.hicache_hook import resolve_layout_io_compatibility

        resolve_layout_io_compatibility(self)

    def _resolve_storage_layout_compatibility(self):
        from sglang.srt.arg_groups.hicache_hook import (
            resolve_storage_layout_compatibility,
        )

        resolve_storage_layout_compatibility(self)

    def _resolve_hf_gguf_model_path(self):
        from sglang.srt.arg_groups.model_path_hook import resolve_hf_gguf_model_path

        resolve_hf_gguf_model_path(self)

    def _handle_expert_pack(self):
        from sglang.srt.arg_groups.expert_pack_hook import handle_expert_pack

        handle_expert_pack(self)

    def _handle_load_format(self):
        # The quantization side of the gguf coupling moved to the pipeline
        # (arg_groups/overrides.py: _gguf_quantization); load_format itself is
        # genuine config (runtime user updates write it) and stays imperative.
        from sglang.srt.arg_groups.model_path_hook import handle_load_format

        handle_load_format(self)

    def _is_mistral_native_format(self) -> bool:
        """True iff the checkpoint requires load_format=mistral.

        Looks for consolidated*.safetensors with no competing
        model*.safetensors; when both weight formats ship in the
        same checkpoint (e.g. Mistral-7B-Instruct-v0.3) the HF path is
        preferred to avoid loading Mistral-named weights into an
        HF-named architecture.

        Name override: mistral-large-3 / mistral-small-4 /
        leanstral always treat as Mistral-native when params.json
        is present -- those families need Mistral weight loading
        regardless of which weight files happen to be present.
        """
        cfg = resolving_view(self)
        _MISTRAL_NATIVE_PATTERNS = (
            "mistral-large-3",
            "mistral-small-4",
            "leanstral",
        )
        name_matches = any(
            p in str(cfg.model_path).lower() for p in _MISTRAL_NATIVE_PATTERNS
        )

        def _check_format(has_params, has_consolidated, has_hf_weights) -> bool:
            if has_params and name_matches:
                return True
            return has_consolidated and not has_hf_weights

        if os.path.isdir(cfg.model_path):
            return _check_format(
                has_params=os.path.exists(os.path.join(cfg.model_path, "params.json")),
                has_consolidated=bool(
                    glob.glob(os.path.join(cfg.model_path, "consolidated*.safetensors"))
                ),
                has_hf_weights=bool(
                    glob.glob(os.path.join(cfg.model_path, "model*.safetensors"))
                ),
            )

        try:
            from huggingface_hub import HfApi

            files = {s.rfilename for s in HfApi().model_info(cfg.model_path).siblings}
            return _check_format(
                has_params="params.json" in files,
                has_consolidated=any(
                    f.startswith("consolidated") and f.endswith(".safetensors")
                    for f in files
                ),
                has_hf_weights=any(
                    f.startswith("model")
                    and f.endswith(".safetensors")
                    and "/" not in f
                    for f in files
                ),
            )
        except Exception:
            return False

    LANGUAGE_MODEL_ONLY_ARCHITECTURES = ("MuseGlimmerForConditionalGeneration",)

    def _handle_language_model_only(self):
        from sglang.srt.arg_groups.model_hook import handle_language_model_only

        handle_language_model_only(self)

    def _handle_encoder_disaggregation(self):
        from sglang.srt.arg_groups.pd_disaggregation_hook import (
            handle_encoder_disaggregation,
        )

        handle_encoder_disaggregation(self)

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
        from sglang.srt.arg_groups.serving_hook import handle_tokenizer_batching

        handle_tokenizer_batching(self)

    def _handle_multimodal_feature_transport(self):
        from sglang.srt.arg_groups.serving_hook import (
            handle_multimodal_feature_transport,
        )

        handle_multimodal_feature_transport(self)

    def _handle_environment_variables(self):
        from sglang.srt.arg_groups.serving_hook import handle_environment_variables

        handle_environment_variables(self)

    def _handle_cache_compatibility(self):
        from sglang.srt.arg_groups.kv_cache_hook import handle_cache_compatibility

        handle_cache_compatibility(self)

    def _handle_deterministic_inference(self):
        from sglang.srt.arg_groups.attention_hook import handle_deterministic_inference

        handle_deterministic_inference(self)

    def _handle_unified_memory_pool(self):
        from sglang.srt.arg_groups.kv_cache_hook import handle_unified_memory_pool

        handle_unified_memory_pool(self)
        # The strided-layout Triton requirement is enforced via
        # --enable-page-major-kv-layout (implied by the unified pool in
        # _handle_page_major_kv_layout); the model-family gate is enforced at pool
        # construction in model_runner_kv_cache_mixin._init_pools.

    def _handle_page_major_kv_layout(self):
        from sglang.srt.arg_groups.kv_cache_hook import handle_page_major_kv_layout

        handle_page_major_kv_layout(self)

    def _handle_dllm_inference(self):
        from sglang.srt.arg_groups.dllm_hook import handle_dllm_inference

        handle_dllm_inference(self)

    def _handle_asr_validation(self):
        from sglang.srt.arg_groups.serving_hook import handle_asr_validation

        handle_asr_validation(self)

    def _validate_prefill_decode_interval(self):
        cfg = resolving_view(self)
        if cfg.prefill_decode_interval < 0:
            raise ValueError("--prefill-decode-interval must be non-negative.")

    def _handle_other_validations(self):
        from sglang.srt.arg_groups.serving_hook import handle_other_validations

        handle_other_validations(self)

    def _handle_crash_dump_env(self):
        from sglang.srt.arg_groups.serving_hook import handle_crash_dump_env

        handle_crash_dump_env(self)

    def _handle_debug_utils(self):
        from sglang.srt.arg_groups.serving_hook import handle_debug_utils

        handle_debug_utils(self)

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser):

        # Auto-derived from Annotated[..., Arg(...)] field metadata.
        add_cli_args_from_dataclass(parser, ServerArgs)

        # --- Fields with dynamic choices (computed at add_cli_args time) ---
        sampling_backend_choices = set(SAMPLING_BACKEND_CHOICES)
        if envs.SGLANG_KV_CANARY_ENABLE_TOKEN_ORACLE.get():
            sampling_backend_choices.add("token_oracle")
        parser.add_argument(
            "--sampling-backend",
            type=str,
            choices=sampling_backend_choices,
            default=ServerArgs.sampling_backend,
            help="Choose the kernels for sampling layers.",
        )

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
            "--enable-expert-distribution-metrics",
            action=DeprecatedAction,
            error_message=(
                "--enable-expert-distribution-metrics is no longer supported. Use "
                "--expert-balancedness-report-mode with one of: off, server_log, "
                "prometheus, both."
            ),
            help=(
                "Removed. Use --expert-balancedness-report-mode with one of: "
                "off, server_log, prometheus, both."
            ),
        )
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
            choices=[
                "flashmla_sparse",
                "flashmla_sparse_q8",
                "flashmla_kv",
                "flashmla_auto",
                "flashinfer_sparse_mla",
                "fa3",
                "tilelang",
                "aiter",
                "trtllm",
            ],
            help="[Deprecated] Use --dsa-prefill-backend instead.",
        )
        parser.add_argument(
            "--nsa-decode-backend",
            dest="dsa_decode_backend",
            action=DeprecatedAliasStoreAction,
            new_flag="--dsa-decode-backend",
            default=argparse.SUPPRESS,
            type=str,
            choices=[
                "flashmla_sparse",
                "flashmla_sparse_q8",
                "flashmla_kv",
                "flashmla_auto",
                "flashinfer_sparse_mla",
                "fa3",
                "tilelang",
                "aiter",
                "trtllm",
            ],
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
            "--enable-gdn-replayssm-spec",
            dest="enable_linear_replayssm_spec",
            action=DeprecatedStoreTrueAction,
            new_flag="--enable-linear-replayssm-spec",
            help="[Deprecated] Use --enable-linear-replayssm-spec instead.",
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
            choices=["in-seq-split", "round-robin-split"],
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
            choices=["in-seq-split", "round-robin-split"],
            help="[Deprecated] Use --cp-strategy instead.",
        )
        parser.add_argument(
            "--prefill-cp-mode",
            dest="prefill_cp_mode",
            action=DeprecatedAliasStoreAction,
            new_flag="--cp-strategy",
            type=str,
            default=ServerArgs.prefill_cp_mode,
            choices=["in-seq-split"],
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

    def get_tokenizer_worker_class(self):
        from sglang.srt.managers.multi_tokenizer_mixin import TokenizerWorker

        return TokenizerWorker

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

    @property
    def is_ep_joiner(self) -> bool:
        """True for processes launched as elastic-EP joiners."""
        cfg = resolving_view(self)

        return cfg.ep_join_mode in ("scale", "recover")

    @property
    def is_ep_scale_joiner(self) -> bool:
        cfg = resolving_view(self)

        return cfg.ep_join_mode == "scale"

    @property
    def is_startup_weight_load_overlap(self) -> bool:
        cfg = resolving_view(self)

        return cfg.startup_weight_load_mode == "overlap"

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
        cfg = resolving_view(self)
        from sglang.srt.configs.model_config import ModelConfig

        memo = getattr(self, "_model_config", None)
        if memo is not None:
            # The key is the path this record carried when the cache was
            # filled. The GGUF and ModelScope handlers declare a different
            # `model_path`, and a configuration built before them describes
            # another checkpoint. `ModelConfig` re-points its own `model_path`
            # at the local pull directory when the weights sit behind an
            # object-store URI, so its field is not the key. A configuration a
            # fixture supplied carries no key and is handed back as it is.
            built_from = getattr(self, "_model_config_built_from", None)
            if built_from is None or built_from == cfg.model_path:
                return memo

        model_config = ModelConfig.from_server_args(self)
        self._model_config = model_config
        self._model_config_built_from = cfg.model_path
        if model_config.is_hybrid_swa:
            logger.info(
                "Hybrid SWA model detected. architectures=%s",
                model_config.hf_config.architectures,
            )
        return model_config

    def _resolved(self):
        """Read-only view of the resolving configuration: declared fields
        resolve from the declaration stash."""

        return resolved_view(self)

    def _late_resolution(self, source: str, **fields) -> None:
        """Resolve fields at the launcher's validation stage (pre-publish).

        See ``arg_groups.overrides.declare_late_resolution``: the decision goes
        to this instance's declaration stash, so every holder of it carries the
        decision and publishes bags that answer with it. Refused outright once
        the config is published.
        """
        from sglang.srt.arg_groups.overrides import declare_late_resolution

        declare_late_resolution(self, source, **fields)

    def __setattr__(self, name, value):
        # Once resolution has finished the record is the READ-ONLY raw input
        # the config bags were projected from. Resolved config changes go to the bags via
        # get_context().override(source, ...); a value one runner or worker
        # owns travels as a constructor argument to it.
        if (
            getattr(self, "_resolution_finished", False)
            and not getattr(self, "_internal_write", False)
            and (not name.startswith("_") or name in _underscore_field_names())
        ):
            raise AttributeError(
                f"server_args.{name} assigned after resolution; server_args is "
                "read-only -- use get_context().override(source, ...) to change "
                "resolved config; a value one runner owns travels as a "
                "constructor argument."
            )
        object.__setattr__(self, name, value)

    def _resolved_attention_backends(self):
        """Mid-resolution (prefill, decode) backends: reads through the pass
        view so declared fields resolve from the declaration stash."""
        from sglang.srt.arg_groups.overrides import (
            attention_backends_of,
        )

        return attention_backends_of(resolved_view(self))

    def get_attention_backends(self):
        """The (prefill, decode) pair resolution decided.

        Reads through the declaration stash, not the fields: the model-specific
        overrides declare into the stash without writing the fields, so a field
        read answers with what the operator typed.
        """
        return attention_backends_of(resolved_view(self))

    def use_mla_backend(self):
        from sglang.srt.configs.model_config import AttentionArch

        model_config = self.get_model_config()
        return model_config.attention_arch == AttentionArch.MLA

    def is_attention_backend_not_set(self):
        cfg = resolving_view(self)
        return (
            cfg.attention_backend is None
            and cfg.prefill_attention_backend is None
            and cfg.decode_attention_backend is None
        )

    def enable_mamba_extra_buffer(self) -> bool:
        return mamba_extra_buffer_of(resolving_view(self))

    def enable_mamba_extra_buffer_lazy(self) -> bool:
        return mamba_extra_buffer_lazy_of(resolving_view(self))

    @property
    def max_speculative_num_draft_tokens(self) -> Optional[int]:
        """Return the maximum draft-token count speculative decoding may use.

        Memoized only once the record is resolved: an answer computed off a raw
        record describes inputs resolution is about to rewrite (auto speculative
        sizing fills `speculative_num_draft_tokens` in), and a cache filled that
        early would keep answering with it.
        """
        cfg = resolving_view(self)

        memo = self.__dict__.get("_max_speculative_num_draft_tokens")
        if memo is not None:
            return memo
        if cfg.speculative_num_draft_tokens is None:
            result = None
        elif not cfg.speculative_adaptive:
            result = cfg.speculative_num_draft_tokens
        else:
            from sglang.srt.speculative.adaptive_spec_params import (
                resolve_candidate_steps_from_config,
            )

            candidate_steps = resolve_candidate_steps_from_config(
                cfg_path=cfg.speculative_adaptive_config,
            )
            # TODO: adaptive spec currently requires topk=1, so each runtime
            # state needs steps + 1 draft-token slots. Revisit this if topk>1
            # is supported.
            result = max(candidate_steps) + 1
        if getattr(self, "_resolution_finished", False):
            self._max_speculative_num_draft_tokens = result
        return result

    @property
    def mamba_cache_chunk_size(self) -> int:
        # For mamba cache with extra buffer, the chunk size is the max of FLA_CHUNK_SIZE
        # (or mamba_chunk_size if it is defined in the model's config) and page_size.
        # It is used to determine the caching point in a sequence during prefill.
        # A pre-seeded `_mamba_cache_chunk_size` (fixtures supply one so a dummy
        # model never loads an HF config) is honored as-is; otherwise the memo
        # is only kept once the record is resolved, because `page_size` below
        # is resolution-written.
        if not hasattr(self, "_mamba_cache_chunk_size"):

            try:
                from sglang.kernels.ops.attention.fla.chunk_delta_h import (
                    CHUNK_SIZE as FLA_CHUNK_SIZE,
                )
            except ImportError:
                # Must match sglang.kernels.ops.attention.fla.chunk_delta_h.CHUNK_SIZE
                FLA_CHUNK_SIZE = 64

            hf_config = self.get_model_config().hf_config
            chunk_size = getattr(hf_config, "mamba_chunk_size", FLA_CHUNK_SIZE)
            page_size = resolved_view(self).page_size
            assert (
                max(chunk_size, page_size) % min(chunk_size, page_size) == 0
            ), f"For SSM models, either chunk_size or page_size must be divisible by the other, got {chunk_size=}, {page_size=}"
            if not getattr(self, "_resolution_finished", False):
                return max(chunk_size, page_size)
            self._mamba_cache_chunk_size = max(chunk_size, page_size)
        return self._mamba_cache_chunk_size

    def _check_two_batch_overlap(self):
        # With no EP a2a backend, two-batch-overlap is only valid on the non-EP
        # DP TP-MoE path (overlapping the DP all_gatherv / reduce_scatterv with
        # the other ubatch's compute), which requires DP attention. Enabling it
        # there needs no extra opt-in env flag.
        cfg = resolving_view(self)

        cp_tbo = (
            is_hip()
            and cfg.enable_dsa_prefill_context_parallel
            and cfg.dsa_prefill_cp_mode == "round-robin-split"
        )
        if (
            cfg.enable_two_batch_overlap
            and cfg.moe_a2a_backend == "none"
            and not cfg.enable_dp_attention
            and not cp_tbo
        ):
            raise ValueError(
                "When enabling two batch overlap without an EP a2a backend "
                "(moe_a2a_backend='none'), --enable-dp-attention is required "
                "(DeepSeek-V4 non-EP DP TBO path)."
            )

    def check_server_args(self):
        cfg = resolving_view(self)

        # Check parallel size constraints
        if cfg.ep_join_mode != "scale":
            assert (
                cfg.tp_size * cfg.pp_size
            ) % cfg.nnodes == 0, "tp_size must be divisible by number of nodes"

        assert (
            cfg.pp_max_micro_batch_size is None or cfg.pp_max_micro_batch_size >= 1
        ), (
            "pp_max_micro_batch_size must be a positive integer or None (for auto-compute). "
            f"Got: {cfg.pp_max_micro_batch_size}"
        )

        assert not (cfg.disable_cuda_graph_padding and cfg.enable_torch_compile), (
            "--disable-cuda-graph-padding is incompatible with --enable-torch-compile. "
            "With padding disabled, every distinct batch size gets its own torch.compile + "
            "Triton autotune cycle (O(max_batch_size) compilations) instead of the small fixed "
            "set of padded bucket sizes, causing engine initialisation to stall for many minutes. "
            "Remove --disable-cuda-graph-padding or --enable-torch-compile."
        )

        if cfg.pp_size > 1:
            assert (
                cfg.disable_overlap_schedule and cfg.speculative_algorithm is None
            ), "Pipeline parallelism is not compatible with overlap schedule, speculative decoding"
            assert cfg.min_free_slots_delay is None, (
                "--min-free-slots-delay is not supported with pipeline "
                "parallelism: allocatable slots per microbatch are bounded by "
                "pp-max-micro-batch-size, so the threshold may never be reached"
            )

        assert not (
            cfg.dp_size > 1 and cfg.nnodes != 1 and not cfg.enable_dp_attention
        ), "multi-node data parallel is not supported unless dp attention!"

        assert cfg.base_gpu_id >= 0, "base_gpu_id must be non-negative"
        assert cfg.gpu_id_step >= 1, "gpu_id_step must be positive"

        assert cfg.moe_dense_tp_size in (
            None,
            1,
            cfg.tp_size,
        ), "moe_dense_tp_size only supports None, 1, or tp_size currently"

        # Check served model name to not have colon as it is reserved for LoRA adapter syntax
        if not is_runai_obj_uri(cfg.served_model_name):
            assert ":" not in cfg.served_model_name, (
                "served_model_name cannot contain a colon (':') character. "
                "The colon is reserved for the 'model:adapter' syntax used in LoRA adapter specification. "
                f"Invalid value: '{cfg.served_model_name}'"
            )

        # Check LoRA
        self.check_lora_server_args()

        # Check speculative decoding
        if cfg.speculative_algorithm is not None:
            assert (
                not cfg.enable_mixed_chunk
            ), "enable_mixed_chunk is required for speculative decoding"

        # Check chunked prefill
        # Skip validation if chunked prefill is disabled (i.e., size <= 0).
        # Skip validation if disaggregation mode is decode.
        if cfg.chunked_prefill_size > 0 and cfg.disaggregation_mode != "decode":
            assert (
                cfg.chunked_prefill_size % cfg.page_size == 0
            ), "chunked_prefill_size must be divisible by page_size"

        # Check pdmux
        if cfg.enable_pdmux:
            assert (
                cfg.pp_size == 1
            ), "PD-Multiplexing is only supported with pipeline parallelism disabled (pp_size=1)."
            assert (
                cfg.chunked_prefill_size == -1
            ), "PD-Multiplexing is not compatible with chunked prefill."
            assert (
                cfg.disaggregation_mode == "null"
            ), "PD-Multiplexing is not compatible with disaggregation mode."
            assert (
                cfg.disable_overlap_schedule
            ), "PD-Multiplexing is not compatible with overlap schedule."

            # NOTE: CUDA Green Context may encounter potential issues with CudaGraph on torch 2.7.x – 2.8.x, leading to performance degradation.
            import torch

            if torch_release >= (2, 7):
                logger.warning(
                    "WARNING: PD-Multiplexing may experience performance degradation with torch versions > 2.6.x.\n"
                    f"  Current torch version is {torch.__version__}.\n"
                    "  Please manually install torch 2.6.x."
                )

        assert cfg.tokenizer_worker_num > 0, "Tokenizer worker num must >= 1"
        assert cfg.detokenizer_worker_num > 0, "Detokenizer worker num must >= 1"
        assert (
            cfg.mm_processor_worker_num >= 0
        ), "Multimodal processor worker num must >= 0"
        assert cfg.mm_io_worker_num >= 0, "Multimodal I/O worker num must >= 0"
        self.validate_buckets_rule("--prompt-tokens-buckets", cfg.prompt_tokens_buckets)
        self.validate_buckets_rule(
            "--generation-tokens-buckets", cfg.generation_tokens_buckets
        )

        # Check scheduling policy
        if cfg.enable_priority_scheduling:
            assert cfg.schedule_policy in [
                "fcfs",
                "lof",
            ], f"To use priority scheduling, schedule_policy must be 'fcfs' or 'lof'. '{cfg.schedule_policy}' is not supported."
            if cfg.default_priority_value is None:
                logger.warning(
                    "--default-priority-value is not set while --enable-priority-scheduling is enabled. "
                    "Requests without explicit priority will have priority=None, "
                    "resulting in priority='None' string labels in Prometheus metrics."
                )
        else:
            if cfg.disable_priority_preemption:
                logger.warning(
                    "--disable-priority-preemption has no effect without --enable-priority-scheduling"
                )
            if cfg.default_priority_value is not None:
                logger.warning(
                    "--default-priority-value has no effect without --enable-priority-scheduling"
                )
        if cfg.retraction_policy == "priority" and not cfg.enable_priority_scheduling:
            raise ValueError(
                "--retraction-policy priority requires --enable-priority-scheduling"
            )

        # Check hisparse
        # Moved to the resolution pipeline (arg_groups/overrides.py:
        # _hisparse_validation), invoked here at its legacy slot.
        from sglang.srt.arg_groups.overrides import (
            _hisparse_validation,
            run_post_process_pass,
        )

        run_post_process_pass(self, _hisparse_validation)

        assert (
            cfg.schedule_conservativeness >= 0
        ), "schedule_conservativeness must be non-negative"

        if cfg.model_impl == "mindspore":
            assert is_npu(), "MindSpore model impl is only supported on Ascend npu."

        # Check metrics labels
        if (
            not cfg.tokenizer_metrics_custom_labels_header
            and cfg.tokenizer_metrics_allowed_custom_labels
        ):
            raise ValueError(
                "Please set --tokenizer-metrics-custom-labels-header when setting --tokenizer-metrics-allowed-custom-labels."
            )

        # Check metrics exporters
        if cfg.export_metrics_to_file and cfg.export_metrics_to_file_dir is None:
            raise ValueError(
                "--export-metrics-to-file-dir is required when --export-metrics-to-file is enabled"
            )

        # Check two batch overlap backend requirement.
        self._check_two_batch_overlap()

        # Check communications compression
        if cfg.enable_quant_communications and cfg.tp_size == 1:
            raise ValueError(
                "Communications quantization is only used with tp_size != 1"
            )

        if cfg.enable_quant_communications and cfg.device != "npu":
            raise ValueError(
                "Communications quantization is only supported for NPU device"
            )

        # grpc_port is None for HTTP-only launches, so the == comparison is
        # already False there; no explicit None check needed.
        if not (cfg.smg_grpc_mode or cfg.grpc_mode) and cfg.grpc_port == cfg.port:
            raise ValueError(
                f"--grpc-port ({cfg.grpc_port}) must differ from --port ({cfg.port})"
            )

        # TODO: Also validate grpc_port != metrics_http_port and grpc_port != nccl_port
        # to avoid opaque bind errors at runtime. Deferred because metrics_http_port
        # and nccl_port have dynamic defaults that may not be resolved yet here.

        if cfg.gc_threshold:
            if not (1 <= len(cfg.gc_threshold) <= 3):
                raise ValueError(
                    "When setting gc_threshold, it must contain 1 to 3 integers."
                )

        if cfg.kv_canary_sweep_interval > 0 and cfg.kv_canary == "none":
            raise ValueError(
                "--kv-canary-sweep-interval requires --kv-canary in {log, raise}"
            )

        self.check_load_publish_args()

    def check_load_publish_args(self):
        """Fail fast at the entrypoint on a --load-publish-endpoint the
        scheduler would decline (no active kv-events publisher to advertise
        through, unbindable, overlapping the KV range, u16 overflow) rather
        than only warning — or silently doing nothing — from a scheduler
        subprocess. Routes through the same resolver the scheduler binds and
        /server_info advertises with."""
        mode = (self.load_publish_endpoint or "").strip()
        if not mode or mode.lower() == "off":
            return  # disabled; nothing to validate

        server_cfg = resolving_view(self)

        from sglang.srt.disaggregation.kv_events import (
            KVEventsConfig,
            resolve_load_pub_range,
        )

        if not self.kv_events_config:
            raise ValueError(
                "--load-publish-endpoint requires --kv-events-config: routers"
                " discover the load range through /server_info's kv_events"
                " block, absent without a publisher."
            )
        try:
            cfg = KVEventsConfig.from_cli(self.kv_events_config)
        except Exception as e:
            raise ValueError(f"--kv-events-config is not parseable: {e}")
        if cfg.publisher == "null" or not cfg.endpoint:
            raise ValueError(
                "--load-publish-endpoint needs an active --kv-events-config"
                " publisher; got publisher='null' or an empty endpoint."
            )
        _, reason = resolve_load_pub_range(
            kv_endpoint=cfg.endpoint,
            replay_endpoint=cfg.replay_endpoint,
            dp_size=server_cfg.dp_size,
            load_publish_endpoint=mode,
        )
        if reason:
            raise ValueError(reason)

    def check_lora_server_args(self):
        cfg = resolving_view(self)

        assert cfg.max_loras_per_batch > 0, "max_loras_per_batch must be positive"

        # Enable LoRA if any LoRA paths are provided for backward compatibility.
        if cfg.lora_paths:
            if cfg.enable_lora is None:
                self._late_resolution("check_lora_server_args", enable_lora=True)
                logger.warning(
                    "--enable-lora is set to True because --lora-paths is provided."
                )
            elif cfg.enable_lora is False:
                logger.warning(
                    "--enable-lora is set to False, any provided lora_paths will be ignored."
                )

        if cfg.enable_lora:
            if cfg.enable_lora_overlap_loading is None:
                self._late_resolution(
                    "check_lora_server_args", enable_lora_overlap_loading=False
                )

            if cfg.enable_lora_overlap_loading:
                # TODO (glenliu21): use some sort of buffer with eviction instead of enforcing a limit
                max_loaded_loras_limit = cfg.max_loras_per_batch * 2
                assert (
                    cfg.max_loaded_loras is not None
                    and cfg.max_loaded_loras <= max_loaded_loras_limit
                ), (
                    "Enabling LoRA overlap loading requires pinning LoRA adapter weights in CPU memory, "
                    f"so --max-loaded-loras must be less than or equal to double --max-loras-per-batch: {max_loaded_loras_limit}"
                )

            # Validate compatibility with speculative decoding
            self._check_lora_speculative_compatibility()

            # Parse lora_paths
            if isinstance(cfg.lora_paths, list):
                parsed_lora_paths = []
                for lora_path in cfg.lora_paths:
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
                    parsed_lora_paths.append(lora_ref)
                self._late_resolution(
                    "check_lora_server_args", lora_paths=parsed_lora_paths
                )
            elif isinstance(cfg.lora_paths, dict):
                self._late_resolution(
                    "check_lora_server_args",
                    lora_paths=[
                        LoRARef(
                            lora_id=LoRARef.deterministic_id(k, v),
                            lora_name=k,
                            lora_path=v,
                            pinned=False,
                        )
                        for k, v in cfg.lora_paths.items()
                    ],
                )
            elif cfg.lora_paths is None:
                self._late_resolution("check_lora_server_args", lora_paths=[])
            else:
                raise ValueError(
                    f"Invalid type for --lora-paths: {type(cfg.lora_paths)}. "
                    "Expected a list or a dictionary."
                )

            # Normalize target modules to a set; keep {"all"} as a sentinel
            # that gets resolved model-awarely in lora_manager.init_lora_shapes().
            if cfg.lora_target_modules:
                self._late_resolution(
                    "check_lora_server_args",
                    lora_target_modules=set(cfg.lora_target_modules),
                )
                if "all" in cfg.lora_target_modules:
                    assert (
                        len(cfg.lora_target_modules) == 1
                    ), "If 'all' is specified in --lora-target-modules, it should be the only module specified."

            # Ensure sufficient information is provided for LoRA initialization.
            assert cfg.lora_paths or (
                cfg.max_lora_rank and cfg.lora_target_modules
            ), "When no initial --lora-paths is provided, you need to specify both --max-lora-rank and --lora-target-modules for LoRA initialization."

            # Validate max_loaded_loras
            if cfg.max_loaded_loras is not None:
                assert cfg.max_loaded_loras >= cfg.max_loras_per_batch, (
                    "max_loaded_loras should be greater than or equal to max_loras_per_batch. "
                    f"max_loaded_loras={cfg.max_loaded_loras}, max_loras_per_batch={cfg.max_loras_per_batch}"
                )
                assert len(cfg.lora_paths) <= cfg.max_loaded_loras, (
                    "The number of LoRA paths should not exceed max_loaded_loras. "
                    f"max_loaded_loras={cfg.max_loaded_loras}, lora_paths={len(cfg.lora_paths)}"
                )

            if cfg.max_lora_chunk_size is not None:
                assert (
                    16 <= cfg.max_lora_chunk_size <= 128
                    and (cfg.max_lora_chunk_size & (cfg.max_lora_chunk_size - 1)) == 0
                ), "--max-lora-chunk-size must be a power of 2 between 16 and 128."

            if cfg.lora_use_virtual_experts:
                logger.info("Virtual expert computation enabled.")

            assert (
                cfg.lora_drain_wait_threshold >= 0.0
            ), "--lora-drain-wait-threshold must be non-negative."

    def _check_lora_speculative_compatibility(self):
        """Validate LoRA + speculative decoding combinations.

        Adapters apply to the target only; a shared draft runs unadapted.
        Matches resolved algorithm names (NEXTN has collapsed to EAGLE).
        """
        cfg = resolving_view(self)
        if cfg.speculative_algorithm in ["NGRAM", None]:
            return

        # These algorithms present a uniform per-request token width during
        # verify, which is what the LoRA segment layout assumes.
        lora_spec_algorithms = ("EAGLE", "EAGLE3", "DFLASH", "DSPARK")
        if cfg.speculative_algorithm not in lora_spec_algorithms:
            promoted = (
                " (NEXTN/EAGLE with a Gemma4 assistant draft is automatically "
                "promoted to FROZEN_KV_MTP, which does not support LoRA)"
                if cfg.speculative_algorithm == "FROZEN_KV_MTP"
                else ""
            )
            raise ValueError(
                "LoRA is only compatible with NGRAM, EAGLE, NEXTN, EAGLE3, "
                "DFLASH, or DSPARK speculative decoding, not "
                f"{cfg.speculative_algorithm}{promoted}."
            )

        ragged_mode = envs.SGLANG_RAGGED_VERIFY_MODE.get()

        # Each entry: (is unsupported, why). Reasons are appended to a shared
        # prefix so the message names the combination, not just the flag.
        unsupported = [
            (
                cfg.speculative_algorithm == "DSPARK" and ragged_mode != "static",
                f"does not support SGLANG_RAGGED_VERIFY_MODE={ragged_mode!r}: "
                "the per-request verify lengths it schedules break the "
                "uniform-width LoRA segment layout",
            ),
            (
                cfg.speculative_adaptive,
                "does not support --speculative-adaptive: the draft is built "
                "from a static ServerArgs snapshot, and the runtime-state "
                "swap does not rebuild LoRA cuda-graph metadata",
            ),
            (
                "experimental_sgl_trtllm"
                in (cfg.moe_runner_backend, cfg.speculative_moe_runner_backend),
                "does not support the experimental_sgl_trtllm MoE runner: its "
                "TopK reads the LoRA config per forward, which the draft "
                "resolves against the target's after its own publish ended",
            ),
            (
                envs.SGLANG_ENABLE_OVERLAP_PLAN_STREAM.get(),
                "does not support SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1: LoRA "
                "batch preparation would run on the plan stream, unordered "
                "against in-flight forwards",
            ),
        ]
        for is_unsupported, reason in unsupported:
            if is_unsupported:
                raise ValueError(
                    f"LoRA with EAGLE/NEXTN/EAGLE3 speculative decoding {reason}."
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
        cfg = resolving_view(self)
        vision_config = getattr(model_config.hf_config, "vision_config", None)
        if vision_config is None:
            return

        # roughly reduce the mem_fraction_static base on params of Vit
        original_server_arg_mem_fraction = cfg.mem_fraction_static
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
        self._declare(
            "adjust_mem_fraction_for_vlm",
            mem_fraction_static=original_server_arg_mem_fraction * final_overall_factor,
        )

    def validate_transfer_engine(self):
        cfg = resolving_view(self)
        try:
            mooncake_available = importlib.util.find_spec("mooncake.engine") is not None
        except (ModuleNotFoundError, ValueError):
            mooncake_available = False
        if not mooncake_available:
            logger.warning(
                "Failed to import mooncake.engine. Does not support using TransferEngine as remote instance weight loader backend."
            )
            return False
        elif cfg.enable_memory_saver:
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
        self._mx_config_cache = result
        return result

    @property
    def modelexpress_url(self) -> Optional[str]:
        return self._parsed_modelexpress_config.get("url")

    @property
    def modelexpress_transport(self) -> str:
        """Transport backend for modelexpress."""
        return self._parsed_modelexpress_config.get("transport", "nixl")

    def remote_instance_weight_loader_use_transfer_engine(self, load_format=None):
        """``load_format`` overrides the seed's: a draft runner loading under
        ``--speculative-draft-load-format`` needs its own transfer engine."""
        return remote_instance_transfer_engine_of(resolving_view(self), load_format)

    @property
    def kv_event_block_size(self) -> int:
        """Width KV events are emitted at: under DCP the radix tree pages at
        ``page_size * dcp_size`` (``mem_cache/kv_cache_builder.py``).
        """
        cfg = resolving_view(self)
        return cfg.page_size * self.dcp_size

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
                "block_size": <kv_event_block_size>,  # subscribers MUST
                                                  # hash prompts at this size
                "dp_size": <dp_size>,             # number of SUB sockets to
                                                  # open; not DCP-scaled, as
                                                  # DCP shards within a rank
                                                  # rather than adding
                                                  # publishers
                "load_endpoint_port_base": <resolved>,
                                                  # base TCP port of the load
                                                  # range (load rank r = base
                                                  # + r). Consumers MUST read
                                                  # this key, not re-derive
                                                  # it; present only when
                                                  # --load-publish-endpoint
                                                  # opted in and a range
                                                  # resolved
                "load_topic": "load",             # SUB filter for the load
                                                  # socket; present iff
                                                  # load_endpoint_port_base
                                                  # is present
            }

        Returns None (i.e. "no publisher to describe") when any of:

        * --kv-events-config is unset / empty / malformed JSON,
        * the configured publisher is "null",
        * page_size is missing or non-positive (a placeholder
          block_size would cause silent KV-cache misses by hashing
          prompts at the wrong granularity on the router side),
        * the endpoint is not a routable TCP address (inproc:// /
          ipc://, missing port, non-integer port, port outside
          1..65535, or a bare unbracketed IPv6 host, which is
          ambiguous).

        NOTE for load-socket consumers: pair the load port with the worker's
        own URL host, as with the KV SUB endpoints — endpoint_host is a
        wildcard ("*", "0.0.0.0", "::") whenever the default packing applies,
        so splicing it yields tcp://*:PORT and connects to nothing.

        Reuses parse_advertisable_tcp and resolve_load_pub_range — the same
        helpers the scheduler binds through — so the advertisement cannot
        drift from the sockets.
        """
        # Lazy import so loading server_args doesn't pull in
        # disaggregation / msgspec / zmq at module top level.
        from sglang.srt.disaggregation.kv_events import (
            LOAD_TOPIC,
            KVEventsConfig,
            parse_advertisable_tcp,
            resolve_load_pub_range,
        )

        resolved = resolving_view(self)
        raw = resolved.kv_events_config
        page_size = resolved.page_size
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
        resolved_kv = parse_advertisable_tcp(cfg.endpoint)
        if resolved_kv is None:
            return None
        host, port = resolved_kv

        descriptor = {
            "publisher": cfg.publisher,
            "endpoint_host": host,
            "endpoint_port_base": port,
            "topic": cfg.topic,
            "block_size": resolved.kv_event_block_size,
            "dp_size": resolved.dp_size,
        }
        # Load range, from the same resolver SchedulerLoadPublisher binds
        # with (so the two can't drift). The decline reason is logged once at
        # startup, not here — this runs per /server_info request.
        resolved_range, _reason = resolve_load_pub_range(
            kv_endpoint=cfg.endpoint,
            replay_endpoint=cfg.replay_endpoint,
            dp_size=resolved.dp_size,
            load_publish_endpoint=self.load_publish_endpoint,
        )
        if resolved_range is not None:
            descriptor["load_endpoint_port_base"] = resolved_range[1]
            descriptor["load_topic"] = LOAD_TOPIC
        return descriptor

    def should_report_expert_balancedness(self) -> bool:
        cfg = resolving_view(self)
        return cfg.expert_balancedness_report_mode != "off"

    def should_log_expert_balancedness_to_server_log(self) -> bool:
        cfg = resolving_view(self)

        return cfg.expert_balancedness_report_mode in ("server_log", "both")

    def should_export_expert_balancedness_to_prometheus(self) -> bool:
        cfg = resolving_view(self)

        return cfg.expert_balancedness_report_mode in ("prometheus", "both")


# --------------------------------------------------------------------------
# Module-level ServerArgs helpers and runtime shims.
# --------------------------------------------------------------------------


def resolve_encoder_transfer_backend(
    backend: str, model_arch: str, tp_size: int
) -> str:
    if backend != "auto":
        return backend
    if model_arch == "KimiK3ForConditionalGeneration" and tp_size > 1:
        return "zmq_to_tokenizer"
    return "zmq_to_scheduler"


def compute_world_size(
    *, enable_dp_attention: bool, dp_size: int, tp_size: int, pp_size: int
) -> int:
    """Total GPU count across all data-parallel replicas.

    Takes the values rather than a config object: the two sizes are the widths
    the launch asked for, which the Ray driver needs before any process group
    exists, and passing a context would hand it the live groups instead.
    """
    return (1 if enable_dp_attention else dp_size) * tp_size * pp_size


def m3_fp8_attn_gemm_enabled(args) -> bool:
    """Whether MiniMax-M3 attention GEMMs run in fp8 (no opt-in flag; active
    whenever possible): fp8_e4m3 main + index KV caches, fp8-cast q, fp8
    sparse/MSA kernels, with dense layers on trtllm_mha's fp8-q path. Needs
    kv_cache_dtype fp8_e4m3 (e5m2 would silently mis-dispatch fmha_sm100's
    e4m3 kernel), the trtllm_mha backend (the only dense backend with fp8-q
    GEMMs), and SM100 (MSA fp8 variants and trtllm-gen fp8 dense kernels are
    sm100-only). SGLANG_DISABLE_M3_FP8_ATTN_GEMM=1 is the kill switch:
    it forces the pre-fp8 numerics (bf16 indexer + widening sparse path,
    bf16 q) without having to move off trtllm_mha.
    """
    from sglang.srt.environ import envs
    from sglang.srt.utils.common import is_sm100_supported

    return (
        args.kv_cache_dtype == "fp8_e4m3"
        and args.attention_backend == "trtllm_mha"
        and is_sm100_supported()
        and not envs.SGLANG_DISABLE_M3_FP8_ATTN_GEMM.get()
    )


# NOTE: The process-wide ServerArgs is owned by the runtime context
# (sglang.srt.runtime_context). The two functions below are LEGACY shims kept
# for the existing call-sites; they publish/read the same live object by
# reference. Do not add new call-sites — the counts are ratcheted
# (decrease-only) by test/registered/unit/test_legacy_global_ratchet.py.
# Imports are in-function so the two modules stay cycle-free at import time.
@functools.lru_cache(maxsize=1)
def _underscore_field_names() -> frozenset:
    """Real dataclass fields whose names start with an underscore.

    The read-only guard exempts underscore names because they are the record's
    own bookkeeping (the stash, the flags, the cache keys). A *field* that
    happens to start with an underscore is still resolved configuration --
    `_speculative_draft_quantization_explicitly_set` is one -- and exempting it
    by spelling would leave exactly one leaf writable on a read-only record.
    """
    return frozenset(
        field.name
        for field in dataclasses.fields(ServerArgs)
        if field.name.startswith("_")
    )


def set_global_server_args_for_scheduler(server_args: ServerArgs):
    """Legacy publish shim (role=scheduler) — prefer
    ``runtime_context.publish(server_args, role=...)`` in new code."""
    from sglang.srt.runtime_context import publish

    publish(server_args, role="scheduler")


def set_global_server_args_for_tokenizer(server_args: ServerArgs):
    """Legacy publish shim (role=tokenizer). Not aliased to the scheduler shim:
    the process role differs."""
    from sglang.srt.runtime_context import publish

    publish(server_args, role="tokenizer")


def get_global_server_args() -> ServerArgs:
    """Legacy accessor shim — prefer ``get_server_args()`` from
    ``sglang.srt.runtime_context`` in new code."""
    from sglang.srt.runtime_context import get_context

    return get_context().server_args


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
        from sglang.srt.utils.server_args_config_parser import ConfigArgumentMerger

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


# --------------------------------------------------------------------------
# Networking constants and PortArgs.
# --------------------------------------------------------------------------


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
        cfg = resolving_view(server_args)
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

        if not cfg.enable_dp_attention:
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

            # Reserve port_base+0..NUM_DERIVED_PORTS-1 (6 fixed ports + dp_size
            # rust-path slots); derive from server_args only (never dp_rank) so
            # every init_new call agrees, decrementing below dist_init_port on
            # overflow.
            is_rust_server = envs.SGLANG_RUST_SERVER.get()
            NUM_DERIVED_PORTS = 6 if not is_rust_server else 6 + cfg.dp_size
            if server_args.is_ep_scale_joiner:
                port_base = server_args.port + ZMQ_TCP_PORT_DELTA
                if port_base + NUM_DERIVED_PORTS > 65535:
                    port_base = server_args.port - ZMQ_TCP_PORT_DELTA
            elif dist_init_port + NUM_DERIVED_PORTS > 65535:
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
            elif is_rust_server:
                # Rust server path (SGLANG_RUST_SERVER + dp attention): there is no
                # DataParallelController allocating worker ports.
                scheduler_input_port = port_base + 6 + dp_rank
            else:
                assert worker_ports is not None
                scheduler_input_port = worker_ports[dp_rank]

            is_joiner = server_args.is_ep_joiner
            # Under SGLANG_DISTRIBUTED_INIT_METHOD_OVERRIDE, SGLang never binds
            # dist_init_port / nccl_port (rendezvous uses the externally-managed
            # store; see distributed/bootstrap.py:_resolve_dist_init_method), so
            # their prechecks could only false-positive and are skipped.
            dist_init_overridden = bool(
                envs.SGLANG_DISTRIBUTED_INIT_METHOD_OVERRIDE.get()
            )
            try:
                if dp_rank is None:
                    if not (is_joiner or dist_init_overridden):
                        wait_port_available(dist_init_port, "dist_init_port")
                    wait_port_available(port_base, "port_base")
                    wait_port_available(detokenizer_port, "detokenizer_port")
                    if not dist_init_overridden:
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
