"""Enumerated choices shared by the config field declarations.

These lived in ``server_args.py`` beside the fields that name them. The fields
moved to ``arg_groups/fields/``, and ``server_args`` imports the field modules,
so the lists cannot stay there without a cycle. ``server_args`` re-exports them
for the handful of modules that import them from their old home.
"""

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

# TODO: this list should likely contain only methods that support online quantization, or that support using custom quantization classes compatible with a given `quant_method` in config.json.
# Some of the choices here do NOT support online quantization.
QUANTIZATION_CHOICES = [
    "awq",
    "fp8",  # MOE + linear online quantization.
    "mxfp8",  # MOE + linear online quantization.
    "gptq",
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

DETERMINISTIC_ATTENTION_BACKEND_CHOICES = [
    "ascend",
    "fa3",
    "fa4",
    "flashinfer",
    "intel_xpu",
    "triton",
]

DISAGG_TRANSFER_BACKEND_CHOICES = [
    "mooncake",
    "nixl",
    "ascend",
    "fake",
    "mori",
    "mooncake_tcp",
]

GRAMMAR_BACKEND_CHOICES = ["xgrammar", "outlines", "llguidance", "none"]

SAMPLING_BACKEND_CHOICES = {"flashinfer", "pytorch", "ascend"}

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

MXFP8_MOE_RUNNER_BACKEND_CHOICES = [
    "cutlass",
    "deep_gemm",
    "flashinfer_trtllm",
    "flashinfer_trtllm_routed",
]

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

FP4_GEMM_RUNNER_BACKEND_CHOICES = [
    "auto",
    "flashinfer_cudnn",
    "flashinfer_cutedsl",
    "flashinfer_cutlass",
    "flashinfer_trtllm",
    "marlin",
]

RADIX_EVICTION_POLICY_CHOICES = ["lru", "lfu", "slru", "priority"]

RL_ON_POLICY_TARGET_CHOICES = ["fsdp"]

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


# --------------------------------------------------------------------------
# Extension points: out-of-tree platforms and plugins extend these lists
# before ServerArgs is constructed. Each list owns its adder on the line
# below it. A list with no adder is not an extension point -- inline it into
# the field's Arg(choices=...) instead of hoisting it here.
# --------------------------------------------------------------------------

# --- Model loading and quantization ---

add_load_format_choices = LOAD_FORMAT_CHOICES.extend
# NOTE: LoadFormat.IPC_CACHE intentionally has no public --load-format choice.
# It is an internal dispatch format set automatically by ModelRunner when the
# weight cache is enabled (weight_cache_mode != "off"). Exposing it as a CLI
# choice let users create contradictory combos (see _handle_load_format).

add_quantization_method_choices = QUANTIZATION_CHOICES.extend

# --- Attention backends ---

add_attention_backend_choices = ATTENTION_BACKEND_CHOICES.extend

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

add_deterministic_attention_backend_choices = (
    DETERMINISTIC_ATTENTION_BACKEND_CHOICES.extend
)

RADIX_SUPPORTED_DETERMINISTIC_ATTENTION_BACKEND = ["ascend", "fa3", "fa4", "triton"]
add_radix_supported_deterministic_attention_backend_choices = (
    RADIX_SUPPORTED_DETERMINISTIC_ATTENTION_BACKEND.extend
)

# --- Transport ---

add_disagg_transfer_backend_choices = DISAGG_TRANSFER_BACKEND_CHOICES.extend

# --- Sampling and grammar ---

add_grammar_backend_choices = GRAMMAR_BACKEND_CHOICES.extend


# --- MoE and GEMM runners ---

add_moe_runner_backend_choices = MOE_RUNNER_BACKEND_CHOICES.extend

add_mxfp8_moe_runner_backend_choices = MXFP8_MOE_RUNNER_BACKEND_CHOICES.extend

add_fp8_gemm_runner_backend_choices = FP8_GEMM_RUNNER_BACKEND_CHOICES.extend

add_fp4_gemm_runner_backend_choices = FP4_GEMM_RUNNER_BACKEND_CHOICES.extend

# --- Cache and scheduling policy ---

add_radix_eviction_policy_choices = RADIX_EVICTION_POLICY_CHOICES.extend

# --- Reinforcement learning ---

add_rl_on_policy_target_choices = RL_ON_POLICY_TARGET_CHOICES.extend

# --- Linear attention ---

add_linear_attn_kernel_backend_choices = LINEAR_ATTN_KERNEL_BACKEND_CHOICES.extend

# --------------------------------------------------------------------------
# Add new extension points at the end of the matching group above. A new
# choice list is inlined into its field by default; hoisting one here makes
# it public API for out-of-tree code and is a deliberate decision.
# --------------------------------------------------------------------------
