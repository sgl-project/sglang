# Define constants for the server arguments
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
    "mistral",
    "layered",
    "flash_rl",
    "remote",
    "remote_instance",
    "fastsafetensors",
    "private",
    "runai_streamer",
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
    "modelopt_mixed",
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
    "unquant",
]

SPECULATIVE_DRAFT_MODEL_QUANTIZATION_CHOICES = QUANTIZATION_CHOICES

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

DETERMINISTIC_ATTENTION_BACKEND_CHOICES = ["flashinfer", "fa3", "triton"]

RADIX_SUPPORTED_DETERMINISTIC_ATTENTION_BACKEND = ["fa3", "triton"]

DISAGG_TRANSFER_BACKEND_CHOICES = ["mooncake", "nixl", "ascend", "fake", "mori"]

GRAMMAR_BACKEND_CHOICES = ["xgrammar", "outlines", "llguidance", "none"]

MOE_RUNNER_BACKEND_CHOICES = [
    "auto",
    "deep_gemm",
    "triton",
    "triton_kernel",
    "flashinfer_trtllm",
    "flashinfer_trtllm_routed",
    "flashinfer_cutlass",
    "flashinfer_mxfp4",
    "flashinfer_cutedsl",
    "cutlass",
]

MOE_A2A_BACKEND_CHOICES = [
    "none",
    "deepep",
    "mooncake",
    "nixl",
    "mori",
    "ascend_fuseep",
    "flashinfer",
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
    "flashinfer_cutlass",
    "flashinfer_trtllm",
]

RADIX_EVICTION_POLICY_CHOICES = ["lru", "lfu", "slru", "priority"]

RL_ON_POLICY_TARGET_CHOICES = ["fsdp"]

LORA_BACKEND_CHOICES = ["triton", "csgmv", "ascend", "torch_native"]

ENCODER_TRANSFER_BACKEND_CHOICES = ["zmq_to_scheduler", "zmq_to_tokenizer", "mooncake"]

NSA_PREFILL_CP_SPLIT_CHOICES = ["in-seq-split", "round-robin-split"]

PREFILL_CP_SPLIT_CHOICES = ["in-seq-split"]

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

MAMBA_SCHEDULER_STRATEGY_CHOICES = ["auto", "no_buffer", "extra_buffer"]

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
