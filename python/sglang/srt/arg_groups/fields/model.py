"""Config fields of the ``model`` namespace.

One class per namespace. The class *is* the namespace: a field declared here
lands in the ``model`` bag, which is what ``get_model()`` returns, so a reader
spells it exactly as before. ``ServerArgs`` composes these classes, so the
record stays one flat object -- the split moves where declarations live, not
how config is shaped at runtime.
"""

from __future__ import annotations

import dataclasses
from typing import (
    Dict,
    List,
    Literal,
    Optional,
    Union,
)

from sglang.srt.arg_groups.arg_utils import (
    A,
    Arg,
)
from sglang.srt.arg_groups.choices import (
    LOAD_FORMAT_CHOICES,
    QUANTIZATION_CHOICES,
)
from sglang.srt.utils.common import (
    human_readable_int,
    json_list_type,
    nullable_str,
)


@dataclasses.dataclass
class Model:
    """Namespace ``model``."""

    _NS_PATH = "model"

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
    ] = "auto"
    model_loader_extra_config: A[
        str,
        "Extra config for model loader. This will be passed to the model loader "
        "corresponding to the chosen load_format. For load_format=presharded, "
        "JSON may include presharded_path (target cache root), "
        "draft_presharded_path (draft cache root), max_file_bytes, "
        "hash_num_threads, and verify_on_load.",
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
    is_embedding: A[
        bool,
        "Whether to use a CausalLM as an embedding model.",
    ] = False
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
    ] = "auto"
    quantization: A[
        Optional[str],
        Arg(
            help="The quantization method.",
            choices=QUANTIZATION_CHOICES,
            resolvable=True,
        ),
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
    ] = "auto"
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
    ] = "serial"
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
        "Prefetch checkpoint files into OS page cache before loading. Each rank prefetches a fraction of the shards, reducing total network I/O on shared filesystems (NFS/Lustre) from N*checkpoint to 1*checkpoint. Recommended for models on network storage. When enabled, multi-threaded safetensors loading is disabled by default to avoid I/O oversubscription with the prefetch threads; set enable_multithread_load=true in --model-loader-extra-config to keep multi-threaded loading (e.g. on local NVMe where prefetch is a no-op).",
    ] = False
    weight_loader_prefetch_num_threads: A[
        int, "Number of threads per rank for checkpoint prefetching (default: 4)."
    ] = 4
    weight_loader_drop_cache_after_load: A[
        bool, "Call posix_fadvise(DONTNEED) on each safetensors shard after loading it."
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
    download_dir: A[
        Optional[str],
        "Model download directory for huggingface.",
    ] = None
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
    ] = "off"
    weight_cache_socket: A[
        Optional[str],
        Arg(
            help="Unix socket path for weight cache daemon (client mode)."
            "If not set, derives the path from SGLANG_WEIGHT_CACHE_SOCKET_TEMPLATE "
            "using the caller's physical GPU UUID.",
        ),
    ] = None
    weight_cache_timeout: A[
        int,
        Arg(
            help="Timeout in seconds for weight cache daemon readiness (default: 1800).",
        ),
    ] = 1800
