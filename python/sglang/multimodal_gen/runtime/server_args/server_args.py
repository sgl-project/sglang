# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
# Inspired by SGLang: https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/server_args.py
"""The arguments of sglang-diffusion Inference."""

import argparse
import dataclasses
import json
import math
import os
import random
import sys
import tempfile
from dataclasses import field
from enum import Enum
from typing import Any, List, Literal, Optional

import addict
import yaml

from sglang.multimodal_gen import envs
from sglang.multimodal_gen.configs.pipeline_configs.base import PipelineConfig
from sglang.multimodal_gen.configs.pipeline_configs.ltx_2 import (
    LTX2PipelineConfig,
    is_ltx23_native_variant,
)
from sglang.multimodal_gen.configs.quantization.nunchaku import NunchakuSVDQuantArgs
from sglang.multimodal_gen.configs.quantization.qvg_kv import QVGKVQuantArgs
from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
from sglang.multimodal_gen.runtime.layers.quantization.configs.nunchaku_config import (
    NunchakuConfig,
)
from sglang.multimodal_gen.runtime.loader.utils import BYTES_PER_GB
from sglang.multimodal_gen.runtime.managers.memory_managers.component_residency import (
    COMPONENT_OFFLOAD,
    LAYERWISE_OFFLOAD,
    RESIDENT,
    normalize_component_residency,
    resolve_component_residency_mode,
    resolve_diffusers_pipeline_offload,
)
from sglang.multimodal_gen.runtime.managers.memory_managers.layerwise_offload_components import (
    LAYERWISE_OFFLOAD_ALL_COMPONENTS,
    LAYERWISE_OFFLOAD_DIT_GROUP,
    LAYERWISE_OFFLOAD_IMAGE_ENCODER_GROUP,
    LAYERWISE_OFFLOAD_TEXT_ENCODER_GROUP,
    LAYERWISE_OFFLOAD_VAE_GROUP,
    RESIDENCY_POLICIES,
    RESIDENCY_POLICY_LEADING,
    cpu_offload_component_matches,
    is_dit_component_name,
    is_image_encoder_component_name,
    is_legacy_dit_offload_component_name,
    is_text_encoder_component_name,
    is_vae_component_name,
    layerwise_component_matches_any_selection,
    normalize_cpu_offload_components,
    normalize_layerwise_offload_components,
)
from sglang.multimodal_gen.runtime.platforms import (
    AttentionBackendEnum,
    current_platform,
)
from sglang.multimodal_gen.runtime.server_args.auto_tune import (
    PERFORMANCE_MODES,
    ServerArgsAutoTuner,
)
from sglang.multimodal_gen.runtime.server_args.disagg import DisaggServerArgsMixin
from sglang.multimodal_gen.runtime.utils.common import (
    is_port_available,
    is_valid_ipv6_address,
    normalize_gpu_ids,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import (
    _sanitize_for_logging,
    configure_logger,
    init_logger,
)
from sglang.multimodal_gen.utils import (
    FlexibleArgumentParser,
    StoreBoolean,
    expand_path_fields,
)

logger = init_logger(__name__)

LTX2_TWO_STAGE_DEVICE_MODES = ("original", "resident")
LTX2_TWO_STAGE_DEVICE_MODE_CHOICES = LTX2_TWO_STAGE_DEVICE_MODES
LTX2_TWO_STAGE_PIPELINE_NAMES = ("LTX2TwoStagePipeline", "LTX2TwoStageHQPipeline")
# H200-class GPUs (>=130 GiB total) can usually keep both LTX2 DiTs resident.
LTX2_RESIDENT_AUTO_ENABLE_MEM_GB = 130
LORA_MERGE_MODES = ("auto", "merge", "dynamic")
MAX_SCHEDULER_RPC_TIMEOUT_S = 2_147_483
# Mirrors AttentionBackend.supports_ring_rotation; the name-level check
# runs before backend classes are importable on every platform.
RING_CAPABLE_ATTENTION_BACKENDS = ("fa", "sage_attn")


def _normalize_ltx2_two_stage_device_mode(mode: str | None) -> str | None:
    if mode is None:
        return None
    return mode.lower()


def is_ltx2_two_stage_pipeline_name(pipeline_class_name: str | None) -> bool:
    return pipeline_class_name in LTX2_TWO_STAGE_PIPELINE_NAMES


class Backend(str, Enum):
    """
    Enumeration for different model backends.
    - AUTO: Automatically select backend (prefer sglang native, fallback to diffusers)
    - SGLANG: Use sglang's native optimized implementation
    - DIFFUSERS: Use vanilla diffusers pipeline (supports all diffusers models)
    """

    AUTO = "auto"
    SGLANG = "sglang"
    DIFFUSERS = "diffusers"

    @classmethod
    def from_string(cls, value: str) -> "Backend":
        """Convert string to Backend enum."""
        try:
            return cls(value.lower())
        except ValueError:
            raise ValueError(
                f"Invalid backend: {value}. Must be one of: {', '.join([m.value for m in cls])}"
            ) from None

    @classmethod
    def choices(cls) -> list[str]:
        """Get all available choices as strings for argparse."""
        return [backend.value for backend in cls]


WARMUP_MODES = ("off", "request", "server")

# Default prompt sequence-length buckets for breakable CUDA graph (BCG) padding.
# Prompt-conditioning is padded up to the smallest bucket that fits so prompts
# of different lengths share one captured graph.
DEFAULT_BCG_TEXT_BUCKETS = (64, 128, 256, 512, 1024)

BREAKABLE_CUDA_GRAPH_SUPPORTED_MODEL_IDS = frozenset(
    {
        "comfy-org/ideogram-4",
        "efficient-large-model/sana1.5_1.6b_1024px_diffusers",
        "sana1.5_1.6b_1024px_diffusers",
        "fal/ideogram-v4-fast",
        "fal/ideogram-v4-instant",
        "glm-image",
        "ideogram-4",
        "ideogram-4-fp8",
        "ideogram-4-nf4",
        "ideogram-v4-fast",
        "ideogram-v4-instant",
        "ideogram-ai/ideogram-4-fp8",
        "ideogram-ai/ideogram-4-nf4",
        "lightricks/ltx-2",
        "lightricks/ltx-2.3",
        "ltx-2",
        "ltx-2.3",
        "minimax-h3",
        "minimaxai/minimax-h3",
        "qwen/qwen-image",
        "qwen/qwen-image-2512",
        "qwen-image",
        "qwen-image-2512",
        "tongyi-mai/z-image",
        "tongyi-mai/z-image-turbo",
        "zai-org/glm-image",
        "z-image",
        "z-image-turbo",
    }
)

BREAKABLE_CUDA_GRAPH_SUPPORTED_PIPELINE_CONFIGS = frozenset(
    {
        "GlmImagePipelineConfig",
        "Ideogram4PipelineConfig",
        "LTX2PipelineConfig",
        "LTX23PipelineConfig",
        "MiniMaxH3PipelineConfig",
        "QwenImagePipelineConfig",
        "SanaPipelineConfig",
        "ZImagePipelineConfig",
    }
)


def _normalized_bcg_model_refs(model_ref: str | None) -> set[str]:
    if not model_ref:
        return set()

    normalized = str(model_ref).strip().rstrip("/").lower()
    refs = {normalized, os.path.basename(normalized)}

    if "models--" in normalized:
        hf_cache_name = normalized.split("models--", 1)[1].split("/", 1)[0]
        refs.add(hf_cache_name.replace("--", "/"))

    return refs


@dataclasses.dataclass
class ServerArgs(DisaggServerArgsMixin):
    # Model and path configuration (for convenience)
    model_path: str
    model_subfolder: str | None = None
    model_variant: str | None = None

    # explicit model ID override (e.g. "Qwen-Image")
    model_id: str | None = None

    # served model name exposed via /v1/models and generation responses
    served_model_name: str | None = None

    # Model backend (sglang native or diffusers)
    backend: Backend = Backend.AUTO

    # Attention
    attention_backend: str = None
    attention_backend_config: addict.Dict | None = None
    component_attention_backends: dict[str, str] | str | None = field(
        default_factory=dict
    )
    cache_dit_config: str | dict[str, Any] | None = (
        None  # cache-dit config for diffusers
    )

    # Distributed executor backend
    nccl_port: Optional[int] = None
    enable_nccl_nvls: bool = False

    # HuggingFace specific parameters
    trust_remote_code: bool = False
    revision: str | None = None

    # Parallelism
    num_gpus: int = 1
    performance_mode: str = "auto"
    base_gpu_id: int = 0
    gpu_ids: list[int] | None = None
    # cross-node: num_gpus is the total world size across all nodes; each
    # node runs num_gpus // nnodes local GPU workers (mirrors srt's
    # tp_size_per_node convention)
    nnodes: int = 1
    node_rank: int = 0
    dist_init_addr: str | None = None
    tp_size: Optional[int] = None
    sp_degree: Optional[int] = None
    # sequence parallelism
    ulysses_degree: Optional[int] = None
    ring_degree: Optional[int] = None
    # rows split inside attention, exchanged with one K/V all-gather instead
    # of Ulysses a2a or ring rotation; auto-assigned at sp_degree=2 when no SP
    # degree is set explicitly
    kv_gather_degree: Optional[int] = None
    # whether the SP split was auto-assigned (lets layers fall back per call)
    sp_split_auto: bool = False
    # data parallelism
    # number of data parallelism groups
    dp_size: int = 1
    # number of gpu in a dp group
    # cfg parallel (None = auto-decide based on num_gpus)
    enable_cfg_parallel: Optional[bool] = None
    # number of GPUs in each CFG parallel group (None = auto, 1 = disabled, N > 1 = enabled)
    cfg_parallel_degree: Optional[int] = None

    # encoder layout across a multi-rank replica: auto | fold | dp | replicate
    # (see --encoder-parallel); fold shards the weights at load time, so it is
    # mutually exclusive with dp/replicate for the lifetime of the model
    encoder_parallel: str = "auto"

    hsdp_replicate_dim: int = 1
    hsdp_shard_dim: Optional[int] = None
    dist_timeout: int | None = 3600  # 1 hour
    scheduler_rpc_timeout: int | None = None

    pipeline_config: PipelineConfig = field(default_factory=PipelineConfig, repr=False)

    # Pipeline override
    pipeline_class_name: str | None = (
        None  # Override pipeline class from model_index.json
    )

    # LoRA parameters
    # (Wenxuan) prefer to keep it here instead of in pipeline config to not make it complicated.
    lora_path: str | None = None
    lora_nickname: str = "default"  # for swapping adapters in the pipeline
    lora_scale: float = 1.0  # LoRA scale for merging (e.g., 0.125 for Hyper-SD)
    lora_alpha: int | None = None  # Override training alpha when metadata omits it
    lora_merge_mode: str = "auto"
    lora_weight_name: str | None = None

    # Component path overrides (key = model_index.json component name, value = path)
    component_paths: dict[str, str] = field(default_factory=dict)
    # Optional LTX-2.5 decoder is large enough to load only when requested.
    load_diffusion_decoder: bool = False

    # Pre-quantized transformer weights: safetensors file/directory or GGUF file.
    transformer_weights_path: str | None = None
    # path to precomputed MiniMax H3 AdaLN outputs for inference-only serving.
    minimax_h3_adaln_cache_path: str | None = None
    # Rebuild AdaLN outputs per request from the checkpoint, no sidecar needed.
    minimax_h3_adaln_online: bool = False
    # Widest timestep plan the rebuild slab is sized for; see
    # MINIMAX_H3_ADALN_MAX_PLAN_WIDTH.
    minimax_h3_adaln_plan_width: int = 4
    # Per-component transformer weight overrides (key = model_index.json component name).
    # Pipelines use this when a checkpoint ships separate quantized weights for
    # secondary DiT components; the generic loader consumes it without model-specific
    # filename logic.
    component_transformer_weights_paths: dict[str, str] = field(default_factory=dict)

    # Explicit quantization method override (e.g. "mxfp8", "fp8", "modelslim").
    # When set, the transformer loader uses it instead of auto-detection.
    quantization: str | None = None
    # Layer name patterns to skip during online quantization
    quantization_ignored_layers: list[str] | None = None

    # can restrict layers to adapt, e.g. ["q_proj"]
    # Will adapt only q, k, v, o by default.
    lora_target_modules: list[str] | None = None

    # CPU offload parameters
    # Exact component keys or component groups mapped to a residency mode.
    component_residency: dict[str, str] | list[str] | str | None = None
    # Exact component keys from model_index.json, or a legacy component group.
    cpu_offload_components: list[str] | None = None
    dit_cpu_offload: bool | None = None
    # trade checkpoint-loading peak memory for faster ordinary DiT startup
    direct_gpu_weight_loading: bool = False
    # if true, select the DiT layerwise group
    dit_layerwise_offload: bool | None = None
    layerwise_offload_components: list[str] | None = None
    dit_offload_prefetch_size: float = 0.0
    # If set, keep this many DiT layers resident on GPU
    dit_layerwise_resident_layers: float = 0.0
    # Which layers those are: the leading ones, or spread evenly over the stack.
    dit_layerwise_residency_policy: str = RESIDENCY_POLICY_LEADING
    # Per-component overrides of the three knobs above; an entry wins for that
    # component.
    layerwise_prefetch_size: dict[str, float] | str | None = field(default_factory=dict)
    layerwise_resident_layers: dict[str, float] | str | None = field(
        default_factory=dict
    )
    layerwise_residency_policy: dict[str, str] | str | None = field(
        default_factory=dict
    )
    offload_during_compile: bool = True
    text_encoder_cpu_offload: bool | None = None
    image_encoder_cpu_offload: bool | None = None
    vae_cpu_offload: bool | None = False
    use_fsdp_inference: bool | None = None
    pin_cpu_memory: bool = True
    ltx2_two_stage_device_mode: str | None = None
    _explicit_arg_names: set[str] = field(default_factory=set, repr=False)
    _required_resident_components: set[str] = field(
        default_factory=set, init=False, repr=False
    )
    _fsdp_disabled_components: set[str] = field(
        default_factory=set, init=False, repr=False
    )
    _component_layerwise_capabilities: dict[str, bool] = field(
        default_factory=dict, init=False, repr=False
    )

    # ComfyUI integration
    comfyui_mode: bool = False

    # Compilation
    enable_torch_compile: bool = False
    regional_compile: bool = False

    # Breakable CUDA graph (BCG): capture the DiT forward as CUDA-graph
    # segments split at attention modules (SP all-to-all / dynamic attention
    # stay eager). Mutually exclusive with --enable-torch-compile and
    # Cache-DiT; BCG takes priority when more than one is requested.
    #
    # BCG graphs are resolution-specific, so --warmup-resolutions is required
    # when BCG is enabled: every requested resolution is captured at warmup so
    # serving never triggers a fresh capture.
    enable_breakable_cuda_graph: bool = False
    # Text/prompt sequence-length padding budget for BCG. Prompt-conditioning
    # inputs are padded up to the smallest bucket that fits, so prompts of
    # different lengths reuse one captured graph. Warmup captures one graph per
    # bucket; a prompt longer than the largest bucket falls back to eager.
    # ``None`` resolves to DEFAULT_BCG_TEXT_BUCKETS.
    bcg_text_buckets: list[int] = None

    # NVTX profiling
    enable_layerwise_nvtx_marker: bool = False

    # Warmup is controlled by the canonical `warmup_mode` knob: one of WARMUP_MODES.
    #   - "off":     no warmup.
    #   - "server":  server-based warmup — a synthetic request right after the
    #                server is ready, before real traffic
    #   - "request": request-based warmup — warm on the first real request(s).
    #                This is a BENCHMARK aid.
    # None is resolved by _adjust_warmup from the selected runtime features.
    warmup_mode: str | None = None

    warmup_resolutions: list[str] = None
    warmup_steps: int = 1

    disable_autocast: bool | None = None

    # Quantization / Nunchaku SVDQuant configuration
    nunchaku_config: NunchakuSVDQuantArgs | NunchakuConfig | None = field(
        default_factory=NunchakuSVDQuantArgs, repr=False
    )

    # KV-cache quantization (Quant-VideoGen PRQ). Off by default; mirrors the
    # SRT --kv-cache-dtype pattern (typed config, not a pile of env vars).
    kv_cache_quant_config: QVGKVQuantArgs = field(
        default_factory=QVGKVQuantArgs, repr=False
    )

    # Master port for distributed inference
    master_port: int = 30005

    # http server endpoint config
    host: str | None = "127.0.0.1"
    port: int | None = 30000

    # TODO: webui and their endpoint, check if webui_port is available.
    webui: bool = False
    webui_port: int | None = 12312

    scheduler_port: int = 5555
    # settled ingress ports, one per DP replica; None until ports are settled
    scheduler_ports: list[int] | None = None
    batching_mode: str = "dynamic"
    batching_max_size: int = 1
    batching_delay_ms: float = 0.0
    batching_config: str | None = None
    enable_batching_metrics: bool = False

    # Strict port mode: fail if requested port is unavailable instead of auto-selecting
    strict_ports: bool = False

    output_path: str | None = "outputs/"
    input_save_path: str | None = "inputs/uploads"

    # Prompt text file for batch processing
    prompt_file_path: str | None = None

    # model paths for correct deallocation
    model_paths: dict[str, str] = field(default_factory=dict)
    model_loaded: dict[str, bool] = field(
        default_factory=lambda: {
            "transformer": True,
            "vae": True,
            "video_vae": True,
            "audio_vae": True,
            "video_dit": True,
            "audio_dit": True,
            "dual_tower_bridge": True,
        }
    )

    # MoE parameters used by Wan2.2
    boundary_ratio: float | None = None

    # Disaggregation (pool mode only — launched via launch_pool_disagg_server())
    disagg_role: RoleType = RoleType.MONOLITHIC
    disagg_timeout: int = 3600
    disagg_downstream_wait_timeout: int = 1800
    disagg_dispatch_policy: str = "round_robin"
    disagg_mode: bool = False
    disagg_instance_id: int = 0
    disagg_max_slots_per_instance: int = 8
    disagg_transfer_redundancy: float = 1.25
    disagg_role_device: Literal["auto", "cpu", "cuda"] = "auto"
    disagg_transfer_backend: Literal["auto", "mock", "mooncake"] = "auto"
    disagg_transfer_pool_size: int = 256 * 1024 * 1024
    disagg_transfer_pin_memory: Literal["auto", "off", "required"] = "auto"
    disagg_p2p_hostname: str = "127.0.0.1"
    disagg_ib_device: str | None = None
    disagg_server_addr: str | None = None
    encoder_urls: str | None = None
    denoiser_urls: str | None = None
    decoder_urls: str | None = None
    encoder_tp: int | None = None
    denoiser_tp: int | None = None
    denoiser_sp: int | None = None
    denoiser_ulysses: int | None = None
    denoiser_ring: int | None = None
    decoder_sp: int | None = None
    pool_work_endpoint: str | None = None
    pool_result_endpoint: str | None = None
    pool_control_endpoint: str | None = None
    pool_control_advertised_endpoint: str | None = None

    # Logging
    log_level: str = "info"
    log_requests: bool = False
    log_requests_level: int = 2
    log_requests_format: str = "text"
    log_requests_target: Optional[List[str]] = None
    uvicorn_access_log_exclude_prefixes: list[str] = field(default_factory=list)
    enable_cache_report: bool = False

    # Tracing
    enable_trace: bool = False
    otlp_traces_endpoint: str = "localhost:4317"

    # SGLang backend for encoder stage
    srt_encoder_url: str | None = None
    srt_encoder_connect_timeout: int = 3.05
    srt_encoder_timeout: int = 100

    # SGLang server for PE model inference
    pe_server_url: str | None = None

    @property
    def broker_port(self) -> int:
        return self.port + 1

    @property
    def is_local_mode(self) -> bool:
        """
        If no server is running when a generation task begins, 'local_mode' will be enabled: a dedicated server will be launched
        """
        return self.host is None or self.port is None

    def _adjust_path(self):
        expand_path_fields(self)
        self._adjust_save_paths()

    def _adjust_parameters(self):
        """set defaults and normalize values."""
        self._normalize_component_residency()
        self._adjust_cpu_offload_components()
        auto_tuner = ServerArgsAutoTuner(self)
        auto_tuner.adjust_based_on_performance_mode()
        if auto_tuner.could_override_server_args():
            self._adjust_offload()
            auto_tuner.maybe_adjust_auto_default_layerwise_offload()
        self._adjust_ltx2_two_stage_device_mode()
        if auto_tuner.could_override_server_args():
            auto_tuner.maybe_adjust_auto_fsdp_with_offload_enabled()
            auto_tuner.maybe_adjust_auto_component_residency_after_offload()
            auto_tuner.maybe_replace_cpu_offloaded_components_with_layerwise()
        self._adjust_path()
        if self.served_model_name is None:
            self.served_model_name = self.model_id or self.model_path
        self._adjust_quant_config()
        self._adjust_breakable_cuda_graph_support()
        self._adjust_warmup()
        self._adjust_network_ports()
        # adjust parallelism before attention backend
        self._adjust_parallelism()
        self._adjust_attention_backend()
        self._adjust_platform_specific()
        self._adjust_layerwise_offload_components()
        self._adjust_autocast()
        auto_tuner.finalize_auto_flags()
        self.adjust_pipeline_config()

    def _validate_parameters(self):
        """check consistency and raise errors for invalid configs"""
        self._validate_scheduler_rpc_timeout()
        self._validate_pipeline()
        self._validate_offload()
        self._validate_direct_gpu_weight_loading()
        if self.lora_alpha is not None and self.lora_alpha <= 0:
            raise ValueError("lora_alpha must be a positive integer")
        if not current_platform.is_cpu():
            self._validate_parallelism()
        self._validate_cfg_parallel()
        self._validate_batching()
        self._validate_breakable_cuda_graph()
        self.pipeline_config.validate_server_args(self)

    def _validate_scheduler_rpc_timeout(self) -> None:
        timeout = self.scheduler_rpc_timeout
        if timeout is None:
            return
        if (
            not isinstance(timeout, int)
            or isinstance(timeout, bool)
            or not 0 < timeout <= MAX_SCHEDULER_RPC_TIMEOUT_S
        ):
            raise ValueError(
                "scheduler_rpc_timeout must be None or an integer between "
                f"1 and {MAX_SCHEDULER_RPC_TIMEOUT_S} seconds"
            )

    def resolved_bcg_text_buckets(self) -> tuple[int, ...]:
        """Sorted, de-duplicated, positive BCG text buckets.

        Falls back to :data:`DEFAULT_BCG_TEXT_BUCKETS` when ``--bcg-text-buckets``
        is unset, so both prompt padding and warmup capture share one source of
        truth instead of the legacy ``SGLANG_BCG_TEXT_BUCKETS`` env var.
        """
        raw = self.bcg_text_buckets
        if not raw:
            return DEFAULT_BCG_TEXT_BUCKETS
        buckets = sorted({int(b) for b in raw if int(b) > 0})
        return tuple(buckets) or DEFAULT_BCG_TEXT_BUCKETS

    def _validate_breakable_cuda_graph(self):
        if not self.enable_breakable_cuda_graph:
            return
        # BCG graphs are captured per resolution and only replay for that exact
        # latent shape, so the user must declare the resolutions up front. We
        # capture every one of them at warmup; serving then never re-captures.
        if not self.warmup_resolutions:
            # No explicit resolutions: capture the model's default warmup
            # resolution (derived by build_warmup_reqs) so
            # --enable-breakable-cuda-graph works standalone. BCG graphs are
            # resolution-specific; a request at any other resolution simply
            # falls back to eager (the runner never re-captures at serving
            # time). Pass --warmup-resolutions to capture additional shapes.
            logger.info(
                "[Diffusion BCG] --warmup-resolutions unset; capturing the "
                "model default warmup resolution. Requests at other "
                "resolutions run eager."
            )
        if self.bcg_text_buckets is not None and not any(
            int(b) > 0 for b in self.bcg_text_buckets
        ):
            raise ValueError(
                "--bcg-text-buckets must contain at least one positive integer."
            )

    def _adjust_breakable_cuda_graph_support(self):
        if not self.enable_breakable_cuda_graph:
            return

        pipeline_config = getattr(self, "pipeline_config", None)
        pipeline_config_name = type(pipeline_config).__name__
        if (
            pipeline_config_name in BREAKABLE_CUDA_GRAPH_SUPPORTED_PIPELINE_CONFIGS
            and self._is_breakable_cuda_graph_supported_model()
        ):
            if not self.warmup_resolutions:
                self._default_bcg_warmup_resolution()
            return

        logger.warning(
            "[Diffusion BCG] disabled for %s: only Ideogram-4, Lightricks/LTX-2, MiniMax-H3, "
            "Qwen/Qwen-Image, Qwen/Qwen-Image-2512, SANA1.5, "
            "Tongyi-MAI/Z-Image/Z-Image-Turbo, and zai-org/GLM-Image are "
            "currently supported.",
            pipeline_config_name,
        )
        self.enable_breakable_cuda_graph = False

    def _is_breakable_cuda_graph_supported_model(self) -> bool:
        refs = _normalized_bcg_model_refs(self.model_id)
        refs.update(_normalized_bcg_model_refs(self.model_path))
        return bool(refs & BREAKABLE_CUDA_GRAPH_SUPPORTED_MODEL_IDS)

    def _default_bcg_warmup_resolution(self) -> None:
        """Seed --warmup-resolutions with the model default for BCG.

        BCG graphs are resolution-specific and captured at warmup. When the
        user does not pre-declare resolutions we capture the model's default
        warmup resolution so --enable-breakable-cuda-graph works standalone;
        requests at any other resolution fall back to eager.
        """
        from sglang.multimodal_gen.runtime.warmup_request_builder import (
            _resolve_default_warmup_resolution,
            get_model_sampling_defaults,
        )

        try:
            sampling_defaults = get_model_sampling_defaults(self)
            width, height = _resolve_default_warmup_resolution(
                self, sampling_defaults, server_based_warmup=True
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(
                "[Diffusion BCG] could not derive a default warmup resolution "
                "(%s); no graph will be captured and serving runs eager.",
                exc,
            )
            return
        self.warmup_resolutions = [f"{width}x{height}"]
        logger.info(
            "[Diffusion BCG] --warmup-resolutions unset; capturing the model "
            "default %dx%d. Requests at other resolutions run eager.",
            width,
            height,
        )

    def _adjust_save_paths(self):
        """Normalize empty-string save paths to None (disabled)."""
        if self.output_path is not None and self.output_path.strip() == "":
            self.output_path = None
        if self.input_save_path is not None and self.input_save_path.strip() == "":
            self.input_save_path = None

    def _adjust_quant_config(self):
        """
        resolve, validate and adjust quantization config

        handles only nunchaku for now
        """

        ncfg = self.nunchaku_config
        if ncfg is None or isinstance(ncfg, NunchakuConfig):
            return

        resolution = ncfg.resolve_runtime_config()
        if resolution.transformer_weights_path:
            self.transformer_weights_path = resolution.transformer_weights_path
        self.nunchaku_config = resolution.nunchaku_config

    def adjust_pipeline_config(self):
        tp_size = self.tp_size or 1
        dp_size = self.dp_size or 1
        sp_degree = self.sp_degree or 1
        # one replica = all its GPUs
        replica_size = (self.num_gpus or tp_size) // dp_size

        fold_world = dp_size == 1 and not self.disagg_mode and replica_size > tp_size
        # An explicit fold policy is honored for any multi-rank replica —
        # pure-TP (replica == tp) and dp>1 shapes included. Encoder layout is
        # independent of the DiT's parallelism; the replica group is simply
        # "the ranks that share this batch". `auto` keeps its conservative
        # proposals below and never folds a pure-TP replica.
        fold_replica = (
            self.encoder_parallel == "fold"
            and not self.disagg_mode
            and replica_size > 1
        )

        if fold_replica:
            mode = "replica"
        elif fold_world:
            mode = "world"
        elif tp_size == 1 and sp_degree > 1:
            # Preserve prior behavior for dp>1 / disaggregated SP runs.
            mode = "sp"
        else:
            return

        # propose the fold group from the parallelism alone; the loader keeps it
        # only for encoders worth folding at their real post-load size
        # (finalize_encoder_folding)
        encoder_configs = list(self.pipeline_config.text_encoder_configs) + list(
            getattr(self.pipeline_config, "image_encoder_configs", ()) or ()
        )
        for encoder_config in encoder_configs:
            encoder_config.parallel_folding_mode = mode

        logger.info(
            "Proposed encoder parallel folding (mode=%s) for %s "
            "(tp=%s sp=%s cfg=%s replica=%s); the loader keeps it for encoders "
            "wide enough to benefit.",
            mode,
            self.__class__.__name__,
            tp_size,
            sp_degree,
            self.cfg_parallel_degree or 1,
            replica_size,
        )

    def _adjust_offload(self):
        if current_platform.is_cpu():
            # CPU platform does not need offload
            return

        if self.pipeline_config.task_type.is_action_gen():
            if self.dit_cpu_offload is None:
                self.dit_cpu_offload = False
            if self.text_encoder_cpu_offload is None:
                self.text_encoder_cpu_offload = False
            if self.image_encoder_cpu_offload is None:
                self.image_encoder_cpu_offload = False
            if self.vae_cpu_offload is None:
                self.vae_cpu_offload = False
            return

        # TODO: to be handled by each platform
        if current_platform.get_device_total_memory() / BYTES_PER_GB < 30:
            logger.info(
                "Enabling large component offloading for GPU with low device memory"
            )
            if self.dit_cpu_offload is None:
                self.dit_cpu_offload = True
            if self.text_encoder_cpu_offload is None:
                self.text_encoder_cpu_offload = True
            if self.image_encoder_cpu_offload is None:
                self.image_encoder_cpu_offload = True
        elif self.pipeline_config.task_type.is_image_gen():
            if self.dit_cpu_offload is None:
                self.dit_cpu_offload = True
            if self.text_encoder_cpu_offload is None:
                self.text_encoder_cpu_offload = True
            if self.image_encoder_cpu_offload is None:
                self.image_encoder_cpu_offload = False
        else:
            if self.dit_cpu_offload is None:
                self.dit_cpu_offload = True
            if self.text_encoder_cpu_offload is None:
                self.text_encoder_cpu_offload = True
            if self.image_encoder_cpu_offload is None:
                self.image_encoder_cpu_offload = True

    def _adjust_cpu_offload_components(self) -> None:
        """Normalize the legacy component offload selector, when provided."""
        if self.cpu_offload_components is None:
            return
        normalized = normalize_cpu_offload_components(self.cpu_offload_components)
        self.cpu_offload_components = normalized if normalized is not None else []

    def _normalize_component_residency(self) -> None:
        self.component_residency = normalize_component_residency(
            self.component_residency
        )

    def _adjust_ltx2_two_stage_device_mode(self):
        if not self._is_ltx23_two_stage_pipeline():
            return

        mode = self.ltx2_two_stage_device_mode
        env_mode = None
        if mode is None:
            env_mode = os.getenv("SGLANG_LTX2_TWO_STAGE_DEVICE_MODE")
            mode = (
                _normalize_ltx2_two_stage_device_mode(env_mode)
                if env_mode
                else self._resolve_default_ltx2_two_stage_device_mode()
            )
        else:
            mode = _normalize_ltx2_two_stage_device_mode(mode)

        if mode not in LTX2_TWO_STAGE_DEVICE_MODES:
            raise ValueError(
                f"Invalid ltx2_two_stage_device_mode={mode!r}. "
                f"Expected one of {LTX2_TWO_STAGE_DEVICE_MODE_CHOICES}."
            )

        explicit_nonresident_dits = {
            component_name: residency_mode
            for component_name in ("transformer", "transformer_2")
            if (residency_mode := self.explicit_residency_mode(component_name))
            in (COMPONENT_OFFLOAD, LAYERWISE_OFFLOAD)
        }
        if mode == "resident" and explicit_nonresident_dits:
            configured = ", ".join(
                f"{name}={residency_mode}"
                for name, residency_mode in explicit_nonresident_dits.items()
            )
            if self.is_arg_explicitly_set("ltx2_two_stage_device_mode") or env_mode:
                raise ValueError(
                    "ltx2_two_stage_device_mode=resident conflicts with explicit "
                    f"component residency: {configured}"
                )
            mode = "original"
            logger.info(
                "Using ltx2_two_stage_device_mode=original because DiT offload "
                "was explicitly configured: %s",
                configured,
            )

        self.ltx2_two_stage_device_mode = mode

    def _resolve_default_ltx2_two_stage_device_mode(self) -> str:
        if not current_platform.is_cuda():
            logger.info(
                "Automatically set ltx2_two_stage_device_mode=original on non-CUDA platform"
            )
            return "original"

        device_name = str(current_platform.get_device_name(0)).upper()
        device_total_memory_gb = (
            current_platform.get_device_total_memory() / BYTES_PER_GB
        )
        if (
            "H200" in device_name
            or device_total_memory_gb >= LTX2_RESIDENT_AUTO_ENABLE_MEM_GB
        ):
            logger.info(
                "Automatically set ltx2_two_stage_device_mode=resident for high-memory CUDA GPU (%s, %.2f GiB total)",
                device_name,
                device_total_memory_gb,
            )
            return "resident"

        logger.info(
            "Automatically set ltx2_two_stage_device_mode=original for CUDA GPU (%s, %.2f GiB total)",
            device_name,
            device_total_memory_gb,
        )
        return "original"

    def _is_ltx23_two_stage_pipeline(self) -> bool:
        return is_ltx2_two_stage_pipeline_name(self.pipeline_class_name) and (
            self._is_ltx23_model_path(self.model_path)
            or is_ltx23_native_variant(self.pipeline_config.vae_config.arch_config)
        )

    def _uses_ltx23_high_memory_resident_two_stage_mode(self) -> bool:
        if (
            self.ltx2_two_stage_device_mode != "resident"
            or not self._is_ltx23_two_stage_pipeline()
            or not current_platform.is_cuda()
        ):
            return False
        return (
            current_platform.get_device_total_memory() / BYTES_PER_GB
            >= LTX2_RESIDENT_AUTO_ENABLE_MEM_GB
        )

    def _adjust_attention_backend(self):
        if self.attention_backend in ["fa3", "fa4"]:
            self.attention_backend = "fa"
        self.component_attention_backends = (
            self._normalize_component_attention_backends(
                self.component_attention_backends
            )
        )

        # attention_backend_config
        if self.attention_backend_config is None:
            self.attention_backend_config = addict.Dict()
        elif isinstance(self.attention_backend_config, str):
            self.attention_backend_config = addict.Dict(
                self._parse_attention_backend_config(self.attention_backend_config)
            )

        if self.backend != Backend.DIFFUSERS and isinstance(
            self.pipeline_config, LTX2PipelineConfig
        ):
            text_backend = self.component_attention_backends.get("text_encoder")
            if text_backend != "torch_sdpa":
                if text_backend is None:
                    logger.info(
                        "Automatically set torch_sdpa backend for component text_encoder to preserve LTX2 official attention semantics"
                    )
                else:
                    logger.warning(
                        "Overriding %s backend with torch_sdpa for component text_encoder to preserve LTX2 official attention semantics",
                        text_backend,
                    )
                self.component_attention_backends["text_encoder"] = "torch_sdpa"

        if self.ring_degree > 1:
            if (
                self.attention_backend is not None
                and self.attention_backend not in RING_CAPABLE_ATTENTION_BACKENDS
            ):
                raise ValueError(
                    "Ring Attention requires one of the ring-capable backends "
                    f"({', '.join(RING_CAPABLE_ATTENTION_BACKENDS)}), got "
                    f"{self.attention_backend!r}"
                )
            if self.attention_backend is None:
                self.attention_backend = RING_CAPABLE_ATTENTION_BACKENDS[0]
                logger.info(
                    "Ring Attention requires a ring-capable backend; "
                    "attention_backend has been automatically set to %s",
                    self.attention_backend,
                )

        if self.attention_backend is None and self.backend != Backend.DIFFUSERS:
            if (
                current_platform.is_cuda()
                and self.pipeline_class_name is None
                and self.num_gpus == 1
                and self.tp_size == 1
                and self.sp_degree == 1
                and self.ulysses_degree == 1
                and self.ring_degree == 1
                and self._is_ltx23_model_path(self.model_path)
            ):
                self.attention_backend = "fa"
                logger.info(
                    "Automatically set attention_backend=fa for LTX-2.3 one-stage on 1 GPU to preserve precision"
                )
                return
            self._set_default_attention_backend()

    @staticmethod
    def _normalize_attention_backend_name(backend: str) -> str:
        if not isinstance(backend, str):
            raise ValueError("Attention backend name must be a string")
        normalized = backend.strip().lower()
        if normalized in ("fa3", "fa4"):
            normalized = "fa"
        elif normalized == "cudnn_sdpa":
            normalized = "torch_cudnn_sdpa"
        try:
            return AttentionBackendEnum[normalized.upper()].name.lower()
        except KeyError:
            raise ValueError(
                f"Invalid attention backend '{backend}'. "
                f"Available options are: {[e.name.lower() for e in AttentionBackendEnum]}"
            ) from None

    @staticmethod
    def _parse_component_value_map(
        value: dict[str, Any] | str | None, *, option: str
    ) -> dict[str, str]:
        """Parse a ``component=value`` map, the same shape as component backends."""
        if value is None or value == "":
            return {}
        if isinstance(value, dict):
            return {str(k): str(v) for k, v in value.items()}
        if not isinstance(value, str):
            raise ValueError(
                f"{option} must be a dict or a comma-separated component=value string"
            )
        try:
            parsed = json.loads(value)
            if isinstance(parsed, dict):
                return {str(k): str(v) for k, v in parsed.items()}
        except json.JSONDecodeError:
            pass
        result: dict[str, str] = {}
        for pair in value.split(","):
            pair = pair.strip()
            if not pair:
                continue
            if "=" not in pair:
                raise ValueError(f"{option} must use component=value entries")
            component, entry = pair.split("=", 1)
            result[component.strip()] = entry.strip()
        return result

    def layerwise_tuning_for(
        self, component_name: str | None, *, dit_group: bool
    ) -> tuple[float, float, str]:
        """Prefetch size, resident layers and residency policy for one component."""
        prefetch_map = self._parse_component_value_map(
            self.layerwise_prefetch_size, option="--layerwise-prefetch-size"
        )
        resident_map = self._parse_component_value_map(
            self.layerwise_resident_layers, option="--layerwise-resident-layers"
        )
        policy_map = self._parse_component_value_map(
            self.layerwise_residency_policy, option="--layerwise-residency-policy"
        )

        def _pick(mapping: dict[str, str], group_default, aux_default):
            if component_name is not None and component_name in mapping:
                return mapping[component_name]
            return group_default if dit_group else aux_default

        prefetch = float(_pick(prefetch_map, self.dit_offload_prefetch_size, 0.0))
        resident = float(_pick(resident_map, self.dit_layerwise_resident_layers, 0.0))
        policy = str(
            _pick(
                policy_map,
                self.dit_layerwise_residency_policy,
                RESIDENCY_POLICY_LEADING,
            )
        )
        if policy not in RESIDENCY_POLICIES:
            raise ValueError(
                f"unknown residency policy {policy!r} for component "
                f"{component_name!r}, expected one of {RESIDENCY_POLICIES}"
            )
        return prefetch, resident, policy

    @staticmethod
    def _parse_component_attention_backend_map(
        value: dict[str, str] | str | None,
    ) -> dict[str, str]:
        if value is None or value == "":
            return {}
        if isinstance(value, dict):
            return dict(value)
        if not isinstance(value, str):
            raise ValueError(
                "component_attention_backends must be a dict or a comma-separated component=backend string"
            )

        try:
            parsed = json.loads(value)
            if not isinstance(parsed, dict):
                raise ValueError
            return parsed
        except (json.JSONDecodeError, ValueError):
            pass

        result: dict[str, str] = {}
        for pair in value.split(","):
            pair = pair.strip()
            if not pair:
                continue
            if "=" not in pair:
                raise ValueError(
                    "component_attention_backends must use component=backend entries"
                )
            component, backend = pair.split("=", 1)
            result[component.strip()] = backend.strip()
        return result

    @classmethod
    def _normalize_component_attention_backends(
        cls, value: dict[str, str] | str | None
    ) -> dict[str, str]:
        raw = cls._parse_component_attention_backend_map(value)
        normalized: dict[str, str] = {}
        for component, backend in raw.items():
            if not isinstance(component, str):
                raise ValueError("Component attention backend key must be a string")
            component_name = component.strip().replace("-", "_")
            if not component_name:
                raise ValueError("Component attention backend key must not be empty")
            normalized[component_name] = cls._normalize_attention_backend_name(backend)
        return normalized

    def resolve_component_attention_backend(
        self, *component_names: str | None
    ) -> tuple[AttentionBackendEnum | None, str | None]:
        for component_name in component_names:
            if component_name is None:
                continue
            key = component_name.replace("-", "_")
            fallback_keys = [key]
            if key.endswith("_2"):
                # Secondary two-stage components inherit the base component
                # backend unless explicitly overridden.
                fallback_keys.append(key[:-2])
            for backend_key in fallback_keys:
                backend = self.component_attention_backends.get(backend_key)
                if backend is not None:
                    return AttentionBackendEnum[backend.upper()], backend_key
        return None, None

    def _adjust_warmup(self):
        if self.warmup_mode is not None and self.warmup_mode not in WARMUP_MODES:
            raise ValueError(
                f"Invalid --warmup-mode {self.warmup_mode!r}; "
                f"expected one of {WARMUP_MODES}."
            )

        if self.enable_torch_compile and self.warmup_mode is None:
            self.warmup_mode = "server"
            logger.info(
                "Automatically enabled server warmup for torch.compile so first "
                "real requests do not pay compile latency. Set --warmup-mode off "
                "to disable this behavior."
            )

        # Explicit resolutions need a request path unless an existing server
        # default already supplies the synthetic startup request.
        if self.warmup_resolutions is not None and self.warmup_mode in (None, "off"):
            self.warmup_mode = "request"

        # BCG captures every graph during a synthetic warmup forward at startup
        # so serving never records a fresh graph.
        if self.enable_breakable_cuda_graph and self.disagg_role == RoleType.MONOLITHIC:
            self.warmup_mode = "server"

        # Disaggregated roles do not host the HTTP startup request. Preserve
        # warmup intent, but schedule it on the first request instead.
        if self.disagg_role != RoleType.MONOLITHIC and self.warmup_mode == "server":
            self.warmup_mode = "request"

        if self.warmup_mode is None:
            self.warmup_mode = "off"

    @staticmethod
    def _require_port(port: int, name: str) -> None:
        """Raise if *port* is occupied (used under ``--strict-ports``)."""
        if not is_port_available(port):
            raise RuntimeError(
                f"{name} port {port} is unavailable and --strict-ports is enabled. "
                f"Either use a different port or disable --strict-ports."
            )

    def _adjust_network_ports(self):
        # Disagg role instances (encoder/denoiser/decoder) don't serve HTTP,
        # so skip settling the HTTP port to avoid unnecessary port collisions.
        needs_http = self.disagg_role in (
            RoleType.MONOLITHIC,
            RoleType.SERVER,
        )

        if self.strict_ports:
            requested_ports = []
            if needs_http:
                requested_ports.append((self.port, "HTTP"))
            for replica in range(self.dp_size or 1):
                requested_ports.append(
                    (self.scheduler_port + replica, f"Scheduler[{replica}]")
                )
            if self.master_port is not None:
                requested_ports.append((self.master_port, "Master"))
            seen_ports: dict[int, str] = {}
            for port, name in requested_ports:
                if port in seen_ports:
                    raise RuntimeError(
                        f"{name} port {port} duplicates {seen_ports[port]} port and "
                        "--strict-ports is enabled."
                    )
                seen_ports[port] = name
                self._require_port(port, name)
        else:
            settled_ports: set[int] = set()
            if needs_http:
                self.port = self.settle_port(self.port)
                settled_ports.add(self.port)
            initial_scheduler_port = self.scheduler_port + (
                random.randint(0, 100) if self.scheduler_port == 5555 else 0
            )
            self.scheduler_port = self.settle_port(
                initial_scheduler_port, avoid=settled_ports
            )
            settled_ports.add(self.scheduler_port)
            self.scheduler_ports = [self.scheduler_port]
            for _ in range((self.dp_size or 1) - 1):
                port = self.settle_port(
                    self.scheduler_ports[-1] + 1, avoid=settled_ports
                )
                settled_ports.add(port)
                self.scheduler_ports.append(port)
            if self.master_port is not None:
                self.master_port = self.settle_port(
                    self.master_port, 37, avoid=settled_ports
                )

    def _adjust_parallelism(self):
        sp_unspecified = self.sp_degree is None
        ulysses_unspecified = self.ulysses_degree is None
        ring_unspecified = self.ring_degree is None
        cfg_unspecified = self.enable_cfg_parallel is None

        if self.tp_size is None:
            self.tp_size = 1

        if current_platform.is_cpu() and self.tp_size > 1:
            # CPU platform reuse num_gpus to represent num cpu numa nodes as devices
            self.num_gpus = self.tp_size

        if self.hsdp_shard_dim is None:
            self.hsdp_shard_dim = self.num_gpus

        # --cfg-parallel-size takes precedence over --enable-cfg-parallel bool.
        if self.cfg_parallel_degree is not None:
            if self.cfg_parallel_degree == 1:
                self.enable_cfg_parallel = False
            elif self.cfg_parallel_degree > 1:
                self.enable_cfg_parallel = True
            cfg_unspecified = False

        # Auto-enable CFG parallel when user hasn't set any parallelism flags
        # and there are enough GPUs.  Only auto-enable for models whose default
        # SamplingParams use classifier-free guidance (negative_prompt is not None),
        # because non-CFG models (e.g. FLUX) crash when CFG parallel splits ranks.
        if cfg_unspecified:
            deployment_config = self.pipeline_config.get_model_deployment_config()
            auto_cfg_parallel_degree = deployment_config.get_auto_cfg_parallel_degree(
                self.num_gpus
            )
            if auto_cfg_parallel_degree < 1:
                self.enable_cfg_parallel = False
            else:
                cfg_group_size = self.dp_size * self.tp_size * auto_cfg_parallel_degree
                if (
                    self.performance_mode != "manual"
                    and deployment_config.auto_enable_cfg_parallel
                    and self.num_gpus >= 2
                    and self.num_gpus % cfg_group_size == 0
                    and sp_unspecified
                    and ulysses_unspecified
                    and ring_unspecified
                    and self._model_default_uses_cfg()
                ):
                    self.cfg_parallel_degree = auto_cfg_parallel_degree
                    self.enable_cfg_parallel = auto_cfg_parallel_degree > 1
                    if self.enable_cfg_parallel:
                        logger.info(
                            "Automatically enabled CFG parallel at degree %d for %d GPUs. "
                            "Use --sp-degree / --ulysses-degree to use sequence "
                            "parallelism instead.",
                            self.cfg_parallel_degree,
                            self.num_gpus,
                        )
                    else:
                        logger.info(
                            "Automatically disabled CFG parallel for %d GPUs based on model deployment config.",
                            self.num_gpus,
                        )
                else:
                    self.enable_cfg_parallel = False

        # Resolve cfg_parallel_degree to a concrete int now that enable_cfg_parallel is settled.
        if self.cfg_parallel_degree is None:
            self.cfg_parallel_degree = 2 if self.enable_cfg_parallel else 1

        # adjust sp_degree: allocate all remaining GPUs after TP and DP
        if self.sp_degree is None:
            num_gpus_per_group = self.dp_size * self.tp_size
            if self.enable_cfg_parallel:
                num_gpus_per_group *= self.cfg_parallel_degree
            if self.num_gpus % num_gpus_per_group == 0:
                self.sp_degree = self.num_gpus // num_gpus_per_group
            else:
                # Will be validated later
                self.sp_degree = 1

        if (
            self.ulysses_degree is None
            and self.ring_degree is None
            and self.kv_gather_degree is None
            and self.sp_degree != 1
        ):
            if self.sp_degree == 2:
                # measured-win zone for the K/V-gather exchange; layers whose
                # calls the gather path cannot take fall back to Ulysses
                self.kv_gather_degree = 2
                self.sp_split_auto = True
                logger.info(
                    "Automatically set kv_gather_degree=sp_degree=2; set "
                    "--ulysses-degree explicitly to keep the Ulysses exchange"
                )
            else:
                self.ulysses_degree = self.sp_degree
                logger.info(
                    "Automatically set ulysses_degree=sp_degree=%d for the "
                    "sequence-parallel process-group layout",
                    self.ulysses_degree,
                )

        if self.kv_gather_degree is None:
            self.kv_gather_degree = 1

        if self.kv_gather_degree > 1:
            if (self.ulysses_degree or 1) != 1 or (self.ring_degree or 1) != 1:
                raise ValueError(
                    "kv_gather_degree does not compose with ulysses_degree or "
                    "ring_degree yet; set exactly one of them above 1"
                )

        if self.ulysses_degree is None:
            self.ulysses_degree = 1
            logger.debug(
                f"Ulysses degree not set, using default value {self.ulysses_degree}"
            )

        if self.ring_degree is None:
            self.ring_degree = 1
            logger.debug(f"Ring degree not set, using default value {self.ring_degree}")

        if self.kv_gather_degree > 1:
            # K/V-gather rows occupy the contiguous inner SP dimension; the
            # process groups are built from ulysses_degree, so alias it until
            # gather gets a first-class dimension (needed only once it
            # composes with Ulysses).
            self.ulysses_degree = self.kv_gather_degree

    def _model_default_uses_cfg(self) -> bool:
        """
        Check whether the model uses classifier-free guidance by default.

        CFG is active when *both* ``negative_prompt is not None`` and ``guidance_scale > 1``.
        """
        from sglang.multimodal_gen.registry import get_model_info

        model_info = get_model_info(self.model_path, self.backend, self.model_id)
        if model_info is None:
            return False
        default_params = model_info.sampling_param_cls()

        return (
            getattr(default_params, "negative_prompt", None) is not None
            and getattr(default_params, "guidance_scale", 0) > 1.0
        )

    @staticmethod
    def _is_ltx23_model_path(model_path: str | None) -> bool:
        if not model_path:
            return False
        normalized = model_path.lower()
        return any(
            token in normalized
            for token in (
                "lightricks/ltx-2.3",
                "models--lightricks--ltx-2.3",
                "lightricks__ltx-2.3",
            )
        )

    def _adjust_platform_specific(self):
        if current_platform.is_mps():
            if self.num_gpus != 1:
                raise ValueError("MPS currently supports only --num-gpus 1")
            if self.component_residency is not None and any(
                mode not in (RESIDENT, LAYERWISE_OFFLOAD)
                for mode in self.component_residency.values()
            ):
                raise ValueError(
                    "MPS supports only resident or layerwise-offload component "
                    "residency"
                )
            self.use_fsdp_inference = False

    def is_arg_explicitly_set(self, arg_name: str) -> bool:
        return arg_name in self._explicit_arg_names

    def canonical_residency_mode(self, component_name: str) -> str | None:
        """Resolve the canonical selector for one component, if present."""
        return resolve_component_residency_mode(
            component_name, self.component_residency
        )

    def explicit_residency_mode(self, component_name: str) -> str | None:
        """Resolve explicit controls in canonical-to-compatibility priority."""
        mode = self.canonical_residency_mode(component_name)
        if mode is not None:
            return mode

        if self.is_explicit_layerwise_offload_component(component_name):
            return LAYERWISE_OFFLOAD

        if self.is_arg_explicitly_set("cpu_offload_components"):
            if not self.cpu_offload_components:
                return RESIDENT
            if cpu_offload_component_matches(
                component_name, self.cpu_offload_components
            ):
                return COMPONENT_OFFLOAD

        legacy_flag = self._legacy_component_offload_flag(component_name)
        if legacy_flag is not None and self.is_arg_explicitly_set(legacy_flag):
            legacy_values = {
                "dit_cpu_offload": self.dit_cpu_offload,
                "text_encoder_cpu_offload": self.text_encoder_cpu_offload,
                "image_encoder_cpu_offload": self.image_encoder_cpu_offload,
                "vae_cpu_offload": self.vae_cpu_offload,
            }
            return COMPONENT_OFFLOAD if legacy_values[legacy_flag] else RESIDENT

        # ``--dit-layerwise-offload false`` historically has no matching CPU
        # flag, but explicitly requests that the DiT not be layerwise-offloaded.
        if is_legacy_dit_offload_component_name(component_name) and (
            self.is_arg_explicitly_set("dit_layerwise_offload")
        ):
            return RESIDENT
        return None

    @staticmethod
    def _legacy_component_offload_flag(component_name: str) -> str | None:
        if is_legacy_dit_offload_component_name(component_name):
            return "dit_cpu_offload"
        if is_text_encoder_component_name(component_name):
            return "text_encoder_cpu_offload"
        if is_image_encoder_component_name(component_name):
            return "image_encoder_cpu_offload"
        if is_vae_component_name(component_name):
            return "vae_cpu_offload"
        return None

    def residency_mode(self, component_name: str) -> str:
        """Return the effective residency mode for a loaded component."""
        if current_platform.is_cpu():
            return RESIDENT
        if component_name in self._required_resident_components:
            return RESIDENT

        explicit_mode = self.explicit_residency_mode(component_name)
        if explicit_mode is not None:
            return explicit_mode

        component_names = normalize_layerwise_offload_components(
            self.layerwise_offload_components
        )
        if self._component_layerwise_capabilities.get(component_name, True):
            if component_names and (
                LAYERWISE_OFFLOAD_ALL_COMPONENTS in component_names
                or layerwise_component_matches_any_selection(
                    component_name, component_names
                )
                or (
                    LAYERWISE_OFFLOAD_DIT_GROUP in component_names
                    and is_dit_component_name(component_name)
                )
            ):
                return LAYERWISE_OFFLOAD

        if self.cpu_offload_components is not None:
            if cpu_offload_component_matches(
                component_name, self.cpu_offload_components
            ):
                return COMPONENT_OFFLOAD
        if is_legacy_dit_offload_component_name(component_name):
            return COMPONENT_OFFLOAD if self.dit_cpu_offload else RESIDENT
        if is_text_encoder_component_name(component_name):
            return COMPONENT_OFFLOAD if self.text_encoder_cpu_offload else RESIDENT
        if is_image_encoder_component_name(component_name):
            return COMPONENT_OFFLOAD if self.image_encoder_cpu_offload else RESIDENT
        if is_vae_component_name(component_name):
            return COMPONENT_OFFLOAD if self.vae_cpu_offload else RESIDENT
        return RESIDENT

    def should_cpu_offload_component(self, component_name: str) -> bool:
        return self.residency_mode(component_name) == COMPONENT_OFFLOAD

    def should_start_component_on_cpu(self, component_name: str) -> bool:
        return self.residency_mode(component_name) in (
            COMPONENT_OFFLOAD,
            LAYERWISE_OFFLOAD,
        )

    def require_component_resident(
        self, component_name: str, *, feature_name: str
    ) -> None:
        configured_mode = self.canonical_residency_mode(component_name)
        if configured_mode is not None and configured_mode != RESIDENT:
            raise ValueError(
                f"{feature_name} requires {component_name!r} to be resident; "
                f"got {configured_mode!r} from --component-residency"
            )
        self._required_resident_components.add(component_name)

    def should_use_fsdp_for_component(self, component_name: str) -> bool:
        return bool(
            self.use_fsdp_inference
            and component_name not in self._fsdp_disabled_components
            and self.residency_mode(component_name) == RESIDENT
        )

    def disable_fsdp_for_component(self, component_name: str) -> None:
        self._fsdp_disabled_components.add(component_name)

    def record_component_layerwise_capability(
        self, component_name: str, *, supported: bool
    ) -> None:
        self._component_layerwise_capabilities[component_name] = supported

    def has_layerwise_offload_components(self) -> bool:
        return bool(
            self.dit_layerwise_offload
            or self.layerwise_offload_components
            or (
                self.component_residency
                and LAYERWISE_OFFLOAD in self.component_residency.values()
            )
        )

    def should_configure_layerwise_offload_for_lazy_component(
        self, component_name: str
    ) -> bool:
        """Return whether a lazy-loaded component needs layerwise setup."""
        return self.residency_mode(component_name) == LAYERWISE_OFFLOAD

    def is_explicit_layerwise_offload_component(self, component_name: str) -> bool:
        if self.canonical_residency_mode(component_name) == LAYERWISE_OFFLOAD:
            return True

        if self.is_arg_explicitly_set("layerwise_offload_components"):
            selected_components = normalize_layerwise_offload_components(
                self.layerwise_offload_components
            )
            if selected_components and (
                LAYERWISE_OFFLOAD_ALL_COMPONENTS in selected_components
                or layerwise_component_matches_any_selection(
                    component_name, selected_components
                )
                or (
                    LAYERWISE_OFFLOAD_DIT_GROUP in selected_components
                    and is_dit_component_name(component_name)
                )
            ):
                return True

        return bool(
            self.is_arg_explicitly_set("dit_layerwise_offload")
            and self.dit_layerwise_offload
            and is_dit_component_name(component_name)
        )

    @property
    def is_dit_layerwise_offload_selected(self) -> bool:
        """Return whether the primary DiT resolves to layerwise offload."""
        return self.residency_mode("transformer") == LAYERWISE_OFFLOAD

    def _adjust_layerwise_offload_components(self):
        selected_component_names = normalize_layerwise_offload_components(
            self.layerwise_offload_components
        )
        if self.dit_layerwise_offload:
            if selected_component_names is None:
                selected_component_names = [LAYERWISE_OFFLOAD_DIT_GROUP]
            elif LAYERWISE_OFFLOAD_DIT_GROUP not in selected_component_names:
                selected_component_names = [
                    LAYERWISE_OFFLOAD_DIT_GROUP,
                    *selected_component_names,
                ]

        self.layerwise_offload_components = selected_component_names
        self._clear_non_dit_component_offload_for_layerwise_groups(
            selected_component_names or ()
        )

        has_explicit_dit_offload = bool(
            self.canonical_residency_mode("transformer")
            in (COMPONENT_OFFLOAD, LAYERWISE_OFFLOAD)
            or self.is_explicit_layerwise_offload_component("transformer")
            or (
                self.is_arg_explicitly_set("cpu_offload_components")
                and cpu_offload_component_matches(
                    "transformer", self.cpu_offload_components
                )
            )
            or (self.is_arg_explicitly_set("dit_cpu_offload") and self.dit_cpu_offload)
        )
        if (
            self.is_arg_explicitly_set("dit_layerwise_offload")
            and not self.dit_layerwise_offload
            and not has_explicit_dit_offload
        ):
            self.dit_cpu_offload = False

    def _clear_non_dit_component_offload_for_layerwise_groups(
        self, selected_component_names: tuple[str, ...] | list[str]
    ) -> None:
        selected = set(selected_component_names)
        select_all = LAYERWISE_OFFLOAD_ALL_COMPONENTS in selected
        disabled_explicit_flags: list[str] = []

        if (select_all or LAYERWISE_OFFLOAD_TEXT_ENCODER_GROUP in selected) and (
            self.text_encoder_cpu_offload is not False
        ):
            self.text_encoder_cpu_offload = False
            if self.is_arg_explicitly_set("text_encoder_cpu_offload"):
                disabled_explicit_flags.append("text_encoder_cpu_offload")
        if (select_all or LAYERWISE_OFFLOAD_IMAGE_ENCODER_GROUP in selected) and (
            self.image_encoder_cpu_offload is not False
        ):
            self.image_encoder_cpu_offload = False
            if self.is_arg_explicitly_set("image_encoder_cpu_offload"):
                disabled_explicit_flags.append("image_encoder_cpu_offload")
        if (select_all or LAYERWISE_OFFLOAD_VAE_GROUP in selected) and (
            self.vae_cpu_offload is not False
        ):
            self.vae_cpu_offload = False
            if self.is_arg_explicitly_set("vae_cpu_offload"):
                disabled_explicit_flags.append("vae_cpu_offload")

        if disabled_explicit_flags:
            logger.info(
                "Ignoring component-offload flags because layerwise offload "
                "controls the same component groups: %s",
                ", ".join(disabled_explicit_flags),
            )

    def _adjust_autocast(self):
        if self.disable_autocast is None:
            self.disable_autocast = not self.pipeline_config.enable_autocast

    def _parse_attention_backend_config(self, config_str: str) -> dict[str, Any]:
        """parse attention backend config from string."""
        if not config_str:
            return {}

        # 1. treat as file path
        if os.path.exists(config_str):
            if config_str.endswith((".yaml", ".yml")):
                with open(config_str, "r") as f:
                    return yaml.safe_load(f)
            elif config_str.endswith(".json"):
                with open(config_str, "r") as f:
                    return json.load(f)

        # 2. treat as JSON string
        try:
            return json.loads(config_str)
        except json.JSONDecodeError:
            pass

        # 3. treat as k=v pairs (simple implementation). e.g., "sparsity=0.5,enable_x=true"
        try:
            config = {}
            pairs = config_str.split(",")
            for pair in pairs:
                k, v = pair.split("=", 1)
                k = k.strip()
                v = v.strip()
                if v.lower() == "true":
                    v = True
                elif v.lower() == "false":
                    v = False
                elif v.replace(".", "", 1).isdigit():
                    v = float(v) if "." in v else int(v)
                config[k] = v
            return config
        except Exception:
            raise ValueError(f"Could not parse attention backend config: {config_str}")

    def __post_init__(self):
        # configure logger before use
        configure_logger(server_args=self)

        # Convert string disagg_role to enum (from CLI/config)
        if isinstance(self.disagg_role, str):
            self.disagg_role = RoleType.from_string(self.disagg_role)
        self._validate_disagg_capability()
        self.gpu_ids = normalize_gpu_ids(self.gpu_ids)

        # 1. adjust parameters
        self._adjust_parameters()

        # 2. Validate parameters
        self._validate_parameters()

        # log clean server_args
        try:
            safe_args = _sanitize_for_logging(self, key_hint="server_args")
            logger.info("server_args: %s", json.dumps(safe_args, ensure_ascii=False))
        except Exception:
            # Fallback to default repr if sanitization fails
            logger.info(f"server_args: {self}")

    @staticmethod
    def add_cli_args(parser: FlexibleArgumentParser) -> FlexibleArgumentParser:
        # Model and path configuration
        parser.add_argument(
            "--model-path",
            type=str,
            help="The path of the model weights. This can be a local folder or a Hugging Face repo ID.",
        )
        parser.add_argument(
            "--model-subfolder",
            type=str,
            default=ServerArgs.model_subfolder,
            help=(
                "Advanced override for a Diffusers pipeline subfolder inside the "
                "model repository. Prefer --model-variant when a model exposes "
                "semantic variant-to-weights routing."
            ),
        )
        parser.add_argument(
            "--model-variant",
            type=str,
            default=ServerArgs.model_variant,
            help=(
                "Semantic checkpoint variant to serve. Models with partitioned "
                "checkpoints use this value to select the compatible weights "
                "without exposing repository subfolder layout."
            ),
        )
        parser.add_argument(
            "--minimax-h3-adaln-online",
            action=StoreBoolean,
            default=ServerArgs.minimax_h3_adaln_online,
            help=(
                "Rebuild MiniMax H3 AdaLN outputs from the checkpoint per "
                "request instead of keeping the 24.2 GiB of adaln_proj weights "
                "resident. Works with any step count or schedule and needs no "
                "prebuilt artifact. Requires unquantized weights."
            ),
        )
        parser.add_argument(
            "--minimax-h3-adaln-plan-width",
            type=int,
            default=ServerArgs.minimax_h3_adaln_plan_width,
            help=(
                "Widest timestep plan --minimax-h3-adaln-online sizes its slab "
                "for. The default 4 covers every task; a deployment serving "
                "only t2va (2) or fl2va (3) can shrink the slab proportionally. "
                "A request exceeding it is rejected rather than truncated."
            ),
        )
        parser.add_argument(
            "--minimax-h3-adaln-cache-path",
            type=str,
            default=ServerArgs.minimax_h3_adaln_cache_path,
            help=(
                "Path to a precomputed MiniMax H3 AdaLN cache. This only "
                "supports the matching unquantized H3 checkpoint and rejects "
                "requests whose timestep embeddings are not present in the cache."
            ),
        )
        parser.add_argument(
            "--model-id",
            type=str,
            default=ServerArgs.model_id,
            help=(
                "Override the model ID used for config resolution. "
                "Useful when --model-path is a local directory whose name does not match "
                "any registered HF repo name. Should be the repo name portion of the HF ID "
                "(e.g. 'Qwen-Image' for 'Qwen/Qwen-Image')."
            ),
        )
        parser.add_argument(
            "--served-model-name",
            type=str,
            default=ServerArgs.served_model_name,
            help=(
                "Override the model name exposed by /v1/models and used in generation "
                "responses. Defaults to --model-id if set, otherwise --model-path."
            ),
        )
        parser.add_argument(
            "--pipeline",
            "--pipeline-class-name",
            dest="pipeline_class_name",
            type=str,
            default=ServerArgs.pipeline_class_name,
            help=(
                "Advanced override for pipeline class selection from the model registry "
                "or model_index.json. Must match a registered pipeline_name."
            ),
        )
        parser.add_argument(
            "--load-diffusion-decoder",
            action=StoreBoolean,
            default=ServerArgs.load_diffusion_decoder,
            help=(
                "Load the optional LTX-2.5 diffusion decoder so requests may set "
                "use_diffusion_decoder. Offline generate enables this automatically "
                "when --use-diffusion-decoder is passed."
            ),
        )
        # attention
        parser.add_argument(
            "--attention-backend",
            type=str,
            default=None,
            help=(
                "The attention backend to use. For SGLang-native pipelines, use "
                "values like fa, torch_sdpa, sage_attn, etc. For diffusers pipelines, "
                "use diffusers attention backend names such as flash, _flash_3_hub, "
                "sage, or xformers."
            ),
        )
        parser.add_argument(
            "--attention-backend-config",
            type=str,
            default=None,
            help="Configuration for the attention backend. Can be a JSON string, a path to a JSON/YAML file, or key=value pairs.",
        )
        parser.add_argument(
            "--component-attention-backends",
            type=str,
            default=None,
            help=(
                "Per-component attention backend overrides for native pipelines. "
                "Use component names from model_index.json, e.g. "
                "'text_encoder=torch_sdpa,transformer=fa'."
            ),
        )
        parser.add_argument(
            "--cache-dit-config",
            type=str,
            default=ServerArgs.cache_dit_config,
            help="Path to a Cache-DiT YAML/JSON config. Enables cache-dit for diffusers backend.",
        )

        # HuggingFace specific parameters
        parser.add_argument(
            "--trust-remote-code",
            action=StoreBoolean,
            default=ServerArgs.trust_remote_code,
            help="Trust remote code when loading HuggingFace models",
        )
        parser.add_argument(
            "--revision",
            type=str,
            default=ServerArgs.revision,
            help="The specific model version to use (can be a branch name, tag name, or commit id)",
        )

        parser.add_argument(
            "--performance-mode",
            "--mode",
            type=str,
            choices=PERFORMANCE_MODES,
            default=ServerArgs.performance_mode,
            help=(
                "Preset for performance and memory defaults. "
                "'manual' keeps performance-related server args under explicit user control, no adjustment is made; "
                "'auto' keeps safe defaults and applies high-confidence FSDP/CFG improvements; "
                "'speed' favors GPU-resident execution for lower latency and higher throughput, and may OOM; "
                "'memory' favors lower GPU memory usage; "
                "Explicit offload/FSDP/parallelism flags take precedence."
            ),
        )
        # Parallelism
        parser.add_argument(
            "--enable-nccl-nvls",
            action=StoreBoolean,
            default=ServerArgs.enable_nccl_nvls,
            help="Enable NCCL NVLS when available.",
        )
        parser.add_argument(
            "--num-gpus",
            type=int,
            default=ServerArgs.num_gpus,
            help="The number of GPUs to use.",
        )
        parser.add_argument(
            "--base-gpu-id",
            type=int,
            default=ServerArgs.base_gpu_id,
            help="The starting GPU ID for this instance. Used with --disagg-role "
            "to place role instances on specific GPUs without CUDA_VISIBLE_DEVICES.",
        )
        parser.add_argument(
            "--nnodes",
            type=int,
            default=ServerArgs.nnodes,
            help="The number of nodes for cross-node parallelism. --num-gpus is "
            "the total GPU count across all nodes; each node runs "
            "num_gpus // nnodes local workers.",
        )
        parser.add_argument(
            "--node-rank",
            type=int,
            default=ServerArgs.node_rank,
            help="The rank of this node among --nnodes nodes, in [0, nnodes).",
        )
        parser.add_argument(
            "--dist-init-addr",
            type=str,
            default=ServerArgs.dist_init_addr,
            help="The host:port distributed rendezvous address, reachable from "
            "every node. Required when --nnodes > 1.",
        )
        parser.add_argument(
            "--gpu-ids",
            nargs="+",
            default=None,
            help=(
                "Physical GPU IDs for this instance, e.g. --gpu-ids 0 1 6 7 "
                "or --gpu-ids 0,1,6,7. Overrides --base-gpu-id for standalone "
                "disagg roles."
            ),
        )
        parser.add_argument(
            "--tp-size",
            type=int,
            default=None,
            help="The tensor parallelism size. Defaults to 1 if not specified.",
        )
        parser.add_argument(
            "--sp-degree",
            type=int,
            default=None,
            help="The sequence parallelism size. If not specified, will use all remaining GPUs after accounting for TP and DP.",
        )
        parser.add_argument(
            "--ulysses-degree",
            type=int,
            default=ServerArgs.ulysses_degree,
            help="Ulysses sequence parallel degree. Used in attention layer.",
        )
        parser.add_argument(
            "--ring-degree",
            type=int,
            default=ServerArgs.ring_degree,
            help="Ring sequence parallel degree. Used in attention layer.",
        )
        parser.add_argument(
            "--encoder-parallel",
            type=str,
            choices=["auto", "fold", "dp", "replicate"],
            default=ServerArgs.encoder_parallel,
            help=(
                "Text/image encoder parallelism across a multi-rank replica. "
                "`auto` folds encoders wide enough to benefit (best "
                "single-request latency) and data-parallels eligible native "
                "text encoders at batch>1; `fold` always tensor-parallels the "
                "encoder weights across the replica; `dp` never folds and "
                "splits the batch across encoder copies inside each replica, "
                "composing with encoder TP (best batched throughput); "
                "`replicate` disables both. The default is `auto`."
            ),
        )
        parser.add_argument(
            "--kv-gather-degree",
            type=int,
            default=ServerArgs.kv_gather_degree,
            help=(
                "Sequence-parallel degree that splits rows inside attention "
                "and exchanges with one K/V all-gather (queries stay local) "
                "instead of Ulysses all-to-all. Non-causal attention only; "
                "does not compose with --ulysses-degree/--ring-degree yet. "
                "When no SP degree is set explicitly, sp_degree=2 defaults to "
                "kv_gather_degree=2 (its measured-win zone) and higher "
                "degrees default to Ulysses."
            ),
        )
        parser.add_argument(
            "--enable-cfg-parallel",
            action=StoreBoolean,
            default=None,
            help="Enable cfg parallel at degree 2. Auto-enabled when num_gpus >= 2 and no SP flags are set. Use false to disable it explicitly.",
        )
        parser.add_argument(
            "--cfg-parallel-size",
            dest="cfg_parallel_degree",
            type=int,
            default=None,
            help=(
                "Number of GPUs per CFG parallel group (1 = disabled, N > 1 = enabled at degree N). "
                "Supersedes --enable-cfg-parallel. Allows 4-branch CFG parallel (e.g., --cfg-parallel-size 4) "
                "for models with cond + neg + perturbed + modality branches."
            ),
        )
        parser.add_argument(
            "--data-parallel-size",
            "--dp-size",
            "--dp",
            dest="dp_size",
            type=int,
            default=ServerArgs.dp_size,
            help="The data parallelism size.",
        )

        parser.add_argument(
            "--hsdp-replicate-dim",
            type=int,
            default=ServerArgs.hsdp_replicate_dim,
            help="The data parallelism size.",
        )
        parser.add_argument(
            "--hsdp-shard-dim",
            type=int,
            default=None,
            help="The data parallelism shards. Defaults to num_gpus if not specified.",
        )
        parser.add_argument(
            "--dist-timeout",
            type=int,
            default=ServerArgs.dist_timeout,
            help="Timeout for torch.distributed operations in seconds. "
            "Increase this value if you encounter 'Connection closed by peer' errors after the service is idle. ",
        )
        parser.add_argument(
            "--scheduler-rpc-timeout",
            type=int,
            default=ServerArgs.scheduler_rpc_timeout,
            help=(
                "Optional end-to-end timeout in seconds for a scheduler RPC, including "
                "time spent in the scheduler queue. By default no transport-level "
                "deadline is imposed; callers may still cancel their request."
            ),
        )

        ServerArgs.add_disagg_cli_args(parser)

        # Prompt text file for batch processing
        parser.add_argument(
            "--prompt-file-path",
            type=str,
            default=ServerArgs.prompt_file_path,
            help="Path to a text file containing prompts (one per line) for batch processing",
        )

        parser.add_argument(
            "--mask-strategy-file-path",
            type=str,
            help="Path to mask strategy JSON file for STA",
        )
        parser.add_argument(
            "--enable-torch-compile",
            action=StoreBoolean,
            default=ServerArgs.enable_torch_compile,
            help="Use torch.compile to speed up diffusion hot paths. "
            + "When no warmup mode is configured, this enables server warmup "
            + "so first real requests do not pay compile latency. "
            + "However, will likely cause precision drifts. See (https://github.com/pytorch/pytorch/issues/145213)",
        )
        parser.add_argument(
            "--regional-compile",
            action=StoreBoolean,
            default=ServerArgs.regional_compile,
            help=(
                "Compile repeated DiT submodules selected by the model's "
                "_compile_conditions instead of compiling the whole transformer. "
                "Requires --enable-torch-compile."
            ),
        )
        parser.add_argument(
            "--offload-during-compile",
            action=StoreBoolean,
            default=ServerArgs.offload_during_compile,
            help="Offload components during the torch.compile warmup (the DiT layerwise) so max-autotune fits on tighter-memory GPUs, then restore the configured residency for serving. Skipped when the DiT is already layerwise-offloaded, or under cache-dit / FSDP.",
        )
        parser.add_argument(
            "--enable-breakable-cuda-graph",
            action=StoreBoolean,
            default=ServerArgs.enable_breakable_cuda_graph,
            help="Capture the DiT forward as breakable CUDA graph segments "
            "(split at attention; SP all-to-all / dynamic attention stay "
            "eager) to cut per-kernel launch overhead. Mutually exclusive "
            "with --enable-torch-compile and Cache-DiT (BCG takes priority). "
            "Requires --warmup-resolutions; all of them are captured at warmup.",
        )
        parser.add_argument(
            "--bcg-text-buckets",
            type=int,
            nargs="+",
            default=ServerArgs.bcg_text_buckets,
            help="Prompt sequence-length padding budget for breakable CUDA "
            "graph. Prompt-conditioning is padded up to the smallest bucket "
            "that fits so different prompt lengths reuse one captured graph; "
            "warmup captures one graph per bucket. Defaults to "
            f"{' '.join(map(str, DEFAULT_BCG_TEXT_BUCKETS))}. "
            "Replaces the legacy SGLANG_BCG_TEXT_BUCKETS env var.",
        )

        parser.add_argument(
            "--enable-layerwise-nvtx-marker",
            action=StoreBoolean,
            default=ServerArgs.enable_layerwise_nvtx_marker,
            help="Enable layerwise NVTX markers for profiling with Nsight Systems. "
            "Adds NVTX ranges around each pipeline stage, the denoising loop, "
            "every denoising step, the predict_noise / scheduler_step "
            "sub-operations, and every transformer submodule forward (recursive). "
            "Warmup steps are excluded to keep captured traces clean.",
        )

        # warmup
        parser.add_argument(
            "--warmup-mode",
            type=str,
            choices=list(WARMUP_MODES),
            default=ServerArgs.warmup_mode,
            help=(
                "Warmup mode. One of: `off` (no warmup); `request` "
                "(request-based: warm on real incoming requests); `server` "
                "(server-based: a synthetic warmup request right after the server "
                "is ready, before traffic). `sglang serve` defaults to `server`; "
                "other entrypoints default "
                "to request-based when warmup is enabled. When enabled, look for "
                "the line ending with `(with warmup excluded)` for actual "
                "processing time."
            ),
        )
        parser.add_argument(
            "--warmup-resolutions",
            type=str,
            nargs="+",
            default=ServerArgs.warmup_resolutions,
            help="Specify explicit warmup resolutions. e.g., `--warmup-resolutions 256x256 720x720`",
        )
        parser.add_argument(
            "--warmup-steps",
            type=int,
            default=ServerArgs.warmup_steps,
            help="The number of warmup steps to perform for each resolution.",
        )
        # component residency and legacy offload controls
        parser.add_argument(
            "--component-residency",
            type=str,
            nargs="+",
            default=ServerArgs.component_residency,
            metavar="COMPONENT=MODE",
            help=(
                "Select resident, component-offload, or layerwise-offload for "
                "pipeline components. Exact model_index.json component keys override "
                "the dit, text_encoder, image_encoder, vae, and all groups. "
                "Components without an assignment keep their automatic placement."
            ),
        )
        parser.add_argument(
            "--dit-cpu-offload",
            action=StoreBoolean,
            help="Use CPU offload for DiT inference. Enable if run out of memory with FSDP.",
        )
        parser.add_argument(
            "--direct-gpu-weight-loading",
            action=StoreBoolean,
            default=ServerArgs.direct_gpu_weight_loading,
            help="Load the full unquantized DiT checkpoint state dict directly "
            "onto GPU before assigning model parameters. This may reduce startup "
            "time depending on the model, but temporarily requires checkpoint "
            "weights and model weights to coexist on GPU. Disabled by default.",
        )
        parser.add_argument(
            "--cpu-offload-components",
            type=str,
            nargs="+",
            default=ServerArgs.cpu_offload_components,
            help=(
                "Select component keys from model_index.json for coarse CPU offload. "
                "Use dit, text_encoder, image_encoder, or vae as group aliases; "
                "all selects every loaded module and none disables component offload. "
                "This compatibility option can be combined with per-component CPU "
                "offload flags; selected components take component-offload while "
                "unmatched components retain their existing settings."
            ),
        )
        parser.add_argument(
            "--dit-layerwise-offload",
            action=StoreBoolean,
            default=ServerArgs.dit_layerwise_offload,
            help="Enable layerwise CPU offload with async H2D prefetch overlap for DiTs. "
            "It selects only the DiT layerwise group. Cannot be used together with cache-dit "
            "(SGLANG_CACHE_DIT_ENABLED) or use_fsdp_inference. If legacy DiT offload "
            "flags are also provided, layerwise offload is the effective DiT mode.",
        )
        parser.add_argument(
            "--layerwise-offload-components",
            "--layerwise-offload-modules",
            type=str,
            nargs="+",
            default=ServerArgs.layerwise_offload_components,
            help="Select pipeline components for layerwise offload. "
            "Use dit to select the DiT layerwise group, default for the default group "
            "(currently text_encoder, image_encoder, and vae), "
            "or all to select every layerwise-offloadable component. "
            "This option does not imply --dit-layerwise-offload. Example: "
            "--layerwise-offload-components text_encoder image_encoder vae.",
        )
        parser.add_argument(
            "--dit-offload-prefetch-size",
            type=float,
            default=ServerArgs.dit_offload_prefetch_size,
            help="The size of prefetch for dit-layerwise-offload. If the value is between 0.0 and 1.0, it is treated as a ratio of the total number of layers. If the value is >= 1, it is treated as the absolute number of layers. 0.0 means prefetch 1 layer (lowest memory). Values above 0.5 might have peak memory close to no offload but worse performance.",
        )
        parser.add_argument(
            "--dit-layerwise-resident-layers",
            type=float,
            default=ServerArgs.dit_layerwise_resident_layers,
            help="With --dit-layerwise-offload, keep this many DiT layers "
            "permanently resident on GPU (retained across denoise steps) and stream "
            "the rest with --dit-offload-prefetch-size; which layers stay resident "
            "is --dit-layerwise-residency-policy. 0.0 = off (pure "
            "streaming). Between 0.0 and 1.0 = ratio of layers; >= 1 = absolute "
            "count. Unlike raising the prefetch size, resident layers are transferred "
            "once (not re-streamed every step), so this trades VRAM for lower denoise "
            "latency when memory is available.",
        )
        parser.add_argument(
            "--layerwise-prefetch-size",
            type=str,
            default=None,
            help="Per-component override of --dit-offload-prefetch-size, as "
            "component=value entries, e.g. --layerwise-prefetch-size "
            "text_encoder=2,vae=2. Same units as the DiT flag. Components with "
            "no entry keep their group default. Prefetch overlaps a layer's "
            "transfer with the previous layer's compute, which happens within a "
            "single pass, so it is worth tuning on any streamed component.",
        )
        parser.add_argument(
            "--layerwise-resident-layers",
            type=str,
            default=None,
            help="Per-component override of --dit-layerwise-resident-layers, as "
            "component=value entries, e.g. --layerwise-resident-layers "
            "text_encoder=4. Resident layers are transferred once at startup "
            "rather than streamed, so they cut the transfer of every pass "
            "including the first -- an auxiliary component that runs once per "
            "request still benefits, it just recovers the VRAM once per request "
            "instead of once per denoising step.",
        )
        parser.add_argument(
            "--layerwise-residency-policy",
            type=str,
            default=None,
            help="Per-component override of --dit-layerwise-residency-policy, as "
            "component=value entries, e.g. --layerwise-residency-policy "
            "text_encoder=strided.",
        )
        parser.add_argument(
            "--dit-layerwise-residency-policy",
            type=str,
            choices=RESIDENCY_POLICIES,
            default=ServerArgs.dit_layerwise_residency_policy,
            help="Which layers --dit-layerwise-resident-layers keeps resident. "
            "'leading' (default) keeps the first N, which crams the whole "
            "weight stream into the tail of each step. 'strided' spreads the "
            "resident layers evenly over the stack so the same bytes move over "
            "the whole step instead: same VRAM, same bytes, only a different "
            "schedule. Worth trying when weight streaming overlaps "
            "memory-bound compute -- the transfers stop competing with it for "
            "L2 and DRAM bandwidth, which is where the gain comes from.",
        )

        # offload flags
        parser.add_argument(
            "--text-encoder-cpu-offload",
            action=StoreBoolean,
            help="Use CPU offload for text encoder. Enable if run out of memory.",
        )
        parser.add_argument(
            "--image-encoder-cpu-offload",
            action=StoreBoolean,
            help="Use CPU offload for image encoder. Enable if run out of memory.",
        )
        parser.add_argument(
            "--vae-cpu-offload",
            action=StoreBoolean,
            help="Use CPU offload for VAE. Enable if run out of memory.",
        )

        parser.add_argument(
            "--use-fsdp-inference",
            action=StoreBoolean,
            help="Use FSDP inference to shard DiT weights across GPUs. For single-GPU memory pressure, prefer CPU or layerwise offload.",
        )
        parser.add_argument(
            "--pin-cpu-memory",
            action=StoreBoolean,
            help='Pin memory for CPU offload. Only added as a temp workaround if it throws "CUDA error: invalid argument". '
            "Should be enabled in almost all cases",
        )
        parser.add_argument(
            "--ltx2-two-stage-device-mode",
            type=str,
            choices=LTX2_TWO_STAGE_DEVICE_MODE_CHOICES,
            default=ServerArgs.ltx2_two_stage_device_mode,
            help=(
                "LTX-2.3 two-stage device residency mode: "
                "'original' keeps official two-stage semantics without premerged stage2, "
                "'resident' keeps both transformers resident on GPU. "
                "Default is auto: resident on H200/high-memory CUDA GPUs, otherwise original."
            ),
        )
        parser.add_argument(
            "--disable-autocast",
            action=StoreBoolean,
            help="Disable autocast for denoising loop and vae decoding in pipeline sampling",
        )

        # KV-cache quantization (Quant-VideoGen PRQ)
        parser.add_argument(
            "--kv-cache-quant",
            type=str,
            default=None,
            choices=["off", "int4", "int2"],
            help="Enable Quant-VideoGen PRQ KV-cache quantization (off|int4|int2). "
            "Defaults reproduce the tuned per-chunk config (stages=1, "
            "centroids=128, block=64, symmetric, iters=2, recent=1, "
            "per-chunk sink).",
        )
        parser.add_argument(
            "--kv-cache-quant-centroids",
            type=int,
            default=None,
            help="PRQ k-means centroids per stage (default 128).",
        )
        parser.add_argument(
            "--kv-cache-quant-block-size",
            type=int,
            default=None,
            help="PRQ residual scale block size (default 64).",
        )
        parser.add_argument(
            "--kv-cache-quant-stages",
            type=int,
            default=None,
            help="PRQ k-means stages (default 1).",
        )
        parser.add_argument(
            "--kv-cache-quant-iters",
            type=int,
            default=None,
            help="PRQ k-means iterations (default 2).",
        )
        parser.add_argument(
            "--kv-cache-quant-asymmetric",
            action="store_true",
            default=None,
            help="Use KIVI-style asymmetric residual quantization.",
        )
        parser.add_argument(
            "--kv-cache-quant-keep-recent",
            type=int,
            default=None,
            help="Completed chunks kept bf16 before quantizing (default 1).",
        )
        parser.add_argument(
            "--kv-cache-quant-sink",
            type=int,
            default=None,
            choices=[0, 1],
            help="Quantize the attention sink too (1, default) " "or keep it bf16 (0).",
        )
        parser.add_argument(
            "--kv-cache-quant-sink-keep",
            type=int,
            default=None,
            help="Leading sink chunks kept bf16 forever (default 0).",
        )

        # quantization
        parser.add_argument(
            "--quantization",
            type=str,
            default=ServerArgs.quantization,
            help=(
                "Quantization method for the transformer. If omitted, the method is "
                "auto-detected from the checkpoint config or safetensors metadata when "
                "possible. Use this flag to override auto-detection. "
                "Online (post-load) quantization from a BF16/FP16 checkpoint "
                "is supported for 'fp8' and 'mxfp4'. Other methods "
                "('modelopt', 'modelopt_fp8', 'modelopt_fp4', 'mxfp8', "
                "'mxfp4_npu', 'modelslim') require a pre-quantized checkpoint. "
                "Note: 'mxfp4' targets ROCm + MI350+ (gfx95x); "
                "'mxfp4_npu' / 'mxfp8' target Ascend NPU (A5 series for mxfp4_npu)."
            ),
        )
        parser.add_argument(
            "--quantization-ignored-layers",
            type=str,
            nargs="+",
            default=ServerArgs.quantization_ignored_layers,
            help=(
                "Layer name patterns to keep unquantized during online quantization "
                "(fp8/mxfp4). Each pattern is matched against the layer prefix. "
                "Example: --quantization-ignored-layers img_mod txt_mod to_out"
            ),
        )

        # Nunchaku SVDQuant quantization parameters
        NunchakuSVDQuantArgs.add_cli_args(parser)

        # Master port for distributed inference
        parser.add_argument(
            "--master-port",
            type=int,
            default=ServerArgs.master_port,
            help="Master port for distributed inference. If not set, a random free port will be used.",
        )
        parser.add_argument(
            "--scheduler-port",
            type=int,
            default=ServerArgs.scheduler_port,
            help="Port for the scheduler server.",
        )
        parser.add_argument(
            "--batching-mode",
            type=str,
            default=ServerArgs.batching_mode,
            choices=["dynamic"],
            help="Request batching scheduler mode. Currently only 'dynamic' is implemented.",
        )
        parser.add_argument(
            "--batching-max-size",
            type=int,
            default=ServerArgs.batching_max_size,
            help="Maximum number of compatible generation requests to merge into one batch.",
        )
        parser.add_argument(
            "--batching-delay-ms",
            type=float,
            default=ServerArgs.batching_delay_ms,
            help="Maximum time (in ms) to wait for forming a larger batch before dispatch.",
        )
        parser.add_argument(
            "--batching-config",
            type=str,
            default=ServerArgs.batching_config,
            help=(
                "Optional JSON file with {'schema_version': 1, 'rules': [...]} "
                "batching admission rules that can cap model/resolution shapes "
                "below --batching-max-size."
            ),
        )
        parser.add_argument(
            "--enable-batching-metrics",
            action="store_true",
            default=ServerArgs.enable_batching_metrics,
            help="Log periodic batch efficiency metrics such as realized batch size and queue wait time.",
        )
        parser.add_argument(
            "--host",
            type=str,
            default=ServerArgs.host,
            help="Host for the HTTP API server.",
        )
        parser.add_argument(
            "--port",
            type=int,
            default=ServerArgs.port,
            help="Port for the HTTP API server.",
        )
        parser.add_argument(
            "--strict-ports",
            action=StoreBoolean,
            default=ServerArgs.strict_ports,
            help="If enabled, fail when requested ports are unavailable instead of auto-selecting.",
        )
        parser.add_argument(
            "--webui",
            action=StoreBoolean,
            default=ServerArgs.webui,
            help="Whether to use webui for better display",
        )

        parser.add_argument(
            "--webui-port",
            type=int,
            default=ServerArgs.webui_port,
            help="Whether to use webui for better display",
        )
        parser.add_argument(
            "--output-path",
            type=str,
            default=ServerArgs.output_path,
            help='Directory path to save generated images/videos. Set to "" to disable persistent saving.',
        )
        parser.add_argument(
            "--input-save-path",
            type=str,
            default=ServerArgs.input_save_path,
            help='Directory path to save uploaded input images/videos. Set to "" to disable persistent saving.',
        )

        # LoRA
        parser.add_argument(
            "--lora-path",
            type=str,
            default=ServerArgs.lora_path,
            help="The path to the LoRA adapter weights (can be local file path or HF hub id) to launch with",
        )
        parser.add_argument(
            "--lora-nickname",
            type=str,
            default=ServerArgs.lora_nickname,
            help="The nickname for the LoRA adapter to launch with",
        )
        parser.add_argument(
            "--lora-scale",
            type=float,
            default=ServerArgs.lora_scale,
            help="LoRA scale for merging (e.g., 0.125 for Hyper-SD). Same as lora_scale in Diffusers",
        )
        parser.add_argument(
            "--lora-alpha",
            type=int,
            default=ServerArgs.lora_alpha,
            help=(
                "Override the LoRA training alpha when neither the checkpoint nor "
                "adapter_config.json records it"
            ),
        )
        parser.add_argument(
            "--lora-merge-mode",
            type=str,
            choices=LORA_MERGE_MODES,
            default=ServerArgs.lora_merge_mode,
            help=(
                "How LoRA is applied: auto keeps static merge for regular weights "
                "and uses dynamic LoRA for FSDP-sharded weights to avoid full-gather; "
                "merge always merges into base weights; dynamic always applies LoRA at forward time."
            ),
        )
        parser.add_argument(
            "--lora-weight-name",
            type=str,
            default=ServerArgs.lora_weight_name,
            help="Specific safetensors filename to load from a multi-file LoRA repo",
        )
        # Add pipeline configuration arguments
        PipelineConfig.add_cli_args(parser)

        # Logging
        parser.add_argument(
            "--log-level",
            type=str,
            default=ServerArgs.log_level,
            help="The logging level of all loggers.",
        )

        # Tracing
        parser.add_argument(
            "--enable-trace",
            action="store_true",
            default=False,
            help="Enable OpenTelemetry tracing.",
        )
        parser.add_argument(
            "--otlp-traces-endpoint",
            type=str,
            default=ServerArgs.otlp_traces_endpoint,
            help="OTLP collector endpoint when --enable-trace is set. Format: <host>:<port>",
        )
        parser.add_argument(
            "--log-requests",
            action="store_true",
            help="Log user-facing fields of all requests (default: False). "
            "Verbosity is controlled by --log-requests-level.",
        )
        parser.add_argument(
            "--log-requests-level",
            type=int,
            default=ServerArgs.log_requests_level,
            choices=[0, 1, 2, 3],
            help="Verbosity level for request logging. "
            "0: Log request metadata only (request_id). "
            "1: Log metadata + sampling config (seed, steps, guidance, resolution, frames, fps, ...). "
            "2: Log metadata + sampling config + prompt/negative prompt (truncated to 2 KiB). "
            "3: Log metadata + sampling config + full prompt/negative prompt.",
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
            default=[],
            help="Exclude uvicorn access logs whose request path starts with any of these prefixes. "
            "Defaults to empty (disabled). "
            "Example: --uvicorn-access-log-exclude-prefixes /metrics /health",
        )
        parser.add_argument(
            "--enable-cache-report",
            action="store_true",
            default=ServerArgs.enable_cache_report,
            help="Return number of cached tokens in usage.prompt_tokens_details for each OpenAI-compatible request.",
        )
        parser.add_argument(
            "--backend",
            type=str,
            choices=Backend.choices(),
            default=ServerArgs.backend.value,
            help="The model backend to use. 'auto' prefers sglang native and falls back to diffusers. "
            "'sglang' uses native optimized implementation. 'diffusers' uses vanilla diffusers pipeline.",
        )

        # SGLang backend for encoder stage
        parser.add_argument(
            "--srt-encoder-url",
            type=str,
            default=ServerArgs.srt_encoder_url,
            help="Url of SGLang server for encoder stage",
        )
        parser.add_argument(
            "--srt-encoder-connection-timeout",
            type=int,
            default=ServerArgs.srt_encoder_connect_timeout,
            help="Timeout (in seconds) for establishing the initial TCP connection to the SGLang encoder server. "
            "Default value is 3.05.",
        )
        parser.add_argument(
            "--srt-encoder-timeout",
            type=int,
            default=ServerArgs.srt_encoder_timeout,
            help="Timeout (in seconds) for HTTP requests to the SGLang encoder server. "
            "Increase value if connection between diffusion server and AR model server is slow.",
        )

        # SGLang server for PE model inference
        parser.add_argument(
            "--pe-server-url",
            type=str,
            default=ServerArgs.pe_server_url,
            help="URL of SGLang server for PE model",
        )

        return parser

    def url(self):
        host = self.host
        if not host or host == "0.0.0.0":
            host = "127.0.0.1"
        elif host == "::":
            host = "::1"
        if is_valid_ipv6_address(host):
            return f"http://[{host}]:{self.port}"
        else:
            return f"http://{host}:{self.port}"

    @property
    def scheduler_endpoint(self):
        """
        Internal endpoint for scheduler.
        Prefers the configured host but normalizes localhost -> 127.0.0.1 to avoid ZMQ issues.
        """
        return self.scheduler_endpoint_for(0)

    def scheduler_endpoint_for(self, replica: int) -> str:
        """Ingress endpoint of one DP replica's driver rank."""
        scheduler_host = self.host
        if scheduler_host is None or scheduler_host == "localhost":
            scheduler_host = "127.0.0.1"
        if self.scheduler_ports is not None:
            port = self.scheduler_ports[replica]
        else:
            port = self.scheduler_port + replica
        return f"tcp://{scheduler_host}:{port}"

    @property
    def scheduler_endpoints(self) -> list[str]:
        return [self.scheduler_endpoint_for(r) for r in range(self.dp_size or 1)]

    def settle_port(
        self,
        port: int,
        port_inc: int = 42,
        max_attempts: int = 100,
        avoid: set[int] | None = None,
    ) -> int:
        """
        Find an available port with retry logic.
        """
        attempts = 0
        original_port = port
        avoid = avoid or set()

        while attempts < max_attempts:
            if port not in avoid and is_port_available(port):
                if attempts > 0:
                    logger.info(
                        f"Port {original_port} was unavailable, using port {port} instead"
                    )
                return port

            attempts += 1
            if port < 60000:
                port += port_inc
            else:
                # Wrap around with randomization to avoid collision
                port = 5000 + random.randint(0, 1000)

        raise RuntimeError(
            f"Failed to find available port after {max_attempts} attempts "
            f"(started from port {original_port})"
        )

    @staticmethod
    def _extract_component_paths(
        unknown_args: list[str],
    ) -> tuple[dict[str, str], list[str]]:
        """
        Extract dynamic component path args from unrecognised CLI args.

        Supported forms:
        - ``--<component>-path /path/to/component``
        - ``--component-paths.<component> /path/to/component`` (expanded from config)
        """
        component_paths: dict[str, str] = {}
        remaining: list[str] = []
        i = 0
        while i < len(unknown_args):
            arg = unknown_args[i]
            key_part = arg.split("=", 1)[0] if "=" in arg else arg
            component = None
            if key_part.startswith("--component-paths."):
                component = key_part[len("--component-paths.") :].replace("-", "_")
            elif key_part.startswith("--component_paths."):
                component = key_part[len("--component_paths.") :].replace("-", "_")
            elif key_part.startswith("--") and key_part.endswith("-path"):
                component = key_part[2:-5].replace("-", "_")

            if component is not None:
                if "=" in arg:
                    component_paths[component] = arg.split("=", 1)[1]
                elif i + 1 < len(unknown_args) and not unknown_args[i + 1].startswith(
                    "-"
                ):
                    i += 1
                    component_paths[component] = unknown_args[i]
                else:
                    remaining.append(arg)
                    i += 1
                    continue
            else:
                remaining.append(arg)
            i += 1

        # canonicalize and validate
        for component, path in component_paths.items():
            path = os.path.expanduser(path)
            component_paths[component] = path
        return component_paths, remaining

    @staticmethod
    def _extract_component_attention_backends(
        unknown_args: list[str],
    ) -> tuple[dict[str, str], list[str]]:
        component_attention_backends: dict[str, str] = {}
        remaining: list[str] = []
        i = 0
        while i < len(unknown_args):
            arg = unknown_args[i]
            key_part = arg.split("=", 1)[0] if "=" in arg else arg
            component = None
            if key_part.startswith("--component-attention-backends."):
                component = key_part[len("--component-attention-backends.") :].replace(
                    "-", "_"
                )
            elif key_part.startswith("--component_attention_backends."):
                component = key_part[len("--component_attention_backends.") :].replace(
                    "-", "_"
                )

            if component is not None:
                if "=" in arg:
                    component_attention_backends[component] = arg.split("=", 1)[1]
                elif i + 1 < len(unknown_args) and not unknown_args[i + 1].startswith(
                    "-"
                ):
                    i += 1
                    component_attention_backends[component] = unknown_args[i]
                else:
                    remaining.append(arg)
                    i += 1
                    continue
            else:
                remaining.append(arg)
            i += 1
        return component_attention_backends, remaining

    @classmethod
    def from_cli_args(
        cls,
        args: argparse.Namespace,
        unknown_args: list[str] | None = None,
        default_args: dict[str, Any] | None = None,
    ) -> "ServerArgs":
        if unknown_args is None:
            unknown_args = []

        # extract dynamic --<component>-path from unknown args
        dynamic_paths, remaining = cls._extract_component_paths(unknown_args)
        dynamic_attention_backends, remaining = (
            cls._extract_component_attention_backends(remaining)
        )
        if remaining:
            raise SystemExit(f"error: unrecognized arguments: {' '.join(remaining)}")

        provided_args = cls.get_provided_args(args, unknown_args)
        explicit_arg_names = set(provided_args)

        # Handle config file
        config_file = provided_args.get("config")
        if config_file:
            config_args = cls.load_config_file(config_file)
            explicit_arg_names.update(config_args)
            provided_args = {**config_args, **provided_args}

        if default_args:
            for key, value in default_args.items():
                provided_args.setdefault(key, value)

        if dynamic_paths:
            existing = dict(provided_args.get("component_paths") or {})
            existing.update(dynamic_paths)
            provided_args["component_paths"] = existing
            explicit_arg_names.add("component_paths")
        if dynamic_attention_backends:
            existing = cls._parse_component_attention_backend_map(
                provided_args.get("component_attention_backends")
            )
            existing.update(dynamic_attention_backends)
            provided_args["component_attention_backends"] = existing
            explicit_arg_names.add("component_attention_backends")

        provided_args["_explicit_arg_names"] = explicit_arg_names
        return cls.from_dict(provided_args)

    @classmethod
    def from_dict(cls, kwargs: dict[str, Any]) -> "ServerArgs":
        """Create a ServerArgs object from a dictionary."""
        cls._reject_retired_args(kwargs)
        attrs = [attr.name for attr in dataclasses.fields(cls) if attr.init]
        server_args_kwargs: dict[str, Any] = {}
        explicit_arg_names = kwargs.get("_explicit_arg_names")
        if explicit_arg_names is None:
            explicit_arg_names = set(kwargs)

        component_paths = dict(kwargs.get("component_paths") or {})
        if component_paths:
            server_args_kwargs["component_paths"] = component_paths
        server_args_kwargs["_explicit_arg_names"] = set(explicit_arg_names)

        for attr in attrs:
            if attr == "_explicit_arg_names":
                continue
            elif attr == "pipeline_config":
                pipeline_config = PipelineConfig.from_kwargs(kwargs)
                logger.debug(f"Using PipelineConfig: {type(pipeline_config)}")
                server_args_kwargs["pipeline_config"] = pipeline_config
            elif attr == "nunchaku_config":
                nunchaku_config = NunchakuSVDQuantArgs.from_dict(kwargs)
                server_args_kwargs["nunchaku_config"] = nunchaku_config
            elif attr == "kv_cache_quant_config":
                kv_quant_config = kwargs.get("kv_cache_quant_config")
                if kv_quant_config is None:
                    kv_quant_config = QVGKVQuantArgs.from_dict(kwargs)
                elif isinstance(kv_quant_config, dict):
                    kv_quant_config = QVGKVQuantArgs(**kv_quant_config).validate()
                elif isinstance(kv_quant_config, QVGKVQuantArgs):
                    kv_quant_config.validate()
                else:
                    raise TypeError(
                        "kv_cache_quant_config must be QVGKVQuantArgs or a dict"
                    )
                server_args_kwargs["kv_cache_quant_config"] = kv_quant_config
            elif attr in kwargs:
                server_args_kwargs[attr] = kwargs[attr]

        return cls(**server_args_kwargs)

    @staticmethod
    def _reject_retired_args(kwargs: dict[str, Any]) -> None:
        retired_args = {
            "decoder_tp": "decoder_sp for decoder/VAE parallel decode",
            "warmup": "warmup_mode=request or warmup_mode=off",
            "server_warmup": "warmup_mode=server or warmup_mode=off",
        }
        removed = [name for name in retired_args if name in kwargs]
        if removed:
            replacements = "; ".join(
                f"{name} -> {retired_args[name]}" for name in removed
            )
            raise ValueError(f"Removed server argument(s): {replacements}")

    @staticmethod
    def load_config_file(config_file: str) -> dict[str, Any]:
        """Load a config file."""
        if config_file.endswith(".json"):
            with open(config_file, "r") as f:
                return json.load(f)
        elif config_file.endswith((".yaml", ".yml")):
            try:
                import yaml
            except ImportError:
                raise ImportError(
                    "Please install PyYAML to use YAML config files. "
                    "`pip install pyyaml`"
                )
            with open(config_file, "r") as f:
                return yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported config file format: {config_file}")

    @classmethod
    def from_kwargs(cls, **kwargs: Any) -> "ServerArgs":
        cls._reject_retired_args(kwargs)
        explicit_arg_names = kwargs.get("_explicit_arg_names")
        if explicit_arg_names is None:
            explicit_arg_names = set(kwargs)
        else:
            explicit_arg_names = set(explicit_arg_names)

        # Convert backend string to enum if necessary
        if "backend" in kwargs and isinstance(kwargs["backend"], str):
            kwargs["backend"] = Backend.from_string(kwargs["backend"])

        # Convert disagg_role string to enum if necessary
        if "disagg_role" in kwargs and isinstance(kwargs["disagg_role"], str):
            kwargs["disagg_role"] = RoleType.from_string(kwargs["disagg_role"])

        kwargs["pipeline_config"] = PipelineConfig.from_kwargs(kwargs)
        kwargs["_explicit_arg_names"] = explicit_arg_names
        return cls(**kwargs)

    @staticmethod
    def get_provided_args(
        args: argparse.Namespace, unknown_args: list[str]
    ) -> dict[str, Any]:
        """Get the arguments provided by the user."""
        provided_args = {}
        # We need to check against the raw command-line arguments to see what was
        # explicitly provided by the user, vs. what's a default value from argparse.
        raw_argv = sys.argv + unknown_args

        # Create a set of argument names that were present on the command line.
        # This handles both styles: '--arg=value' and '--arg value'.
        provided_arg_names = set(getattr(args, "_sglang_explicit_arg_names", ()))
        for arg in raw_argv:
            if arg.startswith("--"):
                # For '--arg=value', this gets 'arg'; for '--arg', this also gets 'arg'.
                arg_name = arg.split("=", 1)[0].replace("-", "_").lstrip("_")
                provided_arg_names.add(arg_name)
        cli_aliases = {
            "cfg_parallel_size": "cfg_parallel_degree",
            "data_parallel_size": "dp_size",
            "dp": "dp_size",
            "layerwise_offload_modules": "layerwise_offload_components",
            "mode": "performance_mode",
        }
        for alias_name, dest_name in cli_aliases.items():
            if alias_name in provided_arg_names:
                provided_arg_names.add(dest_name)

        # Populate provided_args if the argument from the namespace was on the command line.
        for k, v in vars(args).items():
            if k.startswith("_sglang_"):
                continue
            if k in provided_arg_names:
                provided_args[k] = v

        return provided_args

    def _validate_pipeline(self):
        if self.pipeline_config is None:
            raise ValueError("pipeline_config is not set in ServerArgs")

        self.pipeline_config.check_pipeline_config()
        self._validate_disagg_capability()

    def _validate_disagg_capability(self) -> None:
        if self.pipeline_config is None:
            return
        if (
            self.disagg_role != RoleType.MONOLITHIC
            and not self.pipeline_config.supports_disaggregation()
        ):
            raise ValueError(
                f"{type(self.pipeline_config).__name__} only supports monolithic "
                f"deployment; disaggregation role {self.disagg_role.value!r} "
                "is not supported"
            )

    def _validate_offload(self):
        if (
            self.component_residency is not None
            and self.pipeline_config.task_type.is_action_gen()
        ):
            raise ValueError(
                "--component-residency is not supported by action-generation "
                "pipelines; use their existing model-specific offload controls"
            )
        if self.backend == Backend.DIFFUSERS and self.component_residency is not None:
            resolve_diffusers_pipeline_offload(self.component_residency)

        # validate dit_offload_prefetch_size
        if self.dit_offload_prefetch_size > 1 and (
            isinstance(self.dit_offload_prefetch_size, float)
            and not self.dit_offload_prefetch_size.is_integer()
        ):
            self.dit_offload_prefetch_size = int(
                math.floor(self.dit_offload_prefetch_size)
            )
            logger.info(
                f"Invalid --dit-offload-prefetch-size value passed, truncated to: {self.dit_offload_prefetch_size}"
            )

        if 0.5 <= self.dit_offload_prefetch_size < 1.0:
            logger.info(
                "We do not recommend --dit-offload-prefetch-size to be between 0.5 and 1.0"
            )

        # validate dit_layerwise_resident_layers (same ratio/absolute convention)
        if self.dit_layerwise_resident_layers < 0.0:
            raise ValueError("dit_layerwise_resident_layers must be non-negative")
        if self.dit_layerwise_resident_layers >= 1 and (
            isinstance(self.dit_layerwise_resident_layers, float)
            and not self.dit_layerwise_resident_layers.is_integer()
        ):
            self.dit_layerwise_resident_layers = int(
                math.floor(self.dit_layerwise_resident_layers)
            )
            logger.info(
                "Invalid --dit-layerwise-resident-layers value passed, truncated to: "
                f"{self.dit_layerwise_resident_layers}"
            )
        if (
            self.dit_layerwise_resident_layers > 0
            and not self.is_dit_layerwise_offload_selected
        ):
            logger.warning(
                "--dit-layerwise-resident-layers has no effect because the DiT is not "
                "layerwise-offloaded. It only applies together with "
                "--dit-layerwise-offload (or 'dit' in --layerwise-offload-components)."
            )

        if self.dit_layerwise_residency_policy not in RESIDENCY_POLICIES:
            # argparse's choices= only covers the CLI; ServerArgs is also
            # constructed directly by the Python API, and without this the bad
            # value would surface as a ValueError inside the GPU worker at
            # model-load time.
            raise ValueError(
                f"Invalid --dit-layerwise-residency-policy "
                f"{self.dit_layerwise_residency_policy!r}; expected one of "
                f"{RESIDENCY_POLICIES}."
            )

        if self.dit_layerwise_residency_policy != RESIDENCY_POLICY_LEADING:
            if not self.is_dit_layerwise_offload_selected:
                logger.warning(
                    "--dit-layerwise-residency-policy has no effect because the DiT is "
                    "not layerwise-offloaded. It only applies together with "
                    "--dit-layerwise-offload (or 'dit' in "
                    "--layerwise-offload-components)."
                )
            elif self.dit_layerwise_resident_layers <= 0:
                # With nothing resident every layer streams, so there is no
                # layout to choose and the policies are the same run.
                logger.warning(
                    "--dit-layerwise-residency-policy has no effect because "
                    "--dit-layerwise-resident-layers is 0: every layer is streamed, "
                    "so there is no resident set to place."
                )

        # validate layerwise offload conflicts
        if envs.SGLANG_CACHE_DIT_ENABLED and self.use_fsdp_inference:
            if self.is_arg_explicitly_set("use_fsdp_inference"):
                raise ValueError(
                    "FSDP inference cannot be enabled together with cache-dit. "
                    "cache-dit wraps known DiT block structures, while FSDP wraps "
                    "and shards modules before cache-dit can inspect them. "
                    "Please disable --use-fsdp-inference or disable "
                    "SGLANG_CACHE_DIT_ENABLED."
                )
            logger.warning(
                "cache-dit is enabled, automatically disabling use_fsdp_inference."
            )
            self.use_fsdp_inference = False

        if self.has_layerwise_offload_components():
            if self.dit_offload_prefetch_size < 0.0:
                raise ValueError("dit_offload_prefetch_size must be non-negative")

            is_dit_layerwise_offload_selected = self.is_dit_layerwise_offload_selected

            if envs.SGLANG_CACHE_DIT_ENABLED and is_dit_layerwise_offload_selected:
                raise ValueError(
                    "DiT layerwise offload cannot be enabled together with cache-dit. "
                    "cache-dit may reuse skipped blocks whose weights have been released by layerwise offload, "
                    "causing shape mismatch errors. "
                    "Please disable --dit-layerwise-offload, remove DiT from --layerwise-offload-components, "
                    "or disable SGLANG_CACHE_DIT_ENABLED."
                )

            if (
                self.performance_mode == "memory"
                or self.is_arg_explicitly_set("layerwise_offload_components")
                or self.dit_layerwise_offload
                or (
                    self.component_residency
                    and LAYERWISE_OFFLOAD in self.component_residency.values()
                )
            ):
                selected_components = list(self.layerwise_offload_components or ())
                if self.component_residency:
                    selected_components.extend(
                        selector
                        for selector, mode in self.component_residency.items()
                        if mode == LAYERWISE_OFFLOAD
                    )
                selected_components = list(dict.fromkeys(selected_components))
                logger.info_once(
                    "Using layerwise offload components: "
                    f"{', '.join(selected_components or ())}. "
                    "This reduces peak GPU memory and can increase latency; use "
                    "--performance-mode speed for GPU-resident defaults when memory allows."
                )

    def _validate_direct_gpu_weight_loading(self) -> None:
        if not self.direct_gpu_weight_loading:
            return
        if not current_platform.is_cuda():
            raise ValueError("--direct-gpu-weight-loading requires CUDA")
        if (
            self.should_cpu_offload_component("transformer")
            or self.residency_mode("transformer") == LAYERWISE_OFFLOAD
        ):
            raise ValueError(
                "--direct-gpu-weight-loading requires a GPU-resident DiT; disable "
                "DiT CPU and layerwise offload"
            )
        if self.use_fsdp_inference:
            raise ValueError(
                "--direct-gpu-weight-loading does not support FSDP inference"
            )
        if self.tp_size != 1:
            raise ValueError("--direct-gpu-weight-loading requires --tp-size 1")

    def _validate_parallelism(self):
        if self.kv_gather_degree < 1:
            raise ValueError("kv_gather_degree must be >= 1")
        if self.kv_gather_degree > 1 and self.sp_degree != self.kv_gather_degree:
            raise ValueError(
                f"kv_gather_degree ({self.kv_gather_degree}) must equal "
                f"sp_degree ({self.sp_degree}); check how many GPUs remain for "
                "sequence parallelism after dp/tp/cfg"
            )

        if self.nnodes < 1:
            raise ValueError("--nnodes must be a natural number")
        if not (0 <= self.node_rank < self.nnodes):
            raise ValueError(
                f"--node-rank ({self.node_rank}) must be in [0, nnodes={self.nnodes})"
            )
        if self.nnodes > 1 and self.dist_init_addr is None:
            raise ValueError("--dist-init-addr is required when --nnodes > 1")
        if self.num_gpus % self.nnodes != 0:
            raise ValueError(
                f"num_gpus ({self.num_gpus}) must be divisible by nnodes ({self.nnodes})"
            )

        if self.sp_degree > self.num_gpus or self.num_gpus % self.sp_degree != 0:
            raise ValueError(
                f"num_gpus ({self.num_gpus}) must be >= and divisible by sp_degree ({self.sp_degree})"
            )

        if (
            self.hsdp_replicate_dim > self.num_gpus
            or self.num_gpus % self.hsdp_replicate_dim != 0
        ):
            raise ValueError(
                f"num_gpus ({self.num_gpus}) must be >= and divisible by hsdp_replicate_dim ({self.hsdp_replicate_dim})"
            )

        if (
            self.hsdp_shard_dim > self.num_gpus
            or self.num_gpus % self.hsdp_shard_dim != 0
        ):
            raise ValueError(
                f"num_gpus ({self.num_gpus}) must be >= and divisible by hsdp_shard_dim ({self.hsdp_shard_dim})"
            )

        if self.num_gpus % self.dp_size != 0:
            raise ValueError(
                f"num_gpus ({self.num_gpus}) must be divisible by dp_size ({self.dp_size})"
            )

        if self.dp_size < 1:
            raise ValueError("--dp-size must be a natural number")

        if self.dp_size > 1 and self.disagg_role != RoleType.MONOLITHIC:
            raise ValueError(
                "--dp-size > 1 is only supported for monolithic serving; "
                "disaggregated roles scale by adding role instances instead"
            )

        num_gpus_per_group = self.dp_size * self.tp_size
        if self.enable_cfg_parallel:
            num_gpus_per_group *= self.cfg_parallel_degree

        if self.num_gpus % num_gpus_per_group != 0:
            raise ValueError(
                f"num_gpus ({self.num_gpus}) must be divisible by (dp_size * tp_size"
                f"{f' * {self.cfg_parallel_degree}' if self.enable_cfg_parallel else ''}"
                f") = {num_gpus_per_group}"
            )

        if self.sp_degree != self.ring_degree * self.ulysses_degree:
            raise ValueError(
                f"sp_degree ({self.sp_degree}) must equal ring_degree * ulysses_degree "
                f"({self.ring_degree} * {self.ulysses_degree} = {self.ring_degree * self.ulysses_degree})"
            )

        if os.getenv("SGLANG_CACHE_DIT_ENABLED", "").lower() == "true":
            has_sp = self.sp_degree > 1
            has_tp = self.tp_size > 1
            if has_sp and has_tp:
                logger.warning(
                    "cache-dit is enabled with hybrid parallelism (SP + TP). "
                    "Proceeding anyway (SGLang integration may support this mode)."
                )

    def _validate_cfg_parallel(self):
        if not self.enable_cfg_parallel:
            return
        deployment_config = self.pipeline_config.get_model_deployment_config()
        if not deployment_config.supports_cfg_parallel:
            raise ValueError(
                f"{type(self.pipeline_config).__name__} does not support CFG parallelism"
            )
        if self.num_gpus == 1:
            raise ValueError(
                "CFG Parallelism is enabled via `--enable-cfg-parallel`, but num_gpus == 1"
            )

    def _validate_batching(self):
        if self.batching_mode != "dynamic":
            raise ValueError("batching_mode must be one of: dynamic")
        if self.batching_max_size < 1:
            raise ValueError("batching_max_size must be >= 1")
        if self.batching_delay_ms < 0:
            raise ValueError("batching_delay_ms must be >= 0")

    def _set_default_attention_backend(self) -> None:
        """Configure ROCm defaults when users do not specify an attention backend."""
        if current_platform.is_rocm():
            default_backend = AttentionBackendEnum.AITER.name.lower()
            self.attention_backend = default_backend
            logger.info(
                "Attention backend not specified. Using '%s' by default on ROCm "
                "to match SGLang SRT defaults.",
                default_backend,
            )


@dataclasses.dataclass
class PortArgs:
    # The ipc filename for scheduler (rank 0) to receive inputs from tokenizer (zmq)
    scheduler_input_ipc_name: str

    # The port for nccl initialization (torch.dist)
    nccl_port: int

    # The ipc filename for rpc call between Engine and Scheduler
    rpc_ipc_name: str

    # The ipc filename for Scheduler to send metrics
    metrics_ipc_name: str

    # Master port for distributed inference
    master_port: int | None = None

    @staticmethod
    def from_server_args(
        server_args: ServerArgs, dp_rank: Optional[int] = None
    ) -> "PortArgs":
        if server_args.nccl_port is None:
            nccl_port = server_args.scheduler_port + random.randint(100, 1000)
            while True:
                if is_port_available(nccl_port):
                    break
                if nccl_port < 60000:
                    nccl_port += 42
                else:
                    nccl_port -= 43
        else:
            nccl_port = server_args.nccl_port

        # Normal case, use IPC within a single node
        return PortArgs(
            scheduler_input_ipc_name=f"ipc://{tempfile.NamedTemporaryFile(delete=False).name}",
            nccl_port=nccl_port,
            rpc_ipc_name=f"ipc://{tempfile.NamedTemporaryFile(delete=False).name}",
            metrics_ipc_name=f"ipc://{tempfile.NamedTemporaryFile(delete=False).name}",
            master_port=server_args.master_port,
        )


_global_server_args = None


def prepare_server_args(argv: list[str]) -> ServerArgs:
    """
    Prepare the inference arguments from the command line arguments.
    """
    parser = FlexibleArgumentParser()
    ServerArgs.add_cli_args(parser)
    raw_args, unknown_args = parser.parse_known_args(argv)
    server_args = ServerArgs.from_cli_args(raw_args, unknown_args)
    return server_args


def set_global_server_args(server_args: ServerArgs):
    """
    Set the global sgl_diffusion config for each process
    """
    global _global_server_args
    _global_server_args = server_args


def get_global_server_args() -> ServerArgs:
    if _global_server_args is None:
        # in ci, usually when we test custom ops/modules directly,
        # we don't set the sgl_diffusion config. In that case, we set a default
        # config.
        # TODO(will): may need to handle this for CI.
        raise ValueError("Global sgl_diffusion args is not set.")
    return _global_server_args
