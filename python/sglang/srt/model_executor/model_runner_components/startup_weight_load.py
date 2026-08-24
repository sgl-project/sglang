from __future__ import annotations

import dataclasses
import enum
import logging
import time
from typing import TYPE_CHECKING, Callable, Iterable, Optional, Tuple

import torch
from torch import nn

from sglang.srt.configs.device_config import DeviceConfig
from sglang.srt.configs.load_config import LoadConfig, LoadFormat
from sglang.srt.distributed.parallel_state import monkey_patch_vllm_parallel_state
from sglang.srt.model_executor.cuda_graph_config import Backend, Phase
from sglang.srt.model_loader.loader import DefaultModelLoader
from sglang.srt.model_loader.utils import get_model_architecture
from sglang.srt.model_loader.weight_utils import (
    CAPTURE_SAFE_WEIGHT_SENTINEL,
    CheckpointFilePrefetchHandle,
)
from sglang.srt.platforms import current_platform
from sglang.srt.runtime_context import (
    attention_backends,
    configured_attn_cp_size,
    configured_dcp_size,
    configured_pp_size,
    configured_tp_size,
    get_device,
    get_exec,
    get_lora,
    get_model,
    get_parallel,
    get_spec,
)

if TYPE_CHECKING:
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


_SUPPORTED_DTYPES = frozenset({torch.float16, torch.bfloat16})


def _resolve_llama_model_class():
    from sglang.srt.models.llama import LlamaForCausalLM

    return LlamaForCausalLM


def _resolve_qwen2_model_class():
    from sglang.srt.models.qwen2 import Qwen2ForCausalLM

    return Qwen2ForCausalLM


def _resolve_qwen3_model_class():
    from sglang.srt.models.qwen3 import Qwen3ForCausalLM

    return Qwen3ForCausalLM


def _resolve_qwen3_5_model_class():
    from sglang.srt.models.qwen3_5 import Qwen3_5ForConditionalGeneration

    return Qwen3_5ForConditionalGeneration


def _resolve_qwen3_5_moe_model_class():
    from sglang.srt.models.qwen3_5 import Qwen3_5MoeForConditionalGeneration

    return Qwen3_5MoeForConditionalGeneration


def _resolve_qwen3_moe_model_class():
    from sglang.srt.models.qwen3_moe import Qwen3MoeForCausalLM

    return Qwen3MoeForCausalLM


def _resolve_glm_moe_dsa_model_class():
    from sglang.srt.models.glm4_moe import GlmMoeDsaForCausalLM

    return GlmMoeDsaForCausalLM


class StartupWeightLoadState(str, enum.Enum):
    CREATED = "created"
    PREPARING = "preparing"
    CAPTURE_READY = "capture_ready"
    PREFETCHING = "prefetching"
    COMMITTING = "committing"
    READY = "ready"


class StartupWeightLoadProfile(str, enum.Enum):
    NATIVE_DENSE = "native_dense"
    QWEN3_5_HYBRID_VLM = "qwen3_5_hybrid_vlm"
    QWEN3_5_MOE_HYBRID_VLM = "qwen3_5_moe_hybrid_vlm"
    QWEN3_MOE_EP = "qwen3_moe_ep"
    GLM_5_2_DSA_FP8 = "glm_5_2_dsa_fp8"


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class StartupWeightLoadOptions:
    device: str
    is_cuda_platform: bool
    cuda_device_capability: Optional[Tuple[int, int]]
    cuda_graph_enabled: bool
    prefill_cuda_graph_backend: Backend
    is_draft_worker: bool
    speculative_algorithm: Optional[str]
    tp_size: int
    attn_cp_size: int
    dcp_size: int
    pp_size: int
    dp_size: int
    ep_size: int
    moe_dp_size: int
    moe_a2a_backend: str
    moe_runner_backend: str
    fp8_gemm_runner_backend: str
    prefill_attention_backend: Optional[str]
    decode_attention_backend: Optional[str]
    dsa_prefill_backend: Optional[str]
    dsa_decode_backend: Optional[str]
    kv_cache_dtype: str
    disable_shared_experts_fusion: bool
    enable_dp_attention: bool
    enable_two_batch_overlap: bool
    enable_eplb: bool
    ep_num_redundant_experts: int
    init_expert_location: str
    elastic_ep_backend: Optional[str]
    enable_elastic_expert_backup: bool
    ep_join_mode: Optional[str]
    max_ep_size: Optional[int]
    linear_attn_backend: str
    linear_attn_decode_backend: Optional[str]
    linear_attn_prefill_backend: Optional[str]
    cpu_offload_gb: int
    offload_group_size: int
    enable_memory_saver: bool
    enable_weights_cpu_backup: bool
    enable_lora: bool
    has_lora_paths: bool
    weight_loader_disable_mmap: bool
    weight_loader_drop_cache_after_load: bool
    has_custom_weight_loader: bool
    enable_torch_compile: bool
    prefetch_num_threads: int

    @classmethod
    def from_server_args(
        cls,
        *,
        server_args: ServerArgs,
        is_draft_worker: bool,
    ) -> StartupWeightLoadOptions:
        cuda_graph_config = get_exec().graph.cuda_graph_config
        is_cuda_platform = current_platform.is_cuda()
        device_capability = (
            current_platform.get_device_capability() if is_cuda_platform else None
        )
        prefill_attention_backend, decode_attention_backend = attention_backends()
        cuda_graph_enabled = any(
            getattr(cuda_graph_config, phase).backend != Backend.DISABLED
            for phase in Phase.ALL
        )
        return cls(
            device=get_device().device,
            is_cuda_platform=is_cuda_platform,
            cuda_device_capability=(
                tuple(device_capability) if device_capability is not None else None
            ),
            cuda_graph_enabled=cuda_graph_enabled,
            prefill_cuda_graph_backend=cuda_graph_config.prefill.backend,
            is_draft_worker=is_draft_worker,
            speculative_algorithm=get_spec().speculative_algorithm,
            tp_size=configured_tp_size(),
            attn_cp_size=configured_attn_cp_size(),
            dcp_size=configured_dcp_size(),
            pp_size=configured_pp_size(),
            dp_size=get_parallel().dp_size,
            ep_size=get_parallel().ep_size,
            moe_dp_size=get_parallel().moe_dp_size,
            moe_a2a_backend=get_exec().moe.moe_a2a_backend,
            moe_runner_backend=get_exec().moe.moe_runner_backend,
            fp8_gemm_runner_backend=get_exec().kernel.fp8_gemm_runner_backend,
            prefill_attention_backend=prefill_attention_backend,
            decode_attention_backend=decode_attention_backend,
            dsa_prefill_backend=get_exec().kernel.dsa_prefill_backend,
            dsa_decode_backend=get_exec().kernel.dsa_decode_backend,
            kv_cache_dtype=get_model().kv_cache_dtype,
            disable_shared_experts_fusion=(
                get_exec().moe.disable_shared_experts_fusion
            ),
            enable_dp_attention=get_parallel().enable_dp_attention,
            enable_two_batch_overlap=get_exec().overlap.enable_two_batch_overlap,
            enable_eplb=get_exec().moe.enable_eplb,
            ep_num_redundant_experts=get_exec().moe.ep_num_redundant_experts,
            init_expert_location=get_exec().moe.init_expert_location,
            elastic_ep_backend=get_exec().moe.elastic_ep_backend,
            enable_elastic_expert_backup=(
                get_exec().moe.enable_elastic_expert_backup
            ),
            ep_join_mode=get_parallel().ep_join_mode,
            max_ep_size=get_parallel().max_ep_size,
            linear_attn_backend=get_exec().mamba.linear_attn_backend,
            linear_attn_decode_backend=get_exec().mamba.linear_attn_decode_backend,
            linear_attn_prefill_backend=(
                get_exec().mamba.linear_attn_prefill_backend
            ),
            cpu_offload_gb=get_exec().offload.cpu_offload_gb,
            offload_group_size=get_exec().offload.offload_group_size,
            enable_memory_saver=get_exec().features.enable_memory_saver,
            enable_weights_cpu_backup=get_exec().features.enable_weights_cpu_backup,
            enable_lora=get_lora().enable_lora,
            has_lora_paths=bool(get_lora().lora_paths),
            weight_loader_disable_mmap=get_model().weight_loader_disable_mmap,
            weight_loader_drop_cache_after_load=(
                get_model().weight_loader_drop_cache_after_load
            ),
            has_custom_weight_loader=bool(get_model().custom_weight_loader),
            enable_torch_compile=get_exec().graph.enable_torch_compile,
            prefetch_num_threads=get_model().weight_loader_prefetch_num_threads,
        )


@dataclasses.dataclass(frozen=True, slots=True)
class StartupWeightLoadRejection:
    code: str
    message: str


@dataclasses.dataclass(frozen=True, slots=True)
class StartupWeightLoadPlan:
    """Config-level plan before checkpoint source resolution."""

    profile: StartupWeightLoadProfile
    prefetch_num_threads: int


@dataclasses.dataclass(frozen=True, slots=True)
class StartupWeightLoadAdmission:
    plan: Optional[StartupWeightLoadPlan]
    rejections: Tuple[StartupWeightLoadRejection, ...]

    @property
    def supported(self) -> bool:
        return self.plan is not None


@dataclasses.dataclass(frozen=True, slots=True)
class _StartupWeightLoadArchitectureSpec:
    architecture: str
    resolve_model_class: Callable[[], type]


@dataclasses.dataclass(frozen=True, slots=True)
class _StartupWeightLoadProfileSpec:
    profile: StartupWeightLoadProfile
    architectures: Tuple[_StartupWeightLoadArchitectureSpec, ...]
    validate: Callable[
        [ModelConfig, StartupWeightLoadOptions],
        Tuple[StartupWeightLoadRejection, ...],
    ]


def _rejections_from_rules(
    rules: Iterable[Tuple[str, bool, str]],
) -> Tuple[StartupWeightLoadRejection, ...]:
    return tuple(
        StartupWeightLoadRejection(code=code, message=message)
        for code, rejected, message in rules
        if rejected
    )


def _ep_moe_rules(
    *,
    family: str,
    tp_size: int,
    ep_size: int,
    model_config: ModelConfig,
    options: StartupWeightLoadOptions,
) -> Tuple[Tuple[str, bool, str], ...]:
    return (
        (
            "tensor_parallelism",
            options.tp_size != tp_size,
            f"{family} startup overlap requires TP{tp_size}",
        ),
        (
            "dtype",
            model_config.dtype != torch.bfloat16,
            f"{family} startup overlap requires BF16",
        ),
        (
            "quantization",
            model_config.quantization is not None,
            "quantization is not supported",
        ),
        (
            "modelopt",
            bool(getattr(model_config, "modelopt_quant", False)),
            "ModelOpt is not supported",
        ),
        (
            "expert_parallelism",
            options.ep_size != ep_size,
            f"{family} startup overlap requires EP{ep_size}",
        ),
        (
            "moe_data_parallelism",
            options.moe_dp_size != 1,
            "MoE data parallelism is not supported",
        ),
        (
            "moe_a2a_backend",
            options.moe_a2a_backend != "none",
            f"{family} startup overlap requires the standard EP path",
        ),
        (
            "moe_runner_backend",
            options.moe_runner_backend != "triton",
            f"{family} startup overlap requires the Triton MoE runner",
        ),
        (
            "dp_attention",
            options.enable_dp_attention,
            "DP attention is not supported",
        ),
        (
            "two_batch_overlap",
            options.enable_two_batch_overlap,
            "two-batch overlap is not supported",
        ),
        (
            "eplb",
            options.enable_eplb,
            "EPLB is not supported",
        ),
        (
            "redundant_experts",
            options.ep_num_redundant_experts != 0,
            "redundant experts are not supported",
        ),
        (
            "expert_placement",
            options.init_expert_location != "trivial",
            "non-trivial expert placement is not supported",
        ),
        (
            "elastic_expert_parallelism",
            options.elastic_ep_backend is not None
            or options.enable_elastic_expert_backup
            or options.ep_join_mode is not None
            or options.max_ep_size is not None,
            "elastic expert parallelism is not supported",
        ),
    )


def _validate_native_dense(
    model_config: ModelConfig,
    options: StartupWeightLoadOptions,
) -> Tuple[StartupWeightLoadRejection, ...]:
    return _rejections_from_rules(
        (
            (
                "tensor_parallelism",
                options.tp_size not in (1, 2),
                "only TP1 and TP2 are supported",
            ),
            (
                "dtype",
                model_config.dtype not in _SUPPORTED_DTYPES,
                "FP16 or BF16 only",
            ),
            (
                "quantization",
                model_config.quantization is not None,
                "quantization is not supported",
            ),
            (
                "modelopt",
                bool(getattr(model_config, "modelopt_quant", False)),
                "ModelOpt is not supported",
            ),
            (
                "expert_parallelism",
                options.ep_size != 1,
                "expert parallelism is not supported",
            ),
            (
                "multimodal",
                model_config.is_multimodal,
                "multimodal models are not supported",
            ),
        )
    )


def _linear_attention_backends(
    options: StartupWeightLoadOptions,
) -> Tuple[str, str]:
    linear_attn_decode_backend = (
        options.linear_attn_decode_backend or options.linear_attn_backend
    )
    linear_attn_prefill_backend = (
        options.linear_attn_prefill_backend or options.linear_attn_backend
    )

    return linear_attn_decode_backend, linear_attn_prefill_backend


def _validate_qwen3_5_hybrid_vlm(
    model_config: ModelConfig,
    options: StartupWeightLoadOptions,
) -> Tuple[StartupWeightLoadRejection, ...]:
    linear_attn_decode_backend, linear_attn_prefill_backend = (
        _linear_attention_backends(options)
    )
    return _rejections_from_rules(
        (
            (
                "tensor_parallelism",
                options.tp_size not in (2, 4),
                "Qwen3.5-family hybrid VLM startup overlap requires TP2 or TP4",
            ),
            (
                "dtype",
                model_config.dtype != torch.bfloat16,
                "Qwen3.5-family hybrid VLM startup overlap requires BF16",
            ),
            (
                "quantization",
                model_config.quantization is not None,
                "quantization is not supported",
            ),
            (
                "modelopt",
                bool(getattr(model_config, "modelopt_quant", False)),
                "ModelOpt is not supported",
            ),
            (
                "expert_parallelism",
                options.ep_size != 1,
                "expert parallelism is not supported",
            ),
            (
                "multimodal",
                not model_config.is_multimodal,
                "Qwen3.5-family hybrid VLM startup overlap requires multimodal execution",
            ),
            (
                "encoder_only",
                bool(getattr(model_config.hf_config, "encoder_only", False)),
                "encoder-only execution is not supported",
            ),
            (
                "language_only",
                bool(getattr(model_config.hf_config, "language_only", False)),
                "language-only encoder disaggregation is not supported",
            ),
            (
                "language_model_only",
                bool(getattr(model_config.hf_config, "language_model_only", False)),
                "language-model-only execution is not supported",
            ),
            (
                "linear_attention_backend",
                linear_attn_decode_backend != "triton"
                or linear_attn_prefill_backend != "triton",
                "Qwen3.5-family hybrid VLM startup overlap requires Triton linear attention",
            ),
            (
                "full_prefill_cuda_graph",
                options.prefill_cuda_graph_backend == Backend.FULL,
                "Qwen3.5-family hybrid VLM startup overlap does not support full prefill CUDA graphs",
            ),
        )
    )


def _validate_qwen3_5_moe_hybrid_vlm(
    model_config: ModelConfig,
    options: StartupWeightLoadOptions,
) -> Tuple[StartupWeightLoadRejection, ...]:
    linear_attn_decode_backend, linear_attn_prefill_backend = (
        _linear_attention_backends(options)
    )
    rules = _ep_moe_rules(
        family="Qwen3.5 MoE hybrid VLM",
        tp_size=2,
        ep_size=2,
        model_config=model_config,
        options=options,
    ) + (
        (
            "multimodal",
            not model_config.is_multimodal,
            "Qwen3.5 MoE hybrid VLM startup overlap requires multimodal execution",
        ),
        (
            "encoder_only",
            bool(getattr(model_config.hf_config, "encoder_only", False)),
            "encoder-only execution is not supported",
        ),
        (
            "language_only",
            bool(getattr(model_config.hf_config, "language_only", False)),
            "language-only encoder disaggregation is not supported",
        ),
        (
            "language_model_only",
            bool(getattr(model_config.hf_config, "language_model_only", False)),
            "language-model-only execution is not supported",
        ),
        (
            "linear_attention_backend",
            linear_attn_decode_backend != "triton"
            or linear_attn_prefill_backend != "triton",
            "Qwen3.5 MoE hybrid VLM startup overlap requires Triton linear attention",
        ),
        (
            "full_prefill_cuda_graph",
            options.prefill_cuda_graph_backend == Backend.FULL,
            "Qwen3.5 MoE hybrid VLM startup overlap does not support full prefill CUDA graphs",
        ),
    )
    return _rejections_from_rules(rules)


def _validate_qwen3_moe_ep(
    model_config: ModelConfig,
    options: StartupWeightLoadOptions,
) -> Tuple[StartupWeightLoadRejection, ...]:
    rules = _ep_moe_rules(
        family="Qwen3 MoE",
        tp_size=2,
        ep_size=2,
        model_config=model_config,
        options=options,
    ) + (
        (
            "multimodal",
            model_config.is_multimodal,
            "multimodal models are not supported",
        ),
    )
    return _rejections_from_rules(rules)


def _validate_glm_5_2_dsa_fp8(
    model_config: ModelConfig,
    options: StartupWeightLoadOptions,
) -> Tuple[StartupWeightLoadRejection, ...]:
    quantization_config = getattr(model_config.hf_config, "quantization_config", None)
    weight_block_size = (
        tuple(quantization_config.get("weight_block_size") or ())
        if isinstance(quantization_config, dict)
        else ()
    )
    checkpoint_quant_method = (
        quantization_config.get("quant_method")
        if isinstance(quantization_config, dict)
        else None
    )
    checkpoint_activation_scheme = (
        quantization_config.get("activation_scheme")
        if isinstance(quantization_config, dict)
        else None
    )
    checkpoint_fp8_format = (
        quantization_config.get("fmt")
        if isinstance(quantization_config, dict)
        else None
    )
    cli_factor = getattr(model_config.hf_config, "cli_factor", 1)
    if cli_factor is None:
        cli_factor = 1
    is_glm_5_2 = (
        cli_factor == 1
        and getattr(model_config.hf_config, "index_topk_pattern", None) is None
        and getattr(model_config.hf_config, "index_topk_freq", None) == 4
        and getattr(model_config.hf_config, "index_skip_topk_offset", None) == 3
    )
    return _rejections_from_rules(
        (
            (
                "model_variant",
                not is_glm_5_2,
                "startup overlap is validated only for the GLM-5.2 DSA architecture",
            ),
            (
                "cuda_device_capability",
                options.cuda_device_capability != (9, 0),
                "GLM-5.2 DSA FP8 startup overlap is validated only on NVIDIA Hopper (SM90)",
            ),
            (
                "tensor_parallelism",
                options.tp_size not in (8, 16),
                "GLM-5.2 DSA FP8 startup overlap requires TP8 or TP16",
            ),
            (
                "dtype",
                model_config.dtype != torch.bfloat16,
                "GLM-5.2 DSA FP8 startup overlap requires --dtype bfloat16",
            ),
            (
                "quantization",
                model_config.quantization != "fp8"
                or checkpoint_quant_method != "fp8"
                or checkpoint_activation_scheme != "dynamic"
                or checkpoint_fp8_format != "e4m3",
                "GLM-5.2 DSA startup overlap requires its serialized dynamic E4M3 FP8 checkpoint",
            ),
            (
                "fp8_weight_block_size",
                weight_block_size != (128, 128),
                "GLM-5.2 DSA FP8 startup overlap requires 128x128 block scales",
            ),
            (
                "modelopt",
                bool(getattr(model_config, "modelopt_quant", False)),
                "ModelOpt is not supported",
            ),
            (
                "expert_parallelism",
                options.ep_size != 1,
                "GLM-5.2 DSA FP8 startup overlap does not yet support expert parallelism",
            ),
            (
                "moe_data_parallelism",
                options.moe_dp_size != 1,
                "MoE data parallelism is not supported",
            ),
            (
                "moe_a2a_backend",
                options.moe_a2a_backend != "none",
                "GLM-5.2 DSA FP8 startup overlap requires --moe-a2a-backend none",
            ),
            (
                "moe_runner_backend",
                options.moe_runner_backend != "triton",
                "GLM-5.2 DSA FP8 startup overlap requires --moe-runner-backend triton",
            ),
            (
                "fp8_gemm_backend",
                options.fp8_gemm_runner_backend != "triton",
                "GLM-5.2 DSA FP8 startup overlap requires --fp8-gemm-backend triton",
            ),
            (
                "attention_backend",
                options.prefill_attention_backend != "dsa"
                or options.decode_attention_backend != "dsa",
                "GLM-5.2 DSA FP8 startup overlap requires DSA for prefill and decode attention",
            ),
            (
                "dsa_attention_backend",
                options.dsa_prefill_backend != "fa3"
                or options.dsa_decode_backend != "fa3",
                "GLM-5.2 DSA FP8 startup overlap requires FA3 for DSA prefill and decode",
            ),
            (
                "kv_cache_dtype",
                options.kv_cache_dtype != "bfloat16",
                "GLM-5.2 DSA FP8 startup overlap requires --kv-cache-dtype bfloat16",
            ),
            (
                "prefill_cuda_graph",
                options.prefill_cuda_graph_backend != Backend.DISABLED,
                "GLM-5.2 DSA FP8 startup overlap requires prefill CUDA graphs disabled",
            ),
            (
                "shared_experts_fusion",
                not options.disable_shared_experts_fusion,
                "GLM-5.2 DSA FP8 startup overlap requires --disable-shared-experts-fusion",
            ),
            (
                "dp_attention",
                options.enable_dp_attention,
                "DP attention is not supported",
            ),
            (
                "two_batch_overlap",
                options.enable_two_batch_overlap,
                "two-batch overlap is not supported",
            ),
            (
                "eplb",
                options.enable_eplb,
                "EPLB is not supported",
            ),
            (
                "redundant_experts",
                options.ep_num_redundant_experts != 0,
                "redundant experts are not supported",
            ),
            (
                "expert_placement",
                options.init_expert_location != "trivial",
                "non-trivial expert placement is not supported",
            ),
            (
                "elastic_expert_parallelism",
                options.elastic_ep_backend is not None
                or options.enable_elastic_expert_backup
                or options.ep_join_mode is not None
                or options.max_ep_size is not None,
                "elastic expert parallelism is not supported",
            ),
            (
                "multimodal",
                model_config.is_multimodal,
                "multimodal models are not supported",
            ),
        )
    )


# Keep each profile narrow until its storage and startup behavior are validated.
_STARTUP_WEIGHT_LOAD_PROFILE_SPECS = (
    _StartupWeightLoadProfileSpec(
        profile=StartupWeightLoadProfile.NATIVE_DENSE,
        architectures=(
            _StartupWeightLoadArchitectureSpec(
                "LlamaForCausalLM", _resolve_llama_model_class
            ),
            _StartupWeightLoadArchitectureSpec(
                "Qwen2ForCausalLM", _resolve_qwen2_model_class
            ),
            _StartupWeightLoadArchitectureSpec(
                "Qwen3ForCausalLM", _resolve_qwen3_model_class
            ),
        ),
        validate=_validate_native_dense,
    ),
    _StartupWeightLoadProfileSpec(
        profile=StartupWeightLoadProfile.QWEN3_5_HYBRID_VLM,
        architectures=(
            # Qwen3.6 dense checkpoints retain the Qwen3.5 implementation
            # architecture in config.json.
            _StartupWeightLoadArchitectureSpec(
                "Qwen3_5ForConditionalGeneration", _resolve_qwen3_5_model_class
            ),
        ),
        validate=_validate_qwen3_5_hybrid_vlm,
    ),
    _StartupWeightLoadProfileSpec(
        profile=StartupWeightLoadProfile.QWEN3_5_MOE_HYBRID_VLM,
        architectures=(
            _StartupWeightLoadArchitectureSpec(
                "Qwen3_5MoeForConditionalGeneration",
                _resolve_qwen3_5_moe_model_class,
            ),
        ),
        validate=_validate_qwen3_5_moe_hybrid_vlm,
    ),
    _StartupWeightLoadProfileSpec(
        profile=StartupWeightLoadProfile.QWEN3_MOE_EP,
        architectures=(
            _StartupWeightLoadArchitectureSpec(
                "Qwen3MoeForCausalLM", _resolve_qwen3_moe_model_class
            ),
        ),
        validate=_validate_qwen3_moe_ep,
    ),
    _StartupWeightLoadProfileSpec(
        profile=StartupWeightLoadProfile.GLM_5_2_DSA_FP8,
        architectures=(
            _StartupWeightLoadArchitectureSpec(
                "GlmMoeDsaForCausalLM", _resolve_glm_moe_dsa_model_class
            ),
        ),
        validate=_validate_glm_5_2_dsa_fp8,
    ),
)


def _build_startup_weight_load_profile_indexes():
    specs_by_profile = {}
    specs_by_architecture = {}
    for profile_spec in _STARTUP_WEIGHT_LOAD_PROFILE_SPECS:
        if profile_spec.profile in specs_by_profile:
            raise RuntimeError(
                f"Duplicate startup weight-load profile: {profile_spec.profile.value}"
            )
        if not profile_spec.architectures:
            raise RuntimeError(
                f"Startup weight-load profile has no architectures: "
                f"{profile_spec.profile.value}"
            )
        if not callable(profile_spec.validate):
            raise RuntimeError(
                f"Startup weight-load profile has no validator: "
                f"{profile_spec.profile.value}"
            )
        specs_by_profile[profile_spec.profile] = profile_spec
        for architecture_spec in profile_spec.architectures:
            if architecture_spec.architecture in specs_by_architecture:
                raise RuntimeError(
                    "Duplicate startup weight-load architecture: "
                    f"{architecture_spec.architecture}"
                )
            if not callable(architecture_spec.resolve_model_class):
                raise RuntimeError(
                    "Startup weight-load architecture has no model resolver: "
                    f"{architecture_spec.architecture}"
                )
            specs_by_architecture[architecture_spec.architecture] = (
                profile_spec,
                architecture_spec,
            )
    missing_profiles = set(StartupWeightLoadProfile) - set(specs_by_profile)
    if missing_profiles:
        raise RuntimeError(
            "Missing startup weight-load profile registrations: "
            + ", ".join(sorted(profile.value for profile in missing_profiles))
        )
    return specs_by_profile, specs_by_architecture


(
    _STARTUP_WEIGHT_LOAD_PROFILE_SPEC_BY_PROFILE,
    _STARTUP_WEIGHT_LOAD_PROFILE_SPEC_BY_ARCHITECTURE,
) = _build_startup_weight_load_profile_indexes()


def _get_startup_weight_load_profile(
    architecture: Optional[str],
) -> Optional[StartupWeightLoadProfile]:
    registration = _STARTUP_WEIGHT_LOAD_PROFILE_SPEC_BY_ARCHITECTURE.get(architecture)
    return registration[0].profile if registration is not None else None


def _get_canonical_model_class(architecture: str):
    registration = _STARTUP_WEIGHT_LOAD_PROFILE_SPEC_BY_ARCHITECTURE.get(architecture)
    if registration is None:
        raise ValueError(f"Unsupported startup-overlap architecture: {architecture}")
    return registration[1].resolve_model_class()


def _get_profile_rejections(
    *,
    profile: StartupWeightLoadProfile,
    model_config: ModelConfig,
    options: StartupWeightLoadOptions,
) -> Tuple[StartupWeightLoadRejection, ...]:
    profile_spec = _STARTUP_WEIGHT_LOAD_PROFILE_SPEC_BY_PROFILE.get(profile)
    if profile_spec is None:
        raise ValueError(f"Unknown startup weight-load profile: {profile}")
    return profile_spec.validate(model_config, options)


@dataclasses.dataclass(frozen=True, slots=True)
class StartupWeightLoadTimings:
    """Phase timings for deferred startup loading.

    ``weight_load_seconds`` keeps the legacy ``load_weight`` metric limited to
    weight-specific work. ``total_seconds`` covers the end-to-end path, whose
    prefetch window overlaps CUDA graph and KV-cache initialization.
    """

    prepare_seconds: float
    prefetch_start_delay_seconds: float
    prefetch_window_seconds: float
    commit_seconds: float
    prefetch_cleanup_seconds: float
    total_seconds: float

    @property
    def weight_load_seconds(self) -> float:
        return (
            self.prepare_seconds + self.commit_seconds + self.prefetch_cleanup_seconds
        )


@dataclasses.dataclass(frozen=True, slots=True)
class TensorStorageMetadata:
    tensor: torch.Tensor = dataclasses.field(repr=False, compare=False)
    data_ptr: int
    shape: Tuple[int, ...]
    stride: Tuple[int, ...]
    dtype: torch.dtype
    device: torch.device
    storage_offset: int

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor) -> TensorStorageMetadata:
        return cls(
            tensor=tensor,
            data_ptr=tensor.data_ptr(),
            shape=tuple(tensor.shape),
            stride=tuple(tensor.stride()),
            dtype=tensor.dtype,
            device=tensor.device,
            storage_offset=tensor.storage_offset(),
        )

    def matches(self, other: TensorStorageMetadata) -> bool:
        return self.tensor is other.tensor and (
            self.data_ptr,
            self.shape,
            self.stride,
            self.dtype,
            self.device,
            self.storage_offset,
        ) == (
            other.data_ptr,
            other.shape,
            other.stride,
            other.dtype,
            other.device,
            other.storage_offset,
        )


@dataclasses.dataclass(frozen=True, slots=True)
class ModelStorageManifest:
    tensors: Tuple[Tuple[str, TensorStorageMetadata], ...]

    @classmethod
    def capture(cls, model: nn.Module) -> ModelStorageManifest:
        entries = []
        for kind, tensors in (
            ("parameter", model.named_parameters(remove_duplicate=False)),
            ("buffer", model.named_buffers(remove_duplicate=False)),
        ):
            entries.extend(
                (f"{kind}:{name}", TensorStorageMetadata.from_tensor(tensor))
                for name, tensor in tensors
            )
        # CUDA graphs may capture post-load tensors that are not parameters or
        # buffers. Explicit hooks avoid scanning arbitrary runtime state; each
        # implementation must refresh the reported tensors in place.
        derived_names = set()
        for module_name, module in model.named_modules(remove_duplicate=False):
            named_derived_tensors = getattr(
                module, "named_startup_weight_load_derived_tensors", None
            )
            if named_derived_tensors is None:
                continue
            for local_name, tensor in named_derived_tensors():
                if not isinstance(local_name, str) or not local_name:
                    raise ValueError(
                        "Startup weight-load tensor names must be non-empty strings"
                    )
                name = f"{module_name}.{local_name}" if module_name else local_name
                if name in derived_names:
                    raise ValueError(
                        f"Duplicate startup weight-load tensor name: {name!r}"
                    )
                if not isinstance(tensor, torch.Tensor):
                    raise TypeError(
                        f"Startup weight-load tensor {name!r} is not a torch.Tensor"
                    )
                derived_names.add(name)
                entries.append(
                    (
                        f"derived:{name}",
                        TensorStorageMetadata.from_tensor(tensor),
                    )
                )
        # Key explicitly by name because TensorStorageMetadata is not orderable,
        # and stable name ordering keeps diagnostics deterministic for aliases.
        return cls(tensors=tuple(sorted(entries, key=lambda entry: entry[0])))

    def changed_names(self, model: nn.Module) -> Tuple[str, ...]:
        before = dict(self.tensors)
        after = dict(ModelStorageManifest.capture(model).tensors)
        return tuple(
            name
            for name in sorted(before.keys() | after.keys())
            if name not in before
            or name not in after
            or not before[name].matches(after[name])
        )

    def unchanged_parameter_names(self, value: float) -> Tuple[str, ...]:
        """Return required floating-point parameters still equal to ``value``.

        Optional ``_skip_weight_check`` parameters keep their constructor
        values. Every other floating-point parameter must replace the capture
        sentinel; buffers are excluded because they are never overwritten.
        """
        names = []
        checks = []
        seen_tensor_ids = set()
        for name, metadata in self.tensors:
            tensor = metadata.tensor
            if (
                not name.startswith("parameter:")
                or not torch.is_floating_point(tensor)
                or getattr(tensor, "_skip_weight_check", False)
                or id(tensor) in seen_tensor_ids
            ):
                continue
            seen_tensor_ids.add(id(tensor))
            names.append(name)
            checks.append(torch.all(tensor == value))

        if not checks:
            return ()
        unchanged = torch.stack(checks).cpu().tolist()
        return tuple(
            name for name, is_unchanged in zip(names, unchanged) if is_unchanged
        )


def evaluate_startup_weight_load_admission(
    *,
    loader,
    model_config: ModelConfig,
    load_config: LoadConfig,
    options: StartupWeightLoadOptions,
) -> StartupWeightLoadAdmission:
    """Return an overlap plan or a deterministic preflight rejection report."""

    architectures = tuple(model_config.hf_config.architectures or ())
    rules = (
        (
            "non_cuda",
            not options.is_cuda_platform or options.device != "cuda",
            "CUDA only",
        ),
        (
            "cuda_graph_disabled",
            not options.cuda_graph_enabled,
            "CUDA graph capture is disabled",
        ),
        (
            "tc_piecewise_prefill",
            options.prefill_cuda_graph_backend == Backend.TC_PIECEWISE,
            "tc_piecewise prefill CUDA graphs are not supported",
        ),
        (
            "loader",
            type(loader) is not DefaultModelLoader,
            "DefaultModelLoader only",
        ),
        (
            "load_format",
            load_config.load_format not in (LoadFormat.AUTO, LoadFormat.SAFETENSORS),
            "load format must be auto or safetensors",
        ),
        ("draft_worker", options.is_draft_worker, "draft workers are not supported"),
        (
            "draft_model",
            load_config.draft_model_idx is not None,
            "draft model loading is unsupported",
        ),
        (
            "speculative_decoding",
            options.speculative_algorithm is not None,
            "speculative decoding is not supported",
        ),
        (
            "attention_context_parallelism",
            options.attn_cp_size != 1,
            "attention context parallelism is not supported",
        ),
        (
            "decode_context_parallelism",
            options.dcp_size != 1,
            "decode context parallelism is not supported",
        ),
        (
            "pipeline_parallelism",
            options.pp_size != 1,
            "pipeline parallelism is not supported",
        ),
        (
            "data_parallelism",
            options.dp_size != 1,
            "data parallelism is not supported",
        ),
        (
            "cpu_offload",
            options.cpu_offload_gb > 0,
            "CPU offload is not supported",
        ),
        (
            "layer_group_offload",
            options.offload_group_size > 0,
            "layer-group offloading is not supported",
        ),
        (
            "memory_saver",
            options.enable_memory_saver,
            "memory saver is not supported",
        ),
        (
            "cpu_weight_backup",
            options.enable_weights_cpu_backup,
            "CPU weight backup is not supported",
        ),
        (
            "lora",
            options.enable_lora or options.has_lora_paths,
            "LoRA is not supported",
        ),
        (
            "mmap_disabled",
            options.weight_loader_disable_mmap,
            "safetensors mmap must be enabled",
        ),
        (
            "drop_page_cache",
            options.weight_loader_drop_cache_after_load,
            "dropping the page cache during load is not supported",
        ),
        (
            "custom_weight_loader",
            options.has_custom_weight_loader,
            "custom weight loaders are not supported",
        ),
        (
            "torch_compile",
            options.enable_torch_compile,
            "torch.compile is not supported",
        ),
        (
            "non_generation",
            not model_config.is_generation,
            "generation models only",
        ),
        (
            "prefetch_threads",
            options.prefetch_num_threads < 1,
            "checkpoint prefetch requires at least one thread",
        ),
    )
    rejections = [
        StartupWeightLoadRejection(code=code, message=message)
        for code, rejected, message in rules
        if rejected
    ]

    architecture = architectures[0] if len(architectures) == 1 else None
    profile = _get_startup_weight_load_profile(architecture)
    if profile is not None:
        rejections.extend(
            _get_profile_rejections(
                profile=profile,
                model_config=model_config,
                options=options,
            )
        )
    else:
        rejections.append(
            StartupWeightLoadRejection(
                code="architecture",
                message="exactly one supported model architecture is required",
            )
        )

    # Delay implementation imports until cheap preflight passes, so auto can
    # fall back without remote-code imports or config mutation.
    if not rejections:
        assert architecture is not None
        resolved_model_class, resolved_architecture = get_model_architecture(
            model_config
        )
        if (
            resolved_architecture != architecture
            or resolved_model_class is not _get_canonical_model_class(architecture)
        ):
            rejections.append(
                StartupWeightLoadRejection(
                    code="model_implementation",
                    message="the native SGLang model implementation is required",
                )
            )

    if rejections:
        return StartupWeightLoadAdmission(plan=None, rejections=tuple(rejections))
    assert profile is not None
    return StartupWeightLoadAdmission(
        plan=StartupWeightLoadPlan(
            profile=profile,
            prefetch_num_threads=options.prefetch_num_threads,
        ),
        rejections=(),
    )


class StartupWeightLoadManager:
    """Coordinate checkpoint prefetch, graph capture, and real-weight commit.

    After capture-safe mutation, commit failures are terminal. Background
    page-cache prefetch remains best-effort.
    """

    def __init__(
        self,
        *,
        loader: DefaultModelLoader,
        model_config: ModelConfig,
        device_config: DeviceConfig,
        plan: StartupWeightLoadPlan,
        fallback_to_serial: bool = False,
    ) -> None:
        self._loader = loader
        self._model_config = model_config
        self._device_config = device_config
        self._plan = plan
        self._fallback_to_serial = fallback_to_serial
        self._model: Optional[nn.Module] = None
        self._resolved_sources: Tuple[DefaultModelLoader.ResolvedSource, ...] = ()
        self._prefetch_handle: Optional[CheckpointFilePrefetchHandle] = None
        self._state = StartupWeightLoadState.CREATED
        self._created_at = time.perf_counter()
        self._capture_ready_at: Optional[float] = None
        self._prefetch_started_at: Optional[float] = None
        self._prefetch_failure_reported = False
        self._prefetch_stop_timed_out = False
        self._timings: Optional[StartupWeightLoadTimings] = None

    @classmethod
    def create_from_server_args(
        cls,
        *,
        loader,
        model_config: ModelConfig,
        load_config: LoadConfig,
        device_config: DeviceConfig,
        server_args: ServerArgs,
        is_draft_worker: bool,
    ) -> Optional[StartupWeightLoadManager]:
        """Create a manager, or return ``None`` when auto selects serial."""
        options = StartupWeightLoadOptions.from_server_args(
            server_args=server_args,
            is_draft_worker=is_draft_worker,
        )
        if server_args.startup_weight_load_mode == "auto":
            admission = evaluate_startup_weight_load_admission(
                loader=loader,
                model_config=model_config,
                load_config=load_config,
                options=options,
            )
            if not admission.supported:
                logger.info(
                    "Startup weight-load auto mode selected serial loading: %s",
                    cls._format_rejections(admission),
                )
                return None
            assert admission.plan is not None
            return cls(
                loader=loader,
                model_config=model_config,
                device_config=device_config,
                plan=admission.plan,
                fallback_to_serial=True,
            )

        return cls.create(
            loader=loader,
            model_config=model_config,
            load_config=load_config,
            device_config=device_config,
            options=options,
        )

    @classmethod
    def create(
        cls,
        *,
        loader,
        model_config: ModelConfig,
        load_config: LoadConfig,
        device_config: DeviceConfig,
        options: StartupWeightLoadOptions,
    ) -> StartupWeightLoadManager:
        admission = evaluate_startup_weight_load_admission(
            loader=loader,
            model_config=model_config,
            load_config=load_config,
            options=options,
        )
        if not admission.supported:
            raise ValueError(
                "--startup-weight-load-mode=overlap is not supported: "
                f"{cls._format_rejections(admission)}"
            )
        assert admission.plan is not None
        return cls(
            loader=loader,
            model_config=model_config,
            device_config=device_config,
            plan=admission.plan,
        )

    @staticmethod
    def _format_rejections(admission: StartupWeightLoadAdmission) -> str:
        return "; ".join(
            f"{rejection.code}: {rejection.message}"
            for rejection in admission.rejections
        )

    @property
    def state(self) -> StartupWeightLoadState:
        return self._state

    @property
    def is_deferred(self) -> bool:
        return self._state == StartupWeightLoadState.CAPTURE_READY

    def prepare(self) -> nn.Module:
        """Resolve sources, then build capture-safe storage.

        Source-based auto fallback happens before sentinel mutation.
        """

        if self._state != StartupWeightLoadState.CREATED:
            raise RuntimeError(
                f"Cannot prepare startup weights from state {self._state}"
            )
        self._state = StartupWeightLoadState.PREPARING
        model = self._loader.initialize_model_for_startup(
            model_config=self._model_config,
            device_config=self._device_config,
        )
        resolved_sources = self._loader.resolve_model_weights(
            self._model_config,
            model,
        )
        source_rejection = self._get_source_rejection(resolved_sources)
        if source_rejection is not None:
            if not self._fallback_to_serial:
                raise ValueError(source_rejection)
            logger.info(
                "Startup weight-load auto mode selected serial loading: %s",
                source_rejection,
            )
            model = self._loader.load_initialized_model_from_resolved_sources(
                model=model,
                model_config=self._model_config,
                resolved_sources=resolved_sources,
                target_device=torch.device(self._device_config.device),
            )
            self._model = model
            self._resolved_sources = resolved_sources
            self._state = StartupWeightLoadState.READY
            return model

        model = self._loader.prepare_model_for_capture(
            model=model,
            model_config=self._model_config,
        )
        self._model = model
        self._resolved_sources = resolved_sources
        self._capture_ready_at = time.perf_counter()
        self._state = StartupWeightLoadState.CAPTURE_READY
        logger.info(
            "Prepared capture-safe model in %.2f s",
            self._capture_ready_at - self._created_at,
        )
        return model

    @staticmethod
    def _get_source_rejection(
        resolved_sources: Tuple[DefaultModelLoader.ResolvedSource, ...],
    ) -> Optional[str]:
        if len(resolved_sources) != 1:
            return "secondary weights are not supported by startup overlap"
        if not resolved_sources[0].use_safetensors:
            return "startup overlap requires safetensors checkpoints"
        return None

    def start_prefetch(self) -> None:
        if self._state != StartupWeightLoadState.CAPTURE_READY:
            raise RuntimeError(
                f"Cannot prefetch startup weights from state {self._state}"
            )
        assert self._capture_ready_at is not None
        prefetch_started_at = time.perf_counter()
        self._prefetch_handle = self._loader.start_checkpoint_prefetch(
            self._resolved_sources,
            num_threads=self._plan.prefetch_num_threads,
        )
        self._prefetch_started_at = prefetch_started_at
        self._state = StartupWeightLoadState.PREFETCHING
        logger.info(
            "Started checkpoint prefetching %.2f s after capture-safe model prep",
            self._prefetch_started_at - self._capture_ready_at,
        )

    def finalize(self) -> StartupWeightLoadTimings:
        if self._state == StartupWeightLoadState.READY:
            assert self._timings is not None
            return self._timings
        if self._state != StartupWeightLoadState.PREFETCHING:
            raise RuntimeError(
                f"Cannot finalize startup weights from state {self._state}"
            )
        assert self._model is not None
        assert self._capture_ready_at is not None
        assert self._prefetch_started_at is not None
        self._state = StartupWeightLoadState.COMMITTING
        manifest = ModelStorageManifest.capture(self._model)
        prefetch_window_finished_at = time.perf_counter()
        startup_prefetch_active = self._prepare_prefetch_for_commit()
        commit_started_at = time.perf_counter()
        monkey_patch_vllm_parallel_state()
        self._loader.commit_model_weights(
            model=self._model,
            model_config=self._model_config,
            resolved_sources=self._resolved_sources,
            target_device=torch.device(self._device_config.device),
            startup_prefetch_active=startup_prefetch_active,
        )
        torch.cuda.synchronize()
        changed_names = manifest.changed_names(self._model)
        if changed_names:
            preview = ", ".join(changed_names[:8])
            raise RuntimeError(
                "Startup weight commit changed graph-visible tensor storage: "
                f"{preview}"
            )
        unchanged_names = manifest.unchanged_parameter_names(
            CAPTURE_SAFE_WEIGHT_SENTINEL
        )
        if unchanged_names:
            preview = ", ".join(unchanged_names[:8])
            raise RuntimeError(
                "Startup weight commit did not replace capture-safe dummy values: "
                f"{preview}"
            )
        monkey_patch_vllm_parallel_state(reverse=True)
        commit_finished_at = time.perf_counter()
        self._stop_prefetch()
        cleanup_finished_at = time.perf_counter()
        self._timings = StartupWeightLoadTimings(
            prepare_seconds=self._capture_ready_at - self._created_at,
            prefetch_start_delay_seconds=(
                self._prefetch_started_at - self._capture_ready_at
            ),
            prefetch_window_seconds=(
                prefetch_window_finished_at - self._prefetch_started_at
            ),
            commit_seconds=commit_finished_at - commit_started_at,
            prefetch_cleanup_seconds=(
                commit_started_at
                - prefetch_window_finished_at
                + cleanup_finished_at
                - commit_finished_at
            ),
            total_seconds=cleanup_finished_at - self._created_at,
        )
        self._state = StartupWeightLoadState.READY
        logger.info(
            "Load weight end. startup overlap profile=%s, phases: prepare %.2f s, "
            "prefetch start delay %.2f s, prefetch window %.2f s, commit %.2f s, "
            "prefetch cleanup %.2f s, load weight %.2f s, total %.2f s",
            self._plan.profile.value,
            self._timings.prepare_seconds,
            self._timings.prefetch_start_delay_seconds,
            self._timings.prefetch_window_seconds,
            self._timings.commit_seconds,
            self._timings.prefetch_cleanup_seconds,
            self._timings.weight_load_seconds,
            self._timings.total_seconds,
        )
        return self._timings

    def _prepare_prefetch_for_commit(self) -> bool:
        handle = self._prefetch_handle
        assert handle is not None

        if handle.done:
            handle.wait()
            self._report_prefetch_failure(falling_back=True)
            self._prefetch_handle = None
            return False

        try:
            handle.stop()
        except TimeoutError:
            # A blocking file read may outlive cooperative cancellation. Keep
            # reporting it as active so commit uses the conservative policy.
            if handle.done:
                handle.wait()
                self._report_prefetch_failure(falling_back=True)
                self._prefetch_handle = None
                return False

            self._prefetch_stop_timed_out = True
            self._report_prefetch_failure(falling_back=True)
            logger.warning(
                "Checkpoint prefetch did not stop before the weight commit; "
                "continuing weight loading while the cancelled checkpoint "
                "prefetch worker exits."
            )
            return True

        self._report_prefetch_failure(falling_back=True)
        self._prefetch_handle = None
        return False

    def _stop_prefetch(self) -> None:
        handle = self._prefetch_handle
        if handle is None:
            return
        assert self._prefetch_stop_timed_out
        if handle.done:
            handle.wait()
        else:
            # The first stop already cancelled the daemon; do not wait twice.
            logger.warning(
                "Checkpoint prefetch is still exiting after the weight commit; "
                "leaving the cancelled daemon prefetch worker to finish on "
                "its own."
            )
        self._report_prefetch_failure(falling_back=False)
        self._prefetch_handle = None

    def _report_prefetch_failure(self, *, falling_back: bool) -> None:
        handle = self._prefetch_handle
        if handle is None or not handle.failed or self._prefetch_failure_reported:
            return

        if handle.errors:
            path, error = handle.errors[0]
            failure_detail = (
                f"{len(handle.errors)} recorded failure(s), first: {path!r}: {error}"
            )
        else:
            failure_detail = "the background worker terminated before completion"
        action = (
            "falling back to normal weight loading"
            if falling_back
            else "real weight loading completed despite incomplete prefetch"
        )
        logger.warning(
            "Checkpoint prefetch was incomplete because %s; %s",
            failure_detail,
            action,
        )
        self._prefetch_failure_reported = True
