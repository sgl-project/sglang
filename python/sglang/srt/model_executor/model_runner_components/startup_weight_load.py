from __future__ import annotations

import dataclasses
import enum
import logging
import time
from typing import TYPE_CHECKING, Optional, Tuple

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


_NATIVE_DENSE_ARCHITECTURES = frozenset(
    {
        "LlamaForCausalLM",
        "Qwen2ForCausalLM",
        "Qwen3ForCausalLM",
    }
)
_QWEN3_5_HYBRID_VLM_ARCHITECTURES = frozenset(
    {
        # Qwen3.6 dense checkpoints retain the Qwen3.5 implementation
        # architecture in config.json.
        "Qwen3_5ForConditionalGeneration",
    }
)
_QWEN3_5_MOE_HYBRID_VLM_ARCHITECTURES = frozenset(
    {"Qwen3_5MoeForConditionalGeneration"}
)
_QWEN3_MOE_ARCHITECTURES = frozenset({"Qwen3MoeForCausalLM"})
_SUPPORTED_DTYPES = frozenset({torch.float16, torch.bfloat16})


def _get_canonical_model_class(architecture: str):
    if architecture == "LlamaForCausalLM":
        from sglang.srt.models.llama import LlamaForCausalLM

        return LlamaForCausalLM
    if architecture == "Qwen2ForCausalLM":
        from sglang.srt.models.qwen2 import Qwen2ForCausalLM

        return Qwen2ForCausalLM
    if architecture == "Qwen3ForCausalLM":
        from sglang.srt.models.qwen3 import Qwen3ForCausalLM

        return Qwen3ForCausalLM
    if architecture == "Qwen3_5ForConditionalGeneration":
        from sglang.srt.models.qwen3_5 import Qwen3_5ForConditionalGeneration

        return Qwen3_5ForConditionalGeneration
    if architecture == "Qwen3_5MoeForConditionalGeneration":
        from sglang.srt.models.qwen3_5 import Qwen3_5MoeForConditionalGeneration

        return Qwen3_5MoeForConditionalGeneration
    if architecture == "Qwen3MoeForCausalLM":
        from sglang.srt.models.qwen3_moe import Qwen3MoeForCausalLM

        return Qwen3MoeForCausalLM
    raise ValueError(f"Unsupported startup-overlap architecture: {architecture}")


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


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class StartupWeightLoadOptions:
    device: str
    is_cuda_platform: bool
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
        cuda_graph_enabled = any(
            getattr(cuda_graph_config, phase).backend != Backend.DISABLED
            for phase in Phase.ALL
        )
        return cls(
            device=get_device().device,
            is_cuda_platform=current_platform.is_cuda(),
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
    """Config-level plan; resolved checkpoint sources are verified in prepare."""

    profile: StartupWeightLoadProfile
    prefetch_num_threads: int


@dataclasses.dataclass(frozen=True, slots=True)
class StartupWeightLoadAdmission:
    plan: Optional[StartupWeightLoadPlan]
    rejections: Tuple[StartupWeightLoadRejection, ...]

    @property
    def supported(self) -> bool:
        return self.plan is not None


def _get_startup_weight_load_profile(
    architecture: Optional[str],
) -> Optional[StartupWeightLoadProfile]:
    if architecture in _NATIVE_DENSE_ARCHITECTURES:
        return StartupWeightLoadProfile.NATIVE_DENSE
    if architecture in _QWEN3_5_HYBRID_VLM_ARCHITECTURES:
        return StartupWeightLoadProfile.QWEN3_5_HYBRID_VLM
    if architecture in _QWEN3_5_MOE_HYBRID_VLM_ARCHITECTURES:
        return StartupWeightLoadProfile.QWEN3_5_MOE_HYBRID_VLM
    if architecture in _QWEN3_MOE_ARCHITECTURES:
        return StartupWeightLoadProfile.QWEN3_MOE_EP
    return None


def _get_profile_rejections(
    *,
    profile: StartupWeightLoadProfile,
    model_config: ModelConfig,
    options: StartupWeightLoadOptions,
) -> Tuple[StartupWeightLoadRejection, ...]:
    linear_attn_decode_backend = (
        options.linear_attn_decode_backend or options.linear_attn_backend
    )
    linear_attn_prefill_backend = (
        options.linear_attn_prefill_backend or options.linear_attn_backend
    )

    def ep_moe_rules(*, family: str, tp_size: int, ep_size: int):
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

    if profile == StartupWeightLoadProfile.NATIVE_DENSE:
        rules = (
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
    elif profile == StartupWeightLoadProfile.QWEN3_5_HYBRID_VLM:
        rules = (
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
    elif profile == StartupWeightLoadProfile.QWEN3_5_MOE_HYBRID_VLM:
        rules = ep_moe_rules(
            family="Qwen3.5 MoE hybrid VLM",
            tp_size=2,
            ep_size=2,
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
    elif profile == StartupWeightLoadProfile.QWEN3_MOE_EP:
        rules = ep_moe_rules(family="Qwen3 MoE", tp_size=2, ep_size=2) + (
            (
                "multimodal",
                model_config.is_multimodal,
                "multimodal models are not supported",
            ),
        )
    else:
        raise ValueError(f"Unknown startup weight-load profile: {profile}")

    return tuple(
        StartupWeightLoadRejection(code=code, message=message)
        for code, rejected, message in rules
        if rejected
    )


@dataclasses.dataclass(frozen=True, slots=True)
class StartupWeightLoadTimings:
    """Wall times until real weights are ready under overlap.

    ``weight_load_seconds`` is exclusive phase attribution for the existing
    public ``load_weight`` metric, not end-to-end wall time. ``total_seconds``
    is diagnostic critical-path elapsed time and, together with
    ``prefetch_window_seconds``, intentionally overlaps the separately
    reported CUDA graph and KV-cache phases. The prefetch window measures the
    opportunity for overlap; it does not imply that the worker stayed active
    for the whole interval.
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
        # CUDA graphs may also close over plain tensors produced by model
        # post-load code. A module hook enumerates only its local derived
        # tensors; walking hooks explicitly avoids a recursive attribute scan
        # that would mix in runtime state. This manifest verifies storage, while
        # each post-load implementation remains responsible for refreshing the
        # real contents in place. Hooks must not allocate replacement tensors;
        # they return the same graph-visible tensor objects on every call.
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
        """Return floating-point parameters still entirely equal to ``value``.

        This is the capture-sentinel check, and it is deliberately strict: every
        floating-point parameter must be rewritten by ``model.load_weights()``.
        A model that keeps an ``__init__``-computed floating-point parameter with
        no checkpoint entry will fail startup here rather than silently serve the
        sentinel, so this doubles as the admission gate for widening
        ``_NATIVE_DENSE_ARCHITECTURES``. Buffers are excluded because
        ``initialize_capture_safe_weights`` never overwrites them.
        """
        names = []
        checks = []
        seen_tensor_ids = set()
        for name, metadata in self.tensors:
            tensor = metadata.tensor
            if (
                not name.startswith("parameter:")
                or not torch.is_floating_point(tensor)
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
    # NOTE(2026-08): Expand this matrix only with storage-stability,
    # capture-sentinel, and startup-correctness coverage for the new profile.
    # These checks depend on resolved loader/model state, so ServerArgs owns
    # mode selection but not support policy.
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

    # Resolve/import the implementation only after cheap, side-effect-free
    # preflight passes. Future auto mode can therefore fall back to serial for
    # rejected configurations without remote-code imports or config mutation.
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
    """Coordinate native CPU staging with capture and post-capture commit.

    Model commit and validation failures after capture-safe preparation are
    terminal startup failures: the manager fails closed instead of rolling a
    mutated model back for an in-process serial retry. Background page-cache
    prefetch is best-effort and can fall back to the normal checkpoint reader.
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
        """Build a manager straight from ``ServerArgs`` when admitted.

        Callers on the model-loading path only decide *whether* to overlap; the
        knowledge of which server arguments matter, and every support rule,
        stays in this module.
        """
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
        """Whether real-weight loading was deferred past graph capture."""
        return self._state == StartupWeightLoadState.CAPTURE_READY

    def prepare(self) -> nn.Module:
        """Resolve sources and build capture-safe storage.

        Source-dependent rejection stays before ``prepare_model_for_capture``
        mutates parameters with sentinel values.
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
        commit_started_at = time.perf_counter()
        manifest = ModelStorageManifest.capture(self._model)
        startup_prefetch_active = self._prepare_prefetch_for_commit()
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
            prefetch_window_seconds=commit_started_at - self._prefetch_started_at,
            commit_seconds=commit_finished_at - commit_started_at,
            prefetch_cleanup_seconds=cleanup_finished_at - commit_finished_at,
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
        assert self._prefetch_handle is not None
        if not self._prefetch_handle.failed:
            return not self._prefetch_handle.done

        self._prefetch_handle.cancel()
        self._report_prefetch_failure(falling_back=True)
        return not self._prefetch_handle.done

    def _stop_prefetch(self) -> None:
        if self._prefetch_handle is None:
            return
        try:
            if self._prefetch_handle.done:
                self._prefetch_handle.wait()
            else:
                self._prefetch_handle.stop()
        except TimeoutError:
            # Only reached after the real weights are committed and validated,
            # so a stager that outlives its stop timeout must not fail an
            # otherwise-successful startup. The worker is a daemon thread and
            # cannot keep the process alive.
            logger.warning(
                "Checkpoint prefetch did not stop within its timeout after the "
                "weight commit; leaving the daemon stager to exit on its own."
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
            else "real weight loading completed despite incomplete staging"
        )
        logger.warning(
            "Checkpoint prefetch was incomplete because %s; %s",
            failure_detail,
            action,
        )
        self._prefetch_failure_reported = True
