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
from sglang.srt.configs.model_config import ModelImpl
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
    get_device,
    get_exec,
    get_lora,
    get_model,
    get_spec,
)

if TYPE_CHECKING:
    from sglang.srt.configs.model_config import ModelConfig

logger = logging.getLogger(__name__)


def refresh_attention_weight_copies(attn_backend, decode_backend, decode_group) -> None:
    """Refresh backend-owned weight copies without replacing captured storage."""
    seen = set()
    for backend in (attn_backend, decode_backend, *(decode_group or ())):
        if backend is not None and id(backend) not in seen:
            seen.add(id(backend))
            backend.on_after_weight_load()


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
    GLM_MOE_DSA = "glm_moe_dsa"


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class StartupWeightLoadOptions:
    """One-shot admission inputs; the manager retains the resulting plan."""

    device: str
    is_cuda_platform: bool
    cuda_graph_enabled: bool
    moe_a2a_backend: str
    moe_runner_backend: str
    fp8_gemm_runner_backend: str
    offload_group_size: int
    has_lora: bool
    prefetch_num_threads: int

    @classmethod
    def from_published_config(cls) -> StartupWeightLoadOptions:
        """Read only configuration that affects deferred weight loading."""
        cuda_graph_config = get_exec().graph.cuda_graph_config
        is_cuda_platform = current_platform.is_cuda()
        cuda_graph_enabled = any(
            getattr(cuda_graph_config, phase).backend != Backend.DISABLED
            for phase in Phase.ALL
        )
        return cls(
            device=get_device().device,
            is_cuda_platform=is_cuda_platform,
            cuda_graph_enabled=cuda_graph_enabled,
            moe_a2a_backend=get_exec().moe.moe_a2a_backend,
            moe_runner_backend=get_exec().moe.moe_runner_backend,
            fp8_gemm_runner_backend=get_exec().kernel.fp8_gemm_runner_backend,
            offload_group_size=get_exec().offload.offload_group_size,
            has_lora=(
                get_lora().enable_lora
                or bool(get_lora().lora_paths)
                or get_spec().speculative_algorithm == "UNO"
            ),
            prefetch_num_threads=get_model().weight_loader_prefetch_num_threads,
        )


@dataclasses.dataclass(frozen=True, slots=True)
class StartupWeightLoadRejection:
    code: str
    message: str


@dataclasses.dataclass(frozen=True, slots=True)
class StartupWeightLoadPlan:
    """Admitted profile and prefetch settings, before source resolution."""

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
class _StartupWeightLoadProfileSpec:
    profile: StartupWeightLoadProfile
    model_classes: dict[str, Callable[[], type]]
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


def _checkpoint_block_fp8_size(model_config: ModelConfig) -> tuple:
    quantization_config = getattr(model_config.hf_config, "quantization_config", None)
    if (
        model_config.quantization != "fp8"
        or not isinstance(quantization_config, dict)
        or quantization_config.get("quant_method") != "fp8"
    ):
        return ()
    block_size = quantization_config.get("weight_block_size")
    return (
        tuple(block_size)
        if isinstance(block_size, (list, tuple)) and len(block_size) == 2
        else ()
    )


def _requires_fp8_scale_conversion(block_size: tuple) -> bool:
    from sglang.srt.model_loader.utils import should_deepgemm_weight_requant_ue8m0

    return block_size == (128, 128) and should_deepgemm_weight_requant_ue8m0(
        weight_block_size=list(block_size)
    )


def _weight_format_rules(
    model_config: ModelConfig,
    options: StartupWeightLoadOptions,
) -> Tuple[Tuple[str, bool, str], ...]:
    if model_config.quantization is None:
        return ()
    block_size = _checkpoint_block_fp8_size(model_config)
    # Online and per-tensor FP8 processing replace parameters and scales.
    if not block_size:
        return (
            (
                "quantization",
                True,
                "startup overlap requires unquantized or serialized block-FP8 weights",
            ),
        )

    # Block-FP8 linear storage stays intact unless scales are converted to UE8M0.
    converts_scales = _requires_fp8_scale_conversion(block_size)
    uses_deepgemm = options.fp8_gemm_runner_backend == "deep_gemm"
    if converts_scales and options.fp8_gemm_runner_backend == "auto":
        from sglang.srt.layers.quantization.fp8_utils import (
            _dispatch_auto_backend,
            deepgemm_w8a8_block_fp8_linear_with_fallback,
        )

        uses_deepgemm = (
            _dispatch_auto_backend() is deepgemm_w8a8_block_fp8_linear_with_fallback
        )
    return (
        (
            "fp8_gemm_backend",
            converts_scales and uses_deepgemm and model_config.dtype == torch.bfloat16,
            "DeepGEMM UE8M0 scale conversion is not reload-safe during startup overlap",
        ),
    )


def _moe_weight_rules(
    model_config: ModelConfig,
    options: StartupWeightLoadOptions,
) -> Tuple[Tuple[str, bool, str], ...]:
    from sglang.srt.layers.moe.utils import (
        MoeA2ABackend,
        resolve_moe_runner_backend,
    )

    runner = resolve_moe_runner_backend(options.moe_runner_backend)
    rules = (
        (
            "lora",
            options.has_lora,
            "LoRA replaces MoE checkpoint parameter paths before startup commit",
        ),
    )
    block_size = _checkpoint_block_fp8_size(model_config)
    if not block_size:
        return rules

    a2a = MoeA2ABackend(options.moe_a2a_backend)
    converts_scales = False
    if _requires_fp8_scale_conversion(block_size):
        from sglang.srt.layers.quantization.fp8 import Fp8MoEMethod

        converts_scales = Fp8MoEMethod.is_deepgemm_moe_runner_backend_enabled(
            runner, a2a
        )
    return rules + (
        (
            "moe_runner_backend",
            converts_scales,
            "MoE UE8M0 scale conversion is not reload-safe during startup overlap",
        ),
        (
            "moe_a2a_backend",
            a2a.is_megamoe(),
            "MegaMoE packed-weight caches are not refreshed during startup commit",
        ),
    )


def _validate_native_dense(
    model_config: ModelConfig,
    options: StartupWeightLoadOptions,
) -> Tuple[StartupWeightLoadRejection, ...]:
    return _rejections_from_rules(_weight_format_rules(model_config, options))


def _validate_moe(
    model_config: ModelConfig,
    options: StartupWeightLoadOptions,
) -> Tuple[StartupWeightLoadRejection, ...]:
    return _rejections_from_rules(
        _weight_format_rules(model_config, options)
        + _moe_weight_rules(model_config, options)
    )


# Profiles describe weight-loading and post-processing paths, not tested configs.
_STARTUP_WEIGHT_LOAD_PROFILE_SPECS = (
    _StartupWeightLoadProfileSpec(
        profile=StartupWeightLoadProfile.NATIVE_DENSE,
        model_classes={
            "LlamaForCausalLM": _resolve_llama_model_class,
            "Qwen2ForCausalLM": _resolve_qwen2_model_class,
            "Qwen3ForCausalLM": _resolve_qwen3_model_class,
        },
        validate=_validate_native_dense,
    ),
    _StartupWeightLoadProfileSpec(
        profile=StartupWeightLoadProfile.QWEN3_5_HYBRID_VLM,
        model_classes={"Qwen3_5ForConditionalGeneration": _resolve_qwen3_5_model_class},
        validate=_validate_native_dense,
    ),
    _StartupWeightLoadProfileSpec(
        profile=StartupWeightLoadProfile.QWEN3_5_MOE_HYBRID_VLM,
        model_classes={
            "Qwen3_5MoeForConditionalGeneration": _resolve_qwen3_5_moe_model_class,
        },
        validate=_validate_moe,
    ),
    _StartupWeightLoadProfileSpec(
        profile=StartupWeightLoadProfile.QWEN3_MOE_EP,
        model_classes={"Qwen3MoeForCausalLM": _resolve_qwen3_moe_model_class},
        validate=_validate_moe,
    ),
    _StartupWeightLoadProfileSpec(
        profile=StartupWeightLoadProfile.GLM_MOE_DSA,
        model_classes={"GlmMoeDsaForCausalLM": _resolve_glm_moe_dsa_model_class},
        validate=_validate_moe,
    ),
)


_STARTUP_WEIGHT_LOAD_PROFILE_SPEC_BY_PROFILE = {
    spec.profile: spec for spec in _STARTUP_WEIGHT_LOAD_PROFILE_SPECS
}


def _get_startup_weight_load_profile(
    model_class: type,
) -> Optional[StartupWeightLoadProfile]:
    # Empty native aliases (e.g. Mistral -> Llama) share the entire implementation.
    # Any method, class setting, extra base or metaclass change requires review.
    class_metadata = {
        "__module__",
        "__doc__",
        "__qualname__",
        "__firstlineno__",
        "__static_attributes__",
    }
    implementations = [model_class]
    while (
        len(model_class.__bases__) == 1
        and type(model_class) is type(model_class.__bases__[0])
        and not model_class.__dict__.keys() - class_metadata
        and not model_class.__dict__.get("__static_attributes__", ())
    ):
        model_class = model_class.__bases__[0]
        implementations.append(model_class)
    for candidate in implementations:
        for spec in _STARTUP_WEIGHT_LOAD_PROFILE_SPECS:
            resolve = spec.model_classes.get(candidate.__name__)
            if resolve is not None and candidate is resolve():
                return spec.profile
    return None


def _get_native_model_class(architectures) -> Optional[type]:
    from sglang.srt.models.registry import ModelRegistry, import_model_classes

    native_classes = import_model_classes("sglang.srt.models", strict=False)
    # Match registry priority without invoking the Transformers/remote-code fallback.
    for architecture in architectures:
        model_class = ModelRegistry.models.get(architecture)
        if model_class is not None:
            return (
                model_class if model_class is native_classes.get(architecture) else None
            )
    return None


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
    constants: Tuple[Tuple[str, Optional[float]], ...] = ()

    @classmethod
    def capture(cls, model: nn.Module) -> ModelStorageManifest:
        entries = []
        constants = []
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
            named_constants = getattr(
                module, "named_startup_weight_load_constants", None
            )
            if named_constants is not None:
                constants.extend(
                    (f"constant:{module_name}.{name}", value)
                    for name, value in named_constants()
                )
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
        return cls(
            tensors=tuple(sorted(entries, key=lambda entry: entry[0])),
            constants=tuple(sorted(constants)),
        )

    def changed_names(self, model: nn.Module) -> Tuple[str, ...]:
        before = dict(self.tensors)
        current = ModelStorageManifest.capture(model)
        after = dict(current.tensors)
        changed = [
            name
            for name in sorted(before.keys() | after.keys())
            if name not in before
            or name not in after
            or not before[name].matches(after[name])
        ]
        before_constants = dict(self.constants)
        after_constants = dict(current.constants)
        changed.extend(
            name
            for name in before_constants.keys() | after_constants.keys()
            if name not in before_constants
            or name not in after_constants
            or before_constants[name] != after_constants[name]
        )
        return tuple(sorted(changed))

    def unchanged_parameter_names(self, value: float) -> Tuple[str, ...]:
        """Return required floating-point parameters still equal to ``value``.

        Optional ``_skip_weight_check`` parameters keep their constructor
        values. Every other floating-point parameter must replace the capture
        sentinel; buffers are excluded because they are never overwritten.
        """
        checks_by_device = {}
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
            checks_by_device.setdefault(tensor.device, []).append(
                (name, torch.all(tensor == value))
            )

        unchanged_names = []
        for checks in checks_by_device.values():
            names, values = zip(*checks)
            unchanged = torch.stack(values).cpu().tolist()
            unchanged_names.extend(
                name for name, is_unchanged in zip(names, unchanged) if is_unchanged
            )
        return tuple(sorted(unchanged_names))


def evaluate_startup_weight_load_admission(
    *,
    loader,
    model_config: ModelConfig,
    load_config: LoadConfig,
    options: StartupWeightLoadOptions,
) -> StartupWeightLoadAdmission:
    """Reject paths that cannot load real weights after capture.

    Ordinary model, dtype, hardware and parallelism validation stays in the
    normal startup pipeline; these checks concern the changed loading order.
    """

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
            "loader",
            type(loader) is not DefaultModelLoader,
            "DefaultModelLoader only",
        ),
        (
            "load_format",
            load_config.load_format not in (LoadFormat.AUTO, LoadFormat.SAFETENSORS),
            "load format must be auto or safetensors",
        ),
        (
            "layer_group_offload",
            options.offload_group_size > 0,
            "layer-group offloading moves weight storage before weight commit",
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

    if rejections:
        return StartupWeightLoadAdmission(plan=None, rejections=tuple(rejections))

    # Resolve only native registry entries until admission passes. Auto fallback
    # must not import remote model code or mutate the effective configuration.
    model_class = None
    if architectures and model_config.model_impl in (ModelImpl.AUTO, ModelImpl.SGLANG):
        model_class = _get_native_model_class(architectures)
    if model_class is None:
        return StartupWeightLoadAdmission(
            plan=None,
            rejections=(
                StartupWeightLoadRejection(
                    code="model_implementation",
                    message="the native SGLang model implementation is required",
                ),
            ),
        )
    profile = _get_startup_weight_load_profile(model_class)
    if profile is None:
        return StartupWeightLoadAdmission(
            plan=None,
            rejections=(
                StartupWeightLoadRejection(
                    code="architecture",
                    message=f"model implementation {model_class.__name__!r} has no startup overlap loading path",
                ),
            ),
        )
    rejections.extend(
        _get_profile_rejections(
            profile=profile,
            model_config=model_config,
            options=options,
        )
    )
    if not rejections:
        resolved_model_class, _ = get_model_architecture(model_config)
        if resolved_model_class is not model_class:
            rejections.append(
                StartupWeightLoadRejection(
                    code="model_implementation",
                    message="model resolution selected a different implementation",
                )
            )
    if rejections:
        return StartupWeightLoadAdmission(plan=None, rejections=tuple(rejections))
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
    def create_from_published_config(
        cls,
        *,
        loader,
        model_config: ModelConfig,
        load_config: LoadConfig,
        device_config: DeviceConfig,
        is_draft_worker: bool,
    ) -> Optional[StartupWeightLoadManager]:
        """Create a manager, or return ``None`` when auto selects serial."""
        # The scheduler only owns the target runner's prefetch/commit lifecycle.
        if is_draft_worker:
            return None
        options = StartupWeightLoadOptions.from_published_config()
        if get_model().startup_weight_load_mode == "auto":
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
            target_device=torch.device(self._device_config.device),
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
        if not resolved_sources or not all(
            source.use_safetensors for source in resolved_sources
        ):
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

    def finalize(
        self, *, after_weight_load: Optional[Callable[[], None]] = None
    ) -> Optional[StartupWeightLoadTimings]:
        """Return overlap timings, or None after serial fallback."""
        if self._state == StartupWeightLoadState.READY:
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
        if after_weight_load is not None:
            after_weight_load()
        torch.cuda.synchronize()
        changed_names = manifest.changed_names(self._model)
        if changed_names:
            preview = ", ".join(changed_names[:8])
            raise RuntimeError(
                f"Startup weight commit changed graph-visible storage or constants: {preview}"
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
