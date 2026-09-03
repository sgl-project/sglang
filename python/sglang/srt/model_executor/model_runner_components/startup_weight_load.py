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


class StartupWeightLoadState(str, enum.Enum):
    CREATED = "created"
    PREPARING = "preparing"
    CAPTURE_READY = "capture_ready"
    PREFETCHING = "prefetching"
    COMMITTING = "committing"
    READY = "ready"


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class StartupWeightLoadOptions:
    device: str
    is_cuda_platform: bool
    cuda_graph_enabled: bool
    prefill_cuda_graph_backend: Backend
    is_draft_worker: bool
    speculative_algorithm: Optional[str]
    cpu_offload_gb: int
    offload_group_size: int
    enable_weights_cpu_backup: bool
    enable_lora: bool
    has_lora_paths: bool
    weight_loader_disable_mmap: bool
    weight_loader_drop_cache_after_load: bool
    has_custom_weight_loader: bool
    enable_torch_compile: bool
    prefetch_num_threads: int

    @classmethod
    def from_published_config(
        cls,
        *,
        is_draft_worker: bool,
    ) -> StartupWeightLoadOptions:
        """Everything this needs is a published leaf; nothing comes off a record.

        `is_draft_worker` is the exception and travels as an argument: it is
        this runner's role, not the process's configuration.
        """
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
            cpu_offload_gb=get_exec().offload.cpu_offload_gb,
            offload_group_size=get_exec().offload.offload_group_size,
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
        """Return floating parameters still entirely equal to ``value``.

        The sentinel check detects a parameter that the normal checkpoint path
        did not replace. Packed integral parameters are deliberately excluded:
        zero is a valid packed value, and loader invocation alone cannot prove
        that every shard or expert region was written. Their value correctness
        remains the responsibility of the unchanged serial loader contract.
        Buffers and tensors marked ``_skip_weight_check`` are also excluded.
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


class StartupWeightLoadManager:
    """Coordinate CPU staging with capture and post-capture weight commit."""

    def __init__(
        self,
        *,
        loader: DefaultModelLoader,
        model_config: ModelConfig,
        device_config: DeviceConfig,
        options: StartupWeightLoadOptions,
    ) -> None:
        self._loader = loader
        self._model_config = model_config
        self._device_config = device_config
        self._options = options
        self._model: Optional[nn.Module] = None
        self._resolved_sources: Tuple[DefaultModelLoader.ResolvedSource, ...] = ()
        self._prefetch_handle: Optional[CheckpointFilePrefetchHandle] = None
        self._state = StartupWeightLoadState.CREATED
        self._created_at = time.perf_counter()
        self._capture_ready_at: Optional[float] = None
        self._prefetch_started_at: Optional[float] = None
        self._prefetch_failure_reported = False

    @classmethod
    def create_from_published_config(
        cls,
        *,
        loader,
        model_config: ModelConfig,
        load_config: LoadConfig,
        device_config: DeviceConfig,
        is_draft_worker: bool,
    ) -> StartupWeightLoadManager:
        """Build a manager from the published configuration.

        Callers on the model-loading path only decide *whether* to overlap; the
        knowledge of which config leaves matter, and every support rule, stays
        in this module.
        """
        return cls.create(
            loader=loader,
            model_config=model_config,
            load_config=load_config,
            device_config=device_config,
            options=StartupWeightLoadOptions.from_published_config(
                is_draft_worker=is_draft_worker,
            ),
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
        unsupported_reason = cls._get_unsupported_reason(
            loader=loader,
            model_config=model_config,
            load_config=load_config,
            options=options,
        )
        if unsupported_reason is not None:
            raise ValueError(
                "--startup-weight-load-mode=overlap is not supported: "
                f"{unsupported_reason}"
            )
        return cls(
            loader=loader,
            model_config=model_config,
            device_config=device_config,
            options=options,
        )

    @staticmethod
    def _get_unsupported_reason(
        *,
        loader,
        model_config: ModelConfig,
        load_config: LoadConfig,
        options: StartupWeightLoadOptions,
    ) -> Optional[str]:
        basic_rules = (
            (not options.is_cuda_platform or options.device != "cuda", "CUDA only"),
            (not options.cuda_graph_enabled, "CUDA graph capture is disabled"),
            (
                options.prefill_cuda_graph_backend == Backend.TC_PIECEWISE,
                "tc_piecewise prefill CUDA graphs are not supported",
            ),
            (
                load_config.load_format
                not in (LoadFormat.AUTO, LoadFormat.SAFETENSORS),
                "load format must be auto or safetensors",
            ),
            (options.is_draft_worker, "draft workers are not supported"),
            (
                load_config.draft_model_idx is not None,
                "draft model loading is unsupported",
            ),
            (
                options.speculative_algorithm is not None,
                "speculative decoding is not supported",
            ),
            (options.cpu_offload_gb > 0, "CPU offload is not supported"),
            (
                options.offload_group_size > 0,
                "layer-group offloading is not supported",
            ),
            (
                options.enable_weights_cpu_backup,
                "CPU weight backup is not supported",
            ),
            (
                options.enable_lora or options.has_lora_paths,
                "LoRA is not supported",
            ),
            (
                options.weight_loader_disable_mmap,
                "safetensors mmap must be enabled",
            ),
            (
                options.weight_loader_drop_cache_after_load,
                "dropping the page cache during load is not supported",
            ),
            (
                options.has_custom_weight_loader,
                "custom weight loaders are not supported",
            ),
            (options.enable_torch_compile, "torch.compile is not supported"),
        )
        unsupported_reason = next(
            (reason for unsupported, reason in basic_rules if unsupported),
            None,
        )
        if unsupported_reason is not None:
            return unsupported_reason

        supports_overlap = getattr(loader, "supports_startup_weight_load_overlap", None)
        if supports_overlap is None or not supports_overlap(model_config):
            return f"{type(loader).__name__} does not support split startup loading"
        return None

    @property
    def state(self) -> StartupWeightLoadState:
        return self._state

    def prepare(self) -> nn.Module:
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
        if len(resolved_sources) != 1:
            raise ValueError(
                "Startup weight-loading overlap does not support secondary weights"
            )
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

    def start_prefetch(self) -> None:
        if self._state != StartupWeightLoadState.CAPTURE_READY:
            raise RuntimeError(
                f"Cannot prefetch startup weights from state {self._state}"
            )
        assert self._capture_ready_at is not None
        prefetch_started_at = time.perf_counter()
        self._prefetch_handle = self._loader.start_checkpoint_prefetch(
            self._resolved_sources,
            num_threads=self._options.prefetch_num_threads,
        )
        self._prefetch_started_at = prefetch_started_at
        self._state = StartupWeightLoadState.PREFETCHING
        logger.info(
            "Started checkpoint prefetching %.2f s after capture-safe model prep",
            self._prefetch_started_at - self._capture_ready_at,
        )

    def finalize(self) -> None:
        if self._state == StartupWeightLoadState.READY:
            return
        if self._state != StartupWeightLoadState.PREFETCHING:
            raise RuntimeError(
                f"Cannot finalize startup weights from state {self._state}"
            )
        assert self._model is not None
        assert self._capture_ready_at is not None
        assert self._prefetch_started_at is not None
        self._state = StartupWeightLoadState.COMMITTING
        manifest = ModelStorageManifest.capture(self._model)
        startup_prefetch_active = self._prepare_prefetch_for_commit()
        commit_started_at = time.perf_counter()
        monkey_patch_vllm_parallel_state()
        try:
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
                    "Startup weight commit did not populate capture-safe parameters: "
                    f"{preview}"
                )
        finally:
            monkey_patch_vllm_parallel_state(reverse=True)
        self._stop_prefetch()
        self._state = StartupWeightLoadState.READY
        logger.info(
            "Load weight end. Committed real weights after CUDA graph capture in %.2f s "
            "(capture overlap window %.2f s, startup overlap total %.2f s)",
            time.perf_counter() - commit_started_at,
            commit_started_at - self._prefetch_started_at,
            time.perf_counter() - self._created_at,
        )

    def _prepare_prefetch_for_commit(self) -> bool:
        assert self._prefetch_handle is not None
        if not self._prefetch_handle.failed:
            return not self._prefetch_handle.done

        self._prefetch_handle.stop()
        self._report_prefetch_failure(falling_back=True)
        return False

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
