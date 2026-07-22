from __future__ import annotations

import dataclasses
import enum
import logging
import time
from typing import TYPE_CHECKING, Literal, Optional, Tuple

import torch
from torch import nn

from sglang.srt.configs.device_config import DeviceConfig
from sglang.srt.configs.load_config import LoadConfig, LoadFormat
from sglang.srt.distributed.parallel_state import patched_vllm_parallel_state
from sglang.srt.model_executor.cuda_graph_config import Backend, Phase
from sglang.srt.model_loader.loader import DefaultModelLoader
from sglang.srt.model_loader.utils import get_model_architecture
from sglang.srt.model_loader.weight_utils import CheckpointFilePrefetchHandle
from sglang.srt.platforms import current_platform

if TYPE_CHECKING:
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


_SUPPORTED_ARCHITECTURES = frozenset(
    {
        "LlamaForCausalLM",
        "Qwen2ForCausalLM",
        "Qwen3ForCausalLM",
    }
)
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
    raise ValueError(f"Unsupported startup-overlap architecture: {architecture}")


class StartupWeightLoadState(str, enum.Enum):
    CREATED = "created"
    PREPARING = "preparing"
    CAPTURE_READY = "capture_ready"
    PREFETCHING = "prefetching"
    COMMITTING = "committing"
    READY = "ready"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class StartupWeightLoadOptions:
    requested_mode: Literal["serial", "overlap"]
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
    cpu_offload_gb: int
    offload_group_size: int
    enable_memory_saver: bool
    enable_weights_cpu_backup: bool
    torchao_config: str
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
        cuda_graph_config = server_args.cuda_graph_config
        cuda_graph_enabled = any(
            getattr(cuda_graph_config, phase).backend != Backend.DISABLED
            for phase in Phase.ALL
        )
        return cls(
            requested_mode=server_args.startup_weight_load_mode,
            device=server_args.device,
            is_cuda_platform=current_platform.is_cuda(),
            cuda_graph_enabled=cuda_graph_enabled,
            prefill_cuda_graph_backend=cuda_graph_config.prefill.backend,
            is_draft_worker=is_draft_worker,
            speculative_algorithm=server_args.speculative_algorithm,
            tp_size=server_args.tp_size,
            attn_cp_size=server_args.attn_cp_size,
            dcp_size=server_args.dcp_size,
            pp_size=server_args.pp_size,
            dp_size=server_args.dp_size,
            ep_size=server_args.ep_size,
            cpu_offload_gb=server_args.cpu_offload_gb,
            offload_group_size=server_args.offload_group_size,
            enable_memory_saver=server_args.enable_memory_saver,
            enable_weights_cpu_backup=server_args.enable_weights_cpu_backup,
            torchao_config=server_args.torchao_config,
            enable_lora=server_args.enable_lora,
            has_lora_paths=bool(server_args.lora_paths),
            weight_loader_disable_mmap=server_args.weight_loader_disable_mmap,
            weight_loader_drop_cache_after_load=(
                server_args.weight_loader_drop_cache_after_load
            ),
            has_custom_weight_loader=bool(server_args.custom_weight_loader),
            enable_torch_compile=server_args.enable_torch_compile,
            prefetch_num_threads=server_args.weight_loader_prefetch_num_threads,
        )


@dataclasses.dataclass(frozen=True, slots=True)
class TensorStorageMetadata:
    object_id: int
    data_ptr: int
    shape: Tuple[int, ...]
    stride: Tuple[int, ...]
    dtype: torch.dtype
    device: torch.device
    storage_offset: int

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor) -> TensorStorageMetadata:
        return cls(
            object_id=id(tensor),
            data_ptr=tensor.data_ptr(),
            shape=tuple(tensor.shape),
            stride=tuple(tensor.stride()),
            dtype=tensor.dtype,
            device=tensor.device,
            storage_offset=tensor.storage_offset(),
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
        return cls(tensors=tuple(sorted(entries)))

    def changed_names(self, model: nn.Module) -> Tuple[str, ...]:
        before = dict(self.tensors)
        after = dict(ModelStorageManifest.capture(model).tensors)
        return tuple(
            name
            for name in sorted(before.keys() | after.keys())
            if before.get(name) != after.get(name)
        )


class StartupWeightLoadManager:
    """Coordinate native CPU staging with capture and post-capture commit."""

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

    @classmethod
    def create_if_enabled(
        cls,
        *,
        loader,
        model_config: ModelConfig,
        load_config: LoadConfig,
        device_config: DeviceConfig,
        options: StartupWeightLoadOptions,
    ) -> Optional[StartupWeightLoadManager]:
        if options.requested_mode == "serial":
            return None
        if options.requested_mode != "overlap":
            raise ValueError(
                f"Unknown startup weight load mode: {options.requested_mode}"
            )
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
        architectures = tuple(model_config.hf_config.architectures or ())
        basic_rules = (
            (not options.is_cuda_platform or options.device != "cuda", "CUDA only"),
            (not options.cuda_graph_enabled, "CUDA graph capture is disabled"),
            (
                options.prefill_cuda_graph_backend == Backend.TC_PIECEWISE,
                "tc_piecewise prefill CUDA graphs are not supported",
            ),
            (type(loader) is not DefaultModelLoader, "DefaultModelLoader only"),
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
            (options.tp_size not in (1, 2), "only TP1 and TP2 are supported"),
            (
                options.attn_cp_size != 1,
                "attention context parallelism is not supported",
            ),
            (
                options.dcp_size != 1,
                "decode context parallelism is not supported",
            ),
            (options.pp_size != 1, "pipeline parallelism is not supported"),
            (options.dp_size != 1, "data parallelism is not supported"),
            (options.ep_size != 1, "expert parallelism is not supported"),
            (options.cpu_offload_gb > 0, "CPU offload is not supported"),
            (
                options.offload_group_size > 0,
                "layer-group offloading is not supported",
            ),
            (options.enable_memory_saver, "memory saver is not supported"),
            (
                options.enable_weights_cpu_backup,
                "CPU weight backup is not supported",
            ),
            (bool(options.torchao_config), "TorchAO is not supported"),
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

        model_rules = (
            (model_config.dtype not in _SUPPORTED_DTYPES, "FP16 or BF16 only"),
            (model_config.quantization is not None, "quantization is not supported"),
            (
                bool(getattr(model_config, "modelopt_quant", False)),
                "ModelOpt is not supported",
            ),
            (model_config.is_multimodal, "multimodal models are not supported"),
            (not model_config.is_generation, "generation models only"),
            (
                len(architectures) != 1
                or architectures[0] not in _SUPPORTED_ARCHITECTURES,
                "model architecture is not in the startup-overlap allowlist",
            ),
        )
        unsupported_reason = next(
            (reason for unsupported, reason in model_rules if unsupported),
            None,
        )
        if unsupported_reason is not None:
            return unsupported_reason

        architecture = architectures[0]
        resolved_model_class, resolved_architecture = get_model_architecture(
            model_config
        )
        if (
            resolved_architecture != architecture
            or resolved_model_class is not _get_canonical_model_class(architecture)
        ):
            return "the native SGLang model implementation is required"
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
        prepare_succeeded = False
        try:
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
            prepare_succeeded = True
            logger.info(
                "Prepared capture-safe model in %.2f s",
                self._capture_ready_at - self._created_at,
            )
            return model
        finally:
            if not prepare_succeeded:
                self._state = StartupWeightLoadState.FAILED

    def start_prefetch(self) -> None:
        if self._state != StartupWeightLoadState.CAPTURE_READY:
            raise RuntimeError(
                f"Cannot prefetch startup weights from state {self._state}"
            )
        assert self._capture_ready_at is not None
        prefetch_succeeded = False
        try:
            prefetch_started_at = time.perf_counter()
            self._prefetch_handle = self._loader.start_checkpoint_prefetch(
                self._resolved_sources,
                num_threads=self._options.prefetch_num_threads,
            )
            self._prefetch_started_at = prefetch_started_at
            self._state = StartupWeightLoadState.PREFETCHING
            prefetch_succeeded = True
            logger.info(
                "Started checkpoint prefetching %.2f s after capture-safe model prep",
                self._prefetch_started_at - self._capture_ready_at,
            )
        finally:
            if not prefetch_succeeded:
                self._state = StartupWeightLoadState.FAILED

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
        commit_succeeded = False
        try:
            manifest = ModelStorageManifest.capture(self._model)
            commit_started_at = time.perf_counter()
            with patched_vllm_parallel_state():
                self._loader.commit_model_weights(
                    model=self._model,
                    model_config=self._model_config,
                    resolved_sources=self._resolved_sources,
                    target_device=torch.device(self._device_config.device),
                )
                torch.cuda.synchronize()
                changed_names = manifest.changed_names(self._model)
                if changed_names:
                    preview = ", ".join(changed_names[:8])
                    raise RuntimeError(
                        "Startup weight commit changed graph-visible tensor storage: "
                        f"{preview}"
                    )
            commit_succeeded = True
        finally:
            cleanup_succeeded = False
            try:
                self._stop_prefetch()
                cleanup_succeeded = True
            finally:
                self._state = (
                    StartupWeightLoadState.READY
                    if commit_succeeded and cleanup_succeeded
                    else StartupWeightLoadState.FAILED
                )
        logger.info(
            "Load weight end. Committed real weights after CUDA graph capture in %.2f s "
            "(capture overlap window %.2f s, startup overlap total %.2f s)",
            time.perf_counter() - commit_started_at,
            commit_started_at - self._prefetch_started_at,
            time.perf_counter() - self._created_at,
        )

    def cancel(self) -> None:
        if self._state in (
            StartupWeightLoadState.READY,
            StartupWeightLoadState.CANCELLED,
        ):
            return
        failed = self._state == StartupWeightLoadState.FAILED
        try:
            self._stop_prefetch()
        finally:
            self._state = (
                StartupWeightLoadState.FAILED
                if failed
                else StartupWeightLoadState.CANCELLED
            )

    def _stop_prefetch(self) -> None:
        if self._prefetch_handle is None:
            return
        self._prefetch_handle.stop()
        self._prefetch_handle = None
