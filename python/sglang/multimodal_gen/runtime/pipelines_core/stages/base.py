# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
"""
Base classes for pipeline stages.

This module defines the abstract base classes for pipeline stages that can be
composed to create complete diffusion pipelines.
"""

from abc import ABC, abstractmethod
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from enum import Enum, auto

import torch

from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
from sglang.multimodal_gen.runtime.managers.memory_managers.component_manager import (
    ComponentUse,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.dedup import StageDedupMixin
from sglang.multimodal_gen.runtime.pipelines_core.stages.validators import (
    VerificationResult,
)
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.server_args import ServerArgs, get_global_server_args
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.runtime.utils.nvtx_pytorch_hooks import DiffusionNvtxHooks
from sglang.multimodal_gen.runtime.utils.perf_logger import StageProfiler

logger = init_logger(__name__)


class StageParallelismType(Enum):
    # execute on all gpus
    REPLICATED = auto()
    # executed on main rank only
    MAIN_RANK_ONLY = auto()
    # this stage requires a cfg-parallel
    CFG_PARALLEL = auto()
    # executed on main rank only and send result to other ranks
    MAIN_RANK_ONLY_AND_SEND_TO_OTHERS = auto()


class StageVerificationError(Exception):
    """Exception raised when stage verification fails."""

    pass


class PipelineStage(StageDedupMixin, ABC):
    """
    Abstract base class for all pipeline stages.

    A pipeline stage represents a discrete step in the diffusion process that can be
    composed with other stages to create a complete pipeline. Each stage is responsible
    for a specific part of the process, such as prompt encoding, latent preparation, etc.
    """

    # Class-level defaults so subclasses that override __init__ without
    # calling super().__init__() (e.g. lightweight test doubles) still see
    # a consistent NVTX state. All four values are immutable, so sharing
    # the class attribute until a per-instance assignment shadows it is safe.
    _nvtx_hooks: DiffusionNvtxHooks | None = None
    _nvtx_registered_ids: frozenset[int] = frozenset()
    _nvtx_zero_warned: bool = False
    _current_use_nvtx: bool = False

    def __init__(self):
        self.server_args = get_global_server_args()
        self._component_residency_manager = None
        self._registered_stage_name: str | None = None
        self._profile_stage_name: str | None = None

    def log_info(self, msg, *args):
        """Logs an informational message with the stage name as a prefix."""
        if self.server_args.comfyui_mode:
            return
        logger.info(f"[{self.__class__.__name__}] {msg}", *args)

    def log_warning(self, msg, *args):
        """Logs a warning message with the stage name as a prefix."""
        logger.warning(f"[{self.__class__.__name__}] {msg}", *args)

    def log_error(self, msg, *args):
        """Logs an error message with the stage name as a prefix."""
        logger.error(f"[{self.__class__.__name__}] {msg}", *args)

    def log_debug(self, msg, *args):
        """Logs a debug message with the stage name as a prefix."""
        logger.debug(f"[{self.__class__.__name__}] {msg}", *args)

    def verify_input(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        """
        Verify the input for the stage.

        Example:
            from sglang.multimodal_gen.runtime.pipelines.stages.validators import V, VerificationResult

            def verify_input(self, batch, server_args):
                result = VerificationResult()
                result.add_check("height", batch.height, V.positive_int_divisible(8))
                result.add_check("width", batch.width, V.positive_int_divisible(8))
                result.add_check("image_latent", batch.image_latent, V.is_tensor)
                return result

        """
        # Default implementation - no verification
        return VerificationResult()

    def maybe_free_model_hooks(self):
        pass

    def load_model(self):
        """
        Load the model for the stage.
        """
        pass

    def offload_model(self):
        """
        Offload the model for the stage.
        """
        pass

    def set_component_residency_manager(self, manager) -> None:
        self._component_residency_manager = manager

    def set_registered_stage_name(self, stage_name: str) -> None:
        self._registered_stage_name = stage_name

    def set_profile_stage_name(self, stage_name: str) -> None:
        self._profile_stage_name = stage_name

    def _component_stage_name(self, stage_name: str | None = None) -> str:
        return (
            stage_name
            or getattr(self, "_registered_stage_name", None)
            or self.__class__.__name__
        )

    def _active_component_stage_name(self) -> str:
        manager = getattr(self, "_component_residency_manager", None)
        manager_state = getattr(manager, "state", None)
        manager_stage_name = getattr(manager_state, "stage_name", None)
        if manager_stage_name is not None:
            return manager_stage_name
        return self._component_stage_name()

    def _active_profile_stage_name(self) -> str:
        return getattr(self, "_profile_stage_name", None) or self.__class__.__name__

    def _finish_active_component_use(self) -> None:
        if self._component_residency_manager is not None:
            self._component_residency_manager.finish_active_use()

    @contextmanager
    def _use_component(
        self,
        use: ComponentUse,
        module=None,
    ) -> Iterator[object | None]:
        if self._component_residency_manager is None:
            yield module
            return
        with self._component_residency_manager.use_component(use, module) as component:
            yield component

    def _declared_component_use(
        self,
        *,
        component_name: str,
        phase: str | None = None,
        target_dtype: torch.dtype | None = None,
    ) -> ComponentUse:
        manager = self._component_residency_manager
        stage_name = self._active_component_stage_name()
        server_args = manager.server_args if manager is not None else self.server_args
        for use in self.component_uses(server_args, stage_name):
            if use.component_name != component_name:
                continue
            if phase is not None and use.phase != phase:
                continue
            if target_dtype is not None:
                return replace(use, target_dtype=target_dtype)
            return use
        raise ValueError(
            f"{self.__class__.__name__} did not declare component use: "
            f"{component_name}"
        )

    @contextmanager
    def use_declared_component(
        self,
        *,
        component_name: str,
        module=None,
        phase: str | None = None,
        target_dtype: torch.dtype | None = None,
    ) -> Iterator[object | None]:
        """reference a component already declared in `component_uses`"""
        use = self._declared_component_use(
            component_name=component_name,
            phase=phase,
            target_dtype=target_dtype,
        )
        with self._use_component(use, module) as component:
            yield component

    def component_uses(
        self, server_args: ServerArgs, stage_name: str | None = None
    ) -> list[ComponentUse]:
        """Declares component uses of current stage for unified residency scheduling."""
        return []

    # ---- Layerwise NVTX instrumentation (--enable-layerwise-nvtx-marker) ----
    def nvtx_hookable_modules(self) -> list[tuple[torch.nn.Module, str]]:
        """Modules to instrument with layerwise NVTX hooks; default none.

        Override in stages that own forward-able components. Each entry
        is ``(module, prefix)`` where ``prefix`` is prepended to every
        emitted NVTX range name. The base implementation returns an
        empty list — stages without instrumentable components inherit
        a no-op.

        Conventions:

        * ``None`` modules are skipped silently; this is the contract
          for lazy-loaded components (return the attribute before the
          loader runs and the next request retries registration).
        * If a stage may ``del`` its component attribute (e.g. the MPS
          deallocation path on ``DenoisingStage``), use
          ``getattr(self, "<attr>", None)`` so the override does not
          raise ``AttributeError`` between deallocate and re-load.
        * Prefixes must be globally unique across stages that can
          appear in the same pipeline. The current set is:
          ``transformer`` / ``transformer_2`` (DenoisingStage),
          ``text_encoder[_N]`` (TextEncodingStage),
          ``image_encoder`` / ``image_text_encoder``
          (ImageEncodingStage), ``vae`` (DecodingStage),
          ``vae_encoder`` (EncodingStage), ``image_vae_encoder``
          (ImageVAEEncodingStage), ``ltx2_vae_encoder`` /
          ``ltx2_condition_image_encoder`` (LTX2ImageEncodingStage),
          and ``qwen_layered_text_encoder`` / ``qwen_layered_vae``
          (QwenImageLayeredBeforeDenoisingStage).
        * Called every gate (once per request); should be O(modules)
          and side-effect free.
        """
        return []

    def _maybe_register_nvtx_hooks(self) -> None:
        """Idempotently attach hooks; re-register when modules change.

        Tracks the set of declared module identities and re-attaches when
        it changes (lazy load, cache-dit wrap, hot swap), so hooks never
        end up bound to an orphan module after the stage's underlying
        components are rebound between calls. The zero-modules warning
        fires at most once per stage instance.
        """
        if not self.server_args.enable_layerwise_nvtx_marker:
            return
        current = self.nvtx_hookable_modules()
        current_ids = frozenset(id(m) for m, _ in current if m is not None)
        is_rebind = False
        if self._nvtx_hooks is not None:
            if current_ids == self._nvtx_registered_ids:
                return
            # Underlying modules changed identity — detach and re-register.
            self._nvtx_hooks.remove_hooks()
            self._nvtx_hooks = None
            self._nvtx_registered_ids = frozenset()
            self._nvtx_zero_warned = False
            is_rebind = True
        hooks = DiffusionNvtxHooks()
        total = 0
        for module, prefix in current:
            if module is None:
                continue
            total += hooks.register_hooks(module, prefix=prefix)
        if total == 0:
            if not self._nvtx_zero_warned:
                logger.warning(
                    "[%s] NVTX flag set but no modules available; will retry.",
                    self.__class__.__name__,
                )
                self._nvtx_zero_warned = True
            return
        if is_rebind:
            logger.info(
                "[%s] Re-registered NVTX hooks on %d submodules (modules changed)",
                self.__class__.__name__,
                total,
            )
        else:
            logger.info(
                "[%s] Registered NVTX hooks on %d submodules",
                self.__class__.__name__,
                total,
            )
        self._nvtx_hooks = hooks
        self._nvtx_registered_ids = current_ids

    def _apply_nvtx_gate(self, is_warmup: bool) -> bool:
        """Register (if needed) and toggle hooks for this request.

        Caches the resolved ``use_nvtx`` value on ``self`` so
        ``forward`` implementations that need to gate their own explicit
        NVTX ranges can read it via :attr:`current_use_nvtx` instead of
        recomputing.
        """
        self._maybe_register_nvtx_hooks()
        use_nvtx = self.server_args.enable_layerwise_nvtx_marker and not is_warmup
        if self._nvtx_hooks is not None:
            self._nvtx_hooks.set_enabled(use_nvtx)
        self._current_use_nvtx = use_nvtx
        return use_nvtx

    @property
    def current_use_nvtx(self) -> bool:
        """Last resolved ``use_nvtx`` value from :meth:`_apply_nvtx_gate`.

        ``forward`` implementations can read this to gate explicit
        ``maybe_nvtx_range`` blocks without re-evaluating the flag.
        """
        return self._current_use_nvtx

    def _detach_nvtx_hooks(self) -> None:
        """Remove all hooks; call when the underlying module is about to
        be deallocated or replaced (e.g. MPS dealloc)."""
        if self._nvtx_hooks is not None:
            self._nvtx_hooks.remove_hooks()
            self._nvtx_hooks = None
        self._nvtx_registered_ids = frozenset()
        self._nvtx_zero_warned = False

    # Default role affinity: ENCODER. Override in subclasses for DENOISING/DECODER.
    @property
    def role_affinity(self) -> RoleType:
        return RoleType.ENCODER

    # execute on all ranks by default
    @property
    def parallelism_type(self) -> StageParallelismType:
        # if get_global_server_args().enable_cfg_parallel:
        #     return StageParallelismType.MAIN_RANK_ONLY
        return StageParallelismType.REPLICATED

    def verify_output(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        """
        Verify the output for the stage.



        Returns:
            A VerificationResult containing the verification status.
        """
        # Default implementation - no verification
        return VerificationResult()

    def _run_verification(
        self,
        verification_result: VerificationResult,
        stage_name: str,
        verification_type: str,
    ) -> None:
        """
        Run verification and raise errors if any checks fail.

        Args:
            verification_result: Results from verify_input or verify_output
            stage_name: Name of the current stage
            verification_type: "input" or "output"
        """
        if not verification_result.is_valid():
            failed_fields = verification_result.get_failed_fields()
            if failed_fields:
                # Get detailed failure information
                detailed_summary = verification_result.get_failure_summary()

                failed_fields_str = ", ".join(failed_fields)
                error_msg = (
                    f"{verification_type.capitalize()} verification failed for {stage_name}: "
                    f"Failed fields: {failed_fields_str}\n"
                    f"Details: {detailed_summary}"
                )
                raise StageVerificationError(error_msg)

    @property
    def device(self) -> torch.device:
        """Get the device for this stage."""
        return torch.device(
            current_platform.device_type,
        )

    def set_logging(self, enable: bool):
        """
        Enable or disable logging for this stage.

        Args:
            enable: Whether to enable logging.
        """
        self._enable_logging = enable

    def __call__(
        self,
        batch: Req,
        server_args: ServerArgs,
    ) -> Req:
        """
        Execute the stage's processing on the batch with optional verification and logging.
        Should not be overridden by subclasses.

        Returns:
            The updated batch information after this stage's processing.
        """
        stage_name = self._active_profile_stage_name()
        # Check if verification is enabled (simple approach for prototype)

        # Pre-execution input verification
        try:
            input_result = self.verify_input(batch, server_args)
            self._run_verification(input_result, stage_name, "input")
        except Exception as e:
            logger.error("Input verification failed for %s: %s", stage_name, str(e))
            raise

        # Register and toggle layerwise NVTX hooks once per call. Stages
        # whose components are not yet loaded (lazy load) are no-ops here
        # and will register on the next call once modules become available.
        self._apply_nvtx_gate(batch.is_warmup)

        # Execute the actual stage logic with unified profiling. The
        # ``finally`` disables this stage's hooks again so a module
        # shared with another stage (e.g. a VAE referenced by both
        # ImageVAEEncodingStage and DecodingStage) never has more than
        # one stage's hooks armed at a time, even though both stages
        # registered against the same module instance.
        try:
            with StageProfiler(
                stage_name,
                logger=logger,
                metrics=batch.metrics,
                log_stage_start_end=not batch.is_warmup
                and not (self.server_args and self.server_args.comfyui_mode),
                perf_dump_path_provided=batch.perf_dump_path is not None,
            ):
                result = self.forward(batch, server_args)
        finally:
            if self._nvtx_hooks is not None:
                self._nvtx_hooks.set_enabled(False)
            self._current_use_nvtx = False

        # Post-execution output verification
        try:
            output_result = self.verify_output(result, server_args)
            self._run_verification(output_result, stage_name, "output")
        except Exception as e:
            logger.error("Output verification failed for %s: %s", stage_name, str(e))
            raise

        return result

    @abstractmethod
    def forward(
        self,
        batch: Req,
        server_args: ServerArgs,
    ) -> Req:
        """
        Forward pass of the stage's processing.

        This method should be implemented by subclasses to provide the forward
        processing logic for the stage.



        Returns:
            The updated batch information after this stage's processing.
        """
        raise NotImplementedError

    def backward(
        self,
        batch: Req,
        server_args: ServerArgs,
    ) -> Req:
        raise NotImplementedError
