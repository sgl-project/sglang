# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
"""
Base classes for pipeline stages.

This module defines the abstract base classes for pipeline stages that can be
composed to create complete diffusion pipelines.
"""

from abc import ABC, abstractmethod
from copy import deepcopy
from enum import Enum, auto
from typing import Any, ClassVar

import torch

from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.validators import (
    VerificationResult,
)
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.server_args import ServerArgs, get_global_server_args
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.runtime.utils.perf_logger import StageProfiler

logger = init_logger(__name__)


class StageParallelismType(Enum):
    # execute on all gpus
    REPLICATED = auto()
    # executed on main rank only
    MAIN_RANK_ONLY = auto()
    # this stage requires a cfg-parallel
    CFG_PARALLEL = auto()


class StageVerificationError(Exception):
    """Exception raised when stage verification fails."""

    pass


class PipelineStage(ABC):
    """
    Abstract base class for all pipeline stages.

    A pipeline stage represents a discrete step in the diffusion process that can be
    composed with other stages to create a complete pipeline. Each stage is responsible
    for a specific part of the process, such as prompt encoding, latent preparation, etc.
    """

    deduplicated_output_fields: ClassVar[tuple[str, ...]] = ()
    deduplicated_tensor_tree_output_fields: ClassVar[tuple[str, ...]] = ()
    deduplicated_deepcopy_output_fields: ClassVar[tuple[str, ...]] = ()
    deduplicated_extra_tensor_tree_output_keys: ClassVar[tuple[str, ...]] = ()

    def __init__(self):
        self.server_args = get_global_server_args()

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
        stage_name = self.__class__.__name__
        # Check if verification is enabled (simple approach for prototype)

        # Pre-execution input verification
        try:
            input_result = self.verify_input(batch, server_args)
            self._run_verification(input_result, stage_name, "input")
        except Exception as e:
            logger.error("Input verification failed for %s: %s", stage_name, str(e))
            raise

        # Execute the actual stage logic with unified profiling
        with StageProfiler(
            stage_name,
            logger=logger,
            metrics=batch.metrics,
            log_stage_start_end=not batch.is_warmup
            and not (self.server_args and self.server_args.comfyui_mode),
            perf_dump_path_provided=batch.perf_dump_path is not None,
        ):
            result = self.forward(batch, server_args)

        # Post-execution output verification
        try:
            output_result = self.verify_output(result, server_args)
            self._run_verification(output_result, stage_name, "output")
        except Exception as e:
            logger.error("Output verification failed for %s: %s", stage_name, str(e))
            raise

        return result

    def run_grouped_requests(
        self,
        batches: list[Req],
        server_args: ServerArgs,
    ) -> list[Any]:
        """Run this stage for a group of independent requests.

        A grouped request is still a list of normal ``Req`` objects. The group
        boundary only gives a stage the opportunity to reduce duplicate work.
        The default implementation preserves the single-request contract by
        calling ``self(batch, server_args)`` for every request, so stages that do
        not override this method keep exactly the same behavior as before.

        Stage overrides decide their own reuse granularity. A simple stage may
        group by a single full-stage fingerprint, compute once, then copy or
        split the stage-local outputs back to every request. A mixed stage may
        instead reuse only one subprocess, such as positive prompt encoding,
        while still running another subprocess per request. Overrides must
        preserve input order and return one result per input request.

        Full-stage dedup is opt-in: a stage declares the request fields it
        writes through ``deduplicated_output_fields`` (and the clone/deepcopy
        variants when needed), and overrides ``build_dedup_fingerprint`` to
        describe when two requests are equivalent for that stage. The base
        implementation then handles grouping, running the first request, and
        copying only the declared stage outputs to the remaining requests.

        Stages with finer internal reuse should override this method directly.
        For example, latent preparation cannot copy one request's random noise
        to another request, but it can still group requests and batch the
        deterministic packing/scaling subprocess.

        This hook is deliberately not a global cache: deduplication is local to
        the current stage and current group. A dedup fingerprint must only
        contain fields that can change this stage's outputs. Request metadata
        such as request id, output path, or seed should be excluded unless this
        stage actually reads it.
        """
        if self.has_deduplicated_output_fields():
            return self.run_deduplicated_group(batches, server_args)

        return [self(batch, server_args) for batch in batches]

    @classmethod
    def has_deduplicated_output_fields(cls) -> bool:
        """Return whether this stage opts into base full-stage dedup.

        The base class treats declared output fields as the opt-in signal. This
        keeps the common full-stage case declarative: subclasses only name the
        ``Req`` fields that this stage owns and implement
        ``build_dedup_fingerprint``. A stage that needs partial reuse leaves
        these declarations empty and overrides ``run_grouped_requests``.
        """
        return bool(
            cls.deduplicated_output_fields
            or cls.deduplicated_tensor_tree_output_fields
            or cls.deduplicated_deepcopy_output_fields
            or cls.deduplicated_extra_tensor_tree_output_keys
        )

    def build_dedup_fingerprint(self, batch: Req, server_args: ServerArgs) -> Any:
        """Return this stage's semantic input fingerprint for grouped dedup.

        A fingerprint is the stage-local set of input values that fully
        determines the outputs this stage writes. Two requests may share a
        fingerprint only when this stage would produce equivalent outputs for
        both of them.

        The default fingerprint is unique per request, so stages opt into
        deduplication explicitly. Overrides should prefer a named frozen
        dataclass whose field names document the stage inputs. Include every
        request/config field read by this stage, and exclude fields that only
        matter to later stages, such as output path. If this stage reads seed or
        mutable inputs, include them or disable dedup for that request. Use
        ``freeze_for_dedup`` for tensors and nested containers so the
        fingerprint remains hashable and deterministic.
        """
        return id(batch)

    def run_deduplicated_group(
        self,
        batches: list[Req],
        server_args: ServerArgs,
        copy_outputs=None,
    ) -> list[Req]:
        """Run full-stage-equivalent requests once and fan out stage outputs.

        This helper is the implementation behind declarative full-stage dedup.
        It groups requests by ``build_dedup_fingerprint()``, executes the first
        request in each group through the normal ``self(batch, server_args)``
        path, and copies only this stage's declared outputs to the remaining
        requests. The returned list preserves the input order.

        ``copy_outputs`` exists for rare stages that need bespoke full-stage
        copying, but the preferred path is to declare ``deduplicated_*`` fields
        and let ``copy_deduplicated_outputs`` do the copy. Stages with partial
        subprocess reuse should override ``run_grouped_requests`` instead of
        forcing their logic through this full-stage helper.
        """
        if copy_outputs is None:
            copy_outputs = self.copy_deduplicated_outputs

        results: list[Req | None] = [None] * len(batches)

        for _, group in self._group_requests_by_fingerprint(
            batches, lambda batch: self.build_dedup_fingerprint(batch, server_args)
        ):
            first_index, first_batch = group[0]
            first_result = self(first_batch, server_args)
            results[first_index] = first_result

            for index, batch in group[1:]:
                copy_outputs(first_result, batch)
                results[index] = batch

        return [result for result in results if result is not None]

    def copy_deduplicated_outputs(self, src: Req, dst: Req) -> None:
        """Copy declared stage outputs from a computed request to a duplicate.

        This method is intentionally field-based. The stage still owns the
        semantic contract by declaring exactly which ``Req`` fields it writes.
        The helper only applies the requested ownership policy:

        - ``deduplicated_output_fields``: shallow container copy, tensor refs
          shared. This is the default for read-only outputs such as embeddings.
        - ``deduplicated_tensor_tree_output_fields``: recursively clone tensors.
          Use this for small mutable tensor outputs such as timesteps/sigmas.
        - ``deduplicated_deepcopy_output_fields``: Python deepcopy. This is for
          request-local mutable runtime objects such as scheduler instances.
        - ``deduplicated_extra_tensor_tree_output_keys``: clone selected
          ``Req.extra`` entries without replacing the whole extra dict.
        """
        for field in self.deduplicated_output_fields:
            setattr(dst, field, self.copy_stage_output(getattr(src, field)))
        for field in self.deduplicated_tensor_tree_output_fields:
            setattr(dst, field, self.clone_tensor_tree(getattr(src, field)))
        for field in self.deduplicated_deepcopy_output_fields:
            setattr(dst, field, deepcopy(getattr(src, field)))
        for key in self.deduplicated_extra_tensor_tree_output_keys:
            if key in src.extra:
                dst.extra[key] = self.clone_tensor_tree(src.extra[key])

    @classmethod
    def copy_stage_output(cls, value):
        """Copy a reusable output with the default low-overhead ownership model.

        The default is a shallow container copy: lists/tuples/dicts get a new
        container, while tensor objects inside are shared by reference. This is
        deliberate for large read-only outputs, where cloning tensors would add
        GPU memory traffic and defeat much of the dedup benefit.
        """
        if isinstance(value, list):
            return list(value)
        if isinstance(value, tuple):
            return tuple(value)
        if isinstance(value, dict):
            return dict(value)
        return value

    @classmethod
    def clone_tensor_tree(cls, value):
        """Recursively clone tensors in a small output tree.

        This is opt-in because it is more expensive than
        ``copy_stage_output``. Use it when downstream code may mutate the tensor
        or when sharing tensor storage would accidentally couple duplicated
        requests.
        """
        if isinstance(value, torch.Tensor):
            return value.clone()
        if isinstance(value, list):
            return [cls.clone_tensor_tree(item) for item in value]
        if isinstance(value, tuple):
            return tuple(cls.clone_tensor_tree(item) for item in value)
        if isinstance(value, dict):
            return {key: cls.clone_tensor_tree(item) for key, item in value.items()}
        return value

    @staticmethod
    def freeze_for_dedup(value: Any) -> Any:
        """Convert common nested values into a hashable fingerprint fragment.

        Small tensors include their values so scheduler/timestep overrides can
        distinguish user-provided tensors. Larger tensors include shape, dtype,
        and device only; they should not normally be part of a fingerprint
        unless the stage has a stronger equivalence guarantee.
        """
        if isinstance(value, torch.Tensor):
            if value.numel() <= 256:
                return (
                    "tensor",
                    tuple(value.shape),
                    str(value.dtype),
                    tuple(value.detach().cpu().reshape(-1).tolist()),
                )
            return ("tensor", tuple(value.shape), str(value.dtype), value.device.type)
        if isinstance(value, dict):
            return tuple(
                sorted(
                    (key, PipelineStage.freeze_for_dedup(item))
                    for key, item in value.items()
                )
            )
        if isinstance(value, (list, tuple)):
            return tuple(PipelineStage.freeze_for_dedup(item) for item in value)
        if isinstance(value, set):
            return tuple(
                sorted(PipelineStage.freeze_for_dedup(item) for item in value)
            )
        return value

    @staticmethod
    def _group_requests_by_fingerprint(
        batches: list[Req],
        fingerprint_fn,
    ) -> list[tuple[Any, list[tuple[int, Req]]]]:
        """Group requests by a stage-local fingerprint while preserving order.

        The return value is
        ``[(fingerprint, [(original_index, req), ...]), ...]``. Group order
        follows the first appearance of each fingerprint, and requests inside a
        group keep their original relative order. Callers can fill a result
        list by ``original_index`` to preserve input/output ordering.

        The helper does not define request equivalence and does not copy
        outputs; it only provides stable grouping for both full-stage dedup and
        finer-grained stage-local reuse.
        """
        groups: dict[Any, list[tuple[int, Req]]] = {}
        for index, batch in enumerate(batches):
            fingerprint = fingerprint_fn(batch)
            groups.setdefault(fingerprint, []).append((index, batch))
        return list(groups.items())

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
