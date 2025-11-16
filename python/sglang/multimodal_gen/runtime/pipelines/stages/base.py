# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
"""
Base classes for pipeline stages.

This module defines the abstract base classes for pipeline stages that can be
composed to create complete diffusion pipelines.
"""

import time
import traceback
from abc import ABC, abstractmethod
from enum import Enum, auto

import torch

import sglang.multimodal_gen.envs as envs
from sglang.multimodal_gen.runtime.pipelines.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines.stages.validators import VerificationResult
from sglang.multimodal_gen.runtime.server_args import ServerArgs, get_global_server_args
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

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

    def __init__(self):
        self.server_args = get_global_server_args()

    def log_info(self, msg, *args):
        """Logs an informational message with the stage name as a prefix."""
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

        Args:
            batch: The current batch information.
            server_args: The inference arguments.

        Returns:
            A VerificationResult containing the verification status.

        """
        # Default implementation - no verification
        return VerificationResult()

    # execute on all ranks by default
    @property
    def parallelism_type(self) -> StageParallelismType:
        # if get_global_server_args().enable_cfg_parallel:
        #     return StageParallelismType.MAIN_RANK_ONLY
        return StageParallelismType.REPLICATED

    def verify_output(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        """
        Verify the output for the stage.

        Args:
            batch: The current batch information.
            server_args: The inference arguments.

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
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

        Args:
            batch: The current batch information.
            server_args: The inference arguments.

        Returns:
            The updated batch information after this stage's processing.
        """
        stage_name = self.__class__.__name__
        # Check if verification is enabled (simple approach for prototype)
        enable_verification = getattr(server_args, "enable_stage_verification", False)

        if enable_verification:
            # Pre-execution input verification
            try:
                input_result = self.verify_input(batch, server_args)
                self._run_verification(input_result, stage_name, "input")
            except Exception as e:
                logger.error("Input verification failed for %s: %s", stage_name, str(e))
                raise

        # Execute the actual stage logic
        logging_info = getattr(batch, "logging_info", None)

        if envs.SGL_DIFFUSION_STAGE_LOGGING:
            logger.info("[%s] Starting execution", stage_name)
            start_time = time.perf_counter()

            try:
                result = self.forward(batch, server_args)
                execution_time = time.perf_counter() - start_time
                logger.info(
                    "[%s] Execution completed in %s ms",
                    stage_name,
                    execution_time * 1000,
                )
                if logging_info is not None:
                    try:
                        logging_info.add_stage_execution_time(
                            stage_name, execution_time
                        )
                    except Exception:
                        logger.warning(
                            "[%s] Failed to record stage timing on batch.logging_info",
                            stage_name,
                            exc_info=True,
                        )
                perf_logger = getattr(batch, "perf_logger", None)
                if perf_logger is not None:
                    try:
                        perf_logger.log_stage_metric(stage_name, execution_time * 1000)
                    except Exception:
                        logger.warning(
                            "[%s] Failed to log stage metric to performance logger",
                            stage_name,
                            exc_info=True,
                        )
            except Exception as e:
                execution_time = time.perf_counter() - start_time
                logger.error(
                    "[%s] Error during execution after %s ms: %s",
                    stage_name,
                    execution_time * 1000,
                    e,
                )
                logger.error("[%s] Traceback: %s", stage_name, traceback.format_exc())
                raise
        else:
            # Direct execution (current behavior)
            result = self.forward(batch, server_args)

        if enable_verification:
            # Post-execution output verification
            try:
                output_result = self.verify_output(result, server_args)
                self._run_verification(output_result, stage_name, "output")
            except Exception as e:
                logger.error(
                    "Output verification failed for %s: %s", stage_name, str(e)
                )
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

        Args:
            batch: The current batch information.
            server_args: The inference arguments.

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
