"""
In-place weight updates for diffusion pipeline modules.

This module provides WeightsUpdater, which swaps model weights at runtime
without restarting the server.  It is the diffusion-engine counterpart of the
LLM engine's ModelRunner.update_weights_from_disk.

Typical usage (from GPUWorker):

    updater = WeightsUpdater(self.pipeline)
    success, message = updater.update_weights_from_disk(
        model_path,
        original_model_path=self.server_args.model_path,
    )

Key design decisions:

- All-or-nothing: if any module fails to load, all previously updated
  modules are rolled back to the original weights by reloading from
  original_model_path.  No partial updates are left behind.

- Rollback failures propagate: if rollback itself fails, the exception is
  not caught so the caller knows the model is in an inconsistent state.
  This matches the LLM engine behaviour.

- Offload-aware: the diffusion LayerwiseOffloadManager replaces GPU
  parameters with torch.empty((1,)) placeholders while real weights live
  in consolidated pinned CPU buffers.  A naive param.data.copy_() would
  fail with a shape mismatch.  Instead, the updater dynamically detects
  active offload managers and writes new weights directly into their CPU
  buffers via update_cpu_weights(), bypassing the placeholders entirely.
  For any layer that happens to be prefetched on GPU at update time, the
  live GPU tensor is also updated so the change takes effect immediately.
  This requires no extra GPU memory and does not disturb the offload state.

- DTensor-aware: parameters that have been distributed via
  torch.distributed.tensor are updated through distribute_tensor
  so that each shard is correctly placed on the right device mesh.
"""

from __future__ import annotations

import gc
import os
import time

import torch

from sglang.multimodal_gen.runtime.cache.teacache import TeaCacheMixin
from sglang.multimodal_gen.runtime.loader.utils import _list_safetensors_files
from sglang.multimodal_gen.runtime.loader.weight_utils import (
    safetensors_weights_iterator,
)
from sglang.multimodal_gen.runtime.utils.hf_diffusers_utils import maybe_download_model
from sglang.multimodal_gen.runtime.utils.layerwise_offload import OffloadableDiTMixin
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

try:
    from torch.distributed.tensor import DTensor, distribute_tensor
except ImportError:
    DTensor = None
    distribute_tensor = None

logger = init_logger(__name__)


class WeightsUpdater:
    """In-place weight updates for diffusion pipeline modules.

    Args:
        pipeline: A ComposedPipelineBase (or DiffusersPipeline) instance
            whose modules will be updated.
    """

    def __init__(self, pipeline):
        self.pipeline = pipeline

    def update_weights_from_disk(
        self,
        model_path: str,
        original_model_path: str,
        flush_cache: bool = True,
        target_modules: list[str] | None = None,
    ) -> tuple[bool, str]:
        """Update model weights from disk without restarting the server.

        Args:
            model_path: HF repo id or local path to the new weights.
            original_model_path: Path to the currently loaded weights (used
                for rollback on failure).
            flush_cache: If True, reset TeaCache state after a successful
                update so that stale cached residuals are not reused.
            target_modules: Explicit list of module names to update.  None
                or ["all"] updates every nn.Module in the pipeline.

        Returns:
            (success, message) tuple.
        """
        tic = time.perf_counter()
        self._original_model_path = original_model_path
        logger.info(f"Updating weights from disk: {model_path}")

        modules_to_update = self._collect_modules(target_modules)
        if not modules_to_update:
            available = list(self.pipeline.modules.keys())
            error_msg = (
                f"No matching modules found for update. "
                f"Requested: {target_modules}. Available in pipeline: {available}"
            )
            logger.error(error_msg)
            return False, error_msg

        try:
            local_model_path = maybe_download_model(model_path)
        except Exception as e:
            return False, f"Failed to download model: {e}"

        weights_map, missing = _validate_weight_files(
            local_model_path, modules_to_update
        )
        if missing:
            error_msg = (
                f"Cannot update weights: missing weight files for modules: {missing}. "
                f"No partial updates allowed."
            )
            logger.error(error_msg)
            return False, error_msg

        logger.info(
            f"Updating {len(weights_map)} modules: "
            + ", ".join(f"{n} <- {p}" for n, p in weights_map.items())
        )

        success, message = self._apply_weights(modules_to_update, weights_map)

        gc.collect()
        torch.cuda.empty_cache()

        if success and flush_cache:
            for _, module in modules_to_update:
                _reset_cache_state(module)

        elapsed = time.perf_counter() - tic
        message = f"{message} elapsed={elapsed:.2f}s"
        logger.info(message)
        return success, message

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _collect_modules(
        self, target_modules: list[str] | None
    ) -> list[tuple[str, torch.nn.Module]]:
        """Resolve target_modules to (name, module) pairs.

        For ComposedPipelineBase pipelines, modules are looked up via
        pipeline.modules.  For DiffusersPipeline (where pipeline.modules
        is empty), we fall back to diffusers_pipe.components.
        """
        available = self.pipeline.modules.keys()
        if target_modules is None or target_modules == ["all"]:
            names = [
                n
                for n in available
                if isinstance(self.pipeline.get_module(n), torch.nn.Module)
            ]
        else:
            names = target_modules

        result: list[tuple[str, torch.nn.Module]] = []
        for name in names:
            module = self.pipeline.get_module(name)
            if module is not None and isinstance(module, torch.nn.Module):
                result.append((name, module))

        # Fallback for DiffusersPipeline: modules live on the diffusers pipe,
        # not in self.pipeline.modules.
        if not result:
            diffusers_pipe = self.pipeline.get_module("diffusers_pipeline")
            if diffusers_pipe is not None and hasattr(diffusers_pipe, "components"):
                components = diffusers_pipe.components
                if target_modules is None or target_modules == ["all"]:
                    names = list(components.keys())
                for name in names:
                    module = components.get(name)
                    if module is not None and isinstance(module, torch.nn.Module):
                        result.append((name, module))
        return result

    def _apply_weights(
        self,
        modules_to_update: list[tuple[str, torch.nn.Module]],
        weights_map: dict[str, str],
    ) -> tuple[bool, str]:
        """Load weights into each module; rollback on first failure."""
        updated_modules: list[str] = []

        for module_name, module in modules_to_update:
            try:
                weights_iter = _get_weights_iter(weights_map[module_name])
                _load_weights_into_module(module, weights_iter)
                updated_modules.append(module_name)
            except Exception as e:
                error_msg = f"Failed to update {module_name}: {e}. Rolling back."
                logger.error(error_msg, exc_info=True)
                self._rollback(updated_modules)
                return False, error_msg

        names = ", ".join(updated_modules)
        return True, f"Updated {len(updated_modules)} modules ({names})."

    def _rollback(self, updated_modules: list[str]) -> None:
        """Restore updated_modules to original weights.

        If rollback itself fails the exception propagates so the caller
        knows the model is in an inconsistent state.
        """
        if not updated_modules:
            return
        original_path = maybe_download_model(self._original_model_path)
        for name in updated_modules:
            module = self.pipeline.get_module(name)
            if module is None:
                continue
            weights_dir = find_weights_dir(original_path, name)
            if weights_dir is None:
                continue
            weights_iter = _get_weights_iter(weights_dir)
            _load_weights_into_module(module, weights_iter)


# ---------------------------------------------------------------------------
# Module-level utility functions
# ---------------------------------------------------------------------------


def find_weights_dir(local_path: str, module_name: str) -> str | None:
    """Locate the safetensors directory for module_name under local_path.

    Diffusion models store weights in per-module subdirectories (e.g.
    transformer/, vae/, text_encoder/).  This function tries
    <local_path>/<module_name>/ first, then falls back to local_path
    itself if it directly contains safetensors files (common for RL
    checkpoints that save weights in a flat directory).
    """
    dir_path = os.path.join(local_path, module_name)
    if os.path.exists(dir_path):
        return dir_path
    if _list_safetensors_files(local_path):
        return local_path
    return None


def _get_weights_iter(weights_dir: str):
    """Return a (name, tensor) iterator over safetensors in weights_dir."""
    safetensors_files = _list_safetensors_files(weights_dir)
    if not safetensors_files:
        raise FileNotFoundError(f"No safetensors files found in {weights_dir}")
    return safetensors_weights_iterator(safetensors_files)


def _validate_weight_files(
    local_model_path: str,
    modules_to_update: list[tuple[str, torch.nn.Module]],
) -> tuple[dict[str, str], list[str]]:
    """Check that every module has a weights directory with safetensors files.

    Returns:
        (weights_map, missing) where weights_map maps module name to its
        weights directory and missing lists modules without weight files.
    """
    weights_map: dict[str, str] = {}
    missing: list[str] = []
    for module_name, _ in modules_to_update:
        weights_dir = find_weights_dir(local_model_path, module_name)
        if weights_dir and _list_safetensors_files(weights_dir):
            weights_map[module_name] = weights_dir
        else:
            missing.append(module_name)
    return weights_map, missing


def _get_offload_managers(module: torch.nn.Module) -> list:
    """Return active offload managers for the given module, if any."""
    if isinstance(module, OffloadableDiTMixin) and module.layerwise_offload_managers:
        return [m for m in module.layerwise_offload_managers if m.enabled]
    return []


def _load_weights_into_module(module: torch.nn.Module, weights_iter) -> None:
    """Load weights into a module, handling offload-managed parameters.

    When layerwise offload is active, block-layer parameters are stored as
    small placeholders on GPU while the real weights live in consolidated
    CPU buffers.  This function updates those CPU buffers directly and
    falls back to the normal in-place copy for non-offloaded parameters.
    """
    offload_managers = _get_offload_managers(module)
    if offload_managers:
        weight_dict = dict(weights_iter)
        offloaded_names: set[str] = set()
        for manager in offload_managers:
            offloaded_names |= manager.update_cpu_weights(weight_dict)
        remaining = ((n, w) for n, w in weight_dict.items() if n not in offloaded_names)
        load_weights_into_model(remaining, dict(module.named_parameters()))
    else:
        load_weights_into_model(weights_iter, dict(module.named_parameters()))


def load_weights_into_model(weights_iter, model_params: dict) -> None:
    """Copy weights from weights_iter into model_params in-place.

    Handles DTensor parameters by re-distributing the loaded weight
    according to the existing device mesh and placements.

    Raises:
        ValueError: On shape mismatch between model parameter and loaded weight.
    """
    for name, loaded_weight in weights_iter:
        if name not in model_params:
            continue
        param = model_params[name]
        if param.shape != loaded_weight.shape:
            raise ValueError(
                f"Shape mismatch for {name}: model={param.shape}, loaded={loaded_weight.shape}"
            )
        if DTensor is not None and isinstance(param, DTensor):
            distributed_weight = distribute_tensor(
                loaded_weight.to(param.dtype),
                param.device_mesh,
                param.placements,
            )
            param._local_tensor.copy_(distributed_weight._local_tensor)
        else:
            param.data.copy_(loaded_weight.to(param.dtype))


def _reset_cache_state(module: torch.nn.Module) -> None:
    """Reset Cache state after weight updates.

    After weights change, any cached residuals from previous denoising steps
    are invalid and must be cleared.
    """
    if isinstance(module, TeaCacheMixin):
        module.reset_teacache_state()
