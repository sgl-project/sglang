"""
In-place weight updates for diffusion pipeline modules.

This module provides WeightsUpdater, which swaps model weights at runtime
without restarting the server.  It is the diffusion-engine counterpart of the
LLM engine's ModelRunner.update_weights_from_disk.

Detailed usage of higher level API can be found in

/python/sglang/multimodal_gen/test/server/test_update_weights_from_disk.py

Key design decisions:

- All-or-nothing with rollback: modules are updated sequentially.  If
  any module fails (shape mismatch, corrupted file, etc.), every module
  that was already updated is rolled back by reloading its weights from
  pipeline.model_path (the last successfully-loaded checkpoint).  On
  success, pipeline.model_path is updated to the new model_path so
  that future rollbacks target the latest good checkpoint, not the
  originally-launched model.

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
from pathlib import Path
from typing import Any

import torch
from torch.distributed.tensor import DTensor, distribute_tensor

from sglang.multimodal_gen.runtime.cache.teacache import TeaCacheMixin
from sglang.multimodal_gen.runtime.loader.utils import (
    _list_safetensors_files,
)
from sglang.multimodal_gen.runtime.loader.weight_utils import (
    safetensors_weights_iterator,
)
from sglang.multimodal_gen.runtime.pipelines.diffusers_pipeline import DiffusersPipeline
from sglang.multimodal_gen.runtime.utils.hf_diffusers_utils import maybe_download_model
from sglang.multimodal_gen.runtime.utils.layerwise_offload import OffloadableDiTMixin
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.srt.weight_sync.tensor_bucket import (
    FlattenedTensorBucket,
    FlattenedTensorMetadata,
)

logger = init_logger(__name__)
_DEFAULT_TENSOR_TARGET_MODULE = "transformer"


def get_updatable_modules(pipeline) -> dict[str, torch.nn.Module]:
    """Return updatable nn.Module components for the given pipeline.

    Works with both the native ComposedPipelineBase backend and the
    DiffusersPipeline wrapper.
    """
    if isinstance(pipeline, DiffusersPipeline):
        diffusers_pipe = pipeline.get_module("diffusers_pipeline")
        if diffusers_pipe is not None and diffusers_pipe.components is not None:
            raw = diffusers_pipe.components
        else:
            raw = {}
    else:
        raw = pipeline.modules
    return {n: m for n, m in raw.items() if isinstance(m, torch.nn.Module)}


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
        weights_dir = Path(local_model_path) / module_name
        if weights_dir.exists() and _list_safetensors_files(str(weights_dir)):
            weights_map[module_name] = str(weights_dir)
        else:
            missing.append(module_name)
    return weights_map, missing


def _load_weights_into_module(module: torch.nn.Module, weights_iter) -> None:
    """Load weights into a module, handling offload-managed parameters.

    For offloaded modules, updates CPU buffers directly via
    update_cpu_weights(); non-offloaded parameters use in-place copy.
    """
    offload_managers: list = []
    if isinstance(module, OffloadableDiTMixin) and module.layerwise_offload_managers:
        offload_managers = [m for m in module.layerwise_offload_managers if m.enabled]

    if offload_managers:
        weight_dict = dict(weights_iter)
        offloaded_names: set[str] = set()
        for manager in offload_managers:
            offloaded_names.update(manager.update_cpu_weights(weight_dict))
        remaining = ((n, w) for n, w in weight_dict.items() if n not in offloaded_names)
        load_weights_into_model(
            remaining,
            dict(module.named_parameters()),
        )
    else:
        load_weights_into_model(
            weights_iter,
            dict(module.named_parameters()),
        )


def load_weights_into_model(
    weights_iter,
    model_params: dict,
) -> None:
    """Copy weights from weights_iter into model_params in-place."""
    for name, loaded_weight in weights_iter:
        if name not in model_params:
            continue
        param = model_params[name]
        weight_loader = getattr(param, "weight_loader", None)
        if callable(weight_loader):
            weight_loader(param, loaded_weight.to(param.dtype))
        else:
            dtensor_param = param if isinstance(param, DTensor) else None
            if dtensor_param is None and isinstance(getattr(param, "data", None), DTensor):
                dtensor_param = param.data

            if dtensor_param is not None:
                distributed_weight = distribute_tensor(
                    loaded_weight.to(param.dtype),
                    dtensor_param.device_mesh,
                    dtensor_param.placements,
                )
                dtensor_param._local_tensor.copy_(distributed_weight._local_tensor)
            else:
                if param.shape != loaded_weight.shape:
                    raise ValueError(
                        f"Shape mismatch for {name}: model={param.shape}, loaded={loaded_weight.shape}"
                    )
                param.data.copy_(loaded_weight.to(param.dtype))


class WeightsUpdater:
    """In-place weight updates for diffusion pipeline modules.

    Args:
        pipeline: A ComposedPipelineBase (or DiffusersPipeline) instance
            whose modules will be updated.  The pipeline's model_path
            attribute is used for rollback on failure.
    """

    def __init__(self, pipeline):
        self.pipeline = pipeline

    def update_weights_from_disk(
        self,
        model_path: str,
        flush_cache: bool = True,
        target_modules: list[str] | None = None,
    ) -> tuple[bool, str]:
        """Update model weights from disk without restarting the server."""
        logger.info(f"Updating weights from disk: {model_path}")

        try:
            modules_to_update = self._collect_modules(target_modules)
        except ValueError as e:
            logger.error(str(e))
            return False, str(e)

        if not modules_to_update:
            error_msg = (
                f"No matching modules found for update. "
                f"Requested: {target_modules}. "
                f"Available nn.Module(s): {list(get_updatable_modules(self.pipeline).keys())}"
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
                if isinstance(module, TeaCacheMixin):
                    module.reset_teacache_state()

        logger.info(message)
        return success, message

    def _collect_modules(
        self, target_modules: list[str] | None
    ) -> list[tuple[str, torch.nn.Module]]:
        """Resolve target_modules to (name, module) pairs.

        Raises:
            ValueError: If target_modules contains names not found in the pipeline.
        """
        components = get_updatable_modules(self.pipeline)

        if target_modules is None:
            names = list(components.keys())
        else:
            unknown = [n for n in target_modules if n not in components]
            if unknown:
                raise ValueError(
                    f"Module(s) requested for update not found in pipeline: {unknown}. "
                    f"Available Module(s): {list(components.keys())}"
                )
            names = target_modules

        return [(name, components[name]) for name in names]

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
                rollback_list = updated_modules + [module_name]
                logger.error(
                    f"Weight update failed for module '{module_name}': {e}. "
                    f"Rolling back {len(rollback_list)} module(s) "
                    f"(including partially-loaded '{module_name}'): "
                    f"{rollback_list}.",
                    exc_info=True,
                )
                self._rollback(rollback_list)
                return False, (
                    f"Failed to update module '{module_name}': {e}. "
                    f"All modules rolled back to original weights."
                )

        names = ", ".join(updated_modules)
        return True, f"Updated {len(updated_modules)} modules ({names})."

    def _rollback(self, updated_modules: list[str]) -> None:
        """Restore updated_modules to original weights.

        If rollback itself fails the exception propagates so the caller
        knows the model is in an inconsistent state.
        """
        if not updated_modules:
            return
        original_path = maybe_download_model(self.pipeline.model_path)
        for name in updated_modules:
            module = self.pipeline.get_module(name)
            if module is None:
                continue
            weights_dir = Path(original_path) / name
            if not weights_dir.exists():
                continue
            weights_iter = _get_weights_iter(str(weights_dir))
            _load_weights_into_module(module, weights_iter)

    def update_weights_from_tensor(
        self,
        named_tensors: Any,
        load_format: str | None = None,
        target_modules: list[str] | None = None,
    ) -> tuple[bool, str]:
        if target_modules is None:
            target_modules = [_DEFAULT_TENSOR_TARGET_MODULE]
        try:
            modules_to_update = self._collect_modules(target_modules)
        except ValueError as e:
            logger.error(str(e))
            return False, str(e)

        if not modules_to_update:
            error_msg = (
                f"No matching modules found for update. "
                f"Requested: {target_modules}. "
                f"Available nn.Module(s): {list(get_updatable_modules(self.pipeline).keys())}"
            )
            logger.error(error_msg)
            return False, error_msg

        try:
            module_payloads = self._resolve_module_payloads(
                named_tensors=named_tensors,
                modules_to_update=modules_to_update,
            )
        except ValueError as e:
            logger.error(str(e))
            return False, str(e)

        updated_modules: list[str] = []
        for module_name, module in modules_to_update:
            try:
                payload = module_payloads[module_name]
                weights_iter = self._materialize_weights_iter(payload, load_format)
                _load_weights_into_module(module, weights_iter)
                updated_modules.append(module_name)
            except Exception as e:
                error_msg = (
                    f"Failed to update module '{module_name}' from tensor: {e}. "
                    f"The pipeline may be partially updated. "
                    f"Please discard the whole weights and reload from a known-good checkpoint."
                )
                logger.error(error_msg, exc_info=True)
                return False, error_msg

        gc.collect()
        torch.cuda.empty_cache()
        names = ", ".join(updated_modules)
        message = f"Updated {len(updated_modules)} modules from tensor ({names})."
        logger.info(message)
        return True, message

    def _resolve_module_payloads(
        self,
        named_tensors: Any,
        modules_to_update: list[tuple[str, torch.nn.Module]],
    ) -> dict[str, Any]:
        module_names = [name for name, _ in modules_to_update]
        if isinstance(named_tensors, dict):
            missing = [name for name in module_names if name not in named_tensors]
            if missing:
                raise ValueError(
                    f"Missing tensor payload for module(s): {missing}. "
                    f"Provided modules: {list(named_tensors.keys())}"
                )
            return {name: named_tensors[name] for name in module_names}

        if len(module_names) == 1:
            return {module_names[0]: named_tensors}

        raise ValueError(
            "Ambiguous tensor payload for multi-module update. "
            "Provide a dict mapping module_name -> module payload, "
            f"requested modules: {module_names}."
        )

    def _materialize_weights_iter(self, module_payload: Any, load_format: str | None):
        if load_format == "flattened_bucket":
            if not isinstance(module_payload, dict):
                raise ValueError(
                    "flattened_bucket payload must be a dict with "
                    "'flattened_tensor' and 'metadata'."
                )
            flattened_tensor = module_payload.get("flattened_tensor")
            metadata = module_payload.get("metadata")
            if flattened_tensor is None or metadata is None:
                raise ValueError(
                    "flattened_bucket payload missing 'flattened_tensor' or 'metadata'."
                )
            return self._reconstruct_from_flattened_bucket(flattened_tensor, metadata)

        if isinstance(module_payload, (list, tuple)):
            return iter(module_payload)

        raise ValueError(
            f"Unsupported module payload type for load_format={load_format}: "
            f"{type(module_payload).__name__}"
        )

    def _reconstruct_from_flattened_bucket(self, flattened_tensor: Any, metadata: Any):
        if not isinstance(flattened_tensor, torch.Tensor):
            raise ValueError(
                "flattened_bucket 'flattened_tensor' must be a torch.Tensor."
            )
        if not isinstance(metadata, list):
            raise ValueError("flattened_bucket 'metadata' must be a list.")

        converted_metadata: list[FlattenedTensorMetadata] = []
        for meta in metadata:
            converted_metadata.append(
                FlattenedTensorMetadata(
                    name=meta.name,
                    shape=torch.Size(meta.shape),
                    dtype=self._normalize_torch_dtype(meta.dtype),
                    start_idx=int(meta.start_idx),
                    end_idx=int(meta.end_idx),
                    numel=int(meta.numel),
                )
            )

        bucket = FlattenedTensorBucket(
            flattened_tensor=flattened_tensor,
            metadata=converted_metadata,
        )
        return bucket.reconstruct_tensors()

    def _normalize_torch_dtype(self, dtype: Any) -> torch.dtype:
        if isinstance(dtype, torch.dtype):
            return dtype
        if isinstance(dtype, str):
            name = dtype.split(".")[-1]
            normalized = getattr(torch, name, None)
            if isinstance(normalized, torch.dtype):
                return normalized
        raise ValueError(f"Unsupported dtype in flattened_bucket metadata: {dtype!r}")
