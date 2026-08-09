"""
In-place weight updates for diffusion pipeline modules.

This module provides WeightsUpdater, which swaps model weights at runtime
without restarting the server.  It is the diffusion-engine counterpart of the
LLM engine's ModelRunner.update_weights_from_disk.

Detailed usage of higher level API can be found in

/python/sglang/multimodal_gen/test/single_test_file/test_update_weights_from_disk.py

Key design decisions:

- All-or-nothing with rollback: modules are updated sequentially.  If
  any module fails (shape mismatch, corrupted file, etc.), every module
  that was already updated is rolled back by reloading its weights from
  that module's last successfully-loaded weights directory.  On a full
  successful update, pipeline.model_path is updated to the new model_path;
  target_modules updates keep per-module rollback state for hybrid models.

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
    BYTES_PER_GB,
    _list_safetensors_files,
    get_param_names_mapping,
)
from sglang.multimodal_gen.runtime.loader.weight_utils import (
    safetensors_weights_iterator,
)
from sglang.multimodal_gen.runtime.managers.memory_managers.layerwise_offload import (
    is_layerwise_offloaded_module,
)
from sglang.multimodal_gen.runtime.models.encoders.base import TextEncoder
from sglang.multimodal_gen.runtime.pipelines.diffusers_pipeline import DiffusersPipeline
from sglang.multimodal_gen.runtime.pipelines_core.lora_pipeline import LoRAPipeline
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.utils.hf_diffusers_utils import maybe_download_model
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.srt.weight_sync.tensor_bucket import (
    FlattenedTensorBucket,
    FlattenedTensorMetadata,
)

logger = init_logger(__name__)
_DEFAULT_TENSOR_TARGET_MODULE = "transformer"
LORA_MERGE_WEIGHT_UPDATE_MODE = "lora_merge"
_LORA_IPC_TARGET_MODULES = frozenset({"transformer", "transformer_2"})
# Materialising an offloaded module for a weight update makes its full weights
# resident. Require headroom beyond the raw size: free memory is not necessarily
# allocatable as one contiguous block. Both values are heuristics.
_MATERIALIZE_VRAM_MARGIN = 1.2
_MATERIALIZE_VRAM_FLOOR_GB = 1.0


def _get_lora_layer_dict(
    lora_pipeline: LoRAPipeline, target_module: str
) -> dict[str, object]:
    if target_module == "transformer":
        return lora_pipeline.lora_layers
    if target_module == "transformer_2":
        if not lora_pipeline.lora_layers_transformer_2:
            raise ValueError(
                "transformer_2 is not present or has no LoRA layers in this pipeline"
            )
        return lora_pipeline.lora_layers_transformer_2
    raise ValueError(
        f"Unsupported LoRA IPC target_module={target_module!r}; "
        f"expected one of {sorted(_LORA_IPC_TARGET_MODULES)}"
    )


def _group_lora_ab_tensors(
    named_tensors: list[tuple[str, torch.Tensor]],
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Group flattened IPC tensors into {layer_name: (lora_A, lora_B)} pairs."""
    partial: dict[str, dict[str, torch.Tensor]] = {}
    for name, tensor in named_tensors:
        if ".lora_A" in name:
            layer_name = name.split(".lora_A", 1)[0]
            partial.setdefault(layer_name, {})["A"] = tensor
        elif ".lora_B" in name:
            layer_name = name.split(".lora_B", 1)[0]
            partial.setdefault(layer_name, {})["B"] = tensor

    pairs: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for layer_name, ab in partial.items():
        lora_a = ab.get("A")
        lora_b = ab.get("B")
        if lora_a is None or lora_b is None:
            logger.warning(
                "Incomplete LoRA pair for layer %s (has_A=%s has_B=%s); skipping",
                layer_name,
                lora_a is not None,
                lora_b is not None,
            )
            continue
        pairs[layer_name] = (lora_a, lora_b)
    return pairs


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


def _load_weights_into_module(module: torch.nn.Module, weights_iter) -> set[str]:
    """
    Load weights into a module in place, dispatching on how that module stores them.

    Three paths, and the order of the checks matters: an offloaded text encoder satisfies
    more than one condition, so the offload test must come first.

    1. Offloaded modules -> write into the offload manager's consolidated CPU buffers via
       update_cpu_weights(); anything the manager does not claim falls back to in-place copy.
    2. TextEncoder (not offloaded, no DTensor parameters) -> delegate to the module's own
       load_weights().
    3. Everything else -> name mapping followed by in-place copy.

    Buffers are included alongside parameters because module state such as BatchNorm running
    statistics is stored on disk but absent from named_parameters(). For the FLUX.2 VAE the
    BatchNorm is affine=False, so those buffers are the layer's entire state, and they set the
    scale/shift applied to every decode.

    Why TextEncoder delegates: each encoder owns quirks the generic path knows nothing about --
    stripping a "model." prefix, per-model renames, and stacked_params_mapping fusion.  Those
    differ per model and cannot be centralised.

    Why that branch receives the RAW iterator: _iter_module_weight_updates is deliberately NOT
    hoisted above the dispatch, even though it appears in two branches. load_weights needs the
    original checkpoint names -- the fusion mapping is many-to-one (q_proj/k_proj/v_proj ->
    qkv_proj), so shard_id is derived from which original name matched. Rename first and that
    information is unrecoverable.

    Why an offloaded text encoder is materialised instead: under offload the parameters are
    torch.empty((1,)) placeholders, so load_weights has nothing real to write into, and the
    offload path cannot fuse (it copies flattened tensors into fixed byte ranges).
    disable_offload() restores real parameters, and enable_offload() syncs them back to the CPU
    buffers. Same mechanism the LoRA IPC path already uses via LoRAPipeline._temporarily_disable_offload.

    Why DTensor parameters are excluded: load_weights predates sharding. At startup it runs
    before shard_model(), so it never meets a DTensor and has no branch for one; calling it on
    an FSDP-wrapped module raises "got mixed torch.Tensor and DTensor". Those modules fall to
    the generic path, which handles DTensor explicitly.

    Note that _rollback() calls this function too, so a failure here raises again during
    rollback -- and rollback failures propagate by design, turning a recoverable 400 into a 500.

    The in-place copies below (param.data.copy_, DTensor _local_tensor.copy_,
    and the offload manager's CPU-buffer copies) mutate tensors that were
    created while the pipeline ran under ``torch.inference_mode()`` and are
    therefore "inference tensors".  PyTorch forbids in-place mutation of an
    inference tensor outside an inference-mode context, so wrap the whole
    update in ``torch.inference_mode()`` (matching the offload manager's own
    restore path).  Without this the weight-update API raises
    "Inplace update to inference tensor outside InferenceMode is not allowed"
    and returns an HTTP error.
    """
    with torch.inference_mode():
        should_disable_offload = (
            is_layerwise_offloaded_module(module)
            and isinstance(module, TextEncoder)
            and _can_materialize_module(module)
        )
        updated_weights: set[str] = set()
        if should_disable_offload:
            module.disable_offload()
        try:
            model_params = dict(module.named_parameters())
            named_buffers = dict(module.named_buffers())
            target_tensors = model_params | named_buffers
            offload_managers: list = []
            has_distributed_params = any(
                isinstance(p, DTensor) for p in model_params.values()
            )
            if is_layerwise_offloaded_module(module):
                offload_managers = [
                    m for m in module.layerwise_offload_managers if m.enabled
                ]
            if offload_managers:
                # Not hoisted above the dispatch on purpose: the TextEncoder branch
                # below needs the unmapped checkpoint names.
                weights_iter = _iter_module_weight_updates(
                    module, weights_iter, target_tensors
                )
                weight_dict = dict(weights_iter)
                offloaded_names: set[str] = set()
                for manager in offload_managers:
                    offloaded_names.update(manager.update_cpu_weights(weight_dict))
                remaining = (
                    (n, w) for n, w in weight_dict.items() if n not in offloaded_names
                )
                updated_weights.update(offloaded_names)
                updated_weights.update(
                    load_weights_into_model(remaining, target_tensors)
                )
            # TODO(firefly-silvers): FSDP-wrapped text encoders are excluded here and fall
            # through to the generic path, which cannot strip the "model." prefix or fuse
            # q/k/v. So their weights are NOT updated (silently, until the completeness
            # check lands). load_weights() runs before shard_model() at startup, so it has
            # no DTensor branch and raises "got mixed torch.Tensor and DTensor" if called.
            # A fix needs either DTensor handling inside the per-model load_weights(), or
            # the fsdp_load.py pattern: run the weight loader into a plain temp tensor,
            # then distribute_tensor() the result. An offloaded and sharded encoder is
            # materialised above and then lands here too, doing no useful work. See #31924.
            elif isinstance(module, TextEncoder) and not has_distributed_params:
                updated_weights.update(module.load_weights(weights_iter))
            else:
                # See the note above: kept per-branch so the TextEncoder path stays raw.
                weights_iter = _iter_module_weight_updates(
                    module, weights_iter, target_tensors
                )
                updated_weights.update(
                    load_weights_into_model(weights_iter, target_tensors)
                )
        finally:
            # enable_offload() syncs GPU state back into the CPU buffers before
            # re-placeholdering. Without it the update above is silently discarded.
            if should_disable_offload:
                module.enable_offload()
        return updated_weights


def _offloaded_weight_bytes(module: torch.nn.Module) -> int:
    return sum(
        manager.resident_bytes
        for manager in module.layerwise_offload_managers
        if manager.enabled
    )


def _can_materialize_module(module: torch.nn.Module) -> bool:
    """Whether this module's offloaded weights fit in free VRAM with headroom."""
    required_gb = _offloaded_weight_bytes(module) / BYTES_PER_GB
    needed_gb = max(
        required_gb * _MATERIALIZE_VRAM_MARGIN,
        required_gb + _MATERIALIZE_VRAM_FLOOR_GB,
    )
    available_gb = current_platform.get_available_gpu_memory()
    if available_gb < needed_gb:
        logger.warning(
            "Not materialising %s for weight update: needs %.2f GB (%.2f GB weights "
            "plus headroom), only %.2f GB free. Falling back to the offload path, "
            "which cannot strip prefixes or fuse names -- this module's weights will "
            "be skipped.",
            type(module).__name__,
            needed_gb,
            required_gb,
            available_gb,
        )
        return False
    return True


def _build_module_weight_name_mapper(module: torch.nn.Module):
    """Build a chained regex mapper from mapping dicts exposed by the module."""
    mapping_fns = []
    for attr in ("lora_param_names_mapping", "param_names_mapping"):
        mapping = getattr(module, attr, None)
        if not mapping:
            continue
        mapping_fns.append(get_param_names_mapping(mapping))

    if not mapping_fns:
        return None

    def map_name(name: str) -> str:
        mapped_name = name
        for mapping_fn in mapping_fns:
            mapped_name = mapping_fn(mapped_name)[0]
        return mapped_name

    return map_name


def _strip_param_weight_suffix(param_name: str) -> str:
    if param_name.endswith(".weight"):
        return param_name[: -len(".weight")]
    if param_name.endswith(".bias"):
        return param_name[: -len(".bias")]
    return param_name


def _resolve_lora_ipc_layer_dict_key(
    layer_prefix: str,
    layer_dict: dict,
    module: torch.nn.Module,
) -> tuple[Any | None, str]:
    """Map training-side LoRA layer prefix to lora_layers key (Layer 2)."""
    layer = layer_dict.get(layer_prefix)
    if layer is not None:
        return layer, layer_prefix

    map_name = _build_module_weight_name_mapper(module)
    if map_name is None:
        return None, layer_prefix

    mapped = _strip_param_weight_suffix(map_name(f"{layer_prefix}.weight"))
    if mapped != layer_prefix:
        layer = layer_dict.get(mapped)
        if layer is not None:
            return layer, mapped

    return None, layer_prefix


def _iter_module_weight_updates(
    module: torch.nn.Module,
    weights_iter,
    target_tensors: dict,
):
    map_name = _build_module_weight_name_mapper(module)
    module_name = type(module).__name__

    for name, loaded_weight in weights_iter:
        if name in target_tensors:
            yield name, loaded_weight
            continue

        mapped_name = map_name(name) if map_name is not None else name
        if mapped_name in target_tensors:
            yield mapped_name, loaded_weight
            continue

        logger.warning(
            "Skipping weight update for %s: parameter %r not found after mapping to %r",
            module_name,
            name,
            mapped_name,
        )


def load_weights_into_model(
    weights_iter, target_tensors: dict, module_name: str | None = None
) -> set[str]:
    """Copy weights from weights_iter into target_tensors in-place."""
    updated_weights: set[str] = set()
    for name, loaded_weight in weights_iter:
        if name not in target_tensors:
            logger.warning("Skipping weight update: parameter %r not found", name)
            continue
        param = target_tensors[name]
        weight_loader = getattr(param, "weight_loader", None)
        if callable(weight_loader):
            weight_loader(param, loaded_weight.to(param.dtype))
            updated_weights.add(name)
        else:
            dtensor_param = param if isinstance(param, DTensor) else None
            if dtensor_param is None and isinstance(
                getattr(param, "data", None), DTensor
            ):
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
                    module_prefix = f"{module_name}." if module_name else ""
                    raise ValueError(
                        f"Shape mismatch for {module_prefix}{name}: "
                        f"model={param.shape}, loaded={loaded_weight.shape}"
                    )
                param.data.copy_(loaded_weight.to(param.dtype))
            updated_weights.add(name)
    return updated_weights


class WeightsUpdater:
    """In-place weight updates for diffusion pipeline modules.

    Args:
        pipeline: A ComposedPipelineBase (or DiffusersPipeline) instance
            whose modules will be updated.
    """

    def __init__(self, pipeline):
        self.pipeline = pipeline
        try:
            self._module_weight_dirs = pipeline._weights_updater_module_weight_dirs
        except AttributeError:
            self._module_weight_dirs = {}
            pipeline._weights_updater_module_weight_dirs = self._module_weight_dirs

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

        if success:
            for module_name, _ in modules_to_update:
                self._module_weight_dirs[module_name] = weights_map[module_name]
            if target_modules is None:
                self.pipeline.model_path = local_model_path

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
        updated_modules: dict[str, int] = {}

        for module_name, module in modules_to_update:
            try:
                weights_iter = _get_weights_iter(weights_map[module_name])
                loaded_weights = _load_weights_into_module(module, weights_iter)
                if len(loaded_weights) == 0:
                    logger.warning(f"0 weights loaded for module {module_name}")
                updated_modules[module_name] = len(loaded_weights)
            except Exception as e:
                rollback_list = list(updated_modules) + [module_name]
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

        updated_module_name = ", ".join(
            f"{key}: {value}" for key, value in updated_modules.items()
        )
        return True, f"Updated {len(updated_modules)} modules ({updated_module_name})."

    def _rollback(self, updated_modules: list[str]) -> None:
        """Restore updated_modules to original weights.

        If rollback itself fails the exception propagates so the caller
        knows the model is in an inconsistent state.
        """
        if not updated_modules:
            return
        original_path: str | None = None
        for name in updated_modules:
            module = self.pipeline.get_module(name)
            if module is None:
                continue
            weights_dir = self._module_weight_dirs.get(name)
            if weights_dir is None:
                if original_path is None:
                    original_path = maybe_download_model(self.pipeline.model_path)
                weights_dir = str(Path(original_path) / name)
            weights_dir = Path(weights_dir)
            if not weights_dir.exists():
                continue
            weights_iter = _get_weights_iter(str(weights_dir))
            _load_weights_into_module(module, weights_iter)

    def update_weights_from_tensor(
        self,
        named_tensors: Any,
        load_format: str | None = None,
        target_modules: list[str] | None = None,
        weight_update_mode: str | None = None,
        lora_alpha: int | None = None,
        lora_rank: int | None = None,
    ) -> tuple[bool, str]:
        if weight_update_mode == LORA_MERGE_WEIGHT_UPDATE_MODE:
            return self._update_lora_from_tensor(
                named_tensors=named_tensors,
                load_format=load_format,
                target_modules=target_modules,
                lora_alpha=lora_alpha,
                lora_rank=lora_rank,
            )

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

    def _update_lora_from_tensor(
        self,
        named_tensors: Any,
        load_format: str | None,
        target_modules: list[str] | None,
        lora_alpha: int | None,
        lora_rank: int | None,
    ) -> tuple[bool, str]:
        if not isinstance(self.pipeline, LoRAPipeline):
            return (
                False,
                "LoRA merge weight update requires a LoRAPipeline-backed model",
            )

        if target_modules is None:
            target_modules = [_DEFAULT_TENSOR_TARGET_MODULE]
        if len(target_modules) != 1:
            return (
                False,
                "LoRA IPC weight update requires exactly one target module per request",
            )
        target_module = target_modules[0]
        if target_module not in _LORA_IPC_TARGET_MODULES:
            return (
                False,
                f"LoRA IPC weight update supports target_modules in "
                f"{sorted(_LORA_IPC_TARGET_MODULES)}, got {target_module!r}",
            )

        try:
            modules_to_update = self._collect_modules([target_module])
        except ValueError as e:
            logger.error(str(e))
            return False, str(e)

        try:
            module_payloads = self._resolve_module_payloads(
                named_tensors=named_tensors,
                modules_to_update=modules_to_update,
            )
        except ValueError as e:
            logger.error(str(e))
            return False, str(e)

        materialized: list[tuple[str, torch.Tensor]] = []
        for module_name, _module in modules_to_update:
            payload = module_payloads[module_name]
            weights_iter = self._materialize_weights_iter(payload, load_format)
            materialized.extend(list(weights_iter))

        pairs = _group_lora_ab_tensors(materialized)
        if not pairs:
            return False, "No LoRA A/B tensor pairs found in payload"

        lora_pipeline: LoRAPipeline = self.pipeline
        if not lora_pipeline.lora_initialized:
            convert_target = (
                "all"
                if "transformer_2" in get_updatable_modules(lora_pipeline)
                else "transformer"
            )
            # Match disk LoRA loading: wrap all supported Linear layers regardless
            # of lora_target_modules. Training-side HF keys are resolved at write time.
            saved_lora_target_modules = lora_pipeline.lora_target_modules
            lora_pipeline.lora_target_modules = None
            try:
                with lora_pipeline._temporarily_disable_offload(
                    target=convert_target, use_module_names_only=True
                ):
                    lora_pipeline.convert_to_lora_layers()
            finally:
                lora_pipeline.lora_target_modules = saved_lora_target_modules

        try:
            layer_dict = _get_lora_layer_dict(lora_pipeline, target_module)
        except ValueError as e:
            logger.error(str(e))
            return False, str(e)

        dit_module = dict(modules_to_update).get(target_module)
        if dit_module is None:
            return False, f"No DiT module found for LoRA IPC target {target_module!r}"

        updated = 0
        skipped = 0
        unknown_layers: list[str] = []
        with lora_pipeline._temporarily_disable_offload(target=target_module):
            for layer_name, (lora_a, lora_b) in pairs.items():
                layer, _resolved_key = _resolve_lora_ipc_layer_dict_key(
                    layer_name, layer_dict, dit_module
                )
                if layer is None:
                    logger.warning(
                        "Unknown LoRA layer name %s for target %s; skipping",
                        layer_name,
                        target_module,
                    )
                    unknown_layers.append(layer_name)
                    skipped += 1
                    continue
                inferred_rank = int(lora_a.shape[0])
                alpha = lora_alpha if lora_alpha is not None else inferred_rank
                if lora_rank is not None and lora_rank != inferred_rank:
                    logger.warning(
                        "LoRA rank mismatch for %s: payload=%d request=%d; using payload rank",
                        layer_name,
                        inferred_rank,
                        lora_rank,
                    )
                layer.lora_rank = inferred_rank
                layer.lora_alpha = alpha
                layer.set_lora_weights(
                    lora_a, lora_b, merge_weights=True, clear_existing=True
                )
                updated += 1

        gc.collect()
        torch.cuda.empty_cache()

        if updated == 0:
            sample = unknown_layers[:5]
            return (
                False,
                f"No LoRA layers updated for {target_module} ({skipped} unknown layer names"
                f"{f', e.g. {sample}' if sample else ''}); "
                "check training-side layer name mapping",
            )

        message = (
            f"Updated {updated} LoRA layers in {target_module} from IPC tensors "
            f"(skipped {skipped} unknown layers)."
        )
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
