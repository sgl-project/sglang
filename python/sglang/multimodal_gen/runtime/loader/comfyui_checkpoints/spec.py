# SPDX-License-Identifier: Apache-2.0
"""ComfyUI single-file DiT checkpoints: spec, registry, and load.

ComfyUI ships a DiT as one ``.safetensors`` file with no ``model_index.json``
and its own parameter names. A spec supplies what the shared loader cannot
infer: which DiT config to build, how names map onto SGLang, and how to
reshape tensors whose layout differs. Per-model specs live in the sibling
modules of this package.

Everything else -- meta-device init, FSDP sharding, quantization, CPU
offload -- goes through the regular transformer load path.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.loader.fsdp_load import maybe_load_fsdp_model
from sglang.multimodal_gen.runtime.loader.weight_utils import (
    safetensors_weights_iterator,
)
from sglang.multimodal_gen.runtime.models.registry import ModelRegistry
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.runtime.utils.precision import resolve_precision

if TYPE_CHECKING:
    from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
        ComposedPipelineBase,
    )

logger = init_logger(__name__)

WeightIterator = Iterator[tuple[str, torch.Tensor]]

# ComfyUI mapping entries reuse the param_names_mapping format:
#   source_regex -> (target_template, merge_index, num_params_to_merge)
ParamNamesMapping = dict[str, tuple[str, int | None, int | None]]


@dataclass(frozen=True)
class ComfyUICheckpointSpec:
    """Per-model knowledge needed to load a ComfyUI checkpoint."""

    dit_cls_name: str
    build_dit_config: Callable[[ServerArgs], Any]
    param_names_mapping: ParamNamesMapping = field(default_factory=dict)
    # Reshapes tensors that param_names_mapping cannot express. Receives the raw
    # safetensors iterator plus the built dit config, yields SGLang-shaped pairs.
    convert_weights: Callable[[WeightIterator, Any], WeightIterator] | None = None
    # Set False for checkpoints that legitimately omit parameters the model
    # declares, such as optional biases.
    strict: bool = True
    # Whether to layer param_names_mapping on top of the DiT config's own
    # mapping. Keep it True when the two act on different names (the config
    # rules then finish the job, e.g. merging split QKV back into a fused
    # parameter). Set False when both claim the same source names, since name
    # mapping is applied repeatedly until it reaches a fixed point and the
    # config rules would rewrite names this spec already resolved.
    inherit_config_mapping: bool = True


_SPEC_REGISTRY: dict[str, ComfyUICheckpointSpec] = {}
_SPECS_DISCOVERED = False


def register_comfyui_checkpoint(
    pipeline_name: str, spec: ComfyUICheckpointSpec
) -> None:
    _SPEC_REGISTRY[pipeline_name] = spec


def _discover_checkpoint_specs() -> None:
    global _SPECS_DISCOVERED
    if _SPECS_DISCOVERED:
        return
    _SPECS_DISCOVERED = True
    from sglang.multimodal_gen.runtime.loader.comfyui_checkpoints import (  # noqa: F401
        flux,
        qwen_image,
        zimage,
    )


def get_comfyui_checkpoint_spec(pipeline_name: str) -> ComfyUICheckpointSpec | None:
    _discover_checkpoint_specs()
    return _SPEC_REGISTRY.get(pipeline_name)


def get_registered_comfyui_pipeline_names() -> list[str]:
    _discover_checkpoint_specs()
    return sorted(_SPEC_REGISTRY)


def is_comfyui_single_file(model_path: str) -> bool:
    """ComfyUI ships DiTs as one safetensors file with no model_index.json."""
    return os.path.isfile(model_path) and model_path.endswith(".safetensors")


def load_comfyui_transformer(
    pipeline: ComposedPipelineBase,
    server_args: ServerArgs,
    loaded_modules: dict[str, torch.nn.Module] | None = None,
) -> dict[str, Any]:
    """Load the DiT from a single ComfyUI safetensors file.

    Reuses the shared FSDP-aware transformer load path; the spec only supplies
    what the file itself cannot describe.
    """
    if loaded_modules is not None and "transformer" in loaded_modules:
        return {
            "transformer": loaded_modules["transformer"],
            "scheduler": pipeline.get_module("scheduler"),
        }

    spec = get_comfyui_checkpoint_spec(pipeline.pipeline_name)
    if spec is None:
        raise ValueError(
            f"{pipeline.pipeline_name} has no ComfyUI checkpoint spec, so it cannot "
            f"load a single safetensors file. Pipelines with a spec: "
            f"{get_registered_comfyui_pipeline_names()}"
        )

    model_path = pipeline.model_path
    dit_config = spec.build_dit_config(server_args)
    mapping = dict(spec.param_names_mapping)
    if spec.inherit_config_mapping:
        mapping = {
            **(dit_config.arch_config.param_names_mapping or {}),
            **mapping,
        }
    dit_config.arch_config.param_names_mapping = mapping

    model_cls, _ = ModelRegistry.resolve_model_cls(spec.dit_cls_name)
    param_dtype = resolve_precision(server_args, "dit", precision_attr="dit_precision")
    server_args.model_paths["transformer"] = os.path.dirname(model_path) or "."

    # Only override the iterator when tensors need reshaping; leaving it None
    # keeps the rank-local checkpoint fast path available.
    weights_iterator = None
    if spec.convert_weights is not None:
        weights_iterator = spec.convert_weights(
            safetensors_weights_iterator([model_path]), dit_config
        )

    logger.info(
        "Loading %s from ComfyUI checkpoint %s, param_dtype: %s",
        spec.dit_cls_name,
        model_path,
        param_dtype,
    )

    # Weight loading reads param_names_mapping off the model, which inherits it
    # from the class, so the ComfyUI names have to be visible for the whole load.
    original_mapping = model_cls.param_names_mapping
    model_cls.param_names_mapping = mapping
    try:
        model = maybe_load_fsdp_model(
            model_cls=model_cls,
            init_params={"config": dit_config, "hf_config": {}},
            weight_dir_list=[model_path],
            device=get_local_torch_device(),
            hsdp_replicate_dim=server_args.hsdp_replicate_dim,
            hsdp_shard_dim=server_args.hsdp_shard_dim,
            component_starts_on_cpu=server_args.should_start_component_on_cpu(
                "transformer"
            ),
            pin_cpu_memory=server_args.pin_cpu_memory,
            fsdp_inference=server_args.should_use_fsdp_for_component("transformer"),
            param_dtype=param_dtype,
            reduce_dtype=torch.float32,
            output_dtype=None,
            strict=spec.strict,
            weights_iterator=weights_iterator,
        )
    finally:
        model_cls.param_names_mapping = original_mapping

    for param in model.parameters():
        param.requires_grad = False

    logger.info(
        "Loaded transformer with %.2fB parameters",
        sum(p.numel() for p in model.parameters()) / 1e9,
    )

    return {"transformer": model, "scheduler": pipeline.get_module("scheduler")}
