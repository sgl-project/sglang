# SPDX-License-Identifier: Apache-2.0
"""Description of how to read a ComfyUI single-file DiT checkpoint.

ComfyUI ships a DiT as one `.safetensors` file with no `model_index.json` and
its own parameter names. A spec supplies the three things the shared loader
cannot infer from such a file: which DiT config to build, how ComfyUI names map
onto SGLang names, and how to reshape tensors whose layout differs (fused QKV,
swapped scale/shift). Everything else -- meta-device init, FSDP sharding,
quantization, CPU offload -- is handled by the regular transformer load path.
"""

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from sglang.multimodal_gen.runtime.server_args import ServerArgs

WeightIterator = Iterator[tuple[str, torch.Tensor]]

# ComfyUI mapping entries reuse the param_names_mapping format:
#   source_regex -> (target_template, merge_index, num_params_to_merge)
ParamNamesMapping = dict[str, tuple[str, int | None, int | None]]


@dataclass(frozen=True)
class ComfyUICheckpointSpec:
    """Per-model knowledge needed to load a ComfyUI checkpoint."""

    dit_cls_name: str
    build_dit_config: Callable[["ServerArgs"], Any]
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


def get_comfyui_checkpoint_spec(pipeline_name: str) -> ComfyUICheckpointSpec | None:
    global _SPECS_DISCOVERED
    if not _SPECS_DISCOVERED:
        _SPECS_DISCOVERED = True
        import sglang.multimodal_gen.runtime.loader.comfyui_checkpoints  # noqa: F401

    return _SPEC_REGISTRY.get(pipeline_name)


def get_registered_comfyui_pipeline_names() -> list[str]:
    get_comfyui_checkpoint_spec("")
    return sorted(_SPEC_REGISTRY)
