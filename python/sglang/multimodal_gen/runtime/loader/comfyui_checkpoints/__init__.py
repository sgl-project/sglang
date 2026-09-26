# SPDX-License-Identifier: Apache-2.0
"""ComfyUI single-file DiT checkpoints.

``spec`` holds the shared checkpoint description and load path. Each sibling
module registers one DiT family. Importing this package registers every spec.
"""

from sglang.multimodal_gen.runtime.loader.comfyui_checkpoints import (  # noqa: F401
    flux,
    qwen_image,
    zimage,
)
from sglang.multimodal_gen.runtime.loader.comfyui_checkpoints.spec import (
    ComfyUICheckpointSpec,
    ParamNamesMapping,
    WeightIterator,
    get_comfyui_checkpoint_spec,
    get_registered_comfyui_pipeline_names,
    is_comfyui_single_file,
    load_comfyui_transformer,
    register_comfyui_checkpoint,
)

__all__ = [
    "ComfyUICheckpointSpec",
    "ParamNamesMapping",
    "WeightIterator",
    "get_comfyui_checkpoint_spec",
    "get_registered_comfyui_pipeline_names",
    "is_comfyui_single_file",
    "load_comfyui_transformer",
    "register_comfyui_checkpoint",
]
