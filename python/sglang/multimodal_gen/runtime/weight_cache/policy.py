# SPDX-License-Identifier: Apache-2.0
"""Audited pipeline bindings, separate from reusable loader/state capabilities."""

from collections.abc import Callable

import msgspec

from sglang.multimodal_gen.runtime.loader.component_loaders.transformer_loader import (
    TransformerLoader,
)
from sglang.multimodal_gen.runtime.loader.native_dit_state import (
    MINIMAX_H3,
    QWEN_IMAGE,
    WAN,
)


class ComponentBinding(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    name: str
    loader_cls: type
    contract_id: str


class PipelineBinding(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    binding_id: str
    pipeline_module: str
    pipeline_name: str
    components: tuple[ComponentBinding, ...]
    supports_subfolder: bool = False
    validate_model_index: Callable | None = None


def _validate_h3_model_index(model_index):
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.release_metadata import (
        MiniMaxH3ReleaseMetadata,
    )

    metadata = MiniMaxH3ReleaseMetadata.from_model_index(model_index)
    if metadata.partition != "fl2va":
        raise ValueError("Weight cache supports original MiniMax-H3 FL2VA only")


BINDINGS = (
    PipelineBinding(
        "wan2_1_t2v_1_3b.v1",
        "sglang.multimodal_gen.runtime.pipelines.wan_pipeline",
        "WanPipeline",
        (ComponentBinding("transformer", TransformerLoader, WAN.contract_id),),
    ),
    PipelineBinding(
        "qwen_image_original.v1",
        "sglang.multimodal_gen.runtime.pipelines.qwen_image",
        "QwenImagePipeline",
        (ComponentBinding("transformer", TransformerLoader, QWEN_IMAGE.contract_id),),
    ),
    PipelineBinding(
        "minimax_h3_fl2va.v1",
        "sglang.multimodal_gen.runtime.pipelines.minimax_h3_pipeline",
        "MiniMaxH3Pipeline",
        (ComponentBinding("transformer", TransformerLoader, MINIMAX_H3.contract_id),),
        supports_subfolder=True,
        validate_model_index=_validate_h3_model_index,
    ),
)


def for_pipeline(pipeline_cls):
    for binding in BINDINGS:
        if (pipeline_cls.__module__, pipeline_cls.__name__) == (
            binding.pipeline_module,
            binding.pipeline_name,
        ):
            return binding
    return None
