# SPDX-License-Identifier: Apache-2.0
"""Adapters are admitted by resolved capability, not model-name heuristics."""

from . import dit_minimax_h3, dit_qwen_image, dit_wan

_ADAPTERS = {
    module.ADAPTER_ID: module for module in (dit_wan, dit_qwen_image, dit_minimax_h3)
}

for _adapter in _ADAPTERS.values():
    # Every adapter must deliberately declare its optional capabilities; a
    # missing member is an import error, never silent admission of a new row.
    if type(_adapter.SUPPORTS_SUBFOLDER) is not bool or (
        _adapter.validate_model_index is not None
        and not callable(_adapter.validate_model_index)
    ):
        raise TypeError(f"Invalid weight-cache adapter contract: {_adapter.ADAPTER_ID}")


def for_pipeline(pipeline_cls):
    for adapter in _ADAPTERS.values():
        if (pipeline_cls.__module__, pipeline_cls.__name__) == (
            adapter.PIPELINE_MODULE,
            adapter.PIPELINE_NAME,
        ):
            return adapter
    return None


def by_id(adapter_id):
    # IDs are frozen in PreparedPipeline; materialization never rediscovers a
    # pipeline or chooses an adapter from the mutable model registry.
    try:
        return _ADAPTERS[adapter_id]
    except KeyError:
        raise ValueError(
            f"Unknown prepared weight-cache adapter: {adapter_id}"
        ) from None
