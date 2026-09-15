# SPDX-License-Identifier: Apache-2.0
"""Adapters are admitted by resolved capability, not model-name heuristics."""

from . import dit_qwen_image, dit_wan

_ADAPTERS = {module.ADAPTER_ID: module for module in (dit_wan, dit_qwen_image)}


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
