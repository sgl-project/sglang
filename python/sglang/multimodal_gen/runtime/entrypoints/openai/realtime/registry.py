# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.realtime_adapter import (
    BaseRealtimeModelAdapter,
)

if TYPE_CHECKING:
    from sglang.multimodal_gen.runtime.server_args import ServerArgs


_REALTIME_ADAPTER_REGISTRY: dict[type, type[BaseRealtimeModelAdapter]] = {}
_BUILTIN_ADAPTERS_REGISTERED = False


def register_realtime_model_adapter(
    pipeline_config_cls: type,
    adapter_cls: type[BaseRealtimeModelAdapter],
) -> None:
    _REALTIME_ADAPTER_REGISTRY[pipeline_config_cls] = adapter_cls


def _register_builtin_realtime_model_adapters() -> None:
    global _BUILTIN_ADAPTERS_REGISTERED
    if _BUILTIN_ADAPTERS_REGISTERED:
        return

    from sglang.multimodal_gen.configs.pipeline_configs.lingbot_world import (
        LingBotWorldCausalDMDConfig,
    )
    from sglang.multimodal_gen.configs.pipeline_configs.omnidreams import (
        OmniDreamsPipelineConfig,
    )
    from sglang.multimodal_gen.configs.pipeline_configs.sana_wm import (
        SanaWMRealtimeConfig,
    )
    from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.adapters.lingbot_world_realtime_adapter import (
        LingBotWorldRealtimeAdapter,
    )
    from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.adapters.omnidreams_realtime_adapter import (
        OmniDreamsRealtimeAdapter,
    )
    from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.adapters.sana_wm_realtime_adapter import (
        SanaWMRealtimeAdapter,
    )

    register_realtime_model_adapter(
        LingBotWorldCausalDMDConfig,
        LingBotWorldRealtimeAdapter,
    )
    register_realtime_model_adapter(
        OmniDreamsPipelineConfig,
        OmniDreamsRealtimeAdapter,
    )
    register_realtime_model_adapter(
        SanaWMRealtimeConfig,
        SanaWMRealtimeAdapter,
    )
    _BUILTIN_ADAPTERS_REGISTERED = True


def get_realtime_model_adapter(
    server_args: ServerArgs,
) -> BaseRealtimeModelAdapter:
    _register_builtin_realtime_model_adapters()

    pipeline_config = server_args.pipeline_config
    for config_cls in type(pipeline_config).__mro__:
        adapter_cls = _REALTIME_ADAPTER_REGISTRY.get(config_cls)
        if adapter_cls is not None:
            return adapter_cls()

    raise ValueError(
        "Realtime video is not supported for pipeline config "
        f"{type(pipeline_config).__name__}; no realtime adapter is registered."
    )
