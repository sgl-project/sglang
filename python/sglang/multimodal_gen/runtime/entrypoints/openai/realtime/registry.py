# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.realtime_adapter import (
    RealtimeModelAdapter,
)

if TYPE_CHECKING:
    from sglang.multimodal_gen.runtime.server_args import ServerArgs


_REALTIME_ADAPTER_REGISTRY: dict[type, type[RealtimeModelAdapter]] = {}
_BUILTIN_ADAPTERS_REGISTERED = False


def register_realtime_model_adapter(
    pipeline_config_cls: type,
    adapter_cls: type[RealtimeModelAdapter],
) -> None:
    _REALTIME_ADAPTER_REGISTRY[pipeline_config_cls] = adapter_cls


def _register_builtin_realtime_model_adapters() -> None:
    global _BUILTIN_ADAPTERS_REGISTERED
    if _BUILTIN_ADAPTERS_REGISTERED:
        return

    from sglang.multimodal_gen.configs.pipeline_configs.sana_wm import (
        SanaWMRealtimeConfig,
    )
    from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.adapters.sana_wm_realtime_adapter import (
        SanaWMRealtimeAdapter,
    )

    register_realtime_model_adapter(
        SanaWMRealtimeConfig,
        SanaWMRealtimeAdapter,
    )
    _BUILTIN_ADAPTERS_REGISTERED = True


def _get_realtime_model_adapter_cls(
    pipeline_config: object,
) -> type[RealtimeModelAdapter] | None:
    _register_builtin_realtime_model_adapters()

    for config_cls in type(pipeline_config).__mro__:
        adapter_cls = _REALTIME_ADAPTER_REGISTRY.get(config_cls)
        if adapter_cls is not None:
            return adapter_cls
    return None


def has_realtime_model_adapter(server_args: ServerArgs) -> bool:
    return _get_realtime_model_adapter_cls(server_args.pipeline_config) is not None


def get_realtime_model_adapter(
    server_args: ServerArgs,
) -> RealtimeModelAdapter:
    adapter_cls = _get_realtime_model_adapter_cls(server_args.pipeline_config)
    if adapter_cls is not None:
        return adapter_cls()

    raise ValueError(
        "Realtime video is not supported for pipeline config "
        f"{type(server_args.pipeline_config).__name__}; no realtime adapter is registered."
    )
