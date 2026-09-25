"""Worker-side construction of built-in and out-of-tree cache linkers."""

from __future__ import annotations

import importlib
import inspect
from copy import deepcopy
from typing import TYPE_CHECKING

from sglang.srt.mem_cache.unified_cache.linker_config import UnifiedCacheLinkerConfig
from sglang.srt.runtime_context import get_memory

if TYPE_CHECKING:
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
    from sglang.srt.mem_cache.unified_cache.unified_cache_linker import (
        UnifiedCacheLinker,
    )
    from sglang.srt.server_args import ServerArgs


def create_unified_cache_linker(
    server_args: ServerArgs,
    params: CacheInitParams,
    *,
    components: set[ComponentType],
) -> UnifiedCacheLinker:
    memory = get_memory()
    if memory.unified_cache_external_linker_config is not None:
        config = UnifiedCacheLinkerConfig.from_dict(
            memory.unified_cache_external_linker_config
        )
        # Do not mask ImportError from the plugin's own dependencies.
        module = importlib.import_module(config.linker_module_path)
        linker_cls = getattr(module, config.linker, None)
        from sglang.srt.mem_cache.unified_cache.unified_cache_linker import (
            UnifiedCacheLinker,
        )

        if not inspect.isclass(linker_cls) or not issubclass(
            linker_cls, UnifiedCacheLinker
        ):
            raise ValueError(
                f"{config.linker_module_path}.{config.linker} must be a "
                "UnifiedCacheLinker subclass."
            )
        if inspect.isabstract(linker_cls):
            raise ValueError(
                f"{config.linker_module_path}.{config.linker} must implement "
                f"all UnifiedCacheLinker methods: {sorted(linker_cls.__abstractmethods__)}"
            )
        return linker_cls(
            server_args,
            params,
            components=components,
            extra_config=deepcopy(config.linker_extra_config),
        )

    backend = memory.unified_cache_external_linker_backend
    if backend == "mooncake":
        from sglang.srt.mem_cache.storage.mooncake_store.mooncake_direct_linker import (
            MooncakeDirectLinker,
        )

        linker_cls = MooncakeDirectLinker
    elif backend == "mori":
        from sglang.srt.mem_cache.storage.umbp.umbp_direct_linker import (
            UMBPDirectLinker,
        )

        linker_cls = UMBPDirectLinker
    else:
        raise ValueError(f"Unknown unified cache external linker backend: {backend!r}")
    return linker_cls(server_args, params, components=components)
