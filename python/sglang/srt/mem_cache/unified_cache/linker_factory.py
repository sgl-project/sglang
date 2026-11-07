"""Worker-side construction of built-in and out-of-tree cache linkers."""

from __future__ import annotations

import importlib
import inspect
import logging
from copy import deepcopy
from typing import TYPE_CHECKING, Any

from sglang.srt.mem_cache.unified_cache.linker_config import (
    UnifiedCacheLinkerConfig,
    parse_linker_load_options,
)
from sglang.srt.runtime_context import get_memory

if TYPE_CHECKING:
    import torch

    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
    from sglang.srt.mem_cache.unified_cache.unified_cache_linker import (
        UnifiedCacheLinker,
    )
    from sglang.srt.server_args import ServerArgs


logger = logging.getLogger(__name__)


def get_startup_linker_config() -> dict[str, Any] | None:
    """Resolve legacy launch options once, outside the linker implementations."""
    memory = get_memory()
    config = memory.unified_cache_external_linker_config
    if config is None and memory.unified_cache_external_linker_backend in (
        "mooncake",
        "mori",
    ):
        from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
            HybridCacheController,
        )

        # TODO(#46388): Remove this fallback after one full stable-release
        # deprecation cycle following the first release containing #46388.
        legacy_config = memory.hicache_storage_backend_extra_config
        config, *_ = HybridCacheController.parse_storage_backend_extra_config(
            legacy_config
        )
        if legacy_config:
            logger.warning(
                "Using --hicache-storage-backend-extra-config for an external "
                "linker is deprecated. Migrate to "
                "--unified-cache-external-linker-config with a JSON object. "
                "For @file input, pass the file contents as JSON instead. "
                "This fallback will be removed after a full stable-release "
                "deprecation cycle. Ordinary HiCache use of the old flag "
                "is unaffected."
            )
    return deepcopy(config)


def create_unified_cache_linker(
    server_args: ServerArgs,
    params: CacheInitParams,
    *,
    backend: str,
    config: dict[str, Any] | None,
    components: set[ComponentType],
    forward_stream: torch.cuda.Stream | None = None,
) -> UnifiedCacheLinker:
    load_mode, prefetch_policy, backend_config = parse_linker_load_options(config)
    extra_config = backend_config
    if backend == "dynamic":
        config = UnifiedCacheLinkerConfig.from_dict(
            backend_config if config is not None else None
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
        extra_config = config.linker_extra_config
    elif backend == "mooncake":
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
    if backend != "dynamic" and load_mode == "prefetch":
        # Built-ins may need the mode while constructing their load workers.
        extra_config["load_mode"] = load_mode
    linker = linker_cls(
        server_args,
        params,
        components=components,
        extra_config=deepcopy(extra_config),
        forward_stream=forward_stream,
    )
    linker.load_mode = load_mode
    linker.prefetch_policy = prefetch_policy
    return linker
