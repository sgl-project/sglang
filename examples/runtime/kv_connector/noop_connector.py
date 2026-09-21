# SPDX-License-Identifier: Apache-2.0
"""Importable connector example: ordinary local radix caching, no external I/O.

Put this directory on PYTHONPATH and select noop_connector.NoOpKVConnector.
It demonstrates packaging and lifecycle wiring; it is not a storage backend.
"""

from sglang.srt.mem_cache.base_kv_connector import BaseKVConnector
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache


class NoOpKVConnector(UnifiedRadixCache, BaseKVConnector):
    @classmethod
    def validate_config(cls, context, config):
        if context.is_hybrid_swa or context.is_hybrid_ssm or context.is_dsa:
            raise ValueError("The no-op example only supports full-attention models")
        if config.kv_connector_extra_config:
            raise ValueError("The no-op example has no provider options")

    def __init__(self, context, config):
        context.params.tree_components = (ComponentType.FULL,)
        super().__init__(context.params)

    def prefetch_request(self, req):
        pass

    def check_prefetch_progress(self, handle):
        return True

    def pop_prefetch_loaded_span(self, handle):
        return 0, None

    def check_hicache_events(self):
        pass

    def ready_to_load_host_cache(self):
        return -1

    def has_pending_cache_operations(self):
        return False

    def clear_storage_backend(self):
        return False

    def release_host_resources(self):
        super().release_host_resources()
