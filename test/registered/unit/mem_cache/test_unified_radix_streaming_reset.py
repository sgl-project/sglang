from unittest.mock import Mock

from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.mem_cache.unified_cache.components.tree_component import (
    ComponentType,
    EvictLayer,
    TreeComponent,
)
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.session.streaming_session import SessionSlot
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _FakeFullComponent(TreeComponent):
    component_type = ComponentType.FULL

    def create_match_validator(self, match_device_only: bool = False):
        return lambda node: True

    def redistribute_on_node_split(self, new_parent, child):
        return None

    def _dec_session_coverage(self, session_id, leaf) -> None:
        return None

    def _advance_session_coverage(self, session_id, leaf, old_ancestor) -> None:
        return None

    def _recede_session_coverage(self, session_id, leaf, fallback) -> None:
        return None

    def evict_component(
        self,
        node,
        device_frees,
        host_frees,
        target: EvictLayer = EvictLayer.DEVICE,
    ) -> tuple[int, int]:
        return 0, 0

    def acquire_component_lock(self, node, result):
        return result

    def release_component_lock(self, node, params):
        return None

    def _evict_device_start(self, request_cnt) -> None:
        pass

    def _evict_device_next_node(self, tracker, device_frees, host_frees):
        return None

    def _evict_device_end(self) -> None:
        pass


def test_reset_clears_attached_streaming_session_lifecycle() -> None:
    cache = UnifiedRadixCache(
        params=CacheInitParams(
            req_to_token_pool=ReqToTokenPool(
                size=2,
                max_context_len=8,
                device="cpu",
                enable_memory_saver=False,
            ),
            token_to_kv_pool_allocator=None,
            page_size=1,
            disable=True,
            tree_components=(ComponentType.FULL,),
            component_registry_override={ComponentType.FULL: _FakeFullComponent},
        )
    )
    lifecycle = Mock()
    cache.session.attach_session_lifecycle(lifecycle)
    cache.session.slots["session"] = SessionSlot(req_pool_idx=1)

    cache.reset()

    lifecycle.reset.assert_called_once_with()
    assert cache.session.slots == {}
    cache.session.reset_state()
    assert lifecycle.reset.call_count == 2
