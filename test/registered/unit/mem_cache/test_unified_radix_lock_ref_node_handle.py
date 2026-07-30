import unittest
from unittest.mock import MagicMock

from sglang.srt.mem_cache.unified_cache.components import ComponentType
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestUnifiedRadixLockRefNodeHandle(unittest.TestCase):
    def test_resolves_node_id_before_component_locking(self):
        cache = object.__new__(UnifiedRadixCache)
        cache.session = MagicMock()
        cache.session.try_inc_lock_ref.return_value = None
        cache.disable = False
        cache.tree_core = MagicMock()
        node = MagicMock()
        cache.tree_core.node_by_id.return_value = node
        component = MagicMock()
        component.component_type = ComponentType.FULL
        component.acquire_component_lock.side_effect = lambda *, node, result: result
        cache._components_tuple = (component,)

        cache.inc_lock_ref(17)

        cache.tree_core.node_by_id.assert_called_once_with(17)
        component.acquire_component_lock.assert_called_once()
        self.assertIs(component.acquire_component_lock.call_args.kwargs["node"], node)
        cache.tree_core._update_evictable_leaf_sets.assert_called_once_with(node)


if __name__ == "__main__":
    unittest.main()
