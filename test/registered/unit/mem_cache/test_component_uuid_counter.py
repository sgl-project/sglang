import unittest

from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.mem_cache.unified_cache.components import base
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestComponentUuidCounter(unittest.TestCase):
    def setUp(self):
        self._original_counters = base._COMPONENT_UUID_COUNTERS.copy()

    def tearDown(self):
        base._COMPONENT_UUID_COUNTERS.clear()
        base._COMPONENT_UUID_COUNTERS.update(self._original_counters)

    def test_component_ranges_are_disjoint_and_independent(self):
        self.assertEqual(
            base.next_component_uuid(ComponentType.SWA), 100_000_000_000_001
        )
        self.assertEqual(
            base.next_component_uuid(ComponentType.SWA), 100_000_000_000_002
        )
        self.assertEqual(
            base.next_component_uuid(ComponentType.FULL), 200_000_000_000_001
        )
        self.assertEqual(
            base.next_component_uuid(ComponentType.MAMBA), 300_000_000_000_001
        )
        self.assertEqual(
            base.next_component_uuid(ComponentType.C128), 400_000_000_000_001
        )
        self.assertEqual(
            base.next_component_uuid(ComponentType.FULL), 200_000_000_000_002
        )


if __name__ == "__main__":
    unittest.main()
