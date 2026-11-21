import unittest

from sglang.srt.disaggregation.mooncake.utils import ascend_allocation_regions
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, stage="base-a", runner_config="cpu")


class TestAscendAllocationRegistration(CustomTestCase):
    def test_two_views_register_once_and_host_memory_keeps_its_size(self):
        segments = [
            {"address": 4096, "total_size": 1024},
            {"address": 1024, "total_size": 512},
        ]
        regions = [(1056, 64), (1088, 96), (5000, 32), (16000, 32)]
        self.assertEqual(
            ascend_allocation_regions(regions, segments),
            [(1024, 512), (4096, 1024), (16000, 32)],
        )

    def test_reject_a_view_across_two_device_allocations(self):
        segments = [{"address": 1024, "total_size": 512}]
        with self.assertRaisesRegex(ValueError, "one device allocation"):
            ascend_allocation_regions([(1520, 64)], segments)


if __name__ == "__main__":
    unittest.main()
