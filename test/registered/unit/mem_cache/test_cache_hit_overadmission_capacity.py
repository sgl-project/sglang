import unittest

from sglang.srt.mem_cache.kv_cache_configurator import (
    resolve_cache_hit_overadmission_capacity,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestCacheHitOveradmissionCapacity(unittest.TestCase):
    def test_disabled_preserves_normal_capacity(self):
        self.assertEqual(
            resolve_cache_hit_overadmission_capacity(
                normal_capacity=37,
                hard_capacity=128,
                mamba_cache_size=149,
                max_extra_reqs_per_worker=32,
                enabled=False,
            ),
            37,
        )

    def test_provisions_requested_physical_capacity(self):
        self.assertEqual(
            resolve_cache_hit_overadmission_capacity(
                normal_capacity=37,
                hard_capacity=128,
                mamba_cache_size=149,
                max_extra_reqs_per_worker=27,
                enabled=True,
            ),
            64,
        )

    def test_caps_capacity_by_lazy_mamba_live_slots(self):
        self.assertEqual(
            resolve_cache_hit_overadmission_capacity(
                normal_capacity=37,
                hard_capacity=128,
                mamba_cache_size=149,
                max_extra_reqs_per_worker=64,
                enabled=True,
            ),
            74,
        )

    def test_caps_capacity_by_requested_hard_limit(self):
        self.assertEqual(
            resolve_cache_hit_overadmission_capacity(
                normal_capacity=37,
                hard_capacity=50,
                mamba_cache_size=149,
                max_extra_reqs_per_worker=32,
                enabled=True,
            ),
            50,
        )


if __name__ == "__main__":
    unittest.main()
