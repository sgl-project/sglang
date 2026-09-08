"""CPU checks for the opt-in KPool metadata-fusion geometry."""

import unittest

from sglang.srt.layers.attention.dsa.dsa_backend_kpool import (
    _is_kpool_metadata_fusion_supported,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestFusionContract(unittest.TestCase):
    def test_only_supported_pool_page_topk_geometry_is_enabled(self):
        for pool, page, topk, expected in (
            (1, 64, 2048, False),
            (2, 64, 2048, True),
            (4, 64, 2048, True),
            (3, 64, 2048, False),
            (4, 128, 2048, False),
            (4, 64, 2049, False),
        ):
            with self.subTest(pool=pool, page=page, topk=topk):
                self.assertEqual(
                    _is_kpool_metadata_fusion_supported(pool, page, topk), expected
                )


if __name__ == "__main__":
    unittest.main()
