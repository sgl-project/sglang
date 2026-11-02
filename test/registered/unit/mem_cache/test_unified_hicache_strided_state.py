"""Cover strided Mamba state detection and staging for HiCache."""

import unittest

import torch

from sglang.srt.mem_cache.pool_host.mamba import MambaPoolHost
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestStridedStateDetection(CustomTestCase):
    def test_contiguous_slots_are_not_strided(self):
        for shape in ((8, 4), (8, 4, 3), (1, 5)):
            with self.subTest(shape=shape):
                self.assertFalse(
                    MambaPoolHost._slots_are_strided(torch.zeros(shape)),
                    "a contiguous slot array must take the direct path",
                )

    def test_envelope_strided_slots_are_detected(self):
        num_slots, per_slot, envelope = 6, 4, 10
        raw = torch.zeros(num_slots * envelope)
        view = torch.as_strided(raw, size=(num_slots, per_slot), stride=(envelope, 1))
        self.assertTrue(MambaPoolHost._slots_are_strided(view))

    def test_empty_tensor_is_not_strided(self):
        self.assertFalse(MambaPoolHost._slots_are_strided(torch.zeros((0, 4))))


if __name__ == "__main__":
    unittest.main()
