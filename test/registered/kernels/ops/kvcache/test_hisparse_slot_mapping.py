"""Portable HiSparse logical-to-physical slot translation kernel checks."""

import unittest
import weakref

import torch

from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_amd_ci(est_time=20, stage="jit-kernel-unit", runner_config="amd")


class TestHiSparseSlotMapping(CustomTestCase):
    def test_accepts_weak_mapping_used_by_pool(self):
        from sglang.kernels.ops.kvcache.hisparse_slot_mapping import (
            translate_padded_hisparse_locations as translate,
        )

        mapping = torch.arange(32, dtype=torch.int64, device="cuda") + 100
        locations = torch.tensor([17, -1, 18], device="cuda")

        actual = translate(weakref.proxy(mapping), locations)

        torch.testing.assert_close(actual, torch.tensor([117, -1, 118], device="cuda"))

    def test_fused_slot_mapping_matches_padded_gather(self):
        from sglang.kernels.ops.kvcache.hisparse_slot_mapping import (
            translate_padded_hisparse_locations as translate,
        )

        for map_dtype in (torch.int32, torch.int64):
            mapping = torch.arange(257, dtype=map_dtype, device="cuda") * 3 + 1
            for loc_dtype in (torch.int32, torch.int64):
                for count in (0, 1, 3, 127, 129, 1024):
                    for stride in (1, 2):
                        with self.subTest(
                            map_dtype=map_dtype,
                            loc_dtype=loc_dtype,
                            count=count,
                            stride=stride,
                        ):
                            locations = (
                                torch.arange(
                                    count * stride, dtype=loc_dtype, device="cuda"
                                )[::stride]
                                % 257
                            )
                            if stride == 2:
                                storage = torch.zeros(
                                    count * 2, dtype=loc_dtype, device="cuda"
                                )
                                storage[::2] = locations
                                locations = storage[::2]
                            locations[::3] = -1
                            locations[1::7] = -2
                            original = locations.clone()
                            expected = torch.where(
                                locations >= 0,
                                mapping[locations.clamp_min(0)],
                                locations,
                            )
                            actual = translate(mapping, locations)
                            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                            torch.testing.assert_close(
                                locations, original, rtol=0, atol=0
                            )


if __name__ == "__main__":
    unittest.main()
