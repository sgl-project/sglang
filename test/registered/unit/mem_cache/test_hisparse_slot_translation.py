"""Check portable HiSparse pool translation dispatch and gather fallbacks."""

import unittest
from unittest.mock import PropertyMock, patch

import torch

from sglang.srt.mem_cache import hisparse_memory_pool
from sglang.srt.mem_cache.hisparse_memory_pool import HiSparseDSATokenToKVPool
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, stage="base-a", runner_config="cpu")


class TestHiSparseSlotTranslation(CustomTestCase):
    def make_pool(self, pool_type):
        return pool_type.__new__(pool_type)

    def test_pool_translation_selects_fused_kernel_for_gpu_slot_lists(self):
        pool = self.make_pool(HiSparseDSATokenToKVPool)
        mapping = torch.arange(32, dtype=torch.int64) + 100
        pool.register_mapping(mapping)
        storage = torch.tensor([17, 0, -1, 0, 18, 0])
        locations = storage[::2]
        expected = torch.tensor([117, -1, 118])
        with (
            patch.object(
                torch.Tensor, "is_cuda", new_callable=PropertyMock, return_value=True
            ),
            patch.object(
                hisparse_memory_pool,
                "translate_padded_hisparse_locations",
                return_value=expected,
            ) as fused,
        ):
            self.assertIs(pool.translate_loc_to_hisparse_device(locations), expected)
            self.assertIs(fused.call_args.args[0], mapping)
            self.assertIs(fused.call_args.args[1], locations)
            fused.assert_called_once()
        torch.testing.assert_close(storage, torch.tensor([17, 0, -1, 0, 18, 0]))

    def test_pool_translation_keeps_gather_for_page_tables_and_other_devices(self):
        pool = self.make_pool(HiSparseDSATokenToKVPool)
        mapping = torch.arange(32, dtype=torch.int64) + 100
        pool.register_mapping(mapping)
        for gpu, locations in (
            (False, torch.tensor([17, -1, 18])),
            (True, torch.tensor([[17, -1], [18, 0]])),
            (True, torch.tensor(17)),
        ):
            with (
                self.subTest(gpu=gpu, shape=locations.shape),
                patch.object(
                    torch.Tensor, "is_cuda", new_callable=PropertyMock, return_value=gpu
                ),
                patch.object(
                    hisparse_memory_pool, "translate_padded_hisparse_locations"
                ) as fused,
            ):
                actual = pool.translate_loc_to_hisparse_device(locations)
                torch.testing.assert_close(actual, mapping[locations])
                self.assertEqual(actual.shape, locations.shape)
                fused.assert_not_called()


if __name__ == "__main__":
    unittest.main()
