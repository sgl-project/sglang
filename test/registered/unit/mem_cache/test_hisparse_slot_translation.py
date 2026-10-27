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

    def test_device_transfer_copies_physical_slots_without_remapping(self):
        """Accept commit moves target KV, not logical IDs or indexer KV."""
        pool = self.make_pool(HiSparseDSATokenToKVPool)
        pool.layer_num = 2
        pool.bytes_per_token = 4
        pool.kv_buffer = [
            torch.arange(12, dtype=torch.float32) + 100 * layer
            for layer in range(pool.layer_num)
        ]
        pool.data_ptrs = torch.tensor(
            [buf.data_ptr() for buf in pool.kv_buffer], dtype=torch.uint64
        )
        pool.data_strides = torch.full_like(pool.data_ptrs, 4)
        pool.register_mapping(torch.arange(12).flip(0))
        pool.index_key_cache = [torch.arange(12)]
        expected = [buf.clone() for buf in pool.kv_buffer]
        for buf in expected:
            buf[[1, 3]] = buf[[6, 8]]

        def copy_physical_rows(
            data_ptrs,
            strides,
            dst_indices,
            src_indices,
            num_locs,
            num_locs_upper,
            config,
        ):
            self.assertEqual(num_locs, 2)
            self.assertEqual(num_locs_upper, 2)
            torch.testing.assert_close(data_ptrs, pool.data_ptrs)
            torch.testing.assert_close(strides, pool.data_strides)
            for buf in pool.kv_buffer:
                buf[dst_indices] = buf[src_indices]

        with patch.object(
            hisparse_memory_pool,
            "copy_all_layer_kv_cache_func",
            side_effect=copy_physical_rows,
        ):
            pool.transfer_values_on_device(
                dst_indices=torch.tensor([1, 0, 3, 0])[::2],
                src_indices=torch.tensor([6, 0, 8, 0])[::2],
            )

        for actual, reference in zip(pool.kv_buffer, expected):
            torch.testing.assert_close(actual, reference)
        torch.testing.assert_close(
            pool.full_to_hisparse_device_index_mapping, torch.arange(12).flip(0)
        )
        torch.testing.assert_close(pool.index_key_cache[0], torch.arange(12))

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
