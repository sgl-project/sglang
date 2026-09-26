"""Unit tests for the static-shape NPU prefix-valid KV write locations."""

import unittest
from unittest.mock import Mock, patch

import torch

from sglang.srt.hardware_backend.npu.memory_pool_npu import (
    NPUMHATokenToKVPool,
    _build_prefix_valid_write_locs,
)
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestNpuPrefixValidWriteLocs(CustomTestCase):
    def test_invalid_suffixes_use_dummy_slot(self):
        loc_2d = torch.tensor([[10, 11, 12, 13], [20, 21, 22, 23], [30, 31, 32, 33]])
        commit_lens = torch.tensor([0, 2, 4], dtype=torch.int32)

        actual = _build_prefix_valid_write_locs(loc_2d, commit_lens)

        torch.testing.assert_close(
            actual,
            torch.tensor([0, 0, 0, 0, 20, 21, 0, 0, 30, 31, 32, 33]),
            rtol=0,
            atol=0,
        )

    def test_valid_slots_match_dynamic_reference(self):
        loc_2d = torch.tensor([[4, 6, 8, 10], [5, 7, 9, 11]])
        commit_lens = torch.tensor([1, 3], dtype=torch.int32)
        values = torch.arange(8)

        static_cache = torch.full((12,), -1, dtype=torch.int64)
        static_cache.index_copy_(
            0, _build_prefix_valid_write_locs(loc_2d, commit_lens), values
        )

        row_offsets = torch.arange(loc_2d.shape[1])
        valid_mask = row_offsets[None, :] < commit_lens[:, None]
        dynamic_cache = torch.full_like(static_cache, -1)
        dynamic_cache.index_copy_(
            0, loc_2d[valid_mask], values.reshape_as(loc_2d)[valid_mask]
        )

        valid_slots = loc_2d[valid_mask]
        torch.testing.assert_close(
            static_cache.index_select(0, valid_slots),
            dynamic_cache.index_select(0, valid_slots),
            rtol=0,
            atol=0,
        )

    def test_rejects_mismatched_batch(self):
        with self.assertRaisesRegex(ValueError, "must match loc_2d batch size"):
            _build_prefix_valid_write_locs(
                torch.zeros((2, 4), dtype=torch.int64),
                torch.ones((3,), dtype=torch.int32),
            )

    def test_prefix_valid_writer_uses_static_locations(self):
        set_kv_buffer = Mock()
        pool = NPUMHATokenToKVPool.__new__(NPUMHATokenToKVPool)
        pool.use_triton_prefix_kv_cache_store = False
        pool.use_static_prefix_valid_write = True
        pool.set_kv_buffer = set_kv_buffer
        layer = object()
        loc_2d = torch.tensor([[10, 11, 12], [20, 21, 22]])
        commit_lens = torch.tensor([1, 2], dtype=torch.int32)
        cache_k = torch.randn(6, 2, 4)
        cache_v = torch.randn(6, 2, 4)

        NPUMHATokenToKVPool.set_kv_buffer_prefix_valid(
            pool,
            layer,
            loc_2d,
            commit_lens,
            cache_k,
            cache_v,
            0.5,
            0.25,
            3,
        )

        set_kv_buffer.assert_called_once()
        args = set_kv_buffer.call_args.args
        self.assertIs(args[0], layer)
        torch.testing.assert_close(
            args[1], torch.tensor([10, 0, 0, 20, 21, 0]), rtol=0, atol=0
        )
        self.assertIs(args[2], cache_k)
        self.assertIs(args[3], cache_v)
        self.assertEqual(args[4:], (0.5, 0.25))
        self.assertEqual(set_kv_buffer.call_args.kwargs, {"layer_id_override": 3})

    def test_prefix_valid_writer_can_use_dynamic_parent_path(self):
        pool = NPUMHATokenToKVPool.__new__(NPUMHATokenToKVPool)
        pool.use_triton_prefix_kv_cache_store = False
        pool.use_static_prefix_valid_write = False
        layer = object()
        loc_2d = torch.tensor([[10, 11, 12]])
        commit_lens = torch.tensor([2], dtype=torch.int32)
        cache_k = torch.randn(3, 2, 4)
        cache_v = torch.randn(3, 2, 4)

        with patch.object(
            MHATokenToKVPool,
            "set_kv_buffer_prefix_valid",
            return_value="dynamic-result",
        ) as parent_writer:
            result = pool.set_kv_buffer_prefix_valid(
                layer,
                loc_2d,
                commit_lens,
                cache_k,
                cache_v,
                0.5,
                0.25,
                3,
            )

        self.assertEqual(result, "dynamic-result")
        parent_writer.assert_called_once_with(
            layer,
            loc_2d,
            commit_lens,
            cache_k,
            cache_v,
            0.5,
            0.25,
            3,
        )


if __name__ == "__main__":
    unittest.main()
