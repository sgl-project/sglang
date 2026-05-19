"""Unit tests for ``sglang.srt.utils.offloader_v2_mem_pool``.

Run with:
    python -m unittest python.sglang.test.offloader_v2.test_mem_pool -v
"""

from __future__ import annotations

import unittest

import torch

from sglang.srt.utils.offloader_v2_mem_pool import ParamInfo, StaticBufferPool


def _cpu() -> torch.device:
    return torch.device("cpu")


class TestStaticBufferPool(unittest.TestCase):
    def _make_param_infos(self):
        return [
            ParamInfo(name="a", shape=(4, 8), stride=(8, 1), dtype=torch.float32),
            ParamInfo(name="b", shape=(2, 3), stride=(3, 1), dtype=torch.float16),
        ]

    def test_basic_allocation_shape_dtype_stride(self):
        infos = self._make_param_infos()
        pool = StaticBufferPool(param_infos=infos, slot_capacity=2, device=_cpu())

        for info in infos:
            for slot in range(2):
                t = pool.get(
                    name=info.name,
                    shape=info.shape,
                    stride=info.stride,
                    dtype=info.dtype,
                    slot_idx=slot,
                )
                self.assertIsInstance(t, torch.Tensor)
                self.assertEqual(tuple(t.shape), info.shape)
                self.assertEqual(tuple(t.stride()), info.stride)
                self.assertEqual(t.dtype, info.dtype)
                self.assertEqual(t.device, _cpu())

    def test_slots_are_distinct_tensors_per_key(self):
        info = ParamInfo(name="a", shape=(4,), stride=(1,), dtype=torch.float32)
        pool = StaticBufferPool(param_infos=[info], slot_capacity=3, device=_cpu())
        t0 = pool.get("a", (4,), (1,), torch.float32, 0)
        t1 = pool.get("a", (4,), (1,), torch.float32, 1)
        t2 = pool.get("a", (4,), (1,), torch.float32, 2)
        # Distinct storages so concurrent prefetch slots don't collide.
        self.assertNotEqual(t0.data_ptr(), t1.data_ptr())
        self.assertNotEqual(t1.data_ptr(), t2.data_ptr())
        self.assertNotEqual(t0.data_ptr(), t2.data_ptr())

    def test_slot_idx_wraps(self):
        info = ParamInfo(name="a", shape=(2,), stride=(1,), dtype=torch.float32)
        pool = StaticBufferPool(param_infos=[info], slot_capacity=2, device=_cpu())
        t0 = pool.get("a", (2,), (1,), torch.float32, 0)
        t_wrap = pool.get("a", (2,), (1,), torch.float32, 4)  # 4 % 2 == 0
        self.assertIs(t0, t_wrap)

    def test_duplicate_keys_dedup(self):
        # Same (name, shape, stride, dtype) repeated -> one set of buffers.
        info = ParamInfo(name="a", shape=(2,), stride=(1,), dtype=torch.float32)
        pool = StaticBufferPool(
            param_infos=[info, info, info], slot_capacity=2, device=_cpu()
        )
        # total_bytes counts unique keys * slots, not duplicates.
        self.assertEqual(pool.total_bytes, info.num_bytes * 2)
        self.assertEqual(len(pool._buffers), 1)

    def test_distinct_names_same_layout_not_dedup(self):
        a = ParamInfo(name="a", shape=(2,), stride=(1,), dtype=torch.float32)
        b = ParamInfo(name="b", shape=(2,), stride=(1,), dtype=torch.float32)
        pool = StaticBufferPool(param_infos=[a, b], slot_capacity=1, device=_cpu())
        self.assertEqual(len(pool._buffers), 2)
        ta = pool.get("a", (2,), (1,), torch.float32, 0)
        tb = pool.get("b", (2,), (1,), torch.float32, 0)
        self.assertNotEqual(ta.data_ptr(), tb.data_ptr())

    def test_total_bytes_accounts_all_slots(self):
        infos = self._make_param_infos()
        slot_cap = 3
        pool = StaticBufferPool(
            param_infos=infos, slot_capacity=slot_cap, device=_cpu()
        )
        expected = sum(info.num_bytes for info in infos) * slot_cap
        self.assertEqual(pool.total_bytes, expected)

    def test_slot_capacity_one(self):
        """
        When slot_capacity=1, all params are loaded to the same buffer.
        """
        info = ParamInfo(name="a", shape=(2,), stride=(1,), dtype=torch.float32)
        pool = StaticBufferPool(param_infos=[info], slot_capacity=1, device=_cpu())
        t0 = pool.get("a", (2,), (1,), torch.float32, 0)
        t1 = pool.get("a", (2,), (1,), torch.float32, 1)  # wraps to 0
        self.assertIs(t0, t1)

    def test_invalid_slot_capacity(self):
        info = ParamInfo(name="a", shape=(2,), stride=(1,), dtype=torch.float32)
        with self.assertRaises(AssertionError):
            StaticBufferPool(param_infos=[info], slot_capacity=0, device=_cpu())

    def test_get_unknown_key_raises(self):
        info = ParamInfo(name="a", shape=(2,), stride=(1,), dtype=torch.float32)
        pool = StaticBufferPool(param_infos=[info], slot_capacity=1, device=_cpu())
        with self.assertRaises(KeyError):
            pool.get("missing", (2,), (1,), torch.float32, 0)
        with self.assertRaises(KeyError):
            pool.get("a", (3,), (1,), torch.float32, 0)
        with self.assertRaises(KeyError):
            pool.get("a", (2,), (1,), torch.float16, 0)

    def test_buffers_are_writable_and_independent(self):
        info = ParamInfo(name="a", shape=(4,), stride=(1,), dtype=torch.float32)
        pool = StaticBufferPool(param_infos=[info], slot_capacity=2, device=_cpu())
        t0 = pool.get("a", (4,), (1,), torch.float32, 0)
        t1 = pool.get("a", (4,), (1,), torch.float32, 1)
        t0.copy_(torch.arange(4, dtype=torch.float32))
        t1.copy_(torch.full((4,), -1.0))
        self.assertTrue(torch.equal(t0, torch.arange(4, dtype=torch.float32)))
        self.assertTrue(torch.equal(t1, torch.full((4,), -1.0)))

    def test_non_contiguous_stride(self):
        # Column-major-ish stride for a (3, 4) tensor.
        info = ParamInfo(name="t", shape=(3, 4), stride=(1, 3), dtype=torch.float32)
        pool = StaticBufferPool(param_infos=[info], slot_capacity=1, device=_cpu())
        t = pool.get("t", (3, 4), (1, 3), torch.float32, 0)
        self.assertEqual(tuple(t.stride()), (1, 3))
        self.assertEqual(tuple(t.shape), (3, 4))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_cuda_device(self):
        device = torch.device("cuda", 0)
        info = ParamInfo(name="a", shape=(8,), stride=(1,), dtype=torch.float16)
        pool = StaticBufferPool(param_infos=[info], slot_capacity=2, device=device)
        t = pool.get("a", (8,), (1,), torch.float16, 1)
        self.assertEqual(t.device.type, "cuda")
        self.assertEqual(t.dtype, torch.float16)


if __name__ == "__main__":
    unittest.main()
