import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=1, suite="stage-a-unit-test-npu")

from sglang.srt.hardware_backend.npu import memory_pool_npu
from sglang.srt.hardware_backend.npu.memory_pool_npu import NPUMHATokenToKVPool
from sglang.srt.mem_cache.memory_pool import KVWriteLoc
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool


class TestNPUScatterPaKVCache(unittest.TestCase):
    @staticmethod
    def _make_pool(*, enabled=True, use_fia=True, dtype=torch.bfloat16):
        pool = object.__new__(NPUMHATokenToKVPool)
        pool.use_scatter_pa_kv_cache = enabled
        pool.use_fia = use_fia
        pool.dtype = dtype
        pool.store_dtype = dtype
        pool.start_layer = 0
        pool.page_size = 4
        pool.head_num = 2
        pool.head_dim = 3
        pool.v_head_dim = 5
        if use_fia:
            pool.k_buffer = [torch.zeros((8, 1, 2, 3), dtype=dtype)]
            pool.v_buffer = [torch.zeros((8, 1, 2, 5), dtype=dtype)]
        else:
            pool.k_buffer = [torch.zeros((2, 4, 2, 3), dtype=dtype)]
            pool.v_buffer = [torch.zeros((2, 4, 2, 5), dtype=dtype)]
        return pool

    @staticmethod
    def _inputs(dtype=torch.bfloat16):
        layer = SimpleNamespace(layer_id=0)
        loc = torch.tensor([6, 1], dtype=torch.int64)
        cache_k = torch.arange(12, dtype=torch.float32).to(dtype).view(2, 2, 3)
        cache_v = torch.arange(20, dtype=torch.float32).to(dtype).view(2, 2, 5)
        return layer, loc, cache_k, cache_v

    def test_scatter_pa_receives_norm_paged_layout(self):
        pool = self._make_pool()
        layer, loc, cache_k, cache_v = self._inputs()

        def scatter_pa(key, value, key_cache, value_cache, slot_mapping, **kwargs):
            self.assertEqual(kwargs, {"cache_mode": "Norm"})
            key_cache.view(-1, pool.head_num, pool.head_dim).index_copy_(
                0, slot_mapping.long(), key
            )
            value_cache.view(-1, pool.head_num, pool.v_head_dim).index_copy_(
                0, slot_mapping.long(), value
            )

        fake_torch_npu = SimpleNamespace(
            npu_scatter_pa_kv_cache=MagicMock(side_effect=scatter_pa),
            npu_scatter_nd_update_=MagicMock(),
        )

        with patch.object(memory_pool_npu, "torch_npu", fake_torch_npu, create=True):
            pool.set_kv_buffer(
                layer,
                loc,
                cache_k,
                cache_v,
                use_scatter_pa_kv_cache=True,
            )

        fake_torch_npu.npu_scatter_nd_update_.assert_not_called()
        call = fake_torch_npu.npu_scatter_pa_kv_cache.call_args
        self.assertIsNotNone(call)
        key, value, key_cache, value_cache, slot_mapping = call.args
        self.assertEqual(key.shape, (2, 2, 3))
        self.assertEqual(value.shape, (2, 2, 5))
        self.assertEqual(key_cache.shape, (2, 4, 2, 3))
        self.assertEqual(value_cache.shape, (2, 4, 2, 5))
        self.assertEqual(slot_mapping.dtype, torch.int32)
        self.assertTrue(all(tensor.is_contiguous() for tensor in (key, value)))
        self.assertTrue(slot_mapping.is_contiguous())
        self.assertEqual(call.kwargs, {"cache_mode": "Norm"})
        torch.testing.assert_close(
            pool.k_buffer[0].view(-1, pool.head_num, pool.head_dim)[loc], cache_k
        )
        torch.testing.assert_close(
            pool.v_buffer[0].view(-1, pool.head_num, pool.v_head_dim)[loc], cache_v
        )

    def test_prefill_keeps_scatter_nd_when_decode_hint_is_false(self):
        pool = self._make_pool(enabled=True)
        layer, loc, cache_k, cache_v = self._inputs()
        fake_torch_npu = SimpleNamespace(
            npu_scatter_pa_kv_cache=MagicMock(),
            npu_scatter_nd_update_=MagicMock(),
        )

        with patch.object(memory_pool_npu, "torch_npu", fake_torch_npu, create=True):
            pool.set_kv_buffer(layer, loc, cache_k, cache_v)

        fake_torch_npu.npu_scatter_pa_kv_cache.assert_not_called()
        self.assertEqual(fake_torch_npu.npu_scatter_nd_update_.call_count, 2)

    def test_disabled_environment_flag_keeps_scatter_nd(self):
        pool = self._make_pool(enabled=False)
        layer, loc, cache_k, cache_v = self._inputs()
        fake_torch_npu = SimpleNamespace(
            npu_scatter_pa_kv_cache=MagicMock(),
            npu_scatter_nd_update_=MagicMock(),
        )

        with patch.object(memory_pool_npu, "torch_npu", fake_torch_npu, create=True):
            pool.set_kv_buffer(
                layer,
                loc,
                cache_k,
                cache_v,
                use_scatter_pa_kv_cache=True,
            )

        fake_torch_npu.npu_scatter_pa_kv_cache.assert_not_called()
        self.assertEqual(fake_torch_npu.npu_scatter_nd_update_.call_count, 2)

    def test_scatter_pa_requires_fia(self):
        pool = self._make_pool(use_fia=False)
        layer, loc, cache_k, cache_v = self._inputs()

        with self.assertRaisesRegex(RuntimeError, "requires ASCEND_USE_FIA=1"):
            pool.set_kv_buffer(
                layer,
                loc,
                cache_k,
                cache_v,
                use_scatter_pa_kv_cache=True,
            )

    def test_scatter_pa_requires_torch_npu_operator(self):
        pool = self._make_pool()
        layer, loc, cache_k, cache_v = self._inputs()
        fake_torch_npu = SimpleNamespace(npu_scatter_nd_update_=MagicMock())

        with (
            patch.object(memory_pool_npu, "torch_npu", fake_torch_npu, create=True),
            self.assertRaisesRegex(RuntimeError, "provides npu_scatter_pa_kv_cache"),
        ):
            pool.set_kv_buffer(
                layer,
                loc,
                cache_k,
                cache_v,
                use_scatter_pa_kv_cache=True,
            )

    def test_scatter_pa_rejects_nonportable_cache_dtype(self):
        pool = self._make_pool(dtype=torch.float32)
        layer, loc, cache_k, cache_v = self._inputs(dtype=torch.float32)
        fake_torch_npu = SimpleNamespace(npu_scatter_pa_kv_cache=MagicMock())

        with (
            patch.object(memory_pool_npu, "torch_npu", fake_torch_npu, create=True),
            self.assertRaisesRegex(RuntimeError, "supports fp16, bf16, and int8"),
        ):
            pool.set_kv_buffer(
                layer,
                loc,
                cache_k,
                cache_v,
                use_scatter_pa_kv_cache=True,
            )

    def test_swa_pool_forwards_decode_hint_to_selected_npu_pool(self):
        inner_pool = SimpleNamespace(
            use_scatter_pa_kv_cache=True,
            set_kv_buffer=MagicMock(),
        )
        pool = object.__new__(SWAKVPool)
        pool.full_kv_pool = SimpleNamespace(set_kv_buffer=MagicMock())
        pool.swa_kv_pool = inner_pool
        pool.layers_mapping = {3: (0, True)}
        layer = SimpleNamespace(layer_id=3)
        loc = torch.tensor([4, 5])
        swa_loc = torch.tensor([0, 1])
        cache_k = torch.zeros((2, 2, 3), dtype=torch.bfloat16)
        cache_v = torch.zeros((2, 2, 5), dtype=torch.bfloat16)

        pool.set_kv_buffer(
            layer,
            KVWriteLoc(loc, swa_loc),
            cache_k,
            cache_v,
            use_scatter_pa_kv_cache=True,
        )

        inner_pool.set_kv_buffer.assert_called_once_with(
            None,
            swa_loc,
            cache_k,
            cache_v,
            1.0,
            1.0,
            layer_id_override=0,
            use_scatter_pa_kv_cache=True,
        )


if __name__ == "__main__":
    unittest.main()
