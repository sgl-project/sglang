import importlib
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _fake_mlu_ops():
    return SimpleNamespace(
        reshape_paged_cache=MagicMock(),
        flash_attention=MagicMock(),
        single_query_cached_kv_attn=MagicMock(),
    )


class TestMLUKVCacheUnits(CustomTestCase):
    def test_mha_pool_uses_mlu_paged_cache_writer(self):
        fake_ops = _fake_mlu_ops()
        with patch.dict(sys.modules, {"torch_mlu_ops": fake_ops}):
            memory_pool_mod = importlib.import_module(
                "sglang.srt.hardware_backend.mlu.memory_pool"
            )
        with patch.object(memory_pool_mod, "torch_mlu_ops", fake_ops):
            MLUMHATokenToKVPool = memory_pool_mod.MLUMHATokenToKVPool

            pool = MLUMHATokenToKVPool(
                size=32,
                page_size=16,
                dtype=torch.float32,
                head_num=2,
                head_dim=4,
                layer_num=2,
                device="cpu",
                enable_memory_saver=False,
                enable_alt_stream=False,
            )

            self.assertEqual(pool.kv_buffer.shape, (2, 2, 3, 2, 16, 4))
            self.assertTrue(pool.kv_buffer.is_contiguous())

            layer = SimpleNamespace(layer_id=1)
            loc = torch.tensor([0, 17], dtype=torch.int64)
            cache_k = torch.randn(2, 2, 4)
            cache_v = torch.randn(2, 2, 4)
            pool.set_kv_buffer(layer, loc, cache_k, cache_v)

            fake_ops.reshape_paged_cache.assert_called_once()
            kwargs = fake_ops.reshape_paged_cache.call_args.kwargs
            self.assertEqual(kwargs["slot_mapping"].dtype, torch.int32)
            self.assertEqual(tuple(kwargs["k_cache"].shape), (3, 2, 16, 4))
            self.assertEqual(tuple(kwargs["v_cache"].shape), (3, 2, 16, 4))


class TestMLUPagedAllocator(CustomTestCase):
    def test_alloc_extend_uses_soft_i64_kernel_and_advances_pages(self):
        allocator_mod = importlib.import_module(
            "sglang.srt.hardware_backend.mlu.allocator"
        )

        class FakeKernel:
            def __init__(self):
                self.grid = None
                self.kwargs = None

            def __getitem__(self, grid):
                self.grid = grid

                def launch(*args, **kwargs):
                    self.kwargs = kwargs
                    out_indices = args[4]
                    out_indices.copy_(torch.arange(out_indices.numel()))

                return launch

        fake_kernel = FakeKernel()
        with (
            patch.object(allocator_mod, "alloc_extend_kernel", fake_kernel),
            patch.object(allocator_mod, "get_num_new_pages", return_value=2),
        ):
            allocator = allocator_mod.MLUPagedTokenToKVPoolAllocator(
                size=64,
                page_size=16,
                dtype=torch.float32,
                device="cpu",
                kvcache=MagicMock(),
                need_sort=False,
            )
            out = allocator.alloc_extend(
                prefix_lens=torch.tensor([0, 16], dtype=torch.int64),
                prefix_lens_cpu=torch.tensor([0, 16], dtype=torch.int64),
                seq_lens=torch.tensor([18, 20], dtype=torch.int64),
                seq_lens_cpu=torch.tensor([18, 20], dtype=torch.int64),
                last_loc=torch.tensor([-1, 15], dtype=torch.int64),
                extend_num_tokens=6,
            )

        self.assertEqual(fake_kernel.grid, (2,))
        self.assertEqual(fake_kernel.kwargs, {"enable_soft_i64": True})
        self.assertEqual(out.tolist(), [0, 1, 2, 3, 4, 5])
        self.assertEqual(allocator.free_pages.tolist(), [3, 4])


if __name__ == "__main__":
    unittest.main()
