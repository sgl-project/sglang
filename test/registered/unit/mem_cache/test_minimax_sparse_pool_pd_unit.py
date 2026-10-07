import unittest
from unittest.mock import patch

import torch

from sglang.srt.disaggregation.utils import get_kv_transfer_buf_infos
from sglang.srt.mem_cache.memory_pool import MiniMaxSparseKVPool
from sglang.srt.mem_cache.pool_host.mha import (
    HiSparseMHATokenToKVPoolHost,
    MHATokenToKVPoolHost,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _make_k_only_pool(
    start_layer: int = 0, *, enable_hisparse: bool = False
) -> MiniMaxSparseKVPool:
    """Mirror the released MiniMax-M3 config shape: all sparse layers K-only."""
    dense_layer_ids = [start_layer, start_layer + 1, start_layer + 2]
    sparse_layer_ids = [start_layer + 3 + i for i in range(4)]
    end_layer = sparse_layer_ids[-1] + 1
    return MiniMaxSparseKVPool(
        size=8,
        page_size=4,
        dtype=torch.float32,
        head_num=2,
        head_dim=8,
        idx_head_dim=16,
        dense_layer_ids=dense_layer_ids,
        sparse_layer_ids=sparse_layer_ids,
        disable_value_sparse_layer_ids=sparse_layer_ids,
        device="cpu",
        start_layer=start_layer,
        end_layer=end_layer,
        enable_hisparse=enable_hisparse,
    )


class TestMiniMaxSparsePoolPD(CustomTestCase):
    def test_contiguous_buf_infos_main_only(self):
        pool = _make_k_only_pool()
        ptrs, lens, item_lens = pool.get_contiguous_buf_infos()
        # Main K/V only: 2 entries per main layer (K then V), no index buffers.
        n = pool.main_pool.layer_num
        self.assertEqual(len(ptrs), 2 * n)
        self.assertEqual(len(lens), 2 * n)
        self.assertEqual(len(item_lens), 2 * n)
        self.assertEqual(ptrs, pool.main_pool.get_contiguous_buf_infos()[0])

    def test_index_k_state_buf_infos(self):
        pool = _make_k_only_pool()
        ptrs, lens, item_lens = pool.get_index_k_state_buf_infos()
        n = pool.index_k_pool.layer_num
        self.assertEqual(len(ptrs), n)
        self.assertEqual(len(lens), n)
        self.assertEqual(len(item_lens), n)
        for i in range(n):
            buf = pool.index_k_pool.k_buffer[i]
            self.assertEqual(ptrs[i], buf.data_ptr())
            self.assertEqual(lens[i], buf.nbytes)
            self.assertEqual(item_lens[i], buf[0].nbytes * pool.page_size)

    def test_hisparse_host_registration(self):
        """PD startup must register every sparse host K/V buffer with page strides."""
        pool = _make_k_only_pool(enable_hisparse=True)
        host = HiSparseMHATokenToKVPoolHost.__new__(HiSparseMHATokenToKVPoolHost)
        with patch(
            "sglang.srt.mem_cache.pool_host.base.host_memory_budget_bytes",
            return_value=1 << 30,
        ):
            MHATokenToKVPoolHost.__init__(
                host,
                device_pool=pool.main_pool,
                host_to_device_ratio=2,
                host_size=0,
                page_size=pool.page_size,
                layout="layer_first",
                pin_memory=False,
            )
        ptrs, lens, item_lens = get_kv_transfer_buf_infos(host)
        buffers = list(host.k_buffer.unbind()) + list(host.v_buffer.unbind())
        self.assertEqual(len(buffers), 8)
        self.assertEqual(ptrs, [buffer.data_ptr() for buffer in buffers])
        self.assertEqual(lens, [buffer.nbytes for buffer in buffers])
        self.assertEqual(
            item_lens, [buffer[0].nbytes * pool.page_size for buffer in buffers]
        )

    def test_pd_registration_separates_dense_and_sparse_layers(self):
        """Both PD peers must keep dense device KV out of the sparse transfer list."""
        for hisparse in (False, True):
            pool = _make_k_only_pool(enable_hisparse=hisparse)
            for layers, infos in (
                (range(3, 7), get_kv_transfer_buf_infos(pool)),
                (range(3), pool.get_dense_kv_state_buf_infos()),
            ):
                with self.subTest(hisparse=hisparse, layers=layers):
                    buffers = [pool.get_key_buffer(i) for i in layers] + [
                        pool.get_value_buffer(i) for i in layers
                    ]
                    ptrs, lens, item_lens = infos
                    self.assertEqual(ptrs, [buffer.data_ptr() for buffer in buffers])
                    self.assertEqual(lens, [buffer.nbytes for buffer in buffers])
                    self.assertEqual(
                        item_lens,
                        [buffer[0].nbytes * pool.page_size for buffer in buffers],
                    )


if __name__ == "__main__":
    unittest.main()
