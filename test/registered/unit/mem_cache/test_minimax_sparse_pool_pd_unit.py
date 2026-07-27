import unittest

import torch

from sglang.srt.mem_cache.memory_pool import MHATokenToKOnlyPool, MiniMaxSparseKVPool
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _make_k_only_pool(start_layer: int = 0) -> MiniMaxSparseKVPool:
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
    )


def _all_buffers(pool: MiniMaxSparseKVPool):
    return [
        *pool.main_pool.k_buffer,
        *pool.main_pool.v_buffer,
        *pool.index_k_pool.k_buffer,
    ]


def _fill_and_snapshot(pool: MiniMaxSparseKVPool):
    snapshots = []
    for buffer_id, buffer in enumerate(_all_buffers(pool)):
        values = torch.arange(buffer.numel(), dtype=torch.int64).reshape(buffer.shape)
        buffer.copy_(((values + 17 * buffer_id) % 251).to(buffer.dtype))
        snapshots.append(buffer.clone())
    return snapshots


class TestMiniMaxSparsePoolPD(unittest.TestCase):
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

    def test_cpu_copy_round_trip(self):
        # Released M3 shape: all sparse layers K-only, so index_kv_pool is None
        # and get/load_cpu_copy must round-trip main K/V + index-K and skip the
        # absent index_kv sub-pool.
        pool = _make_k_only_pool()
        self.assertIsNone(pool.index_kv_pool)
        self.assertIsNotNone(pool.index_k_pool)

        indices = torch.tensor([7, 1, 5], dtype=torch.long)
        # Force >1 chunk so the chunked copy path is exercised.
        pool.main_pool.cpu_offloading_chunk_size = 2
        pool.index_k_pool.cpu_offloading_chunk_size = 2

        snapshots = _fill_and_snapshot(pool)
        cpu_copy = pool.get_cpu_copy(indices)

        self.assertEqual(set(cpu_copy), {"main", "index_kv", "index_k"})
        self.assertIsNone(cpu_copy["index_kv"])
        self.assertEqual(cpu_copy["main"][0][0][0].device.type, "cpu")

        for buffer in _all_buffers(pool):
            buffer[indices] = 0
        pool.load_cpu_copy(cpu_copy, indices)

        for buffer, snapshot in zip(_all_buffers(pool), snapshots):
            torch.testing.assert_close(buffer, snapshot, rtol=0, atol=0)


def _make_extended_pool(
    sparse_value_mode: str, start_layer: int = 0
) -> MiniMaxSparseKVPool:
    dense_layer_ids = [start_layer, start_layer + 1, start_layer + 2]
    sparse_layer_ids = [start_layer + 3 + i for i in range(4)]
    end_layer = sparse_layer_ids[-1] + 1

    if sparse_value_mode == "kv_only":
        disable_value_sparse_layer_ids = []
    elif sparse_value_mode == "mixed":
        disable_value_sparse_layer_ids = sparse_layer_ids[::2]
    else:
        raise ValueError(f"Unknown sparse value mode: {sparse_value_mode}")

    return MiniMaxSparseKVPool(
        size=8,
        page_size=4,
        dtype=torch.float32,
        head_num=2,
        head_dim=8,
        idx_head_dim=16,
        dense_layer_ids=dense_layer_ids,
        sparse_layer_ids=sparse_layer_ids,
        disable_value_sparse_layer_ids=disable_value_sparse_layer_ids,
        device="cpu",
        start_layer=start_layer,
        end_layer=end_layer,
    )


def _named_extended_buffers(pool: MiniMaxSparseKVPool):
    buffers = []
    buffers.extend(
        (f"main.k[{i}]", buffer) for i, buffer in enumerate(pool.main_pool.k_buffer)
    )
    buffers.extend(
        (f"main.v[{i}]", buffer) for i, buffer in enumerate(pool.main_pool.v_buffer)
    )
    if pool.index_kv_pool is not None:
        buffers.extend(
            (f"index_kv.k[{i}]", buffer)
            for i, buffer in enumerate(pool.index_kv_pool.k_buffer)
        )
        buffers.extend(
            (f"index_kv.v[{i}]", buffer)
            for i, buffer in enumerate(pool.index_kv_pool.v_buffer)
        )
    if pool.index_k_pool is not None:
        buffers.extend(
            (f"index_k.k[{i}]", buffer)
            for i, buffer in enumerate(pool.index_k_pool.k_buffer)
        )
    return buffers


def _fill_extended_buffers(pool: MiniMaxSparseKVPool):
    snapshots = {}
    for buffer_id, (name, buffer) in enumerate(_named_extended_buffers(pool)):
        values = torch.arange(buffer.numel(), dtype=torch.int64).reshape(buffer.shape)
        buffer.copy_(((values + 17 * buffer_id) % 251).to(buffer.dtype))
        snapshots[name] = buffer.clone()
    return snapshots


class TestMiniMaxSparsePoolCPUCopies(unittest.TestCase):
    source_indices = torch.tensor([7, 1, 5], dtype=torch.long)
    destination_indices = torch.tensor([2, 6, 0], dtype=torch.long)

    def test_k_only_pool_chunked_copy_to_new_indices(self):
        pool = MHATokenToKOnlyPool(
            size=8,
            page_size=4,
            dtype=torch.float32,
            head_num=1,
            head_dim=16,
            layer_num=2,
            device="cpu",
            enable_memory_saver=False,
        )
        pool.cpu_offloading_chunk_size = 2

        snapshots = []
        for layer_id, buffer in enumerate(pool.k_buffer):
            values = torch.arange(buffer.numel(), dtype=torch.int64).reshape(
                buffer.shape
            )
            buffer.copy_(((values + 17 * layer_id) % 251).to(buffer.dtype))
            snapshots.append(buffer.clone())

        cpu_copy = pool.get_cpu_copy(self.source_indices)

        self.assertEqual(len(cpu_copy), pool.layer_num)
        for layer_id, chunks in enumerate(cpu_copy):
            self.assertEqual([chunk.shape[0] for chunk in chunks], [2, 1])
            self.assertTrue(all(chunk.device.type == "cpu" for chunk in chunks))
            torch.testing.assert_close(
                torch.cat(chunks),
                snapshots[layer_id][self.source_indices],
                rtol=0,
                atol=0,
            )

        for buffer in pool.k_buffer:
            buffer[self.source_indices] = 0
            buffer[self.destination_indices] = 0
        pool.load_cpu_copy(cpu_copy, self.destination_indices)

        for layer_id, buffer in enumerate(pool.k_buffer):
            expected = snapshots[layer_id].clone()
            expected[self.source_indices] = 0
            expected[self.destination_indices] = snapshots[layer_id][
                self.source_indices
            ]
            torch.testing.assert_close(buffer, expected, rtol=0, atol=0)

    def _assert_composite_round_trip(
        self,
        pool: MiniMaxSparseKVPool,
        *,
        has_index_kv: bool,
        has_index_k: bool,
    ):
        for sub_pool in (pool.main_pool, pool.index_kv_pool, pool.index_k_pool):
            if sub_pool is not None:
                sub_pool.cpu_offloading_chunk_size = 2

        snapshots = _fill_extended_buffers(pool)
        cpu_copy = pool.get_cpu_copy(self.source_indices)

        self.assertEqual(set(cpu_copy), {"main", "index_kv", "index_k"})
        self.assertEqual(cpu_copy["index_kv"] is not None, has_index_kv)
        self.assertEqual(cpu_copy["index_k"] is not None, has_index_k)

        for _, buffer in _named_extended_buffers(pool):
            buffer[self.source_indices] = 0
            buffer[self.destination_indices] = 0
        pool.load_cpu_copy(cpu_copy, self.destination_indices)

        for name, buffer in _named_extended_buffers(pool):
            expected = snapshots[name].clone()
            expected[self.source_indices] = 0
            expected[self.destination_indices] = snapshots[name][self.source_indices]
            torch.testing.assert_close(buffer, expected, rtol=0, atol=0, msg=name)

    def test_composite_copy_without_index_k_pool(self):
        pool = _make_extended_pool("kv_only")
        self.assertIsNotNone(pool.index_kv_pool)
        self.assertIsNone(pool.index_k_pool)
        self._assert_composite_round_trip(pool, has_index_kv=True, has_index_k=False)

    def test_composite_copy_with_both_index_pools(self):
        pool = _make_extended_pool("mixed")
        self.assertIsNotNone(pool.index_kv_pool)
        self.assertIsNotNone(pool.index_k_pool)
        self._assert_composite_round_trip(pool, has_index_kv=True, has_index_k=True)


if __name__ == "__main__":
    unittest.main()
