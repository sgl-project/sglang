import unittest
from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import MooncakeStore
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _make_store(dcp_size: int, dcp_rank: int, logical_tokens: int):
    layer_num = 2
    kv_cache_dim = 3
    physical_tokens = logical_tokens // dcp_size
    kv_buffer = torch.empty(
        (physical_tokens, layer_num, 1, kv_cache_dim), dtype=torch.float32
    )
    host_pool = SimpleNamespace(
        kv_buffer=kv_buffer,
        layer_num=layer_num,
        kv_cache_dim=kv_cache_dim,
        dtype=kv_buffer.dtype,
        _storage_physical_page_start=lambda index: int(index) // dcp_size,
    )
    pool_group = SimpleNamespace(
        layout="page_first",
        logical_page_size=64 * dcp_size,
        anchor_entry=SimpleNamespace(host_pool=host_pool),
    )
    store = object.__new__(MooncakeStore)
    store.mem_pool_host = pool_group
    store.canonical_dcp_size = 8
    store.canonical_page_size = 64
    store.canonical_mla_scope = "canonv1_tp8_pp0of1"
    store.attn_dcp_size = dcp_size
    store.attn_dcp_rank = dcp_rank
    return store, host_pool


class TestMooncakeCanonicalDCP(unittest.TestCase):
    def test_dcp1_and_dcp8_produce_the_same_object_keys(self):
        hashes = [f"hash-{i}" for i in range(8)]
        dcp1, _ = _make_store(dcp_size=1, dcp_rank=0, logical_tokens=512)
        dcp1_keys = dcp1._canonical_mla_keys(hashes)

        dcp8_keys = []
        for rank in range(8):
            store, _ = _make_store(
                dcp_size=8, dcp_rank=rank, logical_tokens=512
            )
            dcp8_keys.extend(store._canonical_mla_keys(hashes))

        self.assertEqual(set(dcp1_keys), set(dcp8_keys))
        self.assertEqual(len(dcp1_keys), 64)
        self.assertEqual(len(dcp8_keys), 64)

    def test_dcp1_scatter_and_dcp8_shard_buffer_shapes_match(self):
        row_bytes = 2 * 3 * torch.float32.itemsize
        dcp1, dcp1_pool = _make_store(
            dcp_size=1, dcp_rank=0, logical_tokens=64
        )
        keys1, ptrs1, sizes1 = dcp1._get_canonical_mla_buffer_meta(
            ["hash"], torch.arange(64)
        )

        dcp8, dcp8_pool = _make_store(
            dcp_size=8, dcp_rank=3, logical_tokens=512
        )
        hashes = [f"hash-{i}" for i in range(8)]
        keys8, ptrs8, sizes8 = dcp8._get_canonical_mla_buffer_meta(
            hashes, torch.arange(512)
        )

        shard3 = keys1.index("hash_canonv1_tp8_pp0of1_dcp3of8_k")
        self.assertEqual(
            ptrs1[shard3],
            [
                dcp1_pool.kv_buffer.data_ptr() + row_bytes * token
                for token in range(3, 64, 8)
            ],
        )
        self.assertEqual(sizes1[shard3], [row_bytes] * 8)
        self.assertEqual(keys8[0], "hash-0_canonv1_tp8_pp0of1_dcp3of8_k")
        self.assertEqual(ptrs8[0], dcp8_pool.kv_buffer.data_ptr())
        self.assertEqual(sizes8[0], row_bytes * 8)

    def test_canonical_results_are_grouped_per_runtime_page(self):
        store = object.__new__(MooncakeStore)
        results = [1] * 8 + [1] * 7 + [-1]

        self.assertEqual(
            store._batch_postprocess(results, key_multiplier=8),
            [True, False],
        )

    def test_hash_count_must_match_logical_pages(self):
        store, _ = _make_store(dcp_size=8, dcp_rank=0, logical_tokens=512)

        with self.assertRaisesRegex(ValueError, "Canonical hash/page mismatch"):
            store._get_canonical_mla_buffer_meta(
                ["only-one-hash"], torch.arange(512)
            )


if __name__ == "__main__":
    unittest.main()
