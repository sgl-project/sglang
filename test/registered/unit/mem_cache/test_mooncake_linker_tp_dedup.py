import unittest
from types import SimpleNamespace

from sglang.srt.mem_cache.hicache_storage import PoolName
from sglang.srt.mem_cache.storage.mooncake_store.mooncake_direct_linker import (
    _storage_sharing,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_REPLICATED = {PoolName.KV, PoolName.INDEXER, PoolName.DRAFT}


def _sharing(dcp_size=1, **kwargs):
    pools = {
        name: SimpleNamespace(
            tp_replicated=name in _REPLICATED,
            dcp_sharded=dcp_size > 1 and name in _REPLICATED,
        )
        for name in (*_REPLICATED, PoolName.MAMBA)
    }
    args = dict(
        rank_replicated=False,
        tp_rank=0,
        tp_size=1,
        attn_cp_rank=0,
        pp_rank=0,
        offload_owner=True,
    )
    args.update(kwargs)
    return _storage_sharing(
        pools, dcp_size=dcp_size, dcp_rank=args["tp_rank"] % dcp_size, **args
    )


class TestStorageSharing(CustomTestCase):
    def test_replicated_pools_share_one_key_and_one_writer(self):
        writers = {}
        for rank in range(8):
            suffixes, offload, tp_shared = _sharing(tp_rank=rank, tp_size=8)
            self.assertEqual(tp_shared, _REPLICATED)
            self.assertEqual(
                suffixes, dict.fromkeys(_REPLICATED, "tpshared_v1_cp0_pp0")
            )
            writers[rank] = offload
        self.assertEqual(writers[0], _REPLICATED | {PoolName.MAMBA})
        for rank in range(1, 8):
            self.assertEqual(writers[rank], {PoolName.MAMBA})

    def test_each_cp_rank_keeps_its_own_shared_object(self):
        for cp in range(2):
            suffixes, _, _ = _sharing(tp_rank=1, tp_size=2, attn_cp_rank=cp)
            self.assertEqual(suffixes[PoolName.KV], f"tpshared_v1_cp{cp}_pp0")

    def test_single_tp_rank_or_replicated_group_keeps_group_keys(self):
        for kwargs in ({"tp_size": 1}, {"rank_replicated": True, "tp_size": 8}):
            suffixes, offload, tp_shared = _sharing(**kwargs)
            self.assertEqual((suffixes, tp_shared), ({}, set()))
            self.assertEqual(offload, _REPLICATED | {PoolName.MAMBA})

    def test_dcp_keys_each_rank_and_the_first_block_writes(self):
        for rank in range(16):
            suffixes, offload, _ = _sharing(tp_rank=rank, tp_size=16, dcp_size=8)
            self.assertEqual(
                suffixes,
                dict.fromkeys(_REPLICATED, f"tpshared_v1_dcp8r{rank % 8}_cp0_pp0"),
            )
            self.assertEqual(
                offload,
                _REPLICATED | {PoolName.MAMBA} if rank < 8 else {PoolName.MAMBA},
            )

    def test_non_owner_rank_offloads_nothing(self):
        _, offload, _ = _sharing(tp_rank=3, tp_size=8, offload_owner=False)
        self.assertEqual(offload, set())


if __name__ == "__main__":
    unittest.main()
