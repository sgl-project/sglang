"""The Mooncake direct linker shares one object per pool along the axes whose
pages are byte-identical: attention CP (interleave PCP) and attention TP
(MLA-family pools, which have no head dimension to shard). A hybrid K3.1 group
holds both kinds at once, so the choice is per pool: KV, indexer and an MLA
draft are shared, while TP-sharded KDA state keeps its own rank namespace."""

import unittest
from types import SimpleNamespace

from sglang.srt.mem_cache.hicache_storage import PoolName
from sglang.srt.mem_cache.storage.mooncake_store.mooncake_direct_linker import (
    _shared_pool_suffix,
    _storage_sharing,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _k31_pools(dcp_sharded=False):
    """A K3.1 hybrid group: replicated KV / indexer / MLA draft, sharded KDA."""
    return {
        name: SimpleNamespace(
            pcp_replicated=pcp, tp_replicated=tp, dcp_sharded=dcp_sharded and tp
        )
        for name, pcp, tp in (
            (PoolName.KV, True, True),
            (PoolName.INDEXER, True, True),
            (PoolName.DRAFT, False, True),
            (PoolName.MAMBA, False, False),
        )
    }


def _sharing(dcp_size=1, **kwargs):
    args = dict(
        rank_replicated=False,
        tp_rank=0,
        tp_size=1,
        attn_cp_rank=0,
        attn_cp_size=1,
        pp_rank=0,
        offload_owner=True,
    )
    args.update(kwargs)
    return _storage_sharing(
        _k31_pools(dcp_size > 1),
        dcp_size=dcp_size,
        dcp_rank=args["tp_rank"] % dcp_size,
        **args,
    )


class TestSharedPoolSuffix(CustomTestCase):
    def test_each_dropped_axis_collapses_its_ranks(self):
        for axes, ranks in (
            (dict(tp_shared=True, cp_shared=False), {"tpshared_v1_cp0_pp0"}),
            (dict(tp_shared=True, cp_shared=True), {"pcp_shared_v1_tpshared_v1_pp0"}),
        ):
            with self.subTest(**axes):
                self.assertEqual(
                    {
                        _shared_pool_suffix(
                            tp_rank=rank,
                            attn_cp_rank=0,
                            pp_rank=0,
                            dcp_sharded=False,
                            dcp_size=1,
                            dcp_rank=0,
                            **axes,
                        )
                        for rank in range(8)
                    },
                    ranks,
                )

    def test_the_cp_only_term_keeps_its_original_spelling(self):
        # A store written before the TP axis existed must stay readable.
        self.assertEqual(
            _shared_pool_suffix(
                tp_shared=False,
                cp_shared=True,
                tp_rank=3,
                attn_cp_rank=1,
                pp_rank=2,
                dcp_sharded=False,
                dcp_size=1,
                dcp_rank=0,
            ),
            "pcp_shared_v1_tp3_pp2",
        )

    def test_a_pool_shared_on_no_axis_keeps_every_rank_term(self):
        suffixes = {
            _shared_pool_suffix(
                tp_shared=False,
                cp_shared=False,
                tp_rank=rank,
                attn_cp_rank=1,
                pp_rank=2,
                dcp_sharded=False,
                dcp_size=1,
                dcp_rank=0,
            )
            for rank in range(4)
        }
        self.assertEqual(len(suffixes), 4)
        self.assertIn("tp3_cp1_pp2", suffixes)


class TestStorageSharing(CustomTestCase):
    def test_pure_tp_shares_the_replicated_pools_and_writes_them_once(self):
        # The case this fixes: 8 attention-TP ranks each wrote a full copy of
        # the same MLA KV, indexer and draft pages.
        keys, owners = {}, {}
        for rank in range(8):
            suffixes, offload, cp_shared, tp_shared = _sharing(tp_rank=rank, tp_size=8)
            self.assertEqual(cp_shared, set())
            self.assertEqual(tp_shared, {PoolName.KV, PoolName.INDEXER, PoolName.DRAFT})
            for name, suffix in suffixes.items():
                keys.setdefault(name, set()).add(suffix)
            owners[rank] = offload
        for name in (PoolName.KV, PoolName.INDEXER, PoolName.DRAFT):
            self.assertEqual(keys[name], {"tpshared_v1_cp0_pp0"}, name)
        self.assertNotIn(PoolName.MAMBA, keys)  # keeps the group's tp{rank} suffix
        self.assertEqual(
            owners[0], {PoolName.KV, PoolName.INDEXER, PoolName.DRAFT, PoolName.MAMBA}
        )
        for rank in range(1, 8):
            self.assertEqual(owners[rank], {PoolName.MAMBA}, rank)

    def test_sharded_state_stays_one_object_per_rank(self):
        # Every rank must still write its own KDA shard, or ranks 1-7 would
        # read rank 0's state.
        total = sum(
            PoolName.MAMBA in _sharing(tp_rank=rank, tp_size=8)[1] for rank in range(8)
        )
        self.assertEqual(total, 8)

    def test_both_axes_shared_is_written_by_rank_zero_of_each(self):
        writers = {
            (tp, cp)
            for tp in range(2)
            for cp in range(2)
            if PoolName.KV
            in _sharing(tp_rank=tp, tp_size=2, attn_cp_rank=cp, attn_cp_size=2)[1]
        }
        self.assertEqual(writers, {(0, 0)})

    def test_tp_size_one_changes_nothing(self):
        # v9's shape: attention TP is 1, so there is nothing to collapse and
        # only the CP axis is shared.
        suffixes, offload, cp_shared, tp_shared = _sharing(
            tp_size=1, attn_cp_rank=0, attn_cp_size=2
        )
        self.assertEqual(tp_shared, set())
        self.assertEqual(cp_shared, {PoolName.KV, PoolName.INDEXER})
        self.assertEqual(suffixes[PoolName.KV], "pcp_shared_v1_tp0_pp0")
        self.assertEqual(
            offload, {PoolName.KV, PoolName.INDEXER, PoolName.DRAFT, PoolName.MAMBA}
        )

    def test_a_rank_replicated_group_is_left_to_its_group_suffix(self):
        # No sharded pool, so the group's own suffix already drops the tp term.
        suffixes, _, _, tp_shared = _sharing(rank_replicated=True, tp_rank=3, tp_size=8)
        self.assertEqual(tp_shared, set())
        self.assertEqual(suffixes, {})

    def test_a_non_owner_rank_offloads_nothing(self):
        _, offload, _, _ = _sharing(tp_rank=3, tp_size=8, offload_owner=False)
        self.assertEqual(offload, set())

    def test_dcp_keys_each_rank_and_the_first_block_writes(self):
        sharded = {PoolName.KV, PoolName.INDEXER, PoolName.DRAFT}
        for rank in range(16):
            suffixes, offload, _, _ = _sharing(tp_rank=rank, tp_size=16, dcp_size=8)
            for name in sharded:
                self.assertEqual(suffixes[name], f"tpshared_v1_dcp8r{rank % 8}_cp0_pp0")
            self.assertNotIn(PoolName.MAMBA, suffixes)
            self.assertEqual(
                offload, sharded | {PoolName.MAMBA} if rank < 8 else {PoolName.MAMBA}
            )


if __name__ == "__main__":
    unittest.main()
