"""A rank-sharded sidecar must not share one L3 key across TP ranks.

`HybridCacheController.should_backup` makes every TP rank back up the Mamba
sidecar, including on an MLA model whose primary KV pool is replicated and whose
key therefore carries no rank. Pure CPU test; no server, no CUDA.
"""

import tempfile
import unittest

import torch

from sglang.srt.mem_cache.hicache_storage import (
    HiCacheFile,
    HiCacheStorageConfig,
    PoolName,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

TP_SIZE = 2


def _config(tp_rank, is_mla_model=True):
    return HiCacheStorageConfig(
        tp_rank=tp_rank,
        tp_size=TP_SIZE,
        pp_rank=0,
        pp_size=1,
        attn_cp_rank=0,
        attn_cp_size=1,
        is_mla_model=is_mla_model,
        enable_storage_metrics=False,
        is_page_first_layout=True,
        model_name="testmodel",
    )


class TestRankShardedSidecarKey(CustomTestCase):
    def _backends(self, is_mla_model=True):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        return [
            HiCacheFile(storage_config=_config(r, is_mla_model), file_path=tmp.name)
            for r in range(TP_SIZE)
        ]

    def test_each_rank_keeps_its_own_mamba_shard(self):
        backends = self._backends()
        keys = [b._get_component_key("h0", PoolName.MAMBA) for b in backends]
        self.assertNotEqual(
            keys[0],
            keys[1],
            "every TP rank backs up its own Mamba shard, so the ranks must not "
            "share one key",
        )

        for rank, backend in enumerate(backends):
            backend.set(keys[rank], torch.full((8,), float(rank + 1)))
        for rank, backend in enumerate(backends):
            got = torch.zeros(8)
            backend.get(keys[rank], got)
            self.assertEqual(
                float(got[0]),
                float(rank + 1),
                f"rank {rank} read back another rank's Mamba shard",
            )

    def test_kv_key_stays_replicated(self):
        # The MLA dedup win: one copy of the KV pages for the whole TP group.
        backends = self._backends()
        self.assertEqual(
            backends[0]._get_component_key("h0"),
            backends[1]._get_component_key("h0"),
        )
        self.assertEqual(
            backends[0]._get_component_key("h0", PoolName.KV),
            backends[1]._get_component_key("h0", PoolName.KV),
        )

    def test_non_mla_keys_are_unchanged(self):
        # A non-MLA key already carries the rank; the sidecar must not add a second.
        backends = self._backends(is_mla_model=False)
        for rank, backend in enumerate(backends):
            self.assertEqual(
                backend._get_component_key("h0", PoolName.MAMBA),
                f"h0.mamba_testmodel_{rank}_{TP_SIZE}",
            )

    def test_scoped_key_is_still_visible_to_the_file_scans(self):
        # Both the LRU evictor and the metadata cache select this rank's files
        # with `stem.endswith(config_suffix)`; a scope appended after the suffix
        # would make a sidecar file invisible to eviction.
        backend = self._backends()[0]
        key = backend._get_component_key("h0", PoolName.MAMBA)
        self.assertTrue(
            key.endswith(backend.config_suffix),
            f"{key!r} must end with {backend.config_suffix!r}",
        )


if __name__ == "__main__":
    unittest.main()
