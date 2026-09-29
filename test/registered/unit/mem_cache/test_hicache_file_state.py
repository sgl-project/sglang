"""Rank-owned recurrent file checkpoints with shared MLA KV (CPU)."""

import tempfile
import unittest
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import torch
from test_hicache_file_lru_unit import _make_config

from sglang.srt.environ import envs
from sglang.srt.mem_cache.hicache_storage import (
    HiCacheFile,
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
)
from sglang.srt.mem_cache.pool_host.mamba import MambaPoolHost
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def make_config(tp_rank=0, **overrides):
    fields = dict(
        tp_rank=tp_rank,
        tp_size=4,
        is_mla=True,
        model="test/model",
        extra_config={
            "max_size": "0",
            "min_free_space": "0",
            "enable_metadata_cache": True,
            "metadata_ttl": -1,
        },
    )
    fields.update(overrides)
    return _make_config(**fields)


def _mamba_pool(
    *,
    capacity=8,
    layers=2,
    layout="page_first",
    temporal_dtype=torch.float32,
    temporal_shape=(2, 3),
    conv_dtype=torch.bfloat16,
    conv_shapes=((3, 4), (3, 4)),
):
    device = SimpleNamespace(
        size=capacity,
        device="cpu",
        num_mamba_layers=layers,
        mamba_cache=SimpleNamespace(
            temporal=torch.empty(
                (layers, capacity, *temporal_shape), dtype=temporal_dtype
            ),
            conv=[
                torch.empty((layers, capacity, *shape), dtype=conv_dtype)
                for shape in conv_shapes
            ],
        ),
    )
    return MambaPoolHost(
        device,
        host_to_device_ratio=2,
        host_size=0,
        pin_memory=False,
        device="cpu",
        layout=layout,
    )


class TestFileRecurrentState(CustomTestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.directory = self.stack.enter_context(tempfile.TemporaryDirectory())
        self.stack.enter_context(
            envs.SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR.override(self.directory)
        )

    def backend(self, config):
        return HiCacheFile(config)

    def test_file_state_shards_do_not_alias_mla_replicas(self):
        """TP0 and TP2 share MLA, but must restore different recurrent state."""
        for layout in ("page_first", "page_first_direct"):
            expected = {}
            for rank in (0, 2):
                state = _mamba_pool(layout=layout)
                for i, buffer in enumerate(state.get_hybrid_pool_buffer()):
                    buffer.view(torch.uint8).fill_(17 + rank + i)
                expected[rank] = state.get_data_page(3).clone()
                writer = self.backend(make_config(rank))
                writer.register_mem_host_pool_v2(state, PoolName.MAMBA)
                transfer = PoolTransfer(
                    PoolName.MAMBA,
                    host_indices=torch.tensor([3]),
                    keys=["checkpoint"],
                )
                self.assertEqual(
                    writer.batch_set_v2([transfer])[PoolName.MAMBA], [True]
                )

            for rank in (0, 2):
                # Cache capacity is not part of a compatible tensor schema.
                state = _mamba_pool(capacity=16, layout=layout)
                reader = self.backend(make_config(rank))
                reader.register_mem_host_pool_v2(state, PoolName.MAMBA)
                transfer = PoolTransfer(
                    PoolName.MAMBA,
                    host_indices=torch.tensor([7]),
                    keys=["checkpoint"],
                )
                self.assertEqual(
                    reader.batch_get_v2([transfer])[PoolName.MAMBA], [True]
                )
                torch.testing.assert_close(state.get_data_page(7), expected[rank])

    def test_file_state_schema_mismatch_is_a_miss_without_mutation(self):
        """Equal-sized KDA tensors can have incompatible shapes or precision."""
        writer = self.backend(make_config())
        state = _mamba_pool()
        writer.register_mem_host_pool_v2(state, PoolName.MAMBA)
        transfer = PoolTransfer(
            PoolName.MAMBA, host_indices=torch.tensor([3]), keys=["checkpoint"]
        )
        self.assertEqual(writer.batch_set_v2([transfer])[PoolName.MAMBA], [True])
        reader = self.backend(make_config())
        for changes in (
            {"temporal_shape": (3, 2)},
            {"temporal_dtype": torch.bfloat16, "temporal_shape": (2, 6)},
            {"conv_dtype": torch.float16},
            {"conv_shapes": ((2, 6), (3, 4))},
            {"layers": 1, "temporal_shape": (4, 3), "conv_shapes": ((6, 4), (6, 4))},
            {"layout": "page_first_direct"},
        ):
            with self.subTest(changes=changes):
                target = _mamba_pool(**changes)
                reader.register_mem_host_pool_v2(target, PoolName.MAMBA)
                for buffer in target.get_hybrid_pool_buffer():
                    buffer.view(torch.uint8).fill_(165)
                before = target.get_data_page(3).clone()
                self.assertEqual(
                    reader.batch_get_v2([transfer])[PoolName.MAMBA], [False]
                )
                torch.testing.assert_close(target.get_data_page(3), before)

    def test_file_state_registration_preserves_v1_objects(self):
        """Replacing a pool must preserve compatible files and reject stale schemas."""
        backend = self.backend(make_config(tp_rank=2))
        source = _mamba_pool()
        backend.register_mem_host_pool_v2(source, PoolName.MAMBA)
        for buf in source.get_hybrid_pool_buffer():
            buf.view(torch.uint8).fill_(37)
        expected = source.get_data_page(3).clone()
        transfer = PoolTransfer(
            PoolName.MAMBA, host_indices=torch.tensor([3]), keys=["checkpoint"]
        )
        self.assertEqual(backend.batch_set_v2([transfer])[PoolName.MAMBA], [True])
        self.assertEqual(
            [p.name for p in Path(self.directory).glob("*.bin")],
            [
                "checkpoint.mamba_tp2_4_v1_"
                "8bdd9a3cc60b2b24ade8e664a2a7c00d308e05e1266716c2d7740755939d9384"
                "_test-model_mamba_tp2_4.bin"
            ],
        )
        for changes, hit in (
            ({"temporal_shape": (3, 2)}, False),
            ({"capacity": 16}, True),
        ):
            with self.subTest(changes=changes):
                target = _mamba_pool(**changes)
                backend.register_mem_host_pool_v2(target, PoolName.MAMBA)
                for buf in target.get_hybrid_pool_buffer():
                    buf.view(torch.uint8).fill_(165)
                before = target.get_data_page(3).clone()
                self.assertEqual(
                    backend.batch_get_v2([transfer])[PoolName.MAMBA], [hit]
                )
                torch.testing.assert_close(
                    target.get_data_page(3), expected if hit else before, rtol=0, atol=0
                )

    def test_bounded_mla_replica_can_write_and_evict_only_its_state(self):
        """An MLA replica still owns its TP-sharded recurrent checkpoint."""
        for model in ("bounded/model", "模型" * 70):
            with self.subTest(model=model):
                state = _mamba_pool()
                size = state.get_data_page(3).nbytes
                config = make_config(
                    model=model,
                    extra_config=dict(
                        max_size=size,
                        min_free_space=0,
                        eviction_ratio=1.0,
                        enable_metadata_cache=False,
                    ),
                )
                owner = self.backend(config)
                self.assertTrue(owner.set("kv", torch.ones(size, dtype=torch.uint8)))
                replica_config = replace(config, tp_rank=2)
                replica = self.backend(replica_config)
                replica.register_mem_host_pool_v2(state, PoolName.MAMBA)
                self.assertFalse(replica.set("unowned-kv", torch.ones(1)))
                for key in ("old", "new"):
                    transfer = PoolTransfer(
                        PoolName.MAMBA,
                        host_indices=torch.tensor([3]),
                        keys=[key],
                    )
                    self.assertEqual(
                        replica.batch_set_v2([transfer]), {PoolName.MAMBA: [True]}
                    )
                self.assertFalse(
                    replica.exists(replica._log_key(PoolName.MAMBA, "old"))
                )
                self.assertTrue(owner.exists("kv"))
                # Fresh scans must not adopt another rank's files.
                fresh = self.backend(replica_config)
                fresh.register_mem_host_pool_v2(state, PoolName.MAMBA)
                self.assertTrue(fresh.exists(fresh._log_key(PoolName.MAMBA, "new")))
                self.assertTrue(owner.exists("kv"))
                self.assertEqual(fresh._evictor._total_bytes, size)
                # The primary writer shares one cap across its KV and state,
                # without adopting a replica's independently owned state.
                owner.register_mem_host_pool_v2(state, PoolName.MAMBA)
                self.assertEqual(
                    owner.batch_set_v2([transfer]), {PoolName.MAMBA: [True]}
                )
                self.assertFalse(owner.exists("kv"))
                self.assertTrue(fresh.exists(fresh._log_key(PoolName.MAMBA, "new")))
                self.assertEqual(owner._evictor._total_bytes, size)

    def test_long_model_identity_restores_kv_and_state_from_fresh_backend(self):
        """Snapshot paths plus state fingerprints exceeded NAME_MAX in Kimi serving."""
        for model in ("/cache/models--org--model/snapshots/" + "a" * 100, "模型" * 70):
            with self.subTest(model=model):
                config = make_config(model=model)
                key = "b" * 64
                state = _mamba_pool()
                for buffer in state.get_hybrid_pool_buffer():
                    buffer.view(torch.uint8).fill_(37)
                expected = state.get_data_page(3).clone()
                writer = self.backend(config)
                writer.register_mem_host_pool_v2(state, PoolName.MAMBA)
                self.assertTrue(writer.set(key, torch.tensor([19], dtype=torch.uint8)))
                transfer = PoolTransfer(
                    PoolName.MAMBA,
                    host_indices=torch.tensor([3]),
                    keys=[key],
                    hit_policy=PoolHitPolicy.TRAILING_PAGES,
                )
                self.assertEqual(
                    writer.batch_set_v2([transfer])[PoolName.MAMBA], [True]
                )
                reader = self.backend(config)
                target = _mamba_pool(capacity=16)
                reader.register_mem_host_pool_v2(target, PoolName.MAMBA)
                self.assertEqual(
                    reader.batch_exists_v2([key], [transfer]).kv_hit_pages, 1
                )
                transfer.host_indices = torch.tensor([7])
                self.assertEqual(
                    reader.batch_get_v2([transfer])[PoolName.MAMBA], [True]
                )
                torch.testing.assert_close(target.get_data_page(7), expected)
                torch.testing.assert_close(
                    reader.get(key, torch.empty(1, dtype=torch.uint8)),
                    torch.tensor([19], dtype=torch.uint8),
                )
                other = self.backend(replace(config, model=model + "-other"))
                other.register_mem_host_pool_v2(target, PoolName.MAMBA)
                self.assertEqual(
                    other.batch_exists_v2([key], [transfer]).kv_hit_pages, 0
                )
                self.assertTrue(
                    all(
                        len(p.name.encode()) <= 255
                        for p in Path(self.directory).glob("*.bin")
                    )
                )


if __name__ == "__main__":
    unittest.main()
