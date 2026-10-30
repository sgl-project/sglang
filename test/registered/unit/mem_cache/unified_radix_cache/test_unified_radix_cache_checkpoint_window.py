"""Tests for SWA KV kept only for the window before each cached Mamba checkpoint."""

import unittest

import torch

from sglang.srt.mem_cache.base_prefix_cache import EvictParams
from sglang.srt.mem_cache.unified_cache.components.base import ComponentType
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.kits.unified_radix_cache.checkpoint_window import (
    SWACheckpointWindowFixtures,
)
from sglang.test.kits.unified_radix_cache.config import CacheConfig
from sglang.test.kits.unified_radix_cache.core import (
    MambaFinishedOvershootCheckpointFixtures,
)
from sglang.test.kits.unified_radix_cache.fixtures import (
    match_prefix,
    node_device_value,
    node_host_value,
    node_key_length,
    node_parent,
)
from sglang.test.mem_cache_utils import finish_req
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=10, suite="stage-b-test-1-gpu-small-amd")


class SWACheckpointWindowSuite(SWACheckpointWindowFixtures):
    def test_overlapping_windows_share_swa_kv(self):
        cache, allocator, pool = self._build()
        a, b, leaf = self._two_overlapping_checkpoints(cache, allocator, pool)
        w = self.window
        self.assertEqual(
            self._path(cache, leaf, b),
            [
                (0, a - w, True, False),
                (a - w, b - w, True, False),
                (b - w, a, True, True),
                (a, b, True, True),
            ],
        )
        cache.sanity_check()

    def test_evicting_a_checkpoint_frees_only_its_unshared_window(self):
        cache, allocator, pool = self._build()
        a, b, leaf = self._two_overlapping_checkpoints(cache, allocator, pool)
        available = allocator.swa_available_size()

        result = cache.evict(EvictParams(num_tokens=0, mamba_num=1))

        w = self.window
        self.assertEqual(result.mamba_num_evicted, 1)
        self.assertEqual(result.swa_num_tokens_evicted, b - a)
        self.assertEqual(allocator.swa_available_size(), available + (b - a))
        self.assertEqual(
            self._path(cache, leaf, b)[1:],
            [
                (a - w, b - w, False, False),
                (b - w, a, True, False),
                (a, b, True, True),
            ],
        )
        cache.sanity_check()

    def test_swa_pressure_reclaims_uncovered_kv_then_evicts_checkpoints(self):
        cache, allocator, pool = self._build()
        seqs = [self._seq(1000 * i, 2 * self.window + self.page) for i in range(1, 4)]
        leaves = [self._insert_seq(cache, allocator, pool, seq) for seq in seqs]
        uncovered = 3 * (self.window + self.page)
        available = allocator.swa_available_size()

        result = cache.evict(EvictParams(num_tokens=0, swa_num_tokens=uncovered))
        self.assertEqual(result.swa_num_tokens_evicted, uncovered)
        self.assertEqual(allocator.swa_available_size(), available + uncovered)
        for seq in seqs:
            self.assertEqual(match_prefix(cache, seq).device_prefix_len, len(seq))

        result = cache.evict_for_alloc(
            EvictParams(num_tokens=0, swa_num_tokens=self.page)
        )
        self.assertEqual(result.swa_num_tokens_evicted, self.window)
        self.assertEqual(
            allocator.swa_available_size(), available + uncovered + self.window
        )
        self.assertFalse(cache.tree_core.contains_node(leaves[0]))
        for leaf in leaves[1:]:
            self.assertIsNotNone(node_device_value(cache, leaf, ComponentType.MAMBA))
            self.assertIsNotNone(node_device_value(cache, leaf, ComponentType.SWA))
        cache.sanity_check()

    def test_finished_request_does_not_donate_its_previous_checkpoint(self):
        if not self.cfg.enable_mamba_extra_buffer:
            self.skipTest("the previous checkpoint needs the extra buffer")
        cache, allocator, pool = self._build()
        req, tokens = MambaFinishedOvershootCheckpointFixtures._build_req(
            self, allocator, pool, previous_track_seqlen=8
        )
        req.last_node = cache.root_node_handle()

        finish_req(cache, req, 11)

        self.assertEqual(match_prefix(cache, tokens).device_prefix_len, 0)
        self.assertIsNone(req.kv.mamba_pool_idx)
        cache.sanity_check()

    def test_storage_backend_attach_is_refused(self):
        cache, _, _ = self._build("write_back")

        attached, message = cache.attach_storage_backend("file")

        self.assertFalse(attached)
        self.assertIn("checkpoint windows", message)
        self.assertIsNone(cache.cache_controller.storage_backend)

    def test_host_eviction_counts_window_frees_in_ancestors(self):
        cache, allocator, pool = self._build("write_back")
        _, b, _ = self._two_overlapping_checkpoints(cache, allocator, pool)
        cache.evict(EvictParams(num_tokens=b))
        host_pool = cache.swa_kv_pool_host
        available = host_pool.available_size()

        freed = cache.evict_host(b, ComponentType.SWA)

        self.assertGreater(freed, 0)
        self.assertEqual(freed, host_pool.available_size() - available)
        cache.sanity_check()

    def test_write_through_keeps_window_kv_the_host_copy_missed(self):
        cache, allocator, pool = self._build("write_through")
        seq_r = self._seq(1, 2 * self.window + self.page)
        self._insert_seq(cache, allocator, pool, seq_r)
        # Backed up while seq_r's backup is still in flight.
        seq_q = seq_r[: self.window] + self._seq(1000, self.page)
        leaf = self._insert_seq(cache, allocator, pool, seq_q)
        fragment = node_parent(cache, leaf)
        self.assertEqual(node_key_length(cache, fragment), self.window - self.page)
        self.assertIsNone(node_host_value(cache, fragment, ComponentType.SWA))
        cache.writing_check(write_back=True)

        cache.evict(EvictParams(num_tokens=0, mamba_num=2))

        self.assertIsNone(node_device_value(cache, leaf, ComponentType.MAMBA))
        self.assertIsNotNone(node_device_value(cache, fragment, ComponentType.SWA))
        cache.sanity_check()

    def test_write_back_restores_a_checkpoint_with_its_window(self):
        cache, allocator, pool = self._build("write_back")
        seq = self._seq(1, 2 * self.window + self.page)
        leaf = self._insert_seq(cache, allocator, pool, seq)
        kv_pool = allocator.get_kvcache()
        layers = self.cfg.non_full_layer_ids[:2]
        slots = node_device_value(cache, leaf, ComponentType.SWA)
        written = []
        for layer in layers:
            k_buffer = kv_pool.get_key_buffer(layer)
            written.append(torch.randn_like(k_buffer[slots], dtype=torch.float32))
            k_buffer[slots] = written[-1].to(k_buffer.dtype)

        cache.evict(EvictParams(num_tokens=len(seq)))
        for component in (ComponentType.FULL, ComponentType.MAMBA, ComponentType.SWA):
            self.assertIsNone(node_device_value(cache, leaf, component))
            self.assertIsNotNone(node_host_value(cache, leaf, component))
        parent = node_parent(cache, leaf)
        self.assertIsNone(node_host_value(cache, parent, ComponentType.SWA))
        for layer in layers:
            kv_pool.get_key_buffer(layer).zero_()

        match, loaded, last_node = self._load_back(cache, pool, seq)
        self.assertEqual(last_node, leaf)
        self.assertEqual(match.device_prefix_len + loaded, len(seq))
        slots = node_device_value(cache, leaf, ComponentType.SWA)
        for layer, values in zip(layers, written):
            k_buffer = kv_pool.get_key_buffer(layer)
            self.assertTrue(torch.equal(k_buffer[slots], values.to(k_buffer.dtype)))
        cache.sanity_check()

    def test_random_operations_keep_window_invariants(self):
        """Random inserts, evictions, locks and load-backs on a small SWA pool."""
        for write_policy in ("write_back", "write_through"):
            for seed in range(3):
                with self.subTest(write_policy=write_policy, seed=seed):
                    self._run_random_operations(write_policy, seed)


_SWA_CHECKPOINT_WINDOW_CFG = dict(
    components=(ComponentType.FULL, ComponentType.SWA, ComponentType.MAMBA),
    swa_checkpoint_windows=True,
    kv_size=1024,
    max_num_reqs=400,
    mamba_cache_size=400,
)


@unittest.skipUnless(torch.cuda.is_available(), "cache fixtures need CUDA")
class TestSWACheckpointWindowsPage1(SWACheckpointWindowSuite, CustomTestCase):
    cfg = CacheConfig(page_size=1, sliding_window_size=4, **_SWA_CHECKPOINT_WINDOW_CFG)


@unittest.skipUnless(torch.cuda.is_available(), "cache fixtures need CUDA")
class TestSWACheckpointWindowsPage4(SWACheckpointWindowSuite, CustomTestCase):
    # A 6-token window rounds up to two pages.
    cfg = CacheConfig(
        page_size=4,
        sliding_window_size=6,
        enable_mamba_extra_buffer=True,
        **_SWA_CHECKPOINT_WINDOW_CFG,
    )


if __name__ == "__main__":
    unittest.main()
