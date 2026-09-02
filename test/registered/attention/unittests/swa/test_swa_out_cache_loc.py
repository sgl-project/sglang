"""Unit coverage for SWAKVPool.set_kv_buffer with a pre-translated swa_loc.

The attention backend translates out_cache_loc once per forward and passes it
in via a ``KVWriteLoc`` (loc + swa_loc) on the loc_info argument; set_kv_buffer
uses swa_loc directly for SWA layers and asserts it is provided. The per-backend
cuda-graph buffer plumbing is covered by the backend SWA integration tests.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

import torch

from sglang.srt.layers.attention.dots_hybrid_backend import DotsSWAMLAAttnBackend
from sglang.srt.layers.attention.flashattention_backend import FlashAttentionBackend
from sglang.srt.mem_cache.memory_pool import KVWriteLoc, MLATokenToKVPool
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestSWAKVPoolSetKVBuffer(CustomTestCase):
    """set_kv_buffer: SWA layers require a pre-translated swa_loc; full layers
    use loc unchanged."""

    def _pool_and_record(self):
        pool = object.__new__(SWAKVPool)
        # layer 0 -> full pool, layer 1 -> swa pool
        pool.layers_mapping = {0: (0, False), 1: (0, True)}
        recorded = {}

        def _swa_set(layer, loc, k, v, k_scale, v_scale, layer_id_override):
            recorded["swa_loc"] = loc

        def _full_set(layer, loc, k, v, k_scale, v_scale, layer_id_override):
            recorded["full_loc"] = loc

        pool.swa_kv_pool = SimpleNamespace(set_kv_buffer=_swa_set)
        pool.full_kv_pool = SimpleNamespace(set_kv_buffer=_full_set)
        return pool, recorded

    def test_swa_layer_uses_swa_loc_directly(self):
        pool, recorded = self._pool_and_record()
        swa_loc = torch.tensor([7, 8])
        pool.set_kv_buffer(
            SimpleNamespace(layer_id=1),
            KVWriteLoc(torch.tensor([3, 4]), swa_loc),
            None,
            None,
        )
        self.assertIs(recorded["swa_loc"], swa_loc)

    def test_swa_layer_requires_swa_loc(self):
        # set_kv_buffer never translates internally; SWA layers must be given a
        # pre-translated swa_loc (loc_info without swa_loc, or a bare loc).
        pool, _ = self._pool_and_record()
        with self.assertRaises(AssertionError):
            pool.set_kv_buffer(
                SimpleNamespace(layer_id=1), torch.tensor([3, 4]), None, None
            )
        with self.assertRaises(AssertionError):
            pool.set_kv_buffer(
                SimpleNamespace(layer_id=1),
                KVWriteLoc(torch.tensor([3, 4])),
                None,
                None,
            )

    def test_full_layer_ignores_swa_loc(self):
        pool, recorded = self._pool_and_record()
        loc = torch.tensor([3, 4])
        # Full layer: swa_loc supplied but ignored; loc is used.
        pool.set_kv_buffer(
            SimpleNamespace(layer_id=0),
            KVWriteLoc(loc, torch.tensor([99, 99])),
            None,
            None,
        )
        self.assertIs(recorded["full_loc"], loc)

    def test_composed_mla_pools_route_local_layer_ids(self):
        pool = object.__new__(SWAKVPool)
        pool.layers_mapping = {7: (1, False), 8: (2, True)}
        recorded = {}

        def make_mla_pool(name):
            mla_pool = object.__new__(MLATokenToKVPool)

            def set_kv(layer, loc, k, v, layer_id_override=None):
                recorded[f"{name}_kv"] = (layer, loc, layer_id_override)

            def set_mla(layer, loc, k_nope, k_rope, layer_id_override=None):
                recorded[f"{name}_mla"] = (layer, loc, layer_id_override)

            mla_pool.set_kv_buffer = set_kv
            mla_pool.set_mla_kv_buffer = set_mla
            return mla_pool

        pool.full_kv_pool = make_mla_pool("full")
        pool.swa_kv_pool = make_mla_pool("swa")
        full_loc = torch.tensor([3, 4])
        swa_loc = torch.tensor([7, 8])

        pool.set_kv_buffer(
            SimpleNamespace(layer_id=7), KVWriteLoc(full_loc, swa_loc), None, None
        )
        pool.set_mla_kv_buffer(
            SimpleNamespace(layer_id=8), KVWriteLoc(full_loc, swa_loc), None, None
        )

        full_layer, recorded_full_loc, full_layer_id = recorded["full_kv"]
        swa_layer, recorded_swa_loc, swa_layer_id = recorded["swa_mla"]
        self.assertIsNone(full_layer)
        self.assertIs(recorded_full_loc, full_loc)
        self.assertEqual(full_layer_id, 1)
        self.assertIsNone(swa_layer)
        self.assertIs(recorded_swa_loc, swa_loc)
        self.assertEqual(swa_layer_id, 2)


class TestDotsDraftSWAOutCacheLoc(CustomTestCase):
    def test_metadata_sees_only_current_step_and_forward_batch_is_restored(self):
        backend = object.__new__(FlashAttentionBackend)
        backend.topk = 2
        backend.speculative_num_steps = 3
        backend.speculative_step_id = 1

        seen = []
        backend.init_forward_metadata_out_graph = MagicMock(
            side_effect=lambda forward_batch, in_capture=False: seen.append(
                forward_batch.out_cache_loc.clone()
            )
        )

        wrapper = object.__new__(DotsSWAMLAAttnBackend)
        wrapper.backend = backend
        wrapper._active_backend = backend
        wrapper._prefill_metadata = None

        original = torch.arange(12)
        forward_batch = SimpleNamespace(
            batch_size=2,
            forward_mode=ForwardMode.DECODE,
            out_cache_loc=original,
            spec_info=object(),
        )

        wrapper.init_forward_metadata_out_graph(forward_batch)

        torch.testing.assert_close(seen[0], torch.tensor([1, 4, 7, 10]))
        self.assertIs(forward_batch.out_cache_loc, original)


if __name__ == "__main__":
    unittest.main()
