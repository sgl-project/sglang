"""Unit coverage for SWAKVPool.set_kv_buffer with a pre-translated swa_loc.

The attention backend translates out_cache_loc once per forward and passes it
in via a ``KVWriteLoc`` (loc + swa_loc) on the loc_info argument; set_kv_buffer
uses swa_loc directly for SWA layers and asserts it is provided. The per-backend
cuda-graph buffer plumbing is covered by the backend SWA integration tests.
"""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.memory_pool import KVWriteLoc
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.test.test_utils import CustomTestCase

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=9, suite="base-a-test-cpu")


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


if __name__ == "__main__":
    unittest.main()
