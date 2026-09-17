"""`SWAKVPool.set_kv_buffer_prefix_valid` routing.

The prefix-valid commit (DSpark target-hidden injection over a [bs, width] verify
window) follows `set_kv_buffer`: full layers write the full loc into the full
sub-pool, window layers write the pre-translated SWA loc into the SWA sub-pool,
both under the sub-pool's own layer index. The pool never translates; a window
layer without an SWA loc is a caller bug. Sub-pools are recording stubs (CPU CI).

    python -m pytest test/registered/unit/mem_cache/test_swa_pool_prefix_valid_routing.py -v
"""

import types
import unittest

import torch

from sglang.srt.mem_cache.memory_pool import KVWriteLoc
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _RecordingPool:
    def __init__(self):
        self.calls = []

    def set_kv_buffer_prefix_valid(
        self, layer, loc_2d, commit_lens, cache_k, cache_v, *args, **kwargs
    ):
        self.calls.append((layer, loc_2d, commit_lens, args, kwargs))


def _bare_pool():
    pool = object.__new__(SWAKVPool)
    pool.full_kv_pool = _RecordingPool()
    pool.swa_kv_pool = _RecordingPool()
    # gd36 DSpark draft shape: layer 2 full, the rest sliding.
    pool.layers_mapping = {
        0: (0, True),
        1: (1, True),
        2: (0, False),
        3: (2, True),
        4: (3, True),
    }
    return pool


class TestSWAPoolPrefixValidRouting(unittest.TestCase):
    def setUp(self):
        self.loc_2d = torch.tensor([[10, 11, 12], [20, 21, 22]], dtype=torch.int64)
        self.swa_2d = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int64)
        self.commit = torch.tensor([2, 3], dtype=torch.int32)
        self.k = torch.zeros(6, 2, 8)
        self.v = torch.zeros(6, 2, 8)

    def test_full_layer_commits_full_loc_into_full_pool(self):
        pool = _bare_pool()
        pool.set_kv_buffer_prefix_valid(
            types.SimpleNamespace(layer_id=2),
            KVWriteLoc(self.loc_2d, self.swa_2d),
            self.commit,
            self.k,
            self.v,
            1.0,
            1.0,
        )
        self.assertEqual(pool.swa_kv_pool.calls, [])
        ((layer, loc, commit, args, kwargs),) = pool.full_kv_pool.calls
        self.assertIsNone(layer)
        self.assertIs(loc, self.loc_2d)
        self.assertIs(commit, self.commit)
        self.assertEqual(kwargs, {"layer_id_override": 0})

    def test_window_layer_commits_swa_loc_into_swa_pool(self):
        pool = _bare_pool()
        pool.set_kv_buffer_prefix_valid(
            types.SimpleNamespace(layer_id=3),
            KVWriteLoc(self.loc_2d, self.swa_2d),
            self.commit,
            self.k,
            self.v,
        )
        self.assertEqual(pool.full_kv_pool.calls, [])
        ((_, loc, _, _, kwargs),) = pool.swa_kv_pool.calls
        self.assertIs(loc, self.swa_2d)
        self.assertEqual(kwargs, {"layer_id_override": 2})

    def test_bare_loc_serves_full_layers_only(self):
        pool = _bare_pool()
        pool.set_kv_buffer_prefix_valid(
            types.SimpleNamespace(layer_id=2), self.loc_2d, self.commit, self.k, self.v
        )
        ((_, loc, _, _, _),) = pool.full_kv_pool.calls
        self.assertIs(loc, self.loc_2d)
        with self.assertRaises(AssertionError):
            pool.set_kv_buffer_prefix_valid(
                types.SimpleNamespace(layer_id=0),
                self.loc_2d,
                self.commit,
                self.k,
                self.v,
            )


if __name__ == "__main__":
    unittest.main()
