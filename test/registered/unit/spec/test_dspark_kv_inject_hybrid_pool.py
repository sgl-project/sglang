"""DSpark target-hidden injection into a hybrid (full + SWA) draft pool.

`TargetHiddenKvInjector.inject_target_hidden` is the one place that turns the
verify window's full locs into SWA locs: for an `SWAKVPool` draft pool it looks
the locs up in the target allocator's full->SWA mapping and hands the draft
model the pair (`swa_loc`); a plain pool gets none. The draft model is a
recording stub (CPU CI).

    python -m pytest test/registered/unit/spec/test_dspark_kv_inject_hybrid_pool.py -v
"""

import types
import unittest

import torch

from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.srt.speculative.dspark_components.dspark_kv_inject import (
    TargetHiddenKvInjector,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _RecordingDraft:
    def __init__(self):
        self.calls = []

    def write_target_hidden_kv(self, **kwargs):
        self.calls.append(kwargs)


def _hybrid_pool(mapping):
    pool = object.__new__(SWAKVPool)
    pool.full_to_swa_index_mapping = mapping
    return pool


def _injector(pool):
    draft = _RecordingDraft()
    injector = TargetHiddenKvInjector(
        draft_model=draft,
        draft_model_runner=types.SimpleNamespace(token_to_kv_pool=pool),
        model_runner=types.SimpleNamespace(device="cpu"),
        device="cpu",
        verify_num_draft_tokens=3,
        block_pos_offsets=torch.arange(3),
    )
    return injector, draft


class TestDSparkInjectHybridPool(unittest.TestCase):
    def test_hybrid_pool_gets_translated_swa_loc(self):
        # full slot i -> swa slot 100 + i; trailing -1 sentinel like the allocator.
        mapping = torch.cat([torch.arange(16) + 100, torch.tensor([-1])])
        injector, draft = _injector(_hybrid_pool(mapping))
        cache_loc = torch.tensor([3, 4, 5, 8, 9, 10])
        injector.inject_target_hidden(
            target_hidden=torch.zeros(6, 4),
            cache_loc=cache_loc,
            cache_loc_2d=cache_loc.view(2, 3),
            positions=torch.arange(6),
            commit_lens=torch.tensor([2, 1]),
        )
        (call,) = draft.calls
        torch.testing.assert_close(call["swa_loc"], cache_loc + 100)
        self.assertEqual(tuple(call["cache_loc_2d"].shape), (2, 3))
        self.assertEqual(int(call["commit_lens"].sum()), 3)

    def test_plain_pool_gets_no_swa_loc(self):
        injector, draft = _injector(types.SimpleNamespace())
        cache_loc = torch.tensor([3, 4, 5])
        injector.inject_target_hidden(
            target_hidden=torch.zeros(3, 4),
            cache_loc=cache_loc,
            positions=torch.arange(3),
        )
        (call,) = draft.calls
        self.assertIsNone(call["swa_loc"])
        self.assertIsNone(call["cache_loc_2d"])


if __name__ == "__main__":
    unittest.main()
