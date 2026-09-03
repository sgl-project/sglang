"""CPU/mock tests for unified DSV4 C4 request-state lifecycle."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from sglang.srt.mem_cache.allocation import alloc_req_slots
from sglang.srt.mem_cache.deepseek_v4_compress_state import KVAndScore
from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.model_executor.pool_configurator import DSV4PoolConfigurator
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _request(req_pool_idx=None, *, reused=False):
    return SimpleNamespace(
        kv=SimpleNamespace(
            req_pool_idx=req_pool_idx,
            kv_committed_len=1 if reused else 0,
            kv_allocated_len=1 if reused else 0,
            holds_kv=reused,
        ),
        inflight_middle_chunks=1 if reused else 0,
    )


def _c4_pool(rows: int, width: int, ring_size: int):
    return SimpleNamespace(
        ratio=4,
        ring_size=ring_size,
        kv_score_buffer=KVAndScore(torch.full((rows, width), 7.0)),
    )


class TestUnifiedC4StateLifecycle(unittest.TestCase):
    def test_pool_size_is_exact_request_ring_product(self):
        configurator = object.__new__(DSV4PoolConfigurator)
        configurator.disaggregation_mode = "decode"
        configurator.disaggregation_decode_extra_slots = 3
        configurator.c4_ring_size = 16

        self.assertEqual(configurator._unified_c4_state_pool_size(10), 14 * 16)

    def test_clear_resets_only_selected_request_rings(self):
        ring_size = 8
        logical_rows = 4 * ring_size
        physical_rows = logical_rows + ring_size + 4
        attn = _c4_pool(physical_rows, width=12, ring_size=ring_size)
        indexer = _c4_pool(physical_rows, width=8, ring_size=ring_size)
        c128 = SimpleNamespace(
            ratio=128,
            ring_size=128,
            kv_score_buffer=KVAndScore(torch.full((physical_rows, 8), 9.0)),
        )

        token_pool = object.__new__(DeepSeekV4TokenToKVPool)
        token_pool._unified_kv = True
        token_pool.compress_state_pools = [attn, c128]
        token_pool.indexer_compress_state_pools = [indexer, None]
        token_pool.get_ring_size = MagicMock(return_value=ring_size)

        token_pool.clear_c4_req_states([1, 3])

        selected = torch.tensor(list(range(8, 16)) + list(range(24, 32)))
        untouched = torch.tensor(list(range(0, 8)) + list(range(16, 24)))
        for pool in (attn, indexer):
            state = pool.kv_score_buffer.kv_score
            half = state.shape[-1] // 2
            self.assertTrue(
                torch.equal(
                    state[selected, :half], torch.zeros_like(state[selected, :half])
                )
            )
            self.assertTrue(torch.isneginf(state[selected, half:]).all())
            self.assertTrue((state[untouched] == 7).all())
            self.assertTrue((state[logical_rows:] == 7).all())
        self.assertTrue((c128.kv_score_buffer.kv_score == 9).all())

    def test_alloc_clears_new_slots_but_not_reused_slots(self):
        req_pool = ReqToTokenPool(3, 16, "cpu", enable_memory_saver=False)
        token_pool = MagicMock()
        reused = _request()

        # First admission: a brand-new slot, so its C4 ring must be cleared.
        (reused_idx,) = alloc_req_slots(
            req_pool, [reused], None, token_to_kv_pool=token_pool
        )
        token_pool.clear_c4_req_states.assert_called_once_with([reused_idx])

        # Chunked continuation reuses the same slot -- clearing it here would
        # wipe the state captured by the previous chunk.
        token_pool.clear_c4_req_states.reset_mock()
        reused.kv.req_pool_idx = reused_idx
        reused.kv.kv_committed_len = 1
        reused.kv.kv_allocated_len = 1
        reused.kv.holds_kv = True
        reused.inflight_middle_chunks = 1
        self.assertEqual(
            alloc_req_slots(req_pool, [reused], None, token_to_kv_pool=token_pool),
            [reused_idx],
        )
        token_pool.clear_c4_req_states.assert_not_called()

        # Mixed batch: only the newly allocated slot is cleared.
        fresh = _request()
        indices = alloc_req_slots(
            req_pool, [reused, fresh], None, token_to_kv_pool=token_pool
        )
        self.assertEqual(indices[0], reused_idx)
        self.assertNotEqual(indices[1], reused_idx)
        token_pool.clear_c4_req_states.assert_called_once_with([indices[1]])


if __name__ == "__main__":
    unittest.main()
