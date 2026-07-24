"""Unit tests for the MLX decode-KV pool sync committed-length clamp (#30093).

Chained decode steps advance per-request MLX cache offsets without any
scheduler bookkeeping (no alloc_for_decode, no req_to_token write). The pool
sync must therefore never interpret req_to_token positions beyond the
scheduler-committed bound as slot ids: fresh rows read 0 (the reserved
padding slot) and reused rows read a previous request's slot ids.

A finished request must also flush its committed decode KV before its
req_to_token row is freed and reused, so a later sync cannot read the new
owner's slot ids out of the reused row.
"""

import importlib.util
import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

_HAS_MLX = importlib.util.find_spec("mlx") is not None
_SKIP_REASON = "requires mlx"

if _HAS_MLX:
    import mlx.core as mx

    from sglang.srt.hardware_backend.mlx.kv_cache import (
        ContiguousAttentionKVCache,
        MlxAttentionKVPool,
    )
    from sglang.srt.hardware_backend.mlx.model_runner import MlxModelRunner
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool

    H, D, LAYERS = 2, 4, 1

    class _FakeLayout:
        num_attention_layers = LAYERS
        attention_layer_indices = list(range(LAYERS))
        first_attention_layer_index = 0
        has_auxiliary_state = False

        def attention_pool_index(self, i):
            return i


def _make_runner():
    runner = MlxModelRunner.__new__(MlxModelRunner)
    runner.disable_radix_cache = False
    runner._cache_layout = _FakeLayout()
    runner._attention_kv_pool = MlxAttentionKVPool(
        pool_size=16, num_layers=LAYERS, n_kv_heads=H, head_dim=D, dtype=mx.float32
    )
    runner._req_to_token_pool = ReqToTokenPool(
        size=4, max_context_len=32, device="cpu", enable_memory_saver=False
    )
    runner._req_caches = {}
    runner._req_pool_idx = {}
    runner._req_synced_offset = {}
    runner._req_committed_len = {}
    return runner


def _fill_cache(n_tokens, value):
    cache = ContiguousAttentionKVCache(
        n_kv_heads=H, head_dim=D, max_seq_len=32, dtype=mx.float32
    )
    for i in range(n_tokens):
        k = mx.full((1, H, 1, D), value + i, dtype=mx.float32)
        cache.write_token(k, k)
    return [cache]


@unittest.skipUnless(_HAS_MLX, _SKIP_REASON)
class TestDecodeKvPoolSyncCommittedClamp(CustomTestCase):
    def _add_request(self, runner, req_id, *, n_tokens, value, committed, synced):
        runner._req_caches[req_id] = _fill_cache(n_tokens, value)
        runner._req_pool_idx[req_id] = 0
        runner._req_synced_offset[req_id] = synced
        runner._req_committed_len[req_id] = committed

    def test_fresh_row_chained_positions_are_not_synced(self):
        # Scenario A of #30093: prefill of length 2 (slots 1,2 written and
        # synced), then 3 chained decode steps advanced the cache to offset 5
        # with no req_to_token writes for positions 2..4. Those positions read
        # 0, the reserved padding slot.
        runner = _make_runner()
        pool = runner._attention_kv_pool
        runner._req_to_token_pool.req_to_token[0, 0] = 1
        runner._req_to_token_pool.req_to_token[0, 1] = 2
        self._add_request(runner, "X", n_tokens=5, value=100.0, committed=2, synced=2)

        runner._sync_decode_kv_to_pool("X")
        mx.eval(*pool.all_buffers())

        self.assertEqual(pool.k_buffer[0][0][0][0].item(), 0.0)
        self.assertEqual(runner._req_synced_offset["X"], 2)

    def test_reused_row_stale_slots_are_not_overwritten(self):
        # Scenario B of #30093: request Y owns pool slots 7,8,9; row 0
        # previously belonged to a longer finished request whose positions
        # 2..4 pointed at 7,8,9 (rows are not scrubbed on free). Request Z
        # reuses row 0 with prefill length 2 plus 3 chained steps.
        runner = _make_runner()
        pool = runner._attention_kv_pool
        victim = mx.full((3, H, D), 777.0, dtype=mx.float32)
        pool.set_kv(0, mx.array([7, 8, 9], dtype=mx.int32), victim, victim)
        for pos, slot in [(0, 3), (1, 4), (2, 7), (3, 8), (4, 9)]:
            runner._req_to_token_pool.req_to_token[0, pos] = slot
        self._add_request(runner, "Z", n_tokens=5, value=200.0, committed=2, synced=2)

        runner._sync_decode_kv_to_pool("Z")
        mx.eval(*pool.all_buffers())

        self.assertEqual(pool.k_buffer[0][7][0][0].item(), 777.0)
        self.assertEqual(pool.k_buffer[0][8][0][0].item(), 777.0)
        self.assertEqual(pool.k_buffer[0][9][0][0].item(), 777.0)
        self.assertEqual(runner._req_synced_offset["Z"], 2)

    def test_committed_positions_sync_normally(self):
        # Scheduler-committed decode positions keep flowing to the pool.
        runner = _make_runner()
        pool = runner._attention_kv_pool
        for pos, slot in [(0, 1), (1, 2), (2, 5), (3, 6), (4, 7)]:
            runner._req_to_token_pool.req_to_token[0, pos] = slot
        self._add_request(runner, "W", n_tokens=5, value=100.0, committed=5, synced=2)

        runner._sync_decode_kv_to_pool("W")
        mx.eval(*pool.all_buffers())

        self.assertEqual(pool.k_buffer[0][5][0][0].item(), 102.0)
        self.assertEqual(pool.k_buffer[0][6][0][0].item(), 103.0)
        self.assertEqual(pool.k_buffer[0][7][0][0].item(), 104.0)
        self.assertEqual(runner._req_synced_offset["W"], 5)

    def test_post_chain_commit_resumes_sync(self):
        # After the chain breaks, the scheduler commits further positions;
        # the clamped range from the chained phase must then be synced from
        # the still-live per-request cache instead of being lost.
        runner = _make_runner()
        pool = runner._attention_kv_pool
        runner._req_to_token_pool.req_to_token[0, 0] = 1
        runner._req_to_token_pool.req_to_token[0, 1] = 2
        self._add_request(runner, "X", n_tokens=5, value=100.0, committed=2, synced=2)

        runner._sync_decode_kv_to_pool("X")
        mx.eval(*pool.all_buffers())
        self.assertEqual(runner._req_synced_offset["X"], 2)

        # Scheduler catches up: positions 2..4 get allocated and written.
        for pos, slot in [(2, 10), (3, 11), (4, 12)]:
            runner._req_to_token_pool.req_to_token[0, pos] = slot
        runner.note_committed_len("X", 5)

        runner._sync_decode_kv_to_pool("X")
        mx.eval(*pool.all_buffers())

        self.assertEqual(pool.k_buffer[0][10][0][0].item(), 102.0)
        self.assertEqual(pool.k_buffer[0][11][0][0].item(), 103.0)
        self.assertEqual(pool.k_buffer[0][12][0][0].item(), 104.0)
        self.assertEqual(runner._req_synced_offset["X"], 5)

    def test_release_flush_prevents_stale_resync_after_row_reuse(self):
        # Residual lifecycle window of #30093: a finished request R still has
        # committed decode KV that has not been synced when its req_to_token
        # row is freed and reused by a live request. flush_request_decode_kv
        # runs at the radix-insert point (prepare_for_kv_cache_release) while R
        # still owns the row, so the later remove_request sync cannot read the
        # new owner's slot ids out of the reused row.
        runner = _make_runner()
        pool = runner._attention_kv_pool
        # R: prefill 2 (slots 1,2 already synced) plus 2 committed decode
        # positions (slots 5,6) that are still unsynced at release time.
        for pos, slot in [(0, 1), (1, 2), (2, 5), (3, 6)]:
            runner._req_to_token_pool.req_to_token[0, pos] = slot
        self._add_request(runner, "R", n_tokens=4, value=100.0, committed=4, synced=2)

        runner.flush_request_decode_kv("R")
        mx.eval(*pool.all_buffers())
        # Committed decode KV reached its real slots and the synced bound
        # caught up to the committed length.
        self.assertEqual(pool.k_buffer[0][5][0][0].item(), 102.0)
        self.assertEqual(pool.k_buffer[0][6][0][0].item(), 103.0)
        self.assertEqual(runner._req_synced_offset["R"], 4)

        # Row 0 is freed and reused by a live request whose positions 2,3 now
        # point at live slots 8,9. R's cache is still resident (stale-rid
        # cleanup runs on a later forward).
        live = mx.full((2, H, D), 555.0, dtype=mx.float32)
        pool.set_kv(0, mx.array([8, 9], dtype=mx.int32), live, live)
        for pos, slot in [(2, 8), (3, 9)]:
            runner._req_to_token_pool.req_to_token[0, pos] = slot

        # The stray later sync from remove_request must be a no-op: R's synced
        # bound already equals its committed length, so the reused row is never
        # read and the live slots are untouched.
        runner._sync_decode_kv_to_pool("R")
        mx.eval(*pool.all_buffers())
        self.assertEqual(pool.k_buffer[0][8][0][0].item(), 555.0)
        self.assertEqual(pool.k_buffer[0][9][0][0].item(), 555.0)
        self.assertEqual(runner._req_synced_offset["R"], 4)


if __name__ == "__main__":
    unittest.main()
