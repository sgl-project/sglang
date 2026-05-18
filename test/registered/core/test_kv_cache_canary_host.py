"""Host-side unit tests for KV cache canary (no GPU kernel involved).

These cover the planner, the per-request state machine, and the pool monkey-patch
arithmetic.
"""

from __future__ import annotations

import unittest

import torch

from sglang.srt.kv_cache_canary.config import CanaryConfig, CanaryMode
from sglang.srt.kv_cache_canary.fingerprint import mix_step
from sglang.srt.kv_cache_canary.host_state import CanaryHostState
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, suite="base-b-test-1-gpu-small")


class TestCanaryHostState(unittest.TestCase):
    def _make_state(self, seed: int = 0x9E3779B97F4A7C15) -> CanaryHostState:
        config = CanaryConfig(mode=CanaryMode.LOG, seed=seed)
        return CanaryHostState(config=config, num_req_slots=128)

    def test_plan_emits_write_only_mask_for_first_chunk(self):
        state = self._make_state()
        plan = state.plan_batch(
            req_pool_indices=[3],
            req_token_counts=[4],
            req_start_positions=[0],
            input_tokens_per_req=[[10, 20, 30, 40]],
        )
        self.assertEqual(plan.verify_mask, [0, 0, 0, 0])
        self.assertEqual(plan.expected_req_ids, [3, 3, 3, 3])
        self.assertEqual(plan.expected_token_ids, [10, 20, 30, 40])
        self.assertEqual(plan.expected_positions, [0, 1, 2, 3])
        # prev_hash chain advances locally; the first entry equals SEED.
        self.assertEqual(
            plan.expected_prev_hashes[0] & ((1 << 64) - 1), state._config.seed
        )

    def test_plan_marks_verify_only_for_positions_below_high_water_mark(self):
        state = self._make_state()
        # Step 1: prefill 3 tokens.
        plan1 = state.plan_batch(
            req_pool_indices=[7],
            req_token_counts=[3],
            req_start_positions=[0],
            input_tokens_per_req=[[1, 2, 3]],
        )
        state.commit_plan(plan1)
        # Step 2: decode one new token at position 3 with the prior hash carried over.
        plan2 = state.plan_batch(
            req_pool_indices=[7],
            req_token_counts=[1],
            req_start_positions=[3],
            input_tokens_per_req=[[4]],
        )
        # Position 3 is the new write, so verify_mask must be 0 (write-only).
        self.assertEqual(plan2.verify_mask, [0])
        # The expected_prev_hash for that token equals the chain hash through positions 0..2.
        h = state._config.seed
        for t, p in [(1, 0), (2, 1), (3, 2)]:
            h = mix_step(h, t, p)
        self.assertEqual(plan2.expected_prev_hashes[0] & ((1 << 64) - 1), h)

    def test_chunked_prefill_advance_keeps_chain_continuous(self):
        state = self._make_state()
        # Chunk A: positions 0..1
        plan_a = state.plan_batch(
            req_pool_indices=[5],
            req_token_counts=[2],
            req_start_positions=[0],
            input_tokens_per_req=[[100, 200]],
        )
        state.commit_plan(plan_a)
        # Chunk B: positions 2..3 — prev_hash[0] for the 2nd chunk must be the
        # chain hash from chunk A.
        plan_b = state.plan_batch(
            req_pool_indices=[5],
            req_token_counts=[2],
            req_start_positions=[2],
            input_tokens_per_req=[[300, 400]],
        )
        h = state._config.seed
        h = mix_step(h, 100, 0)
        h = mix_step(h, 200, 1)
        self.assertEqual(plan_b.expected_prev_hashes[0] & ((1 << 64) - 1), h)

    def test_reset_request_restarts_chain_at_seed(self):
        state = self._make_state()
        plan = state.plan_batch(
            req_pool_indices=[9],
            req_token_counts=[2],
            req_start_positions=[0],
            input_tokens_per_req=[[7, 8]],
        )
        state.commit_plan(plan)
        state.reset_request(9)
        plan2 = state.plan_batch(
            req_pool_indices=[9],
            req_token_counts=[1],
            req_start_positions=[0],
            input_tokens_per_req=[[99]],
        )
        self.assertEqual(
            plan2.expected_prev_hashes[0] & ((1 << 64) - 1), state._config.seed
        )


class _FakePool:
    """Stand-in for MHATokenToKVPool with only the attributes pool_patch needs."""

    def __init__(
        self,
        *,
        layer_num: int,
        slot_count: int,
        head_num: int,
        head_dim: int,
        page_size: int,
    ):
        self.layer_num = layer_num
        self.page_size = page_size
        self.k_buffer = [
            torch.zeros(slot_count, head_num, head_dim, dtype=torch.bfloat16)
            for _ in range(layer_num)
        ]
        self.v_buffer = [
            torch.zeros(slot_count, head_num, head_dim, dtype=torch.bfloat16)
            for _ in range(layer_num)
        ]

    def get_contiguous_buf_infos(self):
        k_ptrs = [b.data_ptr() for b in self.k_buffer]
        v_ptrs = [b.data_ptr() for b in self.v_buffer]
        k_lens = [b.nbytes for b in self.k_buffer]
        v_lens = [b.nbytes for b in self.v_buffer]
        k_item_lens = [b[0].nbytes * self.page_size for b in self.k_buffer]
        v_item_lens = [b[0].nbytes * self.page_size for b in self.v_buffer]
        return k_ptrs + v_ptrs, k_lens + v_lens, k_item_lens + v_item_lens


class TestPoolPatch(unittest.TestCase):
    def test_patched_get_contiguous_buf_infos_keeps_len_div_2_midpoint(self):
        from sglang.srt.kv_cache_canary.pool_patch import attach_shadow_buffers

        pool = _FakePool(
            layer_num=4, slot_count=32, head_num=2, head_dim=16, page_size=1
        )
        attach_shadow_buffers(pool)
        ptrs, lens, item_lens = pool.get_contiguous_buf_infos()
        # Original 2*L = 8 entries -> after patch, 2*L + 4 = 12 entries.
        self.assertEqual(len(ptrs), 12)
        self.assertEqual(len(lens), 12)
        self.assertEqual(len(item_lens), 12)
        # len // 2 still bisects K vs V (6 vs 6).
        mid = len(ptrs) // 2
        self.assertEqual(mid, 6)
        # K side ends with the two K canaries (head, tail), in that order.
        self.assertEqual(ptrs[4], pool.canary_k_head.data_ptr())
        self.assertEqual(ptrs[5], pool.canary_k_tail.data_ptr())
        # V side ends with the two V canaries (head, tail), in that order.
        self.assertEqual(ptrs[10], pool.canary_v_head.data_ptr())
        self.assertEqual(ptrs[11], pool.canary_v_tail.data_ptr())
        # All per-slot strides are identical (item_lens are uniform within K and V).
        self.assertEqual(len(set(item_lens[:6])), 1)
        self.assertEqual(len(set(item_lens[6:])), 1)

    def test_attach_is_idempotent(self):
        from sglang.srt.kv_cache_canary.pool_patch import attach_shadow_buffers

        pool = _FakePool(
            layer_num=2, slot_count=16, head_num=1, head_dim=8, page_size=1
        )
        attach_shadow_buffers(pool)
        first_head_ptr = pool.canary_k_head.data_ptr()
        attach_shadow_buffers(pool)
        self.assertEqual(pool.canary_k_head.data_ptr(), first_head_ptr)


if __name__ == "__main__":
    unittest.main()
