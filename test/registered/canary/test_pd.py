"""Host-side unit tests for KV cache canary in PD disaggregation.

PD splits prefill and decode across two nodes; KV is transferred but host
state lives on the originating node. The canary's design here keeps PD
**unaware of canary**:

- The K/V shadow tensors ride the regular KV transfer via the layer-shaped
  ``get_contiguous_buf_infos`` patch — PD's transport sees them as just
  "two extra layers". No PD-protocol fields, no MetadataBuffers extension.
- The decode side's ``K_req`` does NOT need to be transported. The host
  state lazily self-bootstraps on the first decode forward: ``k_req`` starts
  at 0, the chain seed is the configured seed, and the first decode write
  advances ``k_req`` to ``start_position + count``.

These tests pin those invariants so a future change that re-adds PD-aware
canary fields is caught.
"""

from __future__ import annotations

import unittest

import torch

from sglang.srt.disaggregation.utils import MetadataBuffers
from sglang.srt.kv_cache_canary.api import attach
from sglang.srt.kv_cache_canary.config import CanaryConfig, CanaryMode
from sglang.srt.kv_cache_canary.pool_patch import PoolKind
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, suite="base-b-test-1-gpu-small")


class _FakeMHAPool:
    """Stand-in for MHATokenToKVPool exposing the surface attach() needs."""

    def __init__(self, slot_count: int = 32, layer_num: int = 2) -> None:
        self.layer_num = layer_num
        self.page_size = 1
        self.k_buffer = [
            torch.zeros(slot_count, 1, 8, dtype=torch.bfloat16)
            for _ in range(layer_num)
        ]
        self.v_buffer = [
            torch.zeros(slot_count, 1, 8, dtype=torch.bfloat16)
            for _ in range(layer_num)
        ]

    def get_contiguous_buf_infos(self):
        k_ptrs = [b.data_ptr() for b in self.k_buffer]
        v_ptrs = [b.data_ptr() for b in self.v_buffer]
        k_lens = [b.nbytes for b in self.k_buffer]
        v_lens = [b.nbytes for b in self.v_buffer]
        k_item_lens = [b[0].nbytes for b in self.k_buffer]
        v_item_lens = [b[0].nbytes for b in self.v_buffer]
        return k_ptrs + v_ptrs, k_lens + v_lens, k_item_lens + v_item_lens


class _FakeReqToTokenPool:
    def __init__(self, size: int) -> None:
        self.size = size


class TestMetadataBuffersHasNoCanaryFields(unittest.TestCase):
    """PD MetadataBuffers must not be canary-aware.

    Phase 2.4 originally extended MetadataBuffers with ``canary_k_req`` and
    ``canary_prev_hash_tail``. That coupling has been reverted; the buffers
    layout returns to the original 10 entries.
    """

    def test_metadata_buffers_constructs_without_canary_kwarg(self) -> None:
        # No ``token_to_kv_pool`` kwarg — canary doesn't reach into PD.
        mb = MetadataBuffers(size=8, hidden_size=16, hidden_states_dtype=torch.float32)
        self.assertFalse(hasattr(mb, "canary_k_req"))
        self.assertFalse(hasattr(mb, "canary_prev_hash_tail"))

    def test_get_buf_infos_returns_only_original_entries(self) -> None:
        mb = MetadataBuffers(size=4, hidden_size=16, hidden_states_dtype=torch.float32)
        ptrs, lens, item_lens = mb.get_buf_infos()
        # 10 original entries — no canary entries appended.
        self.assertEqual(len(ptrs), 10)
        self.assertEqual(len(lens), 10)
        self.assertEqual(len(item_lens), 10)

    def test_get_buf_returns_only_original_entries(self) -> None:
        mb = MetadataBuffers(size=4, hidden_size=16, hidden_states_dtype=torch.float32)
        outputs = mb.get_buf(2)
        self.assertEqual(len(outputs), 10)


class TestDecodeSideSelfBootstrapsWithoutPDTransport(unittest.TestCase):
    """Decode side host state must not depend on PD transporting anything.

    The chain on the decode rank starts from the configured seed at
    ``k_req=0``; the first decode forward pure-writes (no verify entries)
    and advances ``k_req`` from the request's own bookkeeping.
    """

    def _make_runner(self) -> tuple[_FakeMHAPool, object]:
        pool = _FakeMHAPool(slot_count=32)
        config = CanaryConfig(mode=CanaryMode.LOG)
        req_to_token_pool = _FakeReqToTokenPool(size=16)
        runner = attach(
            pool=pool,
            config=config,
            req_to_token_pool=req_to_token_pool,
            device=torch.device("cpu"),
            pool_kind=PoolKind.FULL,
            launch_capacity=128,
        )
        return pool, runner

    def test_unseen_req_starts_at_seed_and_kreq_zero(self) -> None:
        _pool, runner = self._make_runner()
        # First plan_batch for a fresh decode-side req — no verify entries,
        # writes start from position 0 with prev_hash == config.seed.
        plan = runner.host_state.plan_batch(
            req_pool_indices=[5],
            req_token_counts=[1],
            req_start_positions=[7],  # decode arrives mid-sequence (prefill ran)
            input_tokens_per_req=[[42]],
            write_slot_indices_per_req=[[12]],
        )
        self.assertEqual(plan.num_verify, 0)
        self.assertEqual(plan.num_write, 1)
        u64_mask = (1 << 64) - 1
        # First write's expected_prev_hash equals the chain seed (decode
        # restarted the chain locally — no PD-transported tail needed).
        self.assertEqual(
            plan.expected_prev_hashes[0] & u64_mask, runner.config.seed
        )

    def test_decode_chain_advances_after_first_commit(self) -> None:
        _pool, runner = self._make_runner()
        plan_a = runner.host_state.plan_batch(
            req_pool_indices=[5],
            req_token_counts=[1],
            req_start_positions=[7],
            input_tokens_per_req=[[42]],
            write_slot_indices_per_req=[[12]],
        )
        runner.host_state.commit_plan(plan_a)
        # Second decode forward verifies the just-written entry and writes
        # one more.
        plan_b = runner.host_state.plan_batch(
            req_pool_indices=[5],
            req_token_counts=[1],
            req_start_positions=[8],
            input_tokens_per_req=[[43]],
            write_slot_indices_per_req=[[13]],
        )
        self.assertEqual(plan_b.num_verify, 1)
        self.assertEqual(plan_b.num_write, 1)
        # Verify entry covers the position written by plan_a.
        self.assertEqual(plan_b.verify_seq_positions[0], 7)


if __name__ == "__main__":
    unittest.main()
