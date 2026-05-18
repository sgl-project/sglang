"""Host-side unit tests for KV cache canary in spec decoding (Phase 2.3).

Spec decoding (Eagle / MTP / NEXTN / Medusa) runs two pools:

- Draft pool — owned by ``draft_worker.model_runner.token_to_kv_pool``;
- Target pool — owned by ``target_worker.model_runner.token_to_kv_pool``.

Each ModelRunner goes through ``install_on_model_runner`` independently, so
each gets its own ``CanaryRunner``, its own host state, its own pool patch.
Two independent chains, never mixed.

Reject path: ``token_to_kv_pool_allocator.free(out_cache_loc[evict_mask])`` is
called inside Eagle's verify (see ``speculative/eagle_info.py``). The canary
host state's ``last_committed`` must drop on every free batch — otherwise
the next forward verify-reads against a slot that just got given back.

Depends on Phase 1 fix (C1/C2/C3) for end-to-end mismatch detection; this
file only exercises the host-side bookkeeping + hook plumbing.
"""

from __future__ import annotations

import unittest

import torch

from sglang.srt.kv_cache_canary.api import install_spec_allocator_free_hook
from sglang.srt.kv_cache_canary.config import CanaryConfig, CanaryMode
from sglang.srt.kv_cache_canary.host_state import CanaryHostState
from sglang.srt.kv_cache_canary.pool_patch import PoolKind, attach_shadow_buffers
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, suite="base-b-test-1-gpu-small")


class _FakeAllocator:
    def __init__(self) -> None:
        self.freed: list[int] = []

    def free(self, free_index: torch.Tensor) -> None:
        self.freed.extend(int(x) for x in free_index.tolist())


class _FakeRunner:
    def __init__(self, host_state: CanaryHostState) -> None:
        self.host_state = host_state


class _FakeSpecAlgorithm:
    @staticmethod
    def is_speculative() -> bool:
        return True


class _FakeModelRunner:
    def __init__(self, allocator: _FakeAllocator) -> None:
        self.token_to_kv_pool_allocator = allocator
        self.spec_algorithm = _FakeSpecAlgorithm()


class TestSpecAllocatorFreeHook(unittest.TestCase):
    def _make_host_state(self) -> CanaryHostState:
        config = CanaryConfig(mode=CanaryMode.LOG)
        return CanaryHostState(config=config, num_req_slots=64)

    def test_free_hook_clears_last_committed_after_each_free(self) -> None:
        host_state = self._make_host_state()
        plan = host_state.plan_batch(
            req_pool_indices=[1],
            req_token_counts=[3],
            req_start_positions=[0],
            input_tokens_per_req=[[10, 20, 30]],
            write_slot_indices_per_req=[[100, 101, 102]],
        )
        host_state.commit_plan(plan)
        self.assertTrue(host_state.has_state(1))

        allocator = _FakeAllocator()
        model_runner = _FakeModelRunner(allocator)
        runner = _FakeRunner(host_state)
        install_spec_allocator_free_hook(
            runner=runner,  # type: ignore[arg-type]
            model_runner=model_runner,  # type: ignore[arg-type]
        )

        # Simulate Eagle's reject path freeing some draft slots.
        allocator.free(torch.tensor([101, 102], dtype=torch.int64))

        # Underlying allocator.free still ran (we observe via its bookkeeping).
        self.assertEqual(allocator.freed, [101, 102])
        # Subsequent plan_batch must not emit a verify entry — the host state
        # forgot last_committed when the slots were freed.
        plan2 = host_state.plan_batch(
            req_pool_indices=[1],
            req_token_counts=[1],
            req_start_positions=[3],
            input_tokens_per_req=[[40]],
            write_slot_indices_per_req=[[103]],
        )
        self.assertEqual(plan2.num_verify, 0)

    def test_free_hook_is_idempotent_per_allocator(self) -> None:
        host_state = self._make_host_state()
        allocator = _FakeAllocator()
        model_runner = _FakeModelRunner(allocator)
        runner = _FakeRunner(host_state)
        install_spec_allocator_free_hook(
            runner=runner,  # type: ignore[arg-type]
            model_runner=model_runner,  # type: ignore[arg-type]
        )
        original_free = allocator.free
        # Second install must be a no-op (does not re-wrap).
        install_spec_allocator_free_hook(
            runner=runner,  # type: ignore[arg-type]
            model_runner=model_runner,  # type: ignore[arg-type]
        )
        self.assertIs(allocator.free, original_free)


class TestSpecHostStateRejectReset(unittest.TestCase):
    def _make_state(self) -> CanaryHostState:
        config = CanaryConfig(mode=CanaryMode.LOG)
        return CanaryHostState(config=config, num_req_slots=64)

    def test_reset_request_to_lowers_k_req(self) -> None:
        state = self._make_state()
        # Simulate: draft proposed 5 tokens, target accepted only 2 — reject
        # 3 trailing tokens. Real flow: K_req was 5, must drop to 2.
        plan = state.plan_batch(
            req_pool_indices=[7],
            req_token_counts=[5],
            req_start_positions=[0],
            input_tokens_per_req=[[1, 2, 3, 4, 5]],
            write_slot_indices_per_req=[[10, 11, 12, 13, 14]],
        )
        state.commit_plan(plan)
        state.reset_request_to(req_pool_idx=7, k_req=2)

        # After reset_request_to(k_req=2), history is truncated to positions
        # [0, 2): only the two accepted slots remain. The next plan therefore
        # verifies exactly those two and never reads the freed (rejected)
        # slots 12/13/14.
        plan2 = state.plan_batch(
            req_pool_indices=[7],
            req_token_counts=[1],
            req_start_positions=[2],
            input_tokens_per_req=[[2222]],
            write_slot_indices_per_req=[[15]],
        )
        self.assertEqual(plan2.num_verify, 2)
        self.assertEqual(plan2.verify_slot_indices, [10, 11])

    def test_reset_request_to_zero_drops_request_completely(self) -> None:
        state = self._make_state()
        plan = state.plan_batch(
            req_pool_indices=[3],
            req_token_counts=[2],
            req_start_positions=[0],
            input_tokens_per_req=[[100, 200]],
            write_slot_indices_per_req=[[300, 301]],
        )
        state.commit_plan(plan)
        state.reset_request_to(req_pool_idx=3, k_req=0)
        self.assertFalse(state.has_state(3))

    def test_reset_request_to_unknown_idx_is_noop(self) -> None:
        state = self._make_state()
        # Must not raise on never-tracked req_pool_idx.
        state.reset_request_to(req_pool_idx=99, k_req=1)
        self.assertFalse(state.has_state(99))


class TestDraftAndTargetPoolKindsAreIndependent(unittest.TestCase):
    """Two separate pools = two separate runners; states never alias."""

    def test_draft_and_target_use_separate_canary_attrs(self) -> None:
        # Two fake MHA-style pools; each gets its own shadow set.
        class _FakePool:
            def __init__(self, slot_count: int) -> None:
                self.layer_num = 2
                self.page_size = 1
                self.k_buffer = [
                    torch.zeros(slot_count, 1, 8, dtype=torch.bfloat16)
                    for _ in range(2)
                ]
                self.v_buffer = [
                    torch.zeros(slot_count, 1, 8, dtype=torch.bfloat16)
                    for _ in range(2)
                ]

            def get_contiguous_buf_infos(self):
                k_ptrs = [b.data_ptr() for b in self.k_buffer]
                v_ptrs = [b.data_ptr() for b in self.v_buffer]
                k_lens = [b.nbytes for b in self.k_buffer]
                v_lens = [b.nbytes for b in self.v_buffer]
                k_item_lens = [b[0].nbytes for b in self.k_buffer]
                v_item_lens = [b[0].nbytes for b in self.v_buffer]
                return (
                    k_ptrs + v_ptrs,
                    k_lens + v_lens,
                    k_item_lens + v_item_lens,
                )

        draft_pool = _FakePool(slot_count=8)
        target_pool = _FakePool(slot_count=64)
        attach_shadow_buffers(draft_pool, pool_kind=PoolKind.DRAFT)
        attach_shadow_buffers(target_pool, pool_kind=PoolKind.TARGET)

        # Pools have distinct shadow buffers (different ptr, different sizes).
        self.assertNotEqual(
            draft_pool.canary_k_head.data_ptr(),
            target_pool.canary_k_head.data_ptr(),
        )
        self.assertEqual(draft_pool.canary_k_head.shape[0], 8)
        self.assertEqual(target_pool.canary_k_head.shape[0], 64)


if __name__ == "__main__":
    unittest.main()
