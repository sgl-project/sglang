"""Host-side unit tests for KV cache canary in PD disaggregation (Phase 2.4).

PD splits prefill and decode across two nodes; KV is transferred but host
state lives on the originating node. The canary's K_req high-water mark and
prev_hash_tail must ride along with the KV transfer so the decode side can
continue the chain.

Transport: two new MetadataBuffers fields ``canary_k_req`` (int64) and
``canary_prev_hash_tail`` (int64). Prefill side packs both via
``export_pd_canary_snapshot`` inside ``MetadataBuffers.set_buf``; decode
side rebuilds via ``apply_pd_canary_snapshot`` inside the receive path.

Depends on Phase 1 fix for end-to-end mismatch detection; this file only
exercises the host bookkeeping + MetadataBuffers serialize/deserialize
roundtrip.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.disaggregation.utils import MetadataBuffers
from sglang.srt.kv_cache_canary.api import (
    apply_pd_canary_snapshot,
    attach,
    export_pd_canary_snapshot,
)
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


class TestMetadataBuffersCanaryFields(unittest.TestCase):
    def test_canary_fields_allocated_with_expected_dtype_and_size(self) -> None:
        mb = MetadataBuffers(size=8, hidden_size=16, hidden_states_dtype=torch.float32)
        self.assertEqual(mb.canary_k_req.dtype, torch.int64)
        self.assertEqual(mb.canary_prev_hash_tail.dtype, torch.int64)
        # 8 slots * 8 elements/slot for RDMA padding.
        self.assertEqual(mb.canary_k_req.shape, (8, 8))
        self.assertEqual(mb.canary_prev_hash_tail.shape, (8, 8))

    def test_get_buf_infos_includes_canary_fields(self) -> None:
        mb = MetadataBuffers(size=4, hidden_size=16, hidden_states_dtype=torch.float32)
        ptrs, lens, item_lens = mb.get_buf_infos()
        # 10 original entries + 2 canary entries = 12.
        self.assertEqual(len(ptrs), 12)
        self.assertEqual(len(lens), 12)
        self.assertEqual(len(item_lens), 12)
        self.assertEqual(ptrs[-2], mb.canary_k_req.data_ptr())
        self.assertEqual(ptrs[-1], mb.canary_prev_hash_tail.data_ptr())

    def test_get_buf_returns_canary_fields(self) -> None:
        mb = MetadataBuffers(size=4, hidden_size=16, hidden_states_dtype=torch.float32)
        mb.canary_k_req[2, 0] = 42
        mb.canary_prev_hash_tail[2, 0] = -1  # sign-bit-set negative -> wraps to UINT64
        outputs = mb.get_buf(2)
        self.assertEqual(len(outputs), 12)
        canary_k_req, canary_prev_hash_tail = outputs[-2], outputs[-1]
        self.assertEqual(int(canary_k_req[0].item()), 42)
        self.assertEqual(
            int(canary_prev_hash_tail[0].item()) & ((1 << 64) - 1), (1 << 64) - 1
        )


class TestPDExportImportRoundtrip(unittest.TestCase):
    def _make_runner(self) -> object:
        pool = _FakeMHAPool(slot_count=32)
        config = CanaryConfig(mode=CanaryMode.LOG)
        req_to_token_pool = _FakeReqToTokenPool(size=16)
        runner = attach(
            pool=pool,
            config=config,
            req_to_token_pool=req_to_token_pool,
            device=torch.device("cpu"),
            pool_kind=PoolKind.FULL,
        )
        return pool, runner

    def test_export_returns_zero_for_unknown_req(self) -> None:
        pool, _runner = self._make_runner()
        k_req, prev_hash_tail = export_pd_canary_snapshot(pool=pool, req_pool_idx=99)
        self.assertEqual(k_req, 0)
        self.assertEqual(prev_hash_tail, 0)

    def test_export_after_plan_commit_returns_high_water_mark(self) -> None:
        pool, runner = self._make_runner()
        plan = runner.host_state.plan_batch(
            req_pool_indices=[5],
            req_token_counts=[3],
            req_start_positions=[0],
            input_tokens_per_req=[[111, 222, 333]],
            write_slot_indices_per_req=[[10, 11, 12]],
        )
        runner.host_state.commit_plan(plan)
        k_req, prev_hash_tail = export_pd_canary_snapshot(pool=pool, req_pool_idx=5)
        self.assertEqual(k_req, 3)
        self.assertNotEqual(prev_hash_tail, runner.config.seed)

    def test_apply_rebuilds_state_so_decode_continues_at_k_req(self) -> None:
        # Prefill side: pack a snapshot.
        prefill_pool, prefill_runner = self._make_runner()
        plan = prefill_runner.host_state.plan_batch(
            req_pool_indices=[7],
            req_token_counts=[4],
            req_start_positions=[0],
            input_tokens_per_req=[[10, 20, 30, 40]],
            write_slot_indices_per_req=[[100, 101, 102, 103]],
        )
        prefill_runner.host_state.commit_plan(plan)
        snap_k_req, snap_prev_hash_tail = export_pd_canary_snapshot(
            pool=prefill_pool, req_pool_idx=7
        )

        # Decode side: fresh runner, import the snapshot.
        decode_pool, decode_runner = self._make_runner()
        self.assertFalse(decode_runner.host_state.has_state(7))
        apply_pd_canary_snapshot(
            pool=decode_pool,
            req_pool_idx=7,
            k_req=snap_k_req,
            prev_hash_tail=snap_prev_hash_tail,
        )
        self.assertTrue(decode_runner.host_state.has_state(7))

        # Decode's first plan_batch must pure-write (no verify; last_committed
        # is None after import) but use the imported prev_hash_tail as the
        # chain seed for position k_req.
        plan_decode = decode_runner.host_state.plan_batch(
            req_pool_indices=[7],
            req_token_counts=[1],
            req_start_positions=[snap_k_req],
            input_tokens_per_req=[[50]],
            write_slot_indices_per_req=[[200]],
        )
        self.assertEqual(plan_decode.num_verify, 0)
        self.assertEqual(plan_decode.num_write, 1)
        # The first decode write's expected_prev_hash equals the
        # prev_hash_tail we imported (chain continues seamlessly).
        u64_mask = (1 << 64) - 1
        self.assertEqual(
            plan_decode.expected_prev_hashes[0] & u64_mask,
            snap_prev_hash_tail & u64_mask,
        )

    def test_apply_with_zero_k_req_is_noop(self) -> None:
        # Simulates the prefill side having no canary attached: it sends 0/0
        # and the decode side must NOT clobber its own seed-initialized chain.
        decode_pool, decode_runner = self._make_runner()
        apply_pd_canary_snapshot(
            pool=decode_pool, req_pool_idx=42, k_req=0, prev_hash_tail=0
        )
        self.assertFalse(decode_runner.host_state.has_state(42))


class TestMetadataBuffersSetBufWritesCanaryFromPool(unittest.TestCase):
    def test_set_buf_pulls_canary_snapshot_from_pool(self) -> None:
        pool = _FakeMHAPool(slot_count=32)
        config = CanaryConfig(mode=CanaryMode.LOG)
        runner = attach(
            pool=pool,
            config=config,
            req_to_token_pool=_FakeReqToTokenPool(size=16),
            device=torch.device("cpu"),
            pool_kind=PoolKind.FULL,
        )
        # Plan + commit a state so the snapshot is non-zero.
        plan = runner.host_state.plan_batch(
            req_pool_indices=[3],
            req_token_counts=[2],
            req_start_positions=[0],
            input_tokens_per_req=[[77, 88]],
            write_slot_indices_per_req=[[400, 401]],
        )
        runner.host_state.commit_plan(plan)

        mb = MetadataBuffers(
            size=4,
            hidden_size=16,
            hidden_states_dtype=torch.float32,
            token_to_kv_pool=pool,
        )
        # Build a minimal req that set_buf reads from.
        req = SimpleNamespace(
            metadata_buffer_index=1,
            output_ids=[123],
            cached_tokens=0,
            cached_tokens_device=0,
            cached_tokens_host=0,
            cached_tokens_storage=0,
            return_logprob=False,
            hidden_states_tensor=None,
            bootstrap_room=42,
            req_pool_idx=3,
        )
        mb.set_buf(req)
        self.assertEqual(int(mb.canary_k_req[1, 0].item()), 2)
        # prev_hash_tail is set (non-zero); we don't pin the exact value
        # because the hash function might be tweaked over time.
        self.assertNotEqual(int(mb.canary_prev_hash_tail[1, 0].item()), 0)

    def test_set_buf_writes_zero_when_pool_has_no_canary_attached(self) -> None:
        pool = _FakeMHAPool(slot_count=8)
        mb = MetadataBuffers(
            size=4,
            hidden_size=16,
            hidden_states_dtype=torch.float32,
            token_to_kv_pool=pool,
        )
        req = SimpleNamespace(
            metadata_buffer_index=0,
            output_ids=[5],
            cached_tokens=0,
            cached_tokens_device=0,
            cached_tokens_host=0,
            cached_tokens_storage=0,
            return_logprob=False,
            hidden_states_tensor=None,
            bootstrap_room=1,
            req_pool_idx=2,
        )
        mb.set_buf(req)
        # No canary attached → snapshot is (0, 0) → fields remain at their
        # zero-initialized values.
        self.assertEqual(int(mb.canary_k_req[0, 0].item()), 0)
        self.assertEqual(int(mb.canary_prev_hash_tail[0, 0].item()), 0)

    def test_set_buf_writes_zero_when_no_pool_ref_provided(self) -> None:
        # token_to_kv_pool=None means PD was set up before any canary integration
        # (older code paths). set_buf must not blow up; it just leaves the
        # canary slots at zero.
        mb = MetadataBuffers(size=4, hidden_size=16, hidden_states_dtype=torch.float32)
        req = SimpleNamespace(
            metadata_buffer_index=0,
            output_ids=[5],
            cached_tokens=0,
            cached_tokens_device=0,
            cached_tokens_host=0,
            cached_tokens_storage=0,
            return_logprob=False,
            hidden_states_tensor=None,
            bootstrap_room=1,
            req_pool_idx=2,
        )
        mb.set_buf(req)
        self.assertEqual(int(mb.canary_k_req[0, 0].item()), 0)


if __name__ == "__main__":
    unittest.main()
