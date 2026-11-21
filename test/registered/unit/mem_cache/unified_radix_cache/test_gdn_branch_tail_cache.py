"""Real cache ownership and bounded allocation for optional GDN snapshots."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.environ import envs
from sglang.srt.managers.req_types import FINISH_LENGTH
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.mem_cache.base_prefix_cache import EvictParams
from sglang.srt.mem_cache.common import release_kv_cache
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.runtime_context import get_context
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.unified_radix_cache.config import CacheConfig
from sglang.test.kits.unified_radix_cache.fixtures import (
    build_fixture,
    match_prefix,
    node_device_value,
)
from sglang.test.kits.unified_radix_cache.mixins import UnifiedRadixCacheFixtures
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")


def _rank_rejects_reservation(rank, rendezvous):
    torch.cuda.set_device(rank)
    torch.distributed.init_process_group(
        "gloo", init_method=rendezvous, rank=rank, world_size=2
    )
    try:
        cache, _, pool = build_fixture(TestGDNBranchTailOwnership.cfg)
        held = (
            pool.mamba_allocator.alloc(pool.mamba_allocator.available_size())
            if rank == 1
            else None
        )
        before = pool.mamba_allocator.available_size()
        group = SimpleNamespace(world_size=2, cpu_group=torch.distributed.group.WORLD)
        slot = cache.reserve_gdn_branch_snapshot(
            before > 0, group, is_middle_chunk=False
        )
        assert slot is None
        assert pool.mamba_allocator.available_size() == before
        if held is not None:
            pool.mamba_allocator.free(held)
        cache.sanity_check()
    finally:
        torch.distributed.destroy_process_group()


class TestGDNBranchTailOwnership(UnifiedRadixCacheFixtures, CustomTestCase):
    cfg = CacheConfig(
        page_size=64,
        components=(ComponentType.FULL, ComponentType.MAMBA),
        enable_mamba_extra_buffer=True,
        hybrid_cache_checkpoints_level=2,
        mamba_cache_size=32,
        kv_size=512,
        max_context_len=512,
        num_layers=2,
        full_attention_layer_ids=(1,),
    )

    def setUp(self):
        self.cache, self.allocator, self.pool = build_fixture(self.cfg)
        self.group = SimpleNamespace(world_size=1)
        self.tokens = list(range(192))
        self.req, _ = self._make_inflight_req(
            self.cache, self.allocator, self.pool, self.tokens, allocated=True
        )
        self.req.kv.mamba_last_track_seqlen = 192
        self.req.kv.mamba_last_track_idx = 0
        self.req.kv.mamba_next_track_idx = 1
        self.req.mamba_branching_seqlen = 64

    def _reserve(self, spare=True, middle=False, group=None):
        return self.cache.reserve_gdn_branch_snapshot(
            spare, self.group if group is None else group, is_middle_chunk=middle
        )

    def _batch(self, reqs):
        batch = ScheduleBatch(reqs=reqs)
        batch.tree_cache = self.cache
        batch.token_to_kv_pool_allocator = self.allocator
        batch.req_to_token_pool = self.pool
        batch.model_config = SimpleNamespace(
            hf_config=SimpleNamespace(
                architectures=["Qwen3_5ForConditionalGeneration"]
            ),
            hf_text_config=SimpleNamespace(mamba_chunk_size=64),
        )
        batch.spec_algorithm = SpeculativeAlgorithm.NONE
        return batch

    def _finish(self, checkpoint=True):
        self.req.finished_reason = FINISH_LENGTH(length=0)
        release_kv_cache(self.req, self.cache, checkpoint=checkpoint)
        self.cache.sanity_check()

    def _evict_all(self):
        self.cache.evict(
            EvictParams(
                num_tokens=self.cfg.kv_size, mamba_num=self.cfg.mamba_cache_size
            )
        )
        self.cache.sanity_check()
        self.assertEqual(self.allocator.available_size(), self.cfg.kv_size)
        self.assertEqual(
            self.pool.mamba_allocator.available_size(), self.cfg.mamba_cache_size
        )

    def test_explicit_branch_and_native_tail_have_distinct_tree_owned_slots(self):
        slot = self._reserve()
        native = self.req.kv.mamba_ping_pong_track_buffer[0].clone()
        self.cache.publish_gdn_branch_snapshot(self.req, 64, slot, valid=True)
        self.assertEqual(self.req.kv.cache_protected_len, 64)
        self.assertEqual(self.req.kv.mamba_last_track_seqlen, 192)
        self._finish()
        branch = match_prefix(self.cache, self.tokens[:64] + [999])
        tail = match_prefix(self.cache, self.tokens)
        self.assertEqual(branch.device_prefix_len, 64)
        self.assertEqual(tail.device_prefix_len, 192)
        self.assertTrue(
            torch.equal(
                node_device_value(
                    self.cache, branch.last_device_node, ComponentType.MAMBA
                ),
                slot,
            )
        )
        self.assertTrue(
            torch.equal(
                node_device_value(
                    self.cache, tail.last_device_node, ComponentType.MAMBA
                ),
                native.view(-1),
            )
        )
        self.assertNotEqual(slot.item(), native.item())
        self.assertEqual(
            self.pool.mamba_allocator.available_size(), self.cfg.mamba_cache_size - 2
        )
        self._evict_all()

    def test_duplicate_branch_returns_the_unused_slot_once(self):
        first = self._reserve()
        self.cache.publish_gdn_branch_snapshot(self.req, 64, first, valid=True)
        before = self.pool.mamba_allocator.available_size()
        duplicate = self._reserve()
        self.cache.publish_gdn_branch_snapshot(self.req, 64, duplicate, valid=True)
        self.assertEqual(self.pool.mamba_allocator.available_size(), before)
        match = match_prefix(self.cache, self.tokens[:64] + [999])
        self.assertTrue(
            torch.equal(
                node_device_value(
                    self.cache, match.last_device_node, ComponentType.MAMBA
                ),
                first,
            )
        )
        self._finish()
        self._evict_all()

    def test_discarded_forward_returns_the_request_owned_slot_once(self):
        before = self.pool.mamba_allocator.available_size()
        slot = self._reserve()
        self.assertEqual(self.pool.mamba_allocator.available_size(), before - 1)
        self.cache.publish_gdn_branch_snapshot(self.req, 64, slot, valid=False)
        self.assertEqual(self.pool.mamba_allocator.available_size(), before)
        self.assertEqual(match_prefix(self.cache, self.tokens).device_prefix_len, 0)
        self._finish(checkpoint=False)
        self._evict_all()

    def test_middle_extend_and_no_spare_capacity_do_not_allocate(self):
        before = self.pool.mamba_allocator.available_size()
        group = SimpleNamespace(world_size=2, cpu_group=object())
        with patch("torch.distributed.all_reduce") as collective:
            self.assertIsNone(self._reserve(middle=True, group=group))
            collective.assert_not_called()
        self.assertIsNone(self._reserve(spare=False))
        self.assertEqual(self.pool.mamba_allocator.available_size(), before)
        self._finish(checkpoint=False)
        self._evict_all()

    def test_remote_rank_rejection_preserves_all_free_slots(self):
        before = self.pool.mamba_allocator.available_size()
        group = SimpleNamespace(world_size=2, cpu_group=object())
        with patch(
            "torch.distributed.all_reduce",
            side_effect=lambda capacity, **_: capacity.zero_(),
        ) as collective:
            self.assertIsNone(self._reserve(group=group))
            collective.assert_called_once()
        self.assertEqual(self.pool.mamba_allocator.available_size(), before)
        self._finish(checkpoint=False)
        self._evict_all()

    @unittest.skipUnless(torch.cuda.device_count() >= 2, "Requires two CUDA devices")
    def test_two_real_ranks_agree_when_only_one_has_spare_capacity(self):
        with tempfile.TemporaryDirectory() as directory:
            rendezvous = "file://" + str(Path(directory) / "rendezvous")
            torch.multiprocessing.spawn(
                _rank_rejects_reservation, args=(rendezvous,), nprocs=2, join=True
            )
        self._finish(checkpoint=False)
        self._evict_all()

    def test_same_batch_second_candidate_uses_native_tracking(self):
        other, _ = self._make_inflight_req(
            self.cache, self.allocator, self.pool, self.tokens, allocated=True
        )
        other.mamba_branching_seqlen = 64
        batch = self._batch([self.req, other])
        with (
            envs.SGLANG_EXPERIMENTAL_GDN_BRANCH_AND_TAIL.override(True),
            get_context().override_server_args(
                _mamba_cache_chunk_size=64,
                disable_overlap_schedule=True,
                disable_prefill_cuda_graph=True,
                linear_attn_prefill_backend="triton",
                mamba_ssm_dtype="bfloat16",
            ),
            patch(
                "sglang.srt.distributed.parallel_state.get_tp_group",
                return_value=self.group,
            ),
        ):
            first_entry = batch._mamba_radix_cache_v2_req_prepare_for_extend(
                self.req, 0
            )
            first = batch.gdn_branch_snapshot
            second_entry = batch._mamba_radix_cache_v2_req_prepare_for_extend(other, 1)
            self.assertEqual(first_entry.track_seqlen, 192)
            self.assertEqual(second_entry.track_seqlen, 65)
            self.assertEqual(self.req.kv.mamba_last_track_seqlen, 192)
            self.assertEqual(other.kv.mamba_last_track_seqlen, 64)
            self.assertIs(batch.gdn_branch_snapshot, first)
        self.cache.publish_gdn_branch_snapshot(self.req, 64, first[2], valid=False)
        self._finish(checkpoint=False)
        other.finished_reason = FINISH_LENGTH(length=0)
        release_kv_cache(other, self.cache, checkpoint=False)
        self._evict_all()

    def test_unsupported_combinations_do_not_allocate_or_synchronize(self):
        batch = self._batch([self.req])
        before = self.pool.mamba_allocator.available_size()
        for change in (
            {"disaggregation_mode": "prefill"},
            {"disable_overlap_schedule": False},
            {"disable_prefill_cuda_graph": False},
            {"linear_attn_prefill_backend": "flashinfer"},
            {"mamba_ssm_dtype": "float32"},
        ):
            options = dict(
                _mamba_cache_chunk_size=64,
                disable_overlap_schedule=True,
                disable_prefill_cuda_graph=True,
                linear_attn_prefill_backend="triton",
                mamba_ssm_dtype="bfloat16",
            )
            options.update(change)
            with (
                self.subTest(change=change),
                envs.SGLANG_EXPERIMENTAL_GDN_BRANCH_AND_TAIL.override(True),
                get_context().override_server_args(**options),
                patch("sglang.srt.distributed.parallel_state.get_tp_group") as group,
            ):
                self.assertFalse(
                    batch._reserve_gdn_branch_snapshot(self.req, 0, 64, 192)
                )
                group.assert_not_called()
            self.assertEqual(self.pool.mamba_allocator.available_size(), before)
        self._finish(checkpoint=False)
        self._evict_all()


if __name__ == "__main__":
    unittest.main()
