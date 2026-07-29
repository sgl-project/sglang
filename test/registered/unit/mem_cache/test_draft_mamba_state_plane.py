"""Unit tests for srt/mem_cache/draft_mamba_state_plane.py"""

import unittest
from array import array

import torch

from sglang.srt.configs.mamba_utils import Mamba2CacheParams, Mamba2StateShape
from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.mem_cache.draft_mamba_state_plane import DraftMambaStatePlane
from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool, MambaPool
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.utils import get_device
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=8, stage="base-b", runner_config="1-gpu-small")

MAX_CONTEXT_LEN = 128


def _make_req(rid: int) -> Req:
    return Req(
        rid=rid,
        origin_input_text="",
        origin_input_ids=array("q"),
        sampling_params=SamplingParams(temperature=0, max_new_tokens=1),
    )


def _make_state_shape() -> Mamba2StateShape:
    return Mamba2StateShape.create(
        tp_world_size=1,
        intermediate_size=512,
        n_groups=4,
        num_heads=8,
        head_dim=64,
        state_size=64,
        conv_kernel=4,
    )


def _make_target_pool(num_reqs: int, mamba_size: int) -> HybridReqToTokenPool:
    with envs.SGLANG_MAMBA_SSM_DTYPE.override("bfloat16"):
        cache_params = Mamba2CacheParams(shape=_make_state_shape(), layers=[0, 1, 2])
    return HybridReqToTokenPool(
        size=num_reqs,
        mamba_size=mamba_size,
        mamba_spec_state_size=num_reqs,
        max_context_len=MAX_CONTEXT_LEN,
        device=get_device(),
        enable_memory_saver=False,
        cache_params=cache_params,
        mamba_layer_ids=[0, 1, 2],
        enable_mamba_extra_buffer=False,
    )


def _make_draft_pool(num_reqs: int, mamba_size: int) -> MambaPool:
    with envs.SGLANG_MAMBA_SSM_DTYPE.override("bfloat16"):
        cache_params = Mamba2CacheParams(shape=_make_state_shape(), layers=[0])
    return MambaPool(
        size=mamba_size,
        spec_state_size=num_reqs,
        cache_params=cache_params,
        mamba_layer_ids=[0],
        device=get_device(),
    )


class TestDraftMambaStatePlane(CustomTestCase):
    def _req_indices(self, req: Req) -> torch.Tensor:
        return torch.tensor([req.req_pool_idx], dtype=torch.int64, device=get_device())

    def test_lookup_reports_stale_until_marked_built(self):
        target_pool = _make_target_pool(num_reqs=8, mamba_size=16)
        plane = DraftMambaStatePlane(target_pool, _make_draft_pool(8, 16))

        req = _make_req(0)
        target_pool.alloc([req])
        self.assertEqual(int(target_pool.mamba_slot_generation[req.mamba_pool_idx]), 1)

        mamba_indices, current_mask = plane.lookup(self._req_indices(req))
        self.assertFalse(bool(current_mask[0]))

        plane.mark_built(mamba_indices)
        _, current_mask = plane.lookup(self._req_indices(req))
        self.assertTrue(bool(current_mask[0]))

        # Radix-style reuse: the req keeps its mamba slot across a req-slot
        # free/alloc cycle, so the built draft states stay current (no bump).
        target_pool.free(req)
        target_pool.alloc([req])
        self.assertEqual(int(target_pool.mamba_slot_generation[req.mamba_pool_idx]), 1)
        _, current_mask = plane.lookup(self._req_indices(req))
        self.assertTrue(bool(current_mask[0]))

    def test_slot_recycle_invalidates_built_states(self):
        # mamba_size == num_reqs so a full re-allocation is guaranteed to
        # recycle the freed slot with a fresh assignment.
        target_pool = _make_target_pool(num_reqs=4, mamba_size=4)
        plane = DraftMambaStatePlane(target_pool, _make_draft_pool(4, 4))

        req = _make_req(0)
        target_pool.alloc([req])
        slot = int(req.mamba_pool_idx)
        mamba_indices, _ = plane.lookup(self._req_indices(req))
        plane.mark_built(mamba_indices)

        target_pool.free_mamba_cache(req)
        target_pool.free(req)

        reqs = [_make_req(rid) for rid in range(1, 5)]
        target_pool.alloc(reqs)
        holder = next(r for r in reqs if int(r.mamba_pool_idx) == slot)
        self.assertEqual(int(target_pool.mamba_slot_generation[slot]), 2)
        _, current_mask = plane.lookup(self._req_indices(holder))
        self.assertFalse(bool(current_mask[0]))

    def test_size_mismatch_rejected(self):
        target_pool = _make_target_pool(num_reqs=4, mamba_size=8)
        with self.assertRaises(ValueError):
            DraftMambaStatePlane(target_pool, _make_draft_pool(4, 4))


if __name__ == "__main__":
    unittest.main()
