"""An exhausted unified mamba pool skips the unfinished-request checkpoint
instead of asserting; with a free slot the request's checkpoint slot is
donated and the free slot replaces it, as before."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.mem_cache.base_prefix_cache import InsertParams
from sglang.srt.mem_cache.unified_cache.components.mamba_component import (
    MambaComponent,
)

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def _component(*, slot):
    comp = object.__new__(MambaComponent)
    allocator = MagicMock()
    allocator.alloc.return_value = slot
    allocator.available_size.return_value = 0
    allocator.size = 28
    pool = SimpleNamespace(
        mamba_allocator=allocator,
        mamba_ckpt_pool=None,
        donate_mamba_ping_pong_slot=MagicMock(return_value=torch.tensor([3])),
    )
    comp.cache = SimpleNamespace(
        req_to_token_pool=pool,
        enable_mamba_extra_buffer=True,
        evict_for_alloc=MagicMock(),
    )
    comp.tree_core = SimpleNamespace(mamba_evictable_size=lambda: 0)
    return comp, allocator


def _req():
    return SimpleNamespace(
        rid="r1",
        kv=SimpleNamespace(mamba_last_track_seqlen=4096, mamba_pool_idx=None),
    )


class TestMambaPoolExhaustion(CustomTestCase):
    def test_unfinished_checkpoint_skipped_when_pool_exhausted(self):
        comp, allocator = _component(slot=None)
        params = InsertParams(prev_prefix_len=0, chunked=True, priority=0)
        cache_len = comp.prepare_for_caching_req(
            req=_req(), insert_params=params, token_ids_len=4096, is_finished=False
        )
        self.assertEqual(cache_len, 0)
        self.assertIsNone(params.mamba_value)
        comp.cache.evict_for_alloc.assert_called_once()
        self.assertEqual(allocator.alloc.call_count, 2)

    def test_unfinished_checkpoint_donated_when_replacement_slot_available(self):
        comp, allocator = _component(slot=torch.tensor([7]))
        params = InsertParams(prev_prefix_len=0, chunked=True, priority=0)
        cache_len = comp.prepare_for_caching_req(
            req=_req(), insert_params=params, token_ids_len=4096, is_finished=False
        )
        self.assertEqual(cache_len, 4096)
        self.assertTrue(torch.equal(params.mamba_value, torch.tensor([3])))
        comp.cache.evict_for_alloc.assert_not_called()


if __name__ == "__main__":
    unittest.main()
