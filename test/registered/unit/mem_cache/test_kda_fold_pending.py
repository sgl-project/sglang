# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
"""KDA fold mode (SGLANG_OPT_KDA_SPEC_FOLD_VERIFY): per-slot pending accept lengths across the slot lifecycle.

In fold mode `temporal[slot]` lags by `pending[slot]` ring records between forwards. A slot that is handed out again,
or that receives a copied checkpoint, gets a fresh `temporal` from its writer, so its pending length must be 0;
otherwise the next fold verify would replay the previous owner's ring records into the new request's state. The
allocator runs on the schedule stream while the previous owner's last (overlap) post-verify may still rewrite the
entry on the forward stream, so allocation only queues the slot and the forward-stream drain zeroes it.

What these tests catch (each fails on the named bug):
- the pending tensor is allocated without the env, or for a non-KDA / non-spec pool (wasted fold contract);
- the HybridReqToTokenPool does not register the allocator hook, or the drain does not zero the queued slots, or a
  late write by the previous owner (between allocation and drain) survives (stale pending on slot reuse, including
  grouped pre-allocation);
- copy_from leaves the destination's pending length (stale replay on a copied checkpoint) or clobbers the source's.
"""

import os
from unittest.mock import patch

import pytest
import torch

from sglang.srt.configs.mamba_utils import (
    KimiLinearCacheParams,
    KimiLinearStateShape,
    Mamba2StateDType,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

ENV = "SGLANG_OPT_KDA_SPEC_FOLD_VERIFY"
DTYPE = Mamba2StateDType(conv=torch.bfloat16, temporal=torch.float32)
LAYERS = [0, 1]


def _kda_params():
    shape = KimiLinearStateShape.create(
        tp_world_size=1, num_heads=4, head_dim=8, num_k_heads=4, head_k_dim=8
    )
    return KimiLinearCacheParams(shape=shape, dtype=DTYPE, layers=LAYERS)


def _mamba_pool(*, spec=True, size=5):
    from sglang.srt.mem_cache.memory_pool import MambaPool

    return MambaPool(
        size=size,
        spec_state_size=size,
        cache_params=_kda_params(),
        mamba_layer_ids=LAYERS,
        device="cpu",
        speculative_num_draft_tokens=4,
        speculative_eagle_topk=1,
        linear_replayssm_cache_len=8,
        enable_linear_replayssm_spec=spec,
    )


def _req_pool(size=6):
    from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool

    return HybridReqToTokenPool(
        size=4,
        mamba_size=size,
        mamba_spec_state_size=4,
        max_context_len=32,
        device="cpu",
        enable_memory_saver=False,
        cache_params=_kda_params(),
        mamba_layer_ids=LAYERS,
        enable_mamba_extra_buffer=True,
        speculative_num_draft_tokens=4,
        speculative_eagle_topk=1,
        linear_replayssm_cache_len=8,
        enable_linear_replayssm_spec=True,
    )


class TestKdaFoldPending(CustomTestCase):
    def test_allocated_only_for_kda_spec_with_env(self):
        with patch.dict(os.environ, {ENV: "0"}):
            self.assertIsNone(_mamba_pool().kda_fold_pending)
        with patch.dict(os.environ, {ENV: "1"}):
            self.assertIsNone(_mamba_pool(spec=False).kda_fold_pending)
            pool = _mamba_pool(size=5)
            pend = pool.kda_fold_pending
            self.assertIsNotNone(pend)
            self.assertEqual(pend.dtype, torch.int32)
            # one entry per mamba slot, slot 0 (dummy) included
            self.assertEqual(pend.numel(), 6)
            self.assertEqual(pend.numel(), pool.mamba_cache.temporal.shape[1])
            self.assertFalse(pool.kda_fold_active)

    def test_slot_allocation_zeroes_pending_at_drain(self):
        with patch.dict(os.environ, {ENV: "1"}):
            rp = _req_pool(size=6)
        pool = rp.mamba_pool
        pend = pool.kda_fold_pending
        alloc = rp.mamba_allocator
        a = alloc.alloc(3)
        pool.kda_fold_drain_resets()
        pend[a] = 7 | (1 << 8)  # the owners' pending windows
        alloc.free(a)
        pend[0] = 5  # the dummy slot is never handed out
        # Reallocate every free slot, then the previous owners' late post-verify
        # rewrites their entries before the next forward's drain.
        got = []
        while alloc.available_size() > 0:
            got.append(alloc.alloc(1))
        got = torch.cat(got)
        self.assertTrue(set(a.tolist()) <= set(got.tolist()))
        pend[a] = 4 | (1 << 8)
        pool.kda_fold_drain_resets()
        self.assertEqual(int(pend[got].abs().sum()), 0)
        self.assertEqual(int(pend[0]), 5)
        self.assertEqual(pool.kda_fold_reset_queue, [])
        # Grouped pre-allocation (match_prefix) queues at hand-out as well.
        alloc.free(got)
        pend[got] = 3
        alloc.alloc_group_begin(2)
        s1 = alloc.alloc(1)
        alloc.alloc_group_end()
        pool.kda_fold_drain_resets()
        self.assertEqual(int(pend[s1].abs().sum()), 0)

    def test_copy_from_zeroes_destination_only(self):
        with patch.dict(os.environ, {ENV: "1"}):
            pool = _mamba_pool(size=5)
        pend = pool.kda_fold_pending
        pend[:] = torch.tensor([0, 256, 4, 6, 2, 1], dtype=torch.int32)
        # committed source (0 records, ring half 1)
        pool.copy_from(torch.tensor([1]), torch.tensor([3]))
        self.assertEqual(pend.tolist(), [0, 256, 4, 0, 2, 1])


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
