"""GPU unit test for srt/layers/moe/paged_experts/pager.py.

Exercises the page-in mechanism (decide_keep_warm -> page_in via transfer_kv_per_layer_mla -> remap)
on a synthetic K-slot layer, across multiple per-expert tensors. No server/model launch.
"""

import unittest

import torch

from sglang.srt.layers.moe.paged_experts.pager import ExpertPager
from sglang.srt.layers.moe.paged_experts.policy import LFUPolicy, LRUPolicy
from sglang.srt.layers.moe.paged_experts.store import (
    ExpertStore,
    PageableExpertStore,
    PinnedExpertStore,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-small")


class TestExpertPager(CustomTestCase):
    def test_decide_pagein_remap(self):
        E, K, dev = 8, 4, "cuda"
        layer = torch.nn.Module()
        # per-expert blocks must be 8-byte aligned (transfer_kv requirement); these are.
        layer.w13_weight = torch.nn.Parameter(
            torch.zeros(K, 2, 4, device=dev), requires_grad=False
        )
        layer.w2_weight = torch.nn.Parameter(
            torch.zeros(K, 6, device=dev), requires_grad=False
        )

        store = ExpertPager(layer, E, K, dev)
        for name in store.host:
            for e in range(E):
                store.host[name][e].fill_(float(e + 1))  # expert e's data == e+1

        ar = torch.arange(K, dtype=torch.int64, device=dev)
        store.page_in(ar, ar)  # seed slots 0..K-1 with experts 0..K-1
        torch.cuda.synchronize()

        # decode step routing to expert 1 (resident hit) + 5, 6 (misses)
        src, dst = store.decide_keep_warm(
            torch.tensor([[1, 5, 6]], dtype=torch.int64, device=dev)
        )
        store.page_in(src, dst)
        torch.cuda.synchronize()

        self.assertEqual(src.tolist(), [5, 6])  # misses
        self.assertEqual(dst.tolist(), [0, 2])  # LRU non-needed victims

        # every paged tensor's slots hold the right experts
        want = {
            0: 6.0,
            1: 2.0,
            2: 7.0,
            3: 4.0,
        }  # slot0=exp5, slot1=exp1, slot2=exp6, slot3=exp3
        for name in store.gpu:
            for slot, val in want.items():
                self.assertTrue((store.gpu[name].data[slot] == val).all().item())

        remap = store.logical_to_gpu_index_cuda[
            torch.tensor([1, 5, 6], device=dev)
        ].tolist()
        self.assertEqual(remap, [1, 0, 2])

    def test_pageable_store_page_in(self):
        # pin_host=False (--paged-experts-pageable-store): page_in uses a plain indexed copy instead of
        # transfer_kv (which would read stale data from non-page-locked memory). It must land the same
        # rows in the same slots as the pinned path. For a small-RAM host that can't pin the full store.
        E, K, dev = 8, 4, "cuda"
        layer = torch.nn.Module()
        layer.w13_weight = torch.nn.Parameter(
            torch.zeros(K, 2, 4, device=dev), requires_grad=False
        )
        layer.w2_weight = torch.nn.Parameter(
            torch.zeros(K, 6, device=dev), requires_grad=False
        )

        store = ExpertPager(layer, E, K, dev, pin_host=False)
        self.assertFalse(store.pin_host)
        for name in store.host:
            self.assertFalse(store.host[name].is_pinned())  # pageable, not page-locked
            for e in range(E):
                store.host[name][e].fill_(float(e + 1))

        # page experts [5, 2] into slots [0, 3] via the copy path
        store.page_in(
            torch.tensor([5, 2], dtype=torch.int64, device=dev),
            torch.tensor([0, 3], dtype=torch.int64, device=dev),
        )
        torch.cuda.synchronize()
        for name in store.gpu:
            self.assertTrue((store.gpu[name].data[0] == 6.0).all().item())  # expert 5
            self.assertTrue((store.gpu[name].data[3] == 3.0).all().item())  # expert 2

    def test_distinct_active_and_set_residency(self):
        # distinct_active + set_residency are the wave-path primitives: when distinct active experts
        # > K, forward.py serves them in waves and calls set_residency(last_group) so the NEXT
        # keep-warm step's residency state matches what is physically in the slots. Missing this
        # hand-off is what let the >K case fall through to masked-to-zero output.
        E, K, dev = 8, 4, "cuda"
        layer = torch.nn.Module()
        layer.w13_weight = torch.nn.Parameter(
            torch.zeros(K, 2, 4, device=dev), requires_grad=False
        )
        store = ExpertPager(layer, E, K, dev)

        # distinct_active: dedup, drop the -1 padding the router emits, sorted host list
        topk = torch.tensor([[5, 1, 5], [7, 1, -1]], dtype=torch.int64, device=dev)
        self.assertEqual(store.distinct_active(topk), [1, 5, 7])

        # set_residency(group) makes slot i hold group[i] and rebuilds both maps (host + cuda mirror)
        store.set_residency([6, 2, 7])
        self.assertEqual(store.slot_expert, [6, 2, 7, -1])  # padded to K with -1
        l2g = store.logical_to_gpu_index
        self.assertEqual([int(l2g[e]) for e in (6, 2, 7)], [0, 1, 2])
        self.assertTrue((l2g == store.logical_to_gpu_index_cuda.cpu()).all().item())

        # a following keep-warm step sees expert 2 as a resident hit (no page-in) and pages the miss
        src, dst = store.decide_keep_warm(
            torch.tensor([[2, 0]], dtype=torch.int64, device=dev)
        )
        self.assertEqual(src.tolist(), [0])  # only the miss is paged
        self.assertEqual(
            int(store.logical_to_gpu_index[2]), 1
        )  # resident hit kept its slot

    def test_store_seam(self):
        # The pager delegates host backing + page-in to an ExpertStore (store.py); --paged-experts-store
        # selects the transport (pinned transfer_kv vs pageable copy). The decision/residency stays on the
        # pager. This locks the seam: the right store subclass is composed, and the pager re-exposes it.
        E, K, dev = 8, 4, "cuda"
        layer = torch.nn.Module()
        layer.w13_weight = torch.nn.Parameter(
            torch.zeros(K, 2, 4, device=dev), requires_grad=False
        )

        pinned = ExpertPager(layer, E, K, dev)  # pin_host=True default
        self.assertIsInstance(pinned.store, PinnedExpertStore)
        self.assertTrue(pinned.pin_host and pinned.store.pinned)

        pageable = ExpertPager(layer, E, K, dev, pin_host=False)
        self.assertIsInstance(pageable.store, PageableExpertStore)
        self.assertFalse(pageable.pin_host or pageable.store.pinned)

        # ExpertStore is abstract — page_in is the contract subclasses must implement.
        with self.assertRaises(TypeError):
            ExpertStore(layer, E, K, dev)

    def test_eviction_policy_seam(self):
        # --paged-experts-eviction selects the residency policy the pager composes (default lru).
        E, K, dev = 8, 4, "cuda"
        layer = torch.nn.Module()
        layer.w13_weight = torch.nn.Parameter(
            torch.zeros(K, 2, 4, device=dev), requires_grad=False
        )
        self.assertIsInstance(ExpertPager(layer, E, K, dev).policy, LRUPolicy)
        self.assertIsInstance(
            ExpertPager(layer, E, K, dev, eviction="lfu").policy, LFUPolicy
        )


if __name__ == "__main__":
    unittest.main()
