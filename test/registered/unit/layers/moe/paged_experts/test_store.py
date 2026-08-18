"""GPU unit test for srt/layers/moe/paged_experts/store.py — the WindowedExpertStore fallback.

Covers the pinned-window store (W hot experts page-locked + E-W cold experts pageable) used for stores
that exceed the host's page-lock ceiling: the make_expert_store dispatch, the two-tier layout + pinning,
the membership maps, the fill_tensor/row round-trip, and the mixed page-in (a hot expert via transfer_kv
from the window + a cold expert via the pageable copy must land the same rows in the same slots).
"""

import unittest

import torch

from sglang.srt.layers.moe.paged_experts.store import (
    PageableExpertStore,
    PinnedExpertStore,
    WindowedExpertStore,
    make_expert_store,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-small")


def _layer(K, dev):
    layer = torch.nn.Module()
    # per-expert blocks must be 8-byte aligned (the pinned window feeds transfer_kv); these are.
    layer.w13_weight = torch.nn.Parameter(
        torch.zeros(K, 2, 4, device=dev), requires_grad=False
    )
    layer.w2_weight = torch.nn.Parameter(
        torch.zeros(K, 6, device=dev), requires_grad=False
    )
    return layer


class TestWindowedExpertStore(CustomTestCase):
    def test_make_expert_store_dispatch(self):
        E, K, W, dev = 8, 4, 5, "cuda"
        layer = _layer(K, dev)
        # 0 < W < E and pin_host -> windowed
        self.assertIsInstance(
            make_expert_store(layer, E, K, dev, pin_host=True, window_W=W),
            WindowedExpertStore,
        )
        # W >= E (full pin) or W == 0 (off) -> the plain pinned store
        self.assertIsInstance(
            make_expert_store(layer, E, K, dev, pin_host=True, window_W=E),
            PinnedExpertStore,
        )
        self.assertIsInstance(
            make_expert_store(layer, E, K, dev, pin_host=True, window_W=0),
            PinnedExpertStore,
        )
        # the window is meaningless for a pageable store -> ignored
        self.assertIsInstance(
            make_expert_store(layer, E, K, dev, pin_host=False, window_W=W),
            PageableExpertStore,
        )

    def test_two_tier_layout_and_pinning(self):
        E, K, W, dev = 8, 4, 5, "cuda"
        store = WindowedExpertStore(_layer(K, dev), E, K, dev, window_W=W)
        self.assertEqual((store.W, store.E, store.K), (W, E, K))
        for name in store.gpu:
            self.assertEqual(store.host_hot[name].shape[0], W)
            self.assertEqual(store.host_cold[name].shape[0], E - W)
            self.assertTrue(store.host_hot[name].is_pinned())  # hot window page-locked
            self.assertFalse(store.host_cold[name].is_pinned())  # cold tail pageable
        self.assertTrue(store.pinned)  # captured_ok surface: the window is pinned

    def test_membership_maps(self):
        E, K, W, dev = 8, 4, 5, "cuda"
        store = WindowedExpertStore(_layer(K, dev), E, K, dev, window_W=W)
        for e in range(E):
            if e < W:
                self.assertTrue(store.is_hot(e))
                self.assertEqual(int(store.hot_pos[e]), e)
                self.assertEqual(int(store.cold_pos[e]), -1)
            else:
                self.assertFalse(store.is_hot(e))
                self.assertEqual(int(store.cold_pos[e]), e - W)
                self.assertEqual(int(store.hot_pos[e]), -1)

    def test_fill_tensor_row_roundtrip(self):
        E, K, W, dev = 8, 4, 5, "cuda"
        store = WindowedExpertStore(_layer(K, dev), E, K, dev, window_W=W)
        for name in store.gpu:
            full = torch.randn((E, *store.gpu[name].shape[1:]))
            store.fill_tensor(name, full)
            for e in range(E):
                self.assertTrue(torch.equal(store.row(name, e).cpu(), full[e]))

    def test_mixed_page_in(self):
        # The crux: page in a HOT expert (via transfer_kv from the pinned window) and a COLD expert (via
        # the pageable indexed copy from the tail) in one plan; both must land the right rows in the right
        # slots, identically to the full-pin path. W=5 -> experts 0..4 hot, 5..7 cold.
        E, K, W, dev = 8, 4, 5, "cuda"
        store = WindowedExpertStore(_layer(K, dev), E, K, dev, window_W=W)
        for name in store.gpu:
            full = torch.stack(
                [torch.full(store.gpu[name].shape[1:], float(e + 1)) for e in range(E)]
            )
            store.fill_tensor(name, full)  # expert e's data == e+1, split across tiers

        # route expert 1 (hot) -> slot 0, expert 6 (cold) -> slot 3
        store.page_in(
            torch.tensor([1, 6], dtype=torch.int64, device=dev),
            torch.tensor([0, 3], dtype=torch.int64, device=dev),
        )
        torch.cuda.synchronize()
        for name in store.gpu:
            self.assertTrue(
                (store.gpu[name].data[0] == 2.0).all().item()
            )  # expert 1 (hot)
            self.assertTrue(
                (store.gpu[name].data[3] == 7.0).all().item()
            )  # expert 6 (cold)

        # all-hot and all-cold plans also work (exercise each tier alone)
        store.page_in(
            torch.tensor([2], dtype=torch.int64, device=dev),
            torch.tensor([1], dtype=torch.int64, device=dev),
        )  # hot only
        store.page_in(
            torch.tensor([7], dtype=torch.int64, device=dev),
            torch.tensor([2], dtype=torch.int64, device=dev),
        )  # cold only
        torch.cuda.synchronize()
        for name in store.gpu:
            self.assertTrue(
                (store.gpu[name].data[1] == 3.0).all().item()
            )  # expert 2 (hot)
            self.assertTrue(
                (store.gpu[name].data[2] == 8.0).all().item()
            )  # expert 7 (cold)

    def test_set_window_membership_repins_hottest(self):
        # P3 freq-ranked window: re-pin an arbitrary hot set; the data must follow each expert (row(e) still
        # returns expert e's bytes) and the maps must reflect the new hot/cold split.
        E, K, W, dev = 8, 4, 5, "cuda"
        store = WindowedExpertStore(_layer(K, dev), E, K, dev, window_W=W)
        for name in store.gpu:
            full = torch.stack(
                [torch.full(store.gpu[name].shape[1:], float(e + 1)) for e in range(E)]
            )
            store.fill_tensor(name, full)
        # pick a hot set that is NOT [0, W): hottest = 7,6,5,4,3 (so 0,1,2 become cold)
        new_hot = [7, 6, 5, 4, 3]
        store.set_window_membership(new_hot)
        # Membership is the contract; the Δ-set re-pin keeps surviving experts in their rows (only the
        # promoted/demoted pairs move), so row ORDER within the tier is free — just distinct and valid.
        hot_rows = [int(store.hot_pos[e]) for e in new_hot]
        for e in new_hot:
            self.assertTrue(store.is_hot(e))
        self.assertEqual(sorted(hot_rows), list(range(W)))
        cold_rows = [int(store.cold_pos[e]) for e in (0, 1, 2)]
        for e in (0, 1, 2):
            self.assertFalse(store.is_hot(e))
        self.assertEqual(sorted(cold_rows), list(range(E - W)))
        # data integrity: every expert's bytes are intact and reachable through the new maps
        for name in store.gpu:
            for e in range(E):
                self.assertTrue((store.row(name, e) == float(e + 1)).all().item())

    def test_disk_cold_backing(self):
        # P4: the cold tail is mmap'd to a file (not pinned RAM); the hot window stays pinned, and the
        # fill/row round-trip works the same (writes/reads go through the disk-backed mapping).
        E, K, W, dev = 8, 4, 5, "cuda"
        store = WindowedExpertStore(
            _layer(K, dev), E, K, dev, window_W=W, cold_backing="disk"
        )
        for name in store.gpu:
            self.assertTrue(
                store.host_hot[name].is_pinned()
            )  # hot window still page-locked
            self.assertFalse(
                store.host_cold[name].is_pinned()
            )  # cold tail is the disk mmap
            full = torch.randn((E, *store.gpu[name].shape[1:]))
            store.fill_tensor(name, full)
            for e in range(E):
                self.assertTrue(torch.equal(store.row(name, e).cpu(), full[e]))
        # the disk store also serves a real page-in (hot + cold), like the RAM path
        for name in store.gpu:
            full = torch.stack(
                [torch.full(store.gpu[name].shape[1:], float(e + 1)) for e in range(E)]
            )
            store.fill_tensor(name, full)
        store.page_in(
            torch.tensor([1, 6], dtype=torch.int64, device=dev),
            torch.tensor([0, 3], dtype=torch.int64, device=dev),
        )
        torch.cuda.synchronize()
        for name in store.gpu:
            self.assertTrue((store.gpu[name].data[0] == 2.0).all().item())  # hot
            self.assertTrue(
                (store.gpu[name].data[3] == 7.0).all().item()
            )  # cold (from disk mmap)


if __name__ == "__main__":
    unittest.main()
