import importlib.util
import pathlib
import sys
import unittest
from types import SimpleNamespace

path = (
    pathlib.Path(__file__).resolve().parents[4]
    / "python/sglang/srt/speculative/dflash_independent_cache.py"
)
spec = importlib.util.spec_from_file_location(
    "dflash_independent_cache_under_test", path
)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
DraftSlotLayout = module.DraftSlotLayout


from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class LayoutTests(CustomTestCase):
    def test_page_alignment_and_no_cross_request_alias(self):
        for context in (2047, 2048, 2049, 8192, 1048576):
            layout = DraftSlotLayout(4, context, 64, 12, 2048)
            self.assertEqual(layout.window_stride, 2176)
            for slot in range(1, 5):
                for pos in (0, 1, 63, 64, context - 1, layout.stride - 1):
                    full = layout.full_loc(slot, pos)
                    window = layout.window_loc(slot, pos)
                    self.assertEqual(full % 64, pos % 64)
                    self.assertEqual(window % 64, pos % 64)
                    self.assertLess(full, layout.full_capacity + 64)
                    self.assertLess(window, layout.window_capacity + 64)
                self.assertEqual(layout.full_loc(slot, 0) % 64, 0)
            for slot in range(1, 4):
                self.assertLess(
                    layout.full_loc(slot, layout.stride - 1),
                    layout.full_loc(slot + 1, 0),
                )
                self.assertLess(
                    layout.window_loc(
                        slot, min(layout.window_stride, layout.stride) - 1
                    ),
                    layout.window_loc(slot + 1, 0),
                )

    def test_speculative_window_survives_wrap_and_rejection(self):
        layout = DraftSlotLayout(2, 1048576, 64, 12, 2048)
        for committed in (2047, 2048, 2049, 2175, 2176, 2177, 65535, 1048540):
            positions = range(max(0, committed - 2047), committed + 24)
            locs = [layout.window_loc(1, p) for p in positions]
            self.assertEqual(len(locs), len(set(locs)))
            # Retaining fewer accepted proposals must leave the live prefix intact.
            for accepted in (0, 1, 6, 12):
                retained = range(
                    max(0, committed + accepted - 2047), committed + accepted
                )
                self.assertTrue(
                    set(layout.window_loc(1, p) for p in retained).issubset(set(locs))
                )

    def test_reject_invalid_dimensions_and_locations(self):
        for args in ((0, 10, 64, 12), (1, 0, 64, 12), (1, 10, 0, 12), (1, 10, 64, 0)):
            with self.assertRaises(ValueError):
                DraftSlotLayout(*args)
        layout = DraftSlotLayout(1, 2048, 64, 12, 2048)
        for slot, pos in ((0, 0), (2, 0), (1, -1), (1, layout.stride)):
            with self.assertRaises(ValueError):
                layout.full_loc(slot, pos)

    def test_full_only_layout_is_not_window_truncated(self):
        layout = DraftSlotLayout(2, 8192, 64, 12)
        self.assertEqual(layout.window_stride, layout.stride)
        self.assertNotEqual(layout.full_loc(1, 0), layout.full_loc(1, 8191))

    def test_fixed_budget_counts_all_cohort_slots(self):
        local = DraftSlotLayout(1, 1048576, 64, 12, 2048)
        pair = DraftSlotLayout(2, 1048576, 64, 12, 2048)
        # TP2 halves a row but doubles owner slots: reserved KV is not halved.
        local_kv = (local.full_capacity + 64) * 8192 + (
            local.window_capacity + 64
        ) * 4 * 8192
        pair_kv = (pair.full_capacity + 64) * 4096 + (
            pair.window_capacity + 64
        ) * 4 * 4096
        self.assertLess(abs(local_kv - pair_kv), 3 * 1024 * 1024)
        self.assertGreater(
            pair.reservation_bytes(row_bytes=4096, full_layers=1, window_layers=4),
            pair_kv,
        )


class StorageTests(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        import torch

        cls.torch = torch

    def make_cache(self):
        torch = self.torch
        layout = DraftSlotLayout(2, 8192, 64, 12, 2048)
        requests = SimpleNamespace(
            req_to_token=torch.zeros((3, layout.stride), dtype=torch.int32)
        )
        target = SimpleNamespace(req_generation=torch.tensor([0, 1, 1]))
        allocator = SimpleNamespace(
            full_to_swa_index_mapping=torch.zeros(
                layout.full_capacity + 64, dtype=torch.int64
            )
        )
        return module.IndependentDraftCache(
            layout=layout,
            target_requests=target,
            requests=requests,
            allocator=allocator,
            device="cpu",
        )

    def test_real_tables_mapping_and_prefill_tail_no_duplicate_writes(self):
        torch = self.torch
        cache = self.make_cache()
        indices = torch.tensor([1, 2], dtype=torch.int32)
        lengths = torch.tensor([5000, 3001], dtype=torch.int32)
        positions = torch.cat([torch.arange(5000), torch.arange(100, 3101)])
        full, window, keep = cache.locations(indices, positions, lengths)
        self.assertTrue(
            torch.equal(window, cache.allocator.full_to_swa_index_mapping[full])
        )
        self.assertEqual(keep.sum().item(), 2 * 2176)
        self.assertEqual(window[keep].unique().numel(), keep.sum().item())
        self.assertEqual(full.unique().numel(), positions.numel())
        self.assertTrue(
            torch.equal(cache.requests.req_to_token[1, :5000].long(), full[:5000])
        )
        self.assertEqual(cache.requests.req_to_token[0].count_nonzero().item(), 0)

    def test_finish_retract_generation_and_reservation_bounds(self):
        torch = self.torch
        cache = self.make_cache()
        kwargs = dict(
            req_pool_indices_cpu=torch.tensor([1]),
            target_seq_lens_cpu=torch.tensor([4096]),
        )
        cache.alloc_to_lengths(**kwargs)
        cache.target_requests.req_generation[1] += 1
        with self.assertRaises(RuntimeError):
            cache.alloc_to_lengths(**kwargs)
        cache.free(1)
        cache.alloc_to_lengths(**kwargs)
        self.assertEqual(cache.generations[1], 2)
        with self.assertRaises(RuntimeError):
            cache.alloc_to_lengths(
                req_pool_indices_cpu=torch.tensor([1]),
                target_seq_lens_cpu=torch.tensor([cache.layout.stride + 1]),
            )
        cache.reset_allocator()
        self.assertEqual(cache.allocated_lens, [0, 0, 0])

    def test_owner_lifecycle_cannot_release_a_peer_slot(self):
        torch = self.torch
        layout = DraftSlotLayout(4, 8192, 64, 12, 2048)
        target = SimpleNamespace(req_generation=torch.tensor([0, 7, 9]))
        requests = SimpleNamespace(
            req_to_token=torch.zeros((5, layout.stride), dtype=torch.int32)
        )
        allocator = SimpleNamespace(
            full_to_swa_index_mapping=torch.zeros(
                layout.full_capacity + 64, dtype=torch.int64
            )
        )
        cache = module.IndependentDraftCache(
            layout=layout,
            target_requests=target,
            requests=requests,
            allocator=allocator,
            device="cpu",
            local_slot_offset=2,
        )
        cache.allocated_lens[1] = 200
        cache.alloc_to_lengths(
            req_pool_indices_cpu=torch.tensor([1]),
            target_seq_lens_cpu=torch.tensor([500]),
        )
        self.assertEqual(cache.allocated_lens[3], 500)
        self.assertEqual(cache.generations[3], 7)
        cache.free(1)
        self.assertEqual(cache.allocated_lens[3], 0)
        self.assertEqual(cache.allocated_lens[1], 200)


if __name__ == "__main__":
    unittest.main()
