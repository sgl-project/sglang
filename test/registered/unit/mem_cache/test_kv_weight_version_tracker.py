import unittest
from types import SimpleNamespace
from typing import List

import torch

from sglang.srt.mem_cache.kv_weight_version_tracker import (
    KvWeightVersionRecord,
    KvWeightVersionTracker,
    _StringInterner,
)
from sglang.srt.utils.weight_versions import WeightVersionSpan
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=4, suite="base-a-test-cpu")


class _ReqStub:
    def __init__(self, num_prompt_tokens: int, kv_committed_len: int):
        self.origin_input_ids = [0] * num_prompt_tokens
        self.kv_committed_len = kv_committed_len
        self.req_pool_idx = 1
        self.prefill_weight_versions = None


class _ReqToTokenPoolStub:
    def __init__(self, slots_of_req: List[int]):
        self.req_to_token = torch.zeros((2, 32), dtype=torch.int32)
        self.req_to_token[1, : len(slots_of_req)] = torch.tensor(
            slots_of_req, dtype=torch.int32
        )


def _table(slots_of_req: List[int] = ()) -> KvWeightVersionTracker:
    return KvWeightVersionTracker(
        num_slots=16,
        device="cpu",
        req_to_token_pool=_ReqToTokenPoolStub(list(slots_of_req)),
    )


def _slots(*indices: int) -> torch.Tensor:
    return torch.tensor(indices, dtype=torch.int64)


class TestKvWeightVersionRecord(CustomTestCase):
    def test_capture_without_a_version_fails(self) -> None:
        """A forward cannot publish KV provenance without a weight version."""
        with self.assertRaises(AssertionError):
            KvWeightVersionRecord.capture(slot_indices=_slots(1), version=None)

    def test_capture_survives_source_reuse_and_tensor_mapping(self) -> None:
        """Reusing forward buffers cannot change a pending record's slots."""
        slots = _slots(4, 5, 6)
        record = KvWeightVersionRecord.capture(slot_indices=slots, version="v0")
        slots.fill_(9)
        record.map_device_tensors(lambda tensor: tensor.to(dtype=torch.int32))
        table = _table()

        table.record(slot_indices=record.slot_indices, version=record.version)

        self.assertEqual(
            table._lookup_spans(_slots(4, 5, 6)),
            [WeightVersionSpan(version="v0", start=0, end=3)],
        )
        self.assertEqual(record.slot_indices.dtype, torch.int32)


class TestKvWeightVersionTracker(CustomTestCase):
    def test_pipeline_parallelism_is_rejected_only_when_tracking_is_enabled(
        self,
    ) -> None:
        """The optional tracker rejects PP before allocation without restricting disabled runs."""
        for enabled in (False, True):
            with self.subTest(enabled=enabled):
                kwargs = dict(
                    server_args=SimpleNamespace(
                        enable_prefill_weight_versions=enabled, pp_size=2
                    ),
                    model_config=SimpleNamespace(
                        is_encoder_decoder=False, is_generation=True
                    ),
                    allocator=None,
                    req_to_token_pool=None,
                )
                if enabled:
                    with self.assertRaisesRegex(AssertionError, "pipeline parallelism"):
                        KvWeightVersionTracker.maybe_create(**kwargs)
                else:
                    self.assertIsNone(KvWeightVersionTracker.maybe_create(**kwargs))

    def test_factory_includes_allocator_padding_and_uses_the_request_pool(self) -> None:
        """Factory sizing includes the allocator's padding page and preserves its mapping."""
        tracker = KvWeightVersionTracker.maybe_create(
            server_args=SimpleNamespace(enable_prefill_weight_versions=True, pp_size=1),
            model_config=SimpleNamespace(is_encoder_decoder=False, is_generation=True),
            allocator=SimpleNamespace(size_full=12, page_size=4, device="cpu"),
            req_to_token_pool=_ReqToTokenPoolStub([15]),
        )
        req = _ReqStub(num_prompt_tokens=1, kv_committed_len=1)

        tracker.record(slot_indices=_slots(15), version="v0")
        tracker.fill_req_prefill_weight_versions(req)

        self.assertEqual(
            req.prefill_weight_versions,
            [WeightVersionSpan(version="v0", start=0, end=1)],
        )

    def test_unified_swa_virtual_slots_above_the_static_quota_keep_their_versions(
        self,
    ) -> None:
        """Legal virtual slots above the static token quota remain addressable by the tracker."""
        from sglang.srt.mem_cache.multi_ended_allocator import (
            UnifiedSWATokenToKVPoolAllocator,
        )

        for page_size in (1, 4):
            with self.subTest(page_size=page_size):
                allocator = UnifiedSWATokenToKVPoolAllocator.__new__(
                    UnifiedSWATokenToKVPoolAllocator
                )
                allocator._size_full = 8
                allocator.page_size = page_size
                allocator.device = "cpu"
                allocator.full_attn_allocator = SimpleNamespace(
                    size_full=32, page_size=page_size, device="cpu"
                )
                tracker = KvWeightVersionTracker.maybe_create(
                    server_args=SimpleNamespace(
                        enable_prefill_weight_versions=True, pp_size=1
                    ),
                    model_config=SimpleNamespace(
                        is_encoder_decoder=False, is_generation=True
                    ),
                    allocator=allocator,
                    req_to_token_pool=_ReqToTokenPoolStub([31]),
                )
                req = _ReqStub(num_prompt_tokens=1, kv_committed_len=1)

                tracker.record(slot_indices=_slots(31), version="v2")
                tracker.fill_req_prefill_weight_versions(req)

                self.assertEqual(
                    req.prefill_weight_versions,
                    [WeightVersionSpan(version="v2", start=0, end=1)],
                )

    def test_equal_neighbours_merge_while_a_returning_version_starts_a_new_span(self):
        """Run-length compression merges adjacent equal versions and never merges across a change."""
        table = _table()
        table.record(slot_indices=_slots(4, 5), version="v0")
        table.record(slot_indices=_slots(6), version="v1")
        table.record(slot_indices=_slots(7), version="v0")

        self.assertEqual(
            table._lookup_spans(_slots(4, 5, 6, 7)),
            [
                WeightVersionSpan(version="v0", start=0, end=2),
                WeightVersionSpan(version="v1", start=2, end=3),
                WeightVersionSpan(version="v0", start=3, end=4),
            ],
        )

    def test_a_single_slot_lookup_yields_one_unit_span(self):
        """A one-token prompt maps to exactly one [0, 1) span."""
        table = _table()
        table.record(slot_indices=_slots(7), version="v0")

        self.assertEqual(
            table._lookup_spans(_slots(7)),
            [WeightVersionSpan(version="v0", start=0, end=1)],
        )

    def test_an_empty_lookup_returns_no_spans(self):
        """Looking up an empty prompt yields an empty span list."""
        self.assertEqual(_table()._lookup_spans(_slots()), [])

    def test_unwritten_slots_are_reported_by_index_in_lookup_order(self):
        """The error names exactly the never-stamped slots, ordered by their place in the lookup."""
        table = _table()
        table.record(slot_indices=_slots(1, 3), version="v0")

        with self.assertRaisesRegex(ValueError, r": \[2, 0\]$"):
            table._lookup_spans(_slots(2, 3, 0, 1))

    def test_a_reused_slot_reports_the_version_that_last_wrote_it(self):
        """The table is keyed by slot, so a slot refilled under a new version forgets the old one."""
        table = _table()
        table.record(slot_indices=_slots(3), version="v0")
        table.record(slot_indices=_slots(3), version="v1")

        self.assertEqual(
            table._lookup_spans(_slots(3)),
            [WeightVersionSpan(version="v1", start=0, end=1)],
        )

    def test_int32_slot_indices_are_accepted(self):
        """out_cache_loc arrives as int32 and must index the table like int64 does."""
        table = _table()
        table.record(slot_indices=torch.tensor([2, 3], dtype=torch.int32), version="v0")

        self.assertEqual(
            table._lookup_spans(torch.tensor([2, 3], dtype=torch.int32)),
            [WeightVersionSpan(version="v0", start=0, end=2)],
        )

    def test_the_last_slot_is_addressable(self):
        """The table covers num_slots entries, so index num_slots - 1 is valid."""
        table = _table()
        table.record(slot_indices=_slots(15), version="v0")

        self.assertEqual(
            table._lookup_spans(_slots(15)),
            [WeightVersionSpan(version="v0", start=0, end=1)],
        )

    def test_version_ids_are_dense_and_reused(self):
        """Every new version gets the next id and already seen ones are reused."""
        table = _table()
        for index, version in enumerate(["v0", "v1", "v2", "v1", "v0"]):
            table.record(slot_indices=_slots(index), version=version)

        self.assertEqual(table._slot_version_ids[:5].tolist(), [0, 1, 2, 1, 0])


class TestStringInterner(CustomTestCase):
    def test_first_value_gets_id_zero(self):
        """A fresh interner starts numbering at zero."""
        self.assertEqual(_StringInterner().intern("v0"), 0)

    def test_new_values_get_consecutive_ids_in_first_seen_order(self):
        """Ids are dense and follow the order in which values were first interned."""
        interner = _StringInterner()

        self.assertEqual([interner.intern(v) for v in ["b", "a", "c"]], [0, 1, 2])

    def test_a_seen_value_reuses_its_id(self):
        """Interning a value again returns the id it already has instead of a new one."""
        interner = _StringInterner()
        first = interner.intern("v0")
        interner.intern("v1")

        self.assertEqual(interner.intern("v0"), first)
        self.assertEqual(interner.intern("v2"), 2)

    def test_lookup_returns_the_interned_value(self):
        """lookup is the inverse of intern for every id handed out."""
        interner = _StringInterner()
        values = ["v0", "v1", "v2"]
        ids = [interner.intern(v) for v in values]

        self.assertEqual([interner.lookup(i) for i in ids], values)

    def test_lookup_of_an_unknown_id_fails(self):
        """An id that was never handed out cannot be resolved."""
        interner = _StringInterner()
        interner.intern("v0")

        with self.assertRaises(IndexError):
            interner.lookup(1)

    def test_distinct_strings_never_share_an_id(self):
        """Values that differ only slightly are still distinct entries."""
        interner = _StringInterner()

        ids = {interner.intern(v) for v in ["1", "01", "1 ", "default", ""]}

        self.assertEqual(len(ids), 5)


class TestFillReqPrefillWeightVersions(CustomTestCase):
    def test_prompt_slots_resolve_to_the_versions_that_computed_them(self):
        """Only the prompt's own slots are read, even when more KV is committed than there are prompt tokens."""
        table = _table(slots_of_req=[4, 5, 6, 7, 8])
        table.record(slot_indices=_slots(4, 5), version="v0")
        table.record(slot_indices=_slots(6), version="v1")
        req = _ReqStub(num_prompt_tokens=3, kv_committed_len=5)

        table.fill_req_prefill_weight_versions(req)

        self.assertEqual(
            req.prefill_weight_versions,
            [
                WeightVersionSpan(version="v0", start=0, end=2),
                WeightVersionSpan(version="v1", start=2, end=3),
            ],
        )

    def test_a_partially_committed_prompt_fails_the_fill(self):
        """Filling before the whole prompt is committed is a scheduling bug and must not yield short spans."""
        table = _table(slots_of_req=[4, 5])
        table.record(slot_indices=_slots(4, 5), version="v0")
        req = _ReqStub(num_prompt_tokens=5, kv_committed_len=2)

        with self.assertRaisesRegex(
            AssertionError, "2 committed KV tokens for a 5-token prompt"
        ):
            table.fill_req_prefill_weight_versions(req)

    def test_an_empty_prompt_yields_no_spans(self):
        """A request without prompt tokens reports an empty span list, not an error."""
        table = _table(slots_of_req=[])
        req = _ReqStub(num_prompt_tokens=0, kv_committed_len=0)

        table.fill_req_prefill_weight_versions(req)

        self.assertEqual(req.prefill_weight_versions, [])

    def test_an_unstamped_prompt_slot_fails_the_fill(self):
        """A prompt slot the tracker never saw surfaces as an error on the request path too."""
        table = _table(slots_of_req=[4, 5, 6])
        table.record(slot_indices=_slots(4, 6), version="v0")
        req = _ReqStub(num_prompt_tokens=3, kv_committed_len=3)

        with self.assertRaisesRegex(ValueError, r"\[5\]"):
            table.fill_req_prefill_weight_versions(req)

    def test_a_second_fill_replaces_the_previous_spans(self):
        """Filling again after a re-prefill replaces the stale spans instead of appending."""
        table = _table(slots_of_req=[4, 5])
        table.record(slot_indices=_slots(4, 5), version="v0")
        req = _ReqStub(num_prompt_tokens=2, kv_committed_len=2)
        table.fill_req_prefill_weight_versions(req)
        table.record(slot_indices=_slots(4, 5), version="v1")

        table.fill_req_prefill_weight_versions(req)

        self.assertEqual(
            req.prefill_weight_versions,
            [WeightVersionSpan(version="v1", start=0, end=2)],
        )


if __name__ == "__main__":
    unittest.main()
