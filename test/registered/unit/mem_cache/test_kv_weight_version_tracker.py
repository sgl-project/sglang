import unittest
from typing import List

import torch

from sglang.srt.mem_cache.kv_weight_version_tracker import KvWeightVersionTracker
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

    def effective_kv_committed_len(self) -> int:
        return self.kv_committed_len


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


class TestKvWeightVersionTracker(CustomTestCase):
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

    def test_version_ids_are_dense_and_bidirectionally_consistent(self):
        """Every new version gets the next id, already seen ones are reused, and both maps agree."""
        table = _table()
        for index, version in enumerate(["v0", "v1", "v2", "v1", "v0"]):
            table.record(slot_indices=_slots(index), version=version)

        self.assertEqual(table._version_str_by_id, ["v0", "v1", "v2"])
        self.assertEqual(table._version_id_by_str, {"v0": 0, "v1": 1, "v2": 2})
        self.assertEqual(table._slot_version_ids[:5].tolist(), [0, 1, 2, 1, 0])


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

    def test_the_lookup_is_clamped_to_the_committed_kv_length(self):
        """A prompt whose KV is only partially committed reports only the committed tokens."""
        table = _table(slots_of_req=[4, 5])
        table.record(slot_indices=_slots(4, 5), version="v0")
        req = _ReqStub(num_prompt_tokens=5, kv_committed_len=2)

        table.fill_req_prefill_weight_versions(req)

        self.assertEqual(
            req.prefill_weight_versions,
            [WeightVersionSpan(version="v0", start=0, end=2)],
        )

    def test_nothing_committed_yields_no_spans(self):
        """A request whose KV is not committed at all reports an empty span list, not an error."""
        table = _table(slots_of_req=[4, 5])
        req = _ReqStub(num_prompt_tokens=2, kv_committed_len=0)

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
