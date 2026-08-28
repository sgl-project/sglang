import unittest
from typing import List

import torch

from sglang.srt.mem_cache.kv_slot_weight_versions import KvSlotWeightVersions
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


def _table(slots_of_req: List[int] = ()) -> KvSlotWeightVersions:
    return KvSlotWeightVersions(
        num_slots=16,
        device="cpu",
        req_to_token_pool=_ReqToTokenPoolStub(list(slots_of_req)),
    )


def _slots(*indices: int) -> torch.Tensor:
    return torch.tensor(indices, dtype=torch.int64)


class TestKvSlotWeightVersions(CustomTestCase):
    def test_never_written_slots_fail_the_lookup(self):
        """Looking up a slot no forward ever stamped is a bug, not a version."""
        with self.assertRaisesRegex(ValueError, r"\[0, 1, 2\]"):
<<<<<<< ours
            _table()._lookup_spans(_slots(0, 1, 2))
=======
            _table()._lookup_spans(_slots(0, 1, 2))
>>>>>>> theirs

    def test_single_version_collapses_into_one_span(self):
        """Slots written by one version compress into a single span."""
        table = _table()
        table.record(slot_indices=_slots(3, 4, 5), version="v0")

        self.assertEqual(
            table._lookup_spans(_slots(3, 4, 5)),
            [WeightVersionSpan(version="v0", start=0, end=3)],
        )

    def test_rewriting_slots_under_a_new_version_splits_the_lookup(self):
        """Re-recording the tail of a sequence yields an old-version prefix and a new-version suffix."""
        table = _table()
        table.record(slot_indices=_slots(1, 2, 3, 4), version="v0")
        table.record(slot_indices=_slots(3, 4), version="v1")

        self.assertEqual(
            table._lookup_spans(_slots(1, 2, 3, 4)),
            [
                WeightVersionSpan(version="v0", start=0, end=2),
                WeightVersionSpan(version="v1", start=2, end=4),
            ],
        )

    def test_one_unwritten_slot_among_written_ones_fails_the_lookup(self):
        """A single never-stamped slot inside a prompt is reported by index."""
        table = _table()
        table.record(slot_indices=_slots(1, 3), version="v0")

        with self.assertRaisesRegex(ValueError, r"\[2\]"):
<<<<<<< ours
            table._lookup_spans(_slots(1, 2, 3))
=======
            table._lookup_spans(_slots(1, 2, 3))
>>>>>>> theirs

    def test_non_adjacent_slots_with_the_same_version_merge(self):
        """Compression follows lookup order, not slot order, so the same version merges."""
        table = _table()
        table.record(slot_indices=_slots(9, 2, 5), version="v0")

        self.assertEqual(
            table._lookup_spans(_slots(9, 2, 5)),
            [WeightVersionSpan(version="v0", start=0, end=3)],
        )

    def test_version_ids_are_interned_and_never_reassigned(self):
        """Re-recording an already seen version reuses its id instead of growing the table."""
        table = _table()
        table.record(slot_indices=_slots(0), version="v0")
        table.record(slot_indices=_slots(1), version="v1")
        table.record(slot_indices=_slots(2), version="v0")

        self.assertEqual(table._version_str_by_id, ["v0", "v1"])
        self.assertEqual(
            table._lookup_spans(_slots(0, 2, 1)),
            [
                WeightVersionSpan(version="v0", start=0, end=2),
                WeightVersionSpan(version="v1", start=2, end=3),
            ],
        )

    def test_empty_lookup_returns_no_spans(self):
        """Looking up an empty prompt yields an empty span list."""
        self.assertEqual(_table()._lookup_spans(_slots()), [])

<<<<<<< ours
    def test_fill_req_prefill_weight_versions_resolves_the_prompt_slots_onto_the_request(
=======
    def test_fill_req_prefill_weight_versions_resolves_the_prompt_slots_onto_the_request(
>>>>>>> theirs
        self,
    ):
        """The prompt's KV slots resolve to the versions that computed them."""
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

<<<<<<< ours
    def test_fill_req_prefill_weight_versions_is_clamped_to_the_committed_kv_length(
=======
    def test_fill_req_prefill_weight_versions_is_clamped_to_the_committed_kv_length(
>>>>>>> theirs
        self,
    ):
        """A prompt whose KV is only partially committed reports only the committed tokens."""
        table = _table(slots_of_req=[4, 5])
        table.record(slot_indices=_slots(4, 5), version="v0")
        req = _ReqStub(num_prompt_tokens=5, kv_committed_len=2)

        table.fill_req_prefill_weight_versions(req)

        self.assertEqual(
            req.prefill_weight_versions,
            [WeightVersionSpan(version="v0", start=0, end=2)],
        )


if __name__ == "__main__":
    unittest.main()
