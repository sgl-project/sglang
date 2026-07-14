import unittest
from unittest import mock

import torch

from sglang.srt.mem_cache.allocator.hisparse import _HiSparsePageOwnership
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestHiSparsePageOwnership(unittest.TestCase):
    def test_release_clears_all_owners_before_freeing_canonical_pages(self) -> None:
        """Release clears mapping and buffer owners before one full-page free."""
        mapping = torch.zeros(8, dtype=torch.int64)
        mapping[torch.tensor([1, 2, 3])] = torch.tensor([4, 5, 7])
        buffer_owner = torch.tensor([6, 7], dtype=torch.int64)
        child_allocator = mock.Mock(is_not_in_free_group=True)

        def verify_owner_clear(free_indices: torch.Tensor) -> None:
            self.assertEqual(mapping[torch.tensor([1, 2, 3])].tolist(), [0, 0, 0])
            self.assertEqual(buffer_owner.tolist(), [0, 0])
            self.assertEqual(free_indices.tolist(), [4, 5, 6, 7])

        child_allocator.free.side_effect = verify_owner_clear
        ownership = _HiSparsePageOwnership(
            mapping=mapping,
            child_allocator=child_allocator,
            page_size=4,
        )

        ownership.release(
            mapping_indices=torch.tensor([1, 2, 3]),
            extra_owned_coordinates=buffer_owner,
            clear_extra_owner=lambda: buffer_owner.zero_(),
        )

        child_allocator.free.assert_called_once()

    def test_release_keeps_device_child_out_of_logical_free_group(self) -> None:
        """Device ownership fails loudly if its child joins a logical free group."""
        mapping = torch.tensor([0, 1], dtype=torch.int64)
        child_allocator = mock.Mock(is_not_in_free_group=True)
        ownership = _HiSparsePageOwnership(
            mapping=mapping,
            child_allocator=child_allocator,
            page_size=1,
        )
        child_allocator.is_not_in_free_group = False

        with self.assertRaises(AssertionError):
            ownership.release(mapping_indices=torch.tensor([1]))

        child_allocator.free.assert_not_called()

    def test_unique_hot_path_bypasses_stable_canonicalization(self) -> None:
        """Unique page-size-one owners bypass cold-path canonicalization."""
        mapping = torch.tensor([0, 2, 3], dtype=torch.int64)
        child_allocator = mock.Mock(is_not_in_free_group=True)
        ownership = _HiSparsePageOwnership(
            mapping=mapping,
            child_allocator=child_allocator,
            page_size=1,
        )

        with mock.patch(
            "sglang.srt.mem_cache.allocator.hisparse._stable_unique_page_ids"
        ) as stable_unique:
            ownership.release(
                mapping_indices=torch.tensor([1, 2]),
                unique_page_owners=True,
            )

        stable_unique.assert_not_called()
        self.assertEqual(child_allocator.free.call_args.args[0].tolist(), [2, 3])


if __name__ == "__main__":
    unittest.main()
