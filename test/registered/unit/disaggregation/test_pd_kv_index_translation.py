import unittest

import torch

from sglang.srt.disaggregation.utils import resolve_physical_kv_indices
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _CompactingAllocator:
    def __init__(self):
        self.virtual_to_physical = torch.tensor([0, 11, 12, 13, 14], dtype=torch.int64)

    def translate_kv_loc(self, virtual_indices: torch.Tensor) -> torch.Tensor:
        return self.virtual_to_physical[virtual_indices]


class TestPDKVIndexTranslation(unittest.TestCase):
    def test_uses_current_mapping_after_compaction(self):
        allocator = _CompactingAllocator()
        virtual_indices = torch.tensor([1, 2, 3], dtype=torch.int32)

        before = resolve_physical_kv_indices(allocator, virtual_indices)
        self.assertTrue(torch.equal(before, torch.tensor([11, 12, 13])))

        # Simulate compaction relocating live virtual pages while req_to_token
        # retains the same stable virtual ids.
        allocator.virtual_to_physical[1:4] = torch.tensor([31, 21, 41])
        after = resolve_physical_kv_indices(allocator, virtual_indices)

        self.assertTrue(torch.equal(after, torch.tensor([31, 21, 41])))
        self.assertTrue(
            torch.equal(virtual_indices, torch.tensor([1, 2, 3], dtype=torch.int32))
        )

    def test_non_virtual_allocator_is_identity(self):
        indices = torch.tensor([4, 8], dtype=torch.int32)
        resolved = resolve_physical_kv_indices(object(), indices)

        self.assertIs(resolved, indices)


if __name__ == "__main__":
    unittest.main()
