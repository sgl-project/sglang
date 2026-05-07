import unittest
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.mem_cache import host_allocator as ha


class FakeMooncakeAllocator(ha.HostTensorAllocator):
    pass


class FakeCustomAllocator(ha.HostTensorAllocator):
    pass


class TestMooncakeNPUAllocator(unittest.TestCase):
    def test_alloc_memory_funcs_npu_uses_dedicated_helper(self):
        self.assertIs(ha.ALLOC_MEMORY_FUNCS["npu"], ha.alloc_with_host_register_npu)

    def test_alloc_with_host_register_npu_uses_mooncake_allocator(self):
        expected = MagicMock(name="mooncake_buffer")
        allocator = FakeMooncakeAllocator()
        allocator.allocate = MagicMock(return_value=expected)

        with patch.object(
            ha,
            "get_mooncake_host_tensor_allocator_cls",
            return_value=FakeMooncakeAllocator,
        ), patch.object(
            ha.torch,
            "empty",
            side_effect=AssertionError("torch.empty should not be used"),
        ):
            buffer = ha.alloc_with_host_register_npu(
                dims=(2, 3),
                dtype=torch.float32,
                device="cpu",
                pin_memory=True,
                allocator=allocator,
            )

        self.assertIs(buffer, expected)
        allocator.allocate.assert_called_once_with(
            (2, 3), dtype=torch.float32, device="cpu"
        )

    def test_alloc_with_host_register_npu_falls_back_for_non_mooncake_allocator(self):
        expected = MagicMock(name="pinned_buffer")
        allocator = FakeCustomAllocator()
        allocator.allocate = MagicMock()

        with patch.object(
            ha,
            "get_mooncake_host_tensor_allocator_cls",
            return_value=FakeMooncakeAllocator,
        ), patch.object(ha.torch, "empty", return_value=expected) as mock_empty:
            buffer = ha.alloc_with_host_register_npu(
                dims=(4, 5),
                dtype=torch.bfloat16,
                device="cpu",
                pin_memory=False,
                allocator=allocator,
            )

        self.assertIs(buffer, expected)
        allocator.allocate.assert_not_called()
        mock_empty.assert_called_once_with(
            (4, 5), dtype=torch.bfloat16, device="cpu", pin_memory=False
        )


if __name__ == "__main__":
    unittest.main()
