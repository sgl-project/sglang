import sys
import types
import unittest
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.mem_cache import memory_pool_host as mph


class FakeMooncakeAllocator(mph.HostTensorAllocator):
    pass


class FakeCustomAllocator(mph.HostTensorAllocator):
    pass


class BrokenMooncakeAllocator(mph.HostTensorAllocator):
    def __init__(self):
        raise ImportError("MooncakeHostMemAllocator is unavailable")


@contextmanager
def fake_mooncake_allocator_cls(cls):
    module_name = "sglang.srt.mem_cache.storage.mooncake_store.mooncake_store"
    old_module = sys.modules.get(module_name)
    module = types.ModuleType(module_name)
    module.MooncakeHostTensorAllocator = cls
    sys.modules[module_name] = module
    try:
        yield
    finally:
        if old_module is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = old_module


class TestMooncakeNPUAllocator(unittest.TestCase):
    def test_alloc_memory_funcs_npu_uses_dedicated_helper(self):
        self.assertIs(mph.ALLOC_MEMORY_FUNCS["npu"], mph.alloc_with_host_register_npu)

    def test_get_allocator_from_storage_falls_back_when_mooncake_init_fails(self):
        with fake_mooncake_allocator_cls(BrokenMooncakeAllocator):
            allocator = mph.get_allocator_from_storage("mooncake")

        self.assertIs(type(allocator), mph.HostTensorAllocator)

    def test_alloc_with_host_register_npu_uses_mooncake_allocator(self):
        expected = MagicMock(name="mooncake_buffer")
        allocator = FakeMooncakeAllocator()
        allocator.allocate = MagicMock(return_value=expected)

        with fake_mooncake_allocator_cls(FakeMooncakeAllocator), patch.object(
            mph.torch,
            "empty",
            side_effect=AssertionError("torch.empty should not be used"),
        ):
            buffer = mph.alloc_with_host_register_npu(
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

        with fake_mooncake_allocator_cls(FakeMooncakeAllocator), patch.object(
            mph.torch, "empty", return_value=expected
        ) as mock_empty:
            buffer = mph.alloc_with_host_register_npu(
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
