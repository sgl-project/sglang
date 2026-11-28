import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=3, suite="extra-a-test-npu")

from sglang.srt.mem_cache import memory_pool as mp
from sglang.srt.mem_cache.memory_pool import MambaPool
from sglang.srt.utils import is_npu


def _has_sgl_kernel_clear_mamba_slots() -> bool:
    if not is_npu():
        return False
    try:
        from sgl_kernel_npu.operators.mem_cache import clear_mamba_slots

        return clear_mamba_slots is not None
    except ImportError:
        return False


def _make_dummy_mamba_pool(device: str = "cpu"):
    pool = MambaPool.__new__(MambaPool)
    pool._slot_siblings = []
    pool._should_fuse_slot_ops = lambda: False
    pool.mamba_cache = SimpleNamespace(
        conv=[torch.ones((4, 8, 3, 16), device=device, dtype=torch.bfloat16)],
        temporal=torch.ones((4, 8, 4, 8, 8), device=device, dtype=torch.bfloat16),
    )
    return pool


class TestNPUMambaClearSlots(CustomTestCase):
    def test_clear_slots_dispatches_to_sgl_kernel_npu_when_available(self):
        pool = _make_dummy_mamba_pool(device="cpu")
        indices = torch.tensor([1, 5], dtype=torch.int32)
        mock_clear = MagicMock()

        with (
            patch.object(mp, "_is_npu", True),
            patch.object(mp, "_get_npu_clear_mamba_slots", return_value=mock_clear),
        ):
            pool.clear_slots(indices)

        self.assertEqual(mock_clear.call_count, 2)
        self.assertIs(mock_clear.call_args_list[0].args[0], pool.mamba_cache.conv[0])
        self.assertEqual(mock_clear.call_args_list[0].args[1].dtype, torch.int64)
        self.assertEqual(mock_clear.call_args_list[0].args[1].tolist(), [1, 5])
        self.assertIs(mock_clear.call_args_list[1].args[0], pool.mamba_cache.temporal)

    def test_clear_slots_falls_back_when_sgl_kernel_npu_unavailable(self):
        pool = _make_dummy_mamba_pool(device="cpu")
        indices = torch.tensor([2, 6], dtype=torch.int32)

        with (
            patch.object(mp, "_is_npu", True),
            patch.object(mp, "_get_npu_clear_mamba_slots", return_value=None),
        ):
            pool.clear_slots(indices)

        self.assertEqual(pool.mamba_cache.conv[0][:, indices].abs().max().item(), 0.0)
        self.assertEqual(pool.mamba_cache.temporal[:, indices].abs().max().item(), 0.0)
        self.assertEqual(pool.mamba_cache.conv[0][:, 0].min().item(), 1.0)
        self.assertEqual(pool.mamba_cache.temporal[:, 0].min().item(), 1.0)

    def test_clear_slots_empty_indices_noop(self):
        pool = _make_dummy_mamba_pool(device="cpu")
        indices = torch.empty((0,), dtype=torch.int64)
        mock_clear = MagicMock()

        with (
            patch.object(mp, "_is_npu", True),
            patch.object(mp, "_get_npu_clear_mamba_slots", return_value=mock_clear),
        ):
            pool.clear_slots(indices)

        mock_clear.assert_not_called()
        self.assertEqual(pool.mamba_cache.temporal.min().item(), 1.0)

    @unittest.skipUnless(
        _has_sgl_kernel_clear_mamba_slots(),
        "Requires Ascend NPU hardware with sgl_kernel_npu.operators.mem_cache.clear_mamba_slots",
    )
    def test_clear_slots_npu_integration(self):
        pool = _make_dummy_mamba_pool(device="npu")
        indices = torch.tensor([0, 3, 7], device="npu", dtype=torch.int32)
        pool.clear_slots(indices)
        torch.npu.synchronize()

        self.assertEqual(pool.mamba_cache.conv[0][:, indices].abs().max().item(), 0.0)
        self.assertEqual(pool.mamba_cache.temporal[:, indices].abs().max().item(), 0.0)
        self.assertEqual(pool.mamba_cache.temporal[:, 1].min().item(), 1.0)


if __name__ == "__main__":
    unittest.main()
