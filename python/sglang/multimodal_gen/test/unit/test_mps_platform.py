# SPDX-License-Identifier: Apache-2.0

import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist

from sglang.multimodal_gen.runtime.platforms.mps import MpsPlatform

_GIB = 1 << 30


class TestMpsPlatformMemory(unittest.TestCase):
    """Contract tests for MpsPlatform working-set and memory reporting."""

    def setUp(self):
        super().setUp()
        MpsPlatform.get_device_total_memory.cache_clear()
        self.addCleanup(MpsPlatform.get_device_total_memory.cache_clear)

    def _patch_mps_env(
        self,
        stack: ExitStack,
        *,
        total_ram: int = 48 * _GIB,
        available_ram: int = 40 * _GIB,
        recommended: int | None = 36 * _GIB,
        driver_allocated: int = 0,
        forbid_current_allocated: bool = False,
    ):
        stack.enter_context(
            patch(
                "sglang.multimodal_gen.runtime.platforms.mps.psutil.virtual_memory",
                return_value=SimpleNamespace(total=total_ram, available=available_ram),
            )
        )
        if recommended is None:
            mock_recommended = stack.enter_context(
                patch.object(torch.mps, "recommended_max_memory", None)
            )
        else:
            mock_recommended = stack.enter_context(
                patch.object(
                    torch.mps, "recommended_max_memory", return_value=recommended
                )
            )
        stack.enter_context(
            patch.object(
                torch.mps, "driver_allocated_memory", return_value=driver_allocated
            )
        )
        mock_empty_cache = stack.enter_context(patch.object(torch.mps, "empty_cache"))
        if forbid_current_allocated:
            stack.enter_context(
                patch.object(
                    torch.mps,
                    "current_allocated_memory",
                    side_effect=AssertionError("must use driver_allocated_memory"),
                )
            )
        return SimpleNamespace(
            recommended=mock_recommended,
            empty_cache=mock_empty_cache,
        )

    def test_idle_boot_caps_memory_to_working_set_and_caches_total(self):
        with ExitStack() as stack:
            mocks = self._patch_mps_env(
                stack,
                total_ram=48 * _GIB,
                available_ram=40 * _GIB,
                recommended=36 * _GIB,
                driver_allocated=0,
            )
            self.assertEqual(MpsPlatform.get_device_total_memory(), 36 * _GIB)
            self.assertEqual(MpsPlatform.get_device_total_memory(), 36 * _GIB)
            self.assertEqual(mocks.recommended.call_count, 1)
            self.assertEqual(MpsPlatform.get_available_gpu_memory(), 36.0)

    def test_resident_allocation_uses_driver_accounting_and_deducts_available(self):
        with ExitStack() as stack:
            self._patch_mps_env(
                stack,
                total_ram=48 * _GIB,
                available_ram=40 * _GIB,
                recommended=36 * _GIB,
                driver_allocated=10 * _GIB,
                forbid_current_allocated=True,
            )
            self.assertEqual(MpsPlatform.get_current_memory_usage(), float(10 * _GIB))
            self.assertEqual(MpsPlatform.get_available_gpu_memory(), 26.0)

    def test_host_available_memory_wins_when_tighter(self):
        with ExitStack() as stack:
            self._patch_mps_env(
                stack,
                total_ram=48 * _GIB,
                available_ram=6 * _GIB,
                recommended=36 * _GIB,
                driver_allocated=4 * _GIB,
            )
            self.assertEqual(MpsPlatform.get_available_gpu_memory(), 6.0)

    def test_overcommitted_working_set_clamps_to_zero(self):
        with ExitStack() as stack:
            self._patch_mps_env(
                stack,
                total_ram=48 * _GIB,
                available_ram=20 * _GIB,
                recommended=36 * _GIB,
                driver_allocated=38 * _GIB,
            )
            self.assertEqual(MpsPlatform.get_available_gpu_memory(), 0.0)

    def test_invalid_or_missing_working_set_raises_and_is_not_cached(self):
        for bad_value in (None, 0, -1):
            with self.subTest(recommended=bad_value):
                MpsPlatform.get_device_total_memory.cache_clear()
                with ExitStack() as stack:
                    self._patch_mps_env(stack, recommended=bad_value)
                    with self.assertRaises(RuntimeError):
                        MpsPlatform.get_device_total_memory()
                    with self.assertRaises(RuntimeError):
                        MpsPlatform.get_available_gpu_memory(empty_cache=False)

        with ExitStack() as stack:
            self._patch_mps_env(stack, recommended=16 * _GIB)
            self.assertEqual(MpsPlatform.get_device_total_memory(), 16 * _GIB)

    def test_empty_cache_forwarding_and_distributed_min_reduction(self):
        with ExitStack() as stack:
            mocks = self._patch_mps_env(
                stack,
                available_ram=24 * _GIB,
                recommended=36 * _GIB,
                driver_allocated=16 * _GIB,
            )
            self.assertEqual(MpsPlatform.get_available_gpu_memory(empty_cache=True), 20.0)
            mocks.empty_cache.assert_called_once_with()

            mocks.empty_cache.reset_mock()
            self.assertEqual(
                MpsPlatform.get_available_gpu_memory(empty_cache=False), 20.0
            )
            mocks.empty_cache.assert_not_called()

            cpu_group = object()

            def _fake_all_reduce(tensor, *, op=None, group=None):
                self.assertEqual(op, dist.ReduceOp.MIN)
                self.assertIs(group, cpu_group)
                tensor.fill_(12 * _GIB)

            with patch.object(dist, "all_reduce", side_effect=_fake_all_reduce) as mock_ar:
                reduced = MpsPlatform.get_available_gpu_memory(
                    distributed=True,
                    empty_cache=False,
                    cpu_group=cpu_group,
                )
            self.assertEqual(reduced, 12.0)
            mock_ar.assert_called_once()


if __name__ == "__main__":
    unittest.main()
