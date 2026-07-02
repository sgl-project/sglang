"""Unit tests for MPS memory reporting caps (#21443).

On Apple Silicon the GPU shares unified memory with the CPU, but Metal can
only keep ``recommendedMaxWorkingSetSize`` (~65-75% of RAM) resident on the
GPU. These tests pin the invariant that MPS memory reporting never
advertises more than that working set: ``get_available_gpu_memory`` caps to
the remaining working-set headroom, and the ``_mps_stub`` device properties
report the working set instead of total system RAM. All ``torch.mps`` and
``psutil`` calls are mocked, so the tests run on any host.
"""

import unittest
from contextlib import ExitStack
from unittest.mock import patch

import sglang._mps_stub as mps_stub
from sglang.srt.utils.common import get_available_gpu_memory
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

GiB = 1 << 30


class _VirtualMemory:
    """Minimal stand-in for ``psutil.virtual_memory()``'s return value."""

    def __init__(self, available: int = 0, total: int = 0):
        self.available = available
        self.total = total


class TestGetAvailableGpuMemoryMps(CustomTestCase):
    """The ``device == "mps"`` branch of ``get_available_gpu_memory``."""

    def _free_gib(
        self,
        *,
        available: int,
        recommended: int,
        driver_allocated: int = 0,
        empty_cache: bool = False,
    ) -> float:
        with ExitStack() as stack:
            stack.enter_context(
                patch(
                    "psutil.virtual_memory",
                    return_value=_VirtualMemory(available=available),
                )
            )
            stack.enter_context(
                patch(
                    "torch.mps.recommended_max_memory",
                    create=True,
                    return_value=recommended,
                )
            )
            stack.enter_context(
                patch(
                    "torch.mps.driver_allocated_memory",
                    create=True,
                    return_value=driver_allocated,
                )
            )
            self._empty_cache_mock = stack.enter_context(
                patch("torch.mps.empty_cache", create=True)
            )
            return get_available_gpu_memory("mps", gpu_id=0, empty_cache=empty_cache)

    def test_caps_free_ram_to_working_set(self):
        # Fresh boot: free RAM exceeds what Metal can wire. The old
        # psutil-only report (20 GiB) is exactly the overcommit from #21443.
        free = self._free_gib(available=20 * GiB, recommended=16 * GiB)
        self.assertEqual(free, 16.0)

    def test_subtracts_gpu_memory_already_in_use(self):
        free = self._free_gib(
            available=20 * GiB, recommended=16 * GiB, driver_allocated=10 * GiB
        )
        self.assertEqual(free, 6.0)

    def test_reports_system_available_when_smaller(self):
        # CPU-side pressure: plenty of working-set headroom but little free
        # RAM. Unified memory means the smaller bound wins.
        free = self._free_gib(available=4 * GiB, recommended=16 * GiB)
        self.assertEqual(free, 4.0)

    def test_clamps_to_zero_when_overcommitted(self):
        free = self._free_gib(
            available=20 * GiB, recommended=16 * GiB, driver_allocated=18 * GiB
        )
        self.assertEqual(free, 0.0)

    def test_falls_back_to_system_available_without_working_set(self):
        # recommended_max_memory() == 0 means the limit is unknown; keep the
        # legacy psutil behavior rather than reporting no memory at all.
        free = self._free_gib(available=20 * GiB, recommended=0)
        self.assertEqual(free, 20.0)

    def test_empty_cache_flag_forwards_to_allocator(self):
        self._free_gib(available=GiB, recommended=GiB, empty_cache=True)
        self._empty_cache_mock.assert_called_once()

        self._free_gib(available=GiB, recommended=GiB, empty_cache=False)
        self._empty_cache_mock.assert_not_called()


class TestMpsStubDeviceProperties(CustomTestCase):
    """``_mps_stub.get_device_properties`` reports the Metal working set."""

    def setUp(self):
        self._saved_props = mps_stub._cached_props
        mps_stub._cached_props = None

    def tearDown(self):
        mps_stub._cached_props = self._saved_props

    def test_total_memory_is_working_set_not_system_ram(self):
        with ExitStack() as stack:
            stack.enter_context(
                patch(
                    "torch.mps.recommended_max_memory",
                    create=True,
                    return_value=16 * GiB,
                )
            )
            stack.enter_context(
                patch(
                    "psutil.virtual_memory",
                    return_value=_VirtualMemory(total=24 * GiB),
                )
            )
            props = mps_stub.get_device_properties()
        self.assertEqual(props.total_memory, 16 * GiB)

    def test_falls_back_to_system_ram_when_working_set_unknown(self):
        for recommended in (None, 0):
            with self.subTest(recommended=recommended):
                mps_stub._cached_props = None
                with ExitStack() as stack:
                    if recommended is None:
                        stack.enter_context(
                            patch(
                                "torch.mps.recommended_max_memory",
                                new=None,
                                create=True,
                            )
                        )
                    else:
                        stack.enter_context(
                            patch(
                                "torch.mps.recommended_max_memory",
                                create=True,
                                return_value=recommended,
                            )
                        )
                    stack.enter_context(
                        patch(
                            "psutil.virtual_memory",
                            return_value=_VirtualMemory(total=24 * GiB),
                        )
                    )
                    props = mps_stub.get_device_properties()
                self.assertEqual(props.total_memory, 24 * GiB)

    def test_properties_are_cached(self):
        with ExitStack() as stack:
            stack.enter_context(
                patch(
                    "torch.mps.recommended_max_memory",
                    create=True,
                    return_value=16 * GiB,
                )
            )
            first = mps_stub.get_device_properties()
        # Second call must not re-read the device: same cached object even
        # though the patch is gone.
        self.assertIs(mps_stub.get_device_properties(), first)


if __name__ == "__main__":
    unittest.main()
