"""Unit tests for Apple Silicon memory reporting caps (#21443).

On Apple Silicon the GPU shares unified memory with the CPU, but Metal can
only keep ``recommendedMaxWorkingSetSize`` (~65-75% of RAM) resident on the
GPU.  SGLang has two runtimes on this hardware — native MLX and torch-MPS —
each with its own allocator, so ``sglang._apple_silicon_memory`` answers
every memory query with the APIs of the runtime that actually allocates.

These tests pin two invariants: memory reporting never advertises more than
the Metal working set, and each query is answered by the active runtime
(MLX numbers under ``use_mlx()``, ``torch.mps`` numbers otherwise).  All
runtime APIs (``torch.mps``, ``mlx.core``, ``psutil``) are mocked, so the
tests run on any host.
"""

import sys
import types
import unittest
from contextlib import ExitStack
from unittest.mock import MagicMock, patch

import sglang._apple_silicon_memory as apple_mem
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


def _patch_torch_runtime(
    stack: ExitStack,
    *,
    available: int = 0,
    total: int = 0,
    recommended: int = 0,
    driver_allocated: int = 0,
) -> MagicMock:
    """Pin dispatch to the torch-MPS runtime and mock its memory APIs.

    Returns the ``torch.mps.empty_cache`` mock.
    """
    stack.enter_context(
        patch.object(apple_mem, "_mlx_runtime_active", return_value=False)
    )
    stack.enter_context(
        patch(
            "psutil.virtual_memory",
            return_value=_VirtualMemory(available=available, total=total),
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
    return stack.enter_context(patch("torch.mps.empty_cache", create=True))


def _patch_mlx_runtime(
    stack: ExitStack,
    *,
    available: int = 0,
    total: int = 0,
    working_set: int = 0,
    active: int = 0,
    cache: int = 0,
) -> tuple[types.ModuleType, MagicMock]:
    """Pin dispatch to the MLX runtime and install a fake ``mlx.core``.

    ``torch.mps`` is mocked with poison values, so any test passing under
    this patch proves the torch APIs were not consulted.  Returns the fake
    ``mlx.core`` module and the ``torch.mps.empty_cache`` mock.
    """
    stack.enter_context(
        patch.object(apple_mem, "_mlx_runtime_active", return_value=True)
    )
    stack.enter_context(
        patch(
            "psutil.virtual_memory",
            return_value=_VirtualMemory(available=available, total=total),
        )
    )
    stack.enter_context(
        patch(
            "torch.mps.recommended_max_memory",
            create=True,
            return_value=999 * GiB,
        )
    )
    stack.enter_context(
        patch(
            "torch.mps.driver_allocated_memory",
            create=True,
            return_value=998 * GiB,
        )
    )
    torch_empty_cache = stack.enter_context(patch("torch.mps.empty_cache", create=True))

    device_info = (
        {"max_recommended_working_set_size": working_set} if working_set else {}
    )
    core = types.ModuleType("mlx.core")
    core.device_info = lambda: dict(device_info)
    core.get_active_memory = lambda: active
    core.get_cache_memory = lambda: cache
    core.clear_cache = MagicMock()
    pkg = types.ModuleType("mlx")
    pkg.core = core
    stack.enter_context(patch.dict(sys.modules, {"mlx": pkg, "mlx.core": core}))
    return core, torch_empty_cache


class TestAppleGpuMemoryTorchPath(CustomTestCase):
    """The shared helpers on the torch-MPS path."""

    def test_working_set_size_reads_torch_mps(self):
        with ExitStack() as stack:
            _patch_torch_runtime(stack, recommended=16 * GiB)
            self.assertEqual(apple_mem.apple_gpu_working_set_size(), 16 * GiB)

    def test_working_set_size_zero_when_api_missing(self):
        # torch builds without recommended_max_memory() report 0 so callers
        # fall back to system RAM instead of a zero-sized device.
        with ExitStack() as stack:
            stack.enter_context(
                patch.object(apple_mem, "_mlx_runtime_active", return_value=False)
            )
            stack.enter_context(
                patch("torch.mps.recommended_max_memory", new=None, create=True)
            )
            self.assertEqual(apple_mem.apple_gpu_working_set_size(), 0)

    def test_allocated_memory_reads_driver_allocated(self):
        with ExitStack() as stack:
            _patch_torch_runtime(stack, driver_allocated=3 * GiB)
            self.assertEqual(apple_mem.apple_gpu_allocated_memory(), 3 * GiB)

    def test_total_memory_is_working_set(self):
        with ExitStack() as stack:
            _patch_torch_runtime(stack, recommended=16 * GiB, total=24 * GiB)
            self.assertEqual(apple_mem.apple_gpu_total_memory(), 16 * GiB)

    def test_total_memory_falls_back_to_system_ram(self):
        with ExitStack() as stack:
            _patch_torch_runtime(stack, recommended=0, total=24 * GiB)
            self.assertEqual(apple_mem.apple_gpu_total_memory(), 24 * GiB)


class TestAppleGpuMemoryMlxDispatch(CustomTestCase):
    """Under ``use_mlx()`` every query is answered by MLX, not torch."""

    def test_working_set_size_reads_mlx_device_info(self):
        with ExitStack() as stack:
            _patch_mlx_runtime(stack, working_set=16 * GiB)
            self.assertEqual(apple_mem.apple_gpu_working_set_size(), 16 * GiB)

    def test_allocated_memory_counts_active_plus_cache(self):
        with ExitStack() as stack:
            _patch_mlx_runtime(stack, active=3 * GiB, cache=GiB)
            self.assertEqual(apple_mem.apple_gpu_allocated_memory(), 4 * GiB)

    def test_available_memory_uses_mlx_headroom(self):
        # 16 GiB working set with 5 GiB held by MLX leaves 11 GiB, below the
        # 20 GiB of free system RAM. torch.mps carries poison values, so the
        # result can only come from the MLX runtime.
        with ExitStack() as stack:
            _patch_mlx_runtime(
                stack,
                available=20 * GiB,
                working_set=16 * GiB,
                active=4 * GiB,
                cache=GiB,
            )
            self.assertEqual(apple_mem.apple_gpu_available_memory(), 11 * GiB)

    def test_total_memory_falls_back_when_mlx_limit_unknown(self):
        with ExitStack() as stack:
            _patch_mlx_runtime(stack, total=24 * GiB, working_set=0)
            self.assertEqual(apple_mem.apple_gpu_total_memory(), 24 * GiB)

    def test_empty_cache_clears_mlx_not_torch(self):
        with ExitStack() as stack:
            core, torch_empty_cache = _patch_mlx_runtime(
                stack, available=GiB, working_set=GiB
            )
            apple_mem.apple_gpu_available_memory(empty_cache=True)
            core.clear_cache.assert_called_once()
            torch_empty_cache.assert_not_called()

    def test_get_available_gpu_memory_dispatches_to_mlx(self):
        with ExitStack() as stack:
            _patch_mlx_runtime(
                stack,
                available=20 * GiB,
                working_set=16 * GiB,
                active=6 * GiB,
            )
            free = get_available_gpu_memory("mps", gpu_id=0, empty_cache=False)
        self.assertEqual(free, 10.0)

    def test_stub_total_memory_reads_mlx_working_set(self):
        saved = mps_stub._cached_props
        mps_stub._cached_props = None
        try:
            with ExitStack() as stack:
                _patch_mlx_runtime(stack, working_set=16 * GiB)
                props = mps_stub.get_device_properties()
            self.assertEqual(props.total_memory, 16 * GiB)
        finally:
            mps_stub._cached_props = saved


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
            self._empty_cache_mock = _patch_torch_runtime(
                stack,
                available=available,
                recommended=recommended,
                driver_allocated=driver_allocated,
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
        self._dispatch_patch = patch.object(
            apple_mem, "_mlx_runtime_active", return_value=False
        )
        self._dispatch_patch.start()

    def tearDown(self):
        self._dispatch_patch.stop()
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
