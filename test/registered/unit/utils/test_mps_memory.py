"""Unit tests for Apple Silicon memory reporting caps (#21443).

On Apple Silicon the GPU shares unified memory with the CPU, but Metal can
only keep ``recommendedMaxWorkingSetSize`` (~65-75% of RAM) resident on the
GPU.  SGLang has two runtimes on this hardware — native MLX and torch-MPS —
and ``sglang._apple_silicon_memory`` answers every memory query with the
APIs of the active runtime.  MLX's counters see only MLX buffers while
torch's driver counter is device-level, so the MLX path also folds in the
torch-side view (tensors bridged through torch-MPS stay resident there).

These tests pin those invariants: memory reporting never advertises more
than the Metal working set, working-set queries are answered by the active
runtime, and MLX-mode accounting is not blind to torch-side residency.  All
runtime APIs (``torch.mps``, ``mlx.core``, ``psutil``) are mocked, so the
tests run on any host.  The ``torch.mps`` patches deliberately omit
``create=True``: if torch ever drops one of these APIs, the suite should
fail loudly rather than keep passing against a mock.
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
        patch("torch.mps.recommended_max_memory", return_value=recommended)
    )
    stack.enter_context(
        patch("torch.mps.driver_allocated_memory", return_value=driver_allocated)
    )
    return stack.enter_context(patch("torch.mps.empty_cache"))


def _patch_mlx_runtime(
    stack: ExitStack,
    *,
    available: int = 0,
    total: int = 0,
    working_set: int = 0,
    active: int = 0,
    cache: int = 0,
    torch_driver_allocated: int = 0,
) -> tuple[types.ModuleType, MagicMock]:
    """Pin dispatch to the MLX runtime and install a fake ``mlx.core``.

    ``torch.mps.recommended_max_memory`` carries a poison value: any
    working-set or total-memory assertion passing under this patch proves
    the torch API was not consulted.  ``driver_allocated_memory`` takes a
    real value because MLX-mode accounting legitimately folds it in (the
    torch counter is device-level).  Returns the fake ``mlx.core`` module
    and the ``torch.mps.empty_cache`` mock.
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
    stack.enter_context(patch("torch.backends.mps.is_available", return_value=True))
    stack.enter_context(
        patch("torch.mps.recommended_max_memory", return_value=999 * GiB)
    )
    stack.enter_context(
        patch(
            "torch.mps.driver_allocated_memory",
            return_value=torch_driver_allocated,
        )
    )
    torch_empty_cache = stack.enter_context(patch("torch.mps.empty_cache"))

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
            stack.enter_context(patch("torch.mps.recommended_max_memory", new=None))
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
    """Under ``use_mlx()`` queries are answered by MLX plus the device-level
    torch view; the working set never comes from ``torch.mps``."""

    def test_working_set_size_reads_mlx_device_info(self):
        with ExitStack() as stack:
            _patch_mlx_runtime(stack, working_set=16 * GiB)
            self.assertEqual(apple_mem.apple_gpu_working_set_size(), 16 * GiB)

    def test_allocated_memory_counts_active_plus_cache(self):
        with ExitStack() as stack:
            _patch_mlx_runtime(stack, active=3 * GiB, cache=GiB)
            self.assertEqual(apple_mem.apple_gpu_allocated_memory(), 4 * GiB)

    def test_allocated_memory_folds_in_torch_side_residency(self):
        # Tensors bridged through torch-MPS keep blocks resident that MLX
        # counters cannot see. torch's device-level counter (which already
        # includes the MLX buffers) must win when it is larger.
        with ExitStack() as stack:
            _patch_mlx_runtime(
                stack, active=GiB, cache=GiB, torch_driver_allocated=5 * GiB
            )
            self.assertEqual(apple_mem.apple_gpu_allocated_memory(), 5 * GiB)

    def test_available_memory_uses_mlx_headroom(self):
        # 16 GiB working set with 5 GiB held by MLX leaves 11 GiB, below the
        # 20 GiB of free system RAM. The working set carries a torch-side
        # poison value, so the cap can only come from the MLX runtime.
        with ExitStack() as stack:
            _patch_mlx_runtime(
                stack,
                available=20 * GiB,
                working_set=16 * GiB,
                active=4 * GiB,
                cache=GiB,
            )
            self.assertEqual(apple_mem.apple_gpu_available_memory(), 11 * GiB)

    def test_available_memory_sees_torch_bridged_tensors(self):
        # MLX view says 3 GiB used, but the device-level torch counter says
        # 10 GiB: headroom must shrink to 6 GiB, not 13 GiB.
        with ExitStack() as stack:
            _patch_mlx_runtime(
                stack,
                available=20 * GiB,
                working_set=16 * GiB,
                active=2 * GiB,
                cache=GiB,
                torch_driver_allocated=10 * GiB,
            )
            self.assertEqual(apple_mem.apple_gpu_available_memory(), 6 * GiB)

    def test_total_memory_falls_back_when_mlx_limit_unknown(self):
        with ExitStack() as stack:
            _patch_mlx_runtime(stack, total=24 * GiB, working_set=0)
            self.assertEqual(apple_mem.apple_gpu_total_memory(), 24 * GiB)

    def test_empty_cache_clears_both_allocators(self):
        # MLX serving bridges tensors through torch-MPS, so both allocator
        # caches hold reclaimable working-set memory.
        with ExitStack() as stack:
            core, torch_empty_cache = _patch_mlx_runtime(
                stack, available=GiB, working_set=GiB
            )
            apple_mem.apple_gpu_available_memory(empty_cache=True)
            core.clear_cache.assert_called_once()
            torch_empty_cache.assert_called_once()

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
                patch("torch.mps.recommended_max_memory", return_value=16 * GiB)
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
                            patch("torch.mps.recommended_max_memory", new=None)
                        )
                    else:
                        stack.enter_context(
                            patch(
                                "torch.mps.recommended_max_memory",
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
                patch("torch.mps.recommended_max_memory", return_value=16 * GiB)
            )
            first = mps_stub.get_device_properties()
        # Second call must not re-read the device: same cached object even
        # though the patch is gone.
        self.assertIs(mps_stub.get_device_properties(), first)


if __name__ == "__main__":
    unittest.main()
