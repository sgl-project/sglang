"""Unit tests for the env-gated extend memory profiler (hook-local
caching-allocator allocation deltas plus the whole-extend allocated peak)."""

import contextlib
import unittest
from unittest import mock

import torch

from sglang.srt.utils import extend_mem_profile, mem_forensics
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

MiB = 2**20
LOGGER = "sglang.srt.utils.extend_mem_profile"


class FakeAllocator:
    """Host-side stand-in for the CUDA caching allocator's counters."""

    def __init__(self, allocated: int = 0):
        self.allocated = allocated
        self.peak = allocated
        self.reset_calls = 0
        self.fail_mem_get_info = False
        self.fail_max_memory_allocated = False

    def alloc(self, nbytes: int) -> None:
        self.allocated += nbytes
        self.peak = max(self.peak, self.allocated)

    def free(self, nbytes: int) -> None:
        self.allocated -= nbytes

    def memory_allocated(self) -> int:
        return self.allocated

    def max_memory_allocated(self) -> int:
        if self.fail_max_memory_allocated:
            raise RuntimeError("allocator stats unavailable")
        return self.peak

    def reset_peak_memory_stats(self) -> None:
        self.reset_calls += 1
        self.peak = self.allocated

    def mem_get_info(self):
        if self.fail_mem_get_info:
            raise RuntimeError("cudaMemGetInfo failed")
        return (4 * 2**30, 8 * 2**30)

    @contextlib.contextmanager
    def patched(self):
        with (
            mock.patch.object(torch.cuda, "is_available", return_value=True),
            mock.patch.object(torch.cuda, "memory_allocated", self.memory_allocated),
            mock.patch.object(
                torch.cuda, "max_memory_allocated", self.max_memory_allocated
            ),
            mock.patch.object(
                torch.cuda, "reset_peak_memory_stats", self.reset_peak_memory_stats
            ),
            mock.patch.object(torch.cuda, "mem_get_info", self.mem_get_info),
        ):
            yield


@contextlib.contextmanager
def profiler(enabled: bool):
    """Flip the import-time binding for one test and restore it after."""
    was = extend_mem_profile.ENABLED
    extend_mem_profile._bind(enabled)
    try:
        yield
    finally:
        extend_mem_profile._bind(was)


class ExtendMemProfileTest(unittest.TestCase):
    def setUp(self):
        # Isolation between tests only; the tests below assert the module
        # leaves _active False on its own after every exit path.
        extend_mem_profile._active = False
        extend_mem_profile._phase_peaks = {}
        extend_mem_profile._open_phases.clear()
        extend_mem_profile._extend_count = 0

    def test_disabled_entry_points_are_noops(self):
        with profiler(False):
            self.assertFalse(extend_mem_profile.enabled())
            self.assertFalse(extend_mem_profile.ENABLED)
            with (
                mock.patch.object(torch.cuda, "reset_peak_memory_stats") as reset,
                mock.patch.object(torch.cuda, "memory_allocated") as allocated,
                mock.patch.object(mem_forensics, "maybe_dump_memory_forensics") as dump,
            ):
                extend_mem_profile.begin(4096)
                self.assertFalse(extend_mem_profile._active)
                with extend_mem_profile.phase("layer"):
                    pass
                with self.assertNoLogs(LOGGER):
                    extend_mem_profile.end()
                with self.assertNoLogs(LOGGER):
                    with extend_mem_profile.record(4096):
                        with extend_mem_profile.phase("layer"):
                            pass
        self.assertFalse(reset.called)
        self.assertFalse(allocated.called)
        self.assertFalse(dump.called)
        self.assertEqual(extend_mem_profile._phase_peaks, {})
        self.assertEqual(extend_mem_profile._extend_count, 0)

    def test_disabled_binding_is_the_plain_noop_functions(self):
        # Disabled hot path: module-level functions that return the shared
        # object without reading the env or any module state.
        with profiler(False):
            self.assertIs(extend_mem_profile.phase, extend_mem_profile._phase_disabled)
            self.assertIs(
                extend_mem_profile.record, extend_mem_profile._record_disabled
            )
            with mock.patch("os.getenv") as getenv:
                self.assertIs(
                    extend_mem_profile.phase("a"), extend_mem_profile._NOOP_SCOPE
                )
                self.assertIs(
                    extend_mem_profile.record(4096), extend_mem_profile._NOOP_SCOPE
                )
            self.assertFalse(getenv.called)
        with profiler(True):
            self.assertIs(extend_mem_profile.phase, extend_mem_profile._phase_enabled)
            self.assertIs(extend_mem_profile.record, extend_mem_profile._record_enabled)

    def test_inactive_scopes_are_one_shared_object(self):
        with profiler(False):
            self.assertIs(extend_mem_profile.phase("a"), extend_mem_profile.phase("b"))
            self.assertIs(extend_mem_profile.phase("a"), extend_mem_profile._NOOP_SCOPE)
            self.assertIs(
                extend_mem_profile.record(4096), extend_mem_profile._NOOP_SCOPE
            )
        alloc = FakeAllocator()
        with profiler(True), alloc.patched():
            # Enabled but below min_tokens: still the shared no-op.
            self.assertIs(
                extend_mem_profile.record(1023), extend_mem_profile._NOOP_SCOPE
            )
            # Outside an active extend: still the shared no-op.
            self.assertIs(extend_mem_profile.phase("a"), extend_mem_profile._NOOP_SCOPE)
            with extend_mem_profile.record(4096):
                self.assertIsInstance(
                    extend_mem_profile.phase("a"), extend_mem_profile._Phase
                )

    def test_small_extends_and_missing_cuda_stay_silent(self):
        alloc = FakeAllocator()
        with profiler(True), alloc.patched():
            extend_mem_profile.begin(1023)
            self.assertFalse(extend_mem_profile._active)
            with mock.patch.object(torch.cuda, "is_available", return_value=False):
                extend_mem_profile.begin(4096)
            self.assertFalse(extend_mem_profile._active)
        self.assertEqual(alloc.reset_calls, 0)

    def test_phase_deltas_are_recorded_and_summarized(self):
        alloc = FakeAllocator(allocated=10 * MiB)
        with profiler(True), alloc.patched():
            extend_mem_profile.begin(4096)
            self.assertTrue(extend_mem_profile._active)
            self.assertEqual(extend_mem_profile._base, 10 * MiB)

            with extend_mem_profile.phase("attn"):
                alloc.alloc(30 * MiB)
                alloc.free(30 * MiB)
            with extend_mem_profile.phase("attn"):
                alloc.alloc(20 * MiB)
                alloc.free(20 * MiB)
            with extend_mem_profile.phase("mlp"):
                alloc.alloc(5 * MiB)
                alloc.free(5 * MiB)
            with self.assertLogs(LOGGER, level="INFO") as logs:
                extend_mem_profile.end()

        # Repeated tags keep their maximum; below-threshold phases log nothing
        # until the summary line.
        self.assertEqual(
            extend_mem_profile._phase_peaks, {"attn": 30 * MiB, "mlp": 5 * MiB}
        )
        self.assertEqual(len(logs.output), 1)
        line = logs.output[0]
        self.assertIn("extend-mem-profile #1 tokens=4096 extend_alloc_peak=30MiB", line)
        self.assertIn("top_phase_alloc_deltas[attn=30MiB, mlp=5MiB]", line)
        self.assertFalse(extend_mem_profile._active)

    def test_whole_extend_peak_includes_unwrapped_allocations(self):
        alloc = FakeAllocator(allocated=10 * MiB)
        with profiler(True), alloc.patched():
            with self.assertLogs(LOGGER, level="INFO") as logs:
                with extend_mem_profile.record(4096):
                    # Allocated and freed outside any phase, before the first
                    # phase resets the device-wide counter.
                    alloc.alloc(100 * MiB)
                    alloc.free(100 * MiB)
                    with extend_mem_profile.phase("attn"):
                        alloc.alloc(30 * MiB)
                        alloc.free(30 * MiB)
                    # Again after the last phase, seen only by end().
                    alloc.alloc(120 * MiB)
                    alloc.free(120 * MiB)
        self.assertEqual(extend_mem_profile._phase_peaks, {"attn": 30 * MiB})
        self.assertIn("tokens=4096 extend_alloc_peak=120MiB", logs.output[-1])
        self.assertFalse(extend_mem_profile._active)

    def test_phase_deltas_are_hook_local_not_disjoint_parts_of_the_peak(self):
        # kda:conv retains its outputs, kda:extend then adds a workspace on
        # top of them: the extend peak holds both, each delta holds one, and
        # peak minus a delta is not "everything else".
        alloc = FakeAllocator()
        with profiler(True), alloc.patched():
            with self.assertLogs(LOGGER, level="INFO") as logs:
                with extend_mem_profile.record(4096):
                    with extend_mem_profile.phase("kda:conv"):
                        alloc.alloc(100 * MiB)  # q/k/v outputs, kept alive
                    with extend_mem_profile.phase("kda:extend"):
                        alloc.alloc(60 * MiB)  # workspace
                        alloc.free(60 * MiB)
                    alloc.free(100 * MiB)
        self.assertEqual(
            extend_mem_profile._phase_peaks,
            {"kda:conv": 100 * MiB, "kda:extend": 60 * MiB},
        )
        self.assertIn("extend_alloc_peak=160MiB", logs.output[-1])

    def test_nested_phase_delta_folds_into_outer_phase(self):
        alloc = FakeAllocator()
        with profiler(True), alloc.patched():
            with self.assertLogs(LOGGER, level="INFO") as logs:
                with extend_mem_profile.record(4096):
                    with extend_mem_profile.phase("outer"):
                        alloc.alloc(50 * MiB)  # outer's own transient
                        alloc.free(50 * MiB)
                        alloc.alloc(30 * MiB)  # live across the inner phase
                        with extend_mem_profile.phase("inner"):
                            alloc.alloc(40 * MiB)
                            alloc.free(40 * MiB)
                        alloc.free(30 * MiB)
        # The inner reset must not erase outer's earlier 50 MiB, and the
        # inner peak (30 live + 40) counts towards outer as 70 MiB.
        self.assertEqual(
            extend_mem_profile._phase_peaks, {"outer": 70 * MiB, "inner": 40 * MiB}
        )
        self.assertIn("extend_alloc_peak=70MiB", logs.output[-1])
        self.assertEqual(extend_mem_profile._open_phases, [])

    def test_phase_is_recorded_when_body_raises(self):
        alloc = FakeAllocator()
        with profiler(True), alloc.patched():
            extend_mem_profile.begin(4096)
            with self.assertRaises(RuntimeError):
                with extend_mem_profile.phase("attn"):
                    alloc.alloc(7 * MiB)
                    raise RuntimeError("out of memory")
        self.assertEqual(extend_mem_profile._phase_peaks, {"attn": 7 * MiB})
        self.assertEqual(extend_mem_profile._open_phases, [])

    def test_exception_in_recorded_extend_unwinds_and_propagates_original(self):
        alloc = FakeAllocator()
        with profiler(True), alloc.patched():
            with self.assertRaises(RuntimeError) as raised:
                with extend_mem_profile.record(4096):
                    self.assertTrue(extend_mem_profile._active)
                    with extend_mem_profile.phase("attn"):
                        alloc.alloc(7 * MiB)
                        raise RuntimeError("CUDA out of memory")
            self.assertEqual(str(raised.exception), "CUDA out of memory")
            self.assertFalse(extend_mem_profile._active)
            self.assertEqual(extend_mem_profile._open_phases, [])
            # The following decode/extend does not attribute work to this
            # extend.
            self.assertIs(
                extend_mem_profile.phase("attn"), extend_mem_profile._NOOP_SCOPE
            )

    def test_profiler_failure_never_replaces_original_exception(self):
        alloc = FakeAllocator()
        with profiler(True), alloc.patched():
            with (
                self.assertRaises(RuntimeError) as raised,
                self.assertLogs(LOGGER, level="WARNING") as logs,
            ):
                with extend_mem_profile.record(4096):
                    with extend_mem_profile.phase("attn"):
                        # A CUDA error makes the allocator/device queries in
                        # the profiler fail while the original error unwinds.
                        alloc.fail_max_memory_allocated = True
                        alloc.fail_mem_get_info = True
                        raise RuntimeError("original CUDA error")
        self.assertEqual(str(raised.exception), "original CUDA error")
        self.assertFalse(extend_mem_profile._active)
        self.assertEqual(extend_mem_profile._open_phases, [])
        self.assertTrue(any("disabled for the rest" in line for line in logs.output))

    def test_profiler_failure_without_body_error_is_contained(self):
        alloc = FakeAllocator()
        with profiler(True), alloc.patched():
            alloc.fail_mem_get_info = True
            with self.assertLogs(LOGGER, level="WARNING"):
                with extend_mem_profile.record(4096):
                    with extend_mem_profile.phase("big"):
                        alloc.alloc(300 * MiB)  # crosses the immediate-log threshold
                    # Disabled for the rest of the extend after the failure.
                    self.assertFalse(extend_mem_profile._active)
        self.assertFalse(extend_mem_profile._active)

    def test_large_transient_phase_is_logged_immediately(self):
        alloc = FakeAllocator()
        with profiler(True), alloc.patched():
            extend_mem_profile.begin(4096)
            with (
                mock.patch.object(mem_forensics, "maybe_dump_memory_forensics") as dump,
                self.assertLogs(LOGGER, level="INFO") as logs,
            ):
                with extend_mem_profile.phase("kda:extend"):
                    alloc.alloc(extend_mem_profile.LARGE_PHASE_BYTES)
                    alloc.free(extend_mem_profile.LARGE_PHASE_BYTES)
        self.assertEqual(len(logs.output), 1)
        self.assertIn(
            "extend-mem-profile phase kda:extend alloc_delta=256MiB retained=0MiB",
            logs.output[0],
        )
        # Nothing was retained, so no allocator snapshot is requested.
        self.assertFalse(dump.called)

    def test_retaining_phase_requests_forensics_snapshot(self):
        alloc = FakeAllocator()
        with profiler(True), alloc.patched():
            extend_mem_profile.begin(4096)
            with (
                mock.patch.object(mem_forensics, "maybe_dump_memory_forensics") as dump,
                self.assertLogs(LOGGER, level="INFO") as logs,
            ):
                with extend_mem_profile.phase("mm-embed:image"):
                    alloc.alloc(300 * MiB)
        self.assertIn("alloc_delta=300MiB retained=300MiB", logs.output[0])
        dump.assert_called_once_with("retained-mm-embed:image")

    def test_phase_outside_begin_end_is_noop(self):
        alloc = FakeAllocator()
        with profiler(True), alloc.patched():
            with extend_mem_profile.phase("attn"):
                alloc.alloc(300 * MiB)
        self.assertEqual(extend_mem_profile._phase_peaks, {})
        self.assertEqual(alloc.reset_calls, 0)


if __name__ == "__main__":
    unittest.main()
