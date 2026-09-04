"""Unit tests for the DSA indexer's MQA-logits chunk budget.

``Indexer._get_topk_ragged`` derives ``max_rows`` by dividing the budget by the
logits row size, so **the budget is the chunk allocation size**. Two properties
therefore matter and are pinned here:

* the free-memory reading behind it is taken fresh, so a reading from when
  memory was plentiful cannot keep authorizing that allocation forever;
* it never exceeds ``SGLANG_DSA_MQA_LOGITS_MAX_CHUNK_BYTES`` -- the part a live
  reading cannot give you, since free bytes may be too fragmented to serve one
  large contiguous request, and since a fraction of a nearly-empty device would
  authorize a tens-of-GiB allocation.

The numbers are from a real crash: a DSA + EAGLE-MTP prefill on a 178.35 GiB
device where the cached snapshot authorized a 2.63 GiB chunk while 2.20 GiB was
free, and the scheduler died inside ``deep_gemm.fp8_mqa_logits``.
"""

import types
import unittest
from unittest import mock

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=15, stage="base-b", runner_config="1-gpu-small")

GiB = 1 << 30
MiB = 1 << 20

TOTAL_MEM = int(178.35 * GiB)
FREE_AT_SNAPSHOT = int(13.15 * GiB)
FREE_AT_CRASH = int(2.20 * GiB)
MEM_FRACTION_STATIC = 0.92
FRAC = 0.2
CRASH_ALLOC = int(2.63 * GiB)
CAP = 256 * MiB
CACHE_S = 30.0  # only the reuse tests opt into a window
DEVICE = 0

# The crashing call: a 32768-token prefill chunk with no cached prefix, i.e. a
# 32768 x 32768 fp32 logits matrix (4.00 GiB).
Q_OFFSET = 32768
K_OFFSET = 32768


class _FreeMem:
    """Scriptable ``torch.cuda.mem_get_info`` that counts its calls."""

    def __init__(self, free):
        self.free = free
        self.calls = 0

    def __call__(self, device_index=None):
        self.calls += 1
        return (self.free, TOTAL_MEM)


class TestDSAMqaLogitsBudget(CustomTestCase):
    def setUp(self):
        import sglang.srt.layers.attention.dsa.dsa_indexer as dsa_mod
        from sglang.srt.environ import envs

        self.dsa_mod = dsa_mod
        self.envs = envs
        self.Indexer = dsa_mod.Indexer
        # No __init__: only class attributes and the budget methods are needed.
        self.obj = self.Indexer.__new__(self.Indexer)

        self.Indexer._mqa_logits_budget_bytes.clear()
        self.Indexer._mqa_logits_budget_at.clear()
        self.addCleanup(self.Indexer._mqa_logits_budget_bytes.clear)
        self.addCleanup(self.Indexer._mqa_logits_budget_at.clear)

        envs.SGLANG_DSA_MQA_LOGITS_FREE_MEM_FRACTION.set(FRAC)
        envs.SGLANG_DSA_MQA_LOGITS_MAX_CHUNK_BYTES.set(CAP)
        envs.SGLANG_DSA_MQA_LOGITS_BUDGET_CACHE_S.set(0.0)  # live, the default
        self.addCleanup(envs.SGLANG_DSA_MQA_LOGITS_FREE_MEM_FRACTION.clear)
        self.addCleanup(envs.SGLANG_DSA_MQA_LOGITS_MAX_CHUNK_BYTES.clear)
        self.addCleanup(envs.SGLANG_DSA_MQA_LOGITS_BUDGET_CACHE_S.clear)

        # Deterministic clock, scoped to the module under test.
        self.now = 1000.0
        clock = mock.patch.object(
            dsa_mod, "time", types.SimpleNamespace(monotonic=lambda: self.now)
        )
        clock.start()
        self.addCleanup(clock.stop)

        props = mock.patch.object(
            dsa_mod.torch.cuda,
            "get_device_properties",
            lambda idx: types.SimpleNamespace(total_memory=TOTAL_MEM),
        )
        props.start()
        self.addCleanup(props.stop)

        self._set_mem_fraction_static(MEM_FRACTION_STATIC)
        capture = mock.patch.object(dsa_mod, "get_is_capture_mode", lambda: False)
        capture.start()
        self.addCleanup(capture.stop)

    def _set_mem_fraction_static(self, value):
        patcher = mock.patch.object(
            self.dsa_mod,
            "get_schedule",
            lambda: types.SimpleNamespace(mem_fraction_static=value),
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def _budget(self, free):
        fm = _FreeMem(free)
        with mock.patch.object(self.dsa_mod.torch.cuda, "mem_get_info", fm):
            return self.obj._get_mqa_logits_budget_bytes(DEVICE), fm

    def test_premise_uncapped_snapshot_would_authorize_the_crashing_chunk(self):
        """Without the cap, this snapshot authorizes the 2.63 GiB allocation."""
        self.envs.SGLANG_DSA_MQA_LOGITS_MAX_CHUNK_BYTES.set(0)
        budget, _ = self._budget(FREE_AT_SNAPSHOT)
        self.assertGreaterEqual(budget, CRASH_ALLOC)
        self.assertGreater(budget, FREE_AT_CRASH)

    def test_budget_respects_the_absolute_cap(self):
        budget, _ = self._budget(FREE_AT_SNAPSHOT)
        self.assertLessEqual(budget, CAP)
        self.assertLess(budget, FREE_AT_CRASH)

    def test_chunk_rows_fit_the_cap(self):
        need_chunk, budget = self.obj._should_chunk_mqa_logits(
            Q_OFFSET, K_OFFSET, DEVICE
        )
        self.assertTrue(need_chunk, "a 4 GiB logits matrix must be chunked")
        # Mirror _get_topk_ragged's own sizing.
        bytes_per_row = K_OFFSET * self.obj._MQA_LOGITS_BYTES_PER_ELEM
        max_rows = min(max(1, budget // bytes_per_row), Q_OFFSET)
        self.assertLessEqual(max_rows * bytes_per_row, CAP)
        self.assertGreaterEqual(max_rows, 1)

    def test_small_matrices_still_skip_the_budget_entirely(self):
        self.assertEqual(
            self.obj._should_chunk_mqa_logits(1000, 1000, DEVICE), (False, 0)
        )

    def test_default_takes_a_fresh_reading_every_call(self):
        first, fm1 = self._budget(FREE_AT_SNAPSHOT)
        self.assertEqual(fm1.calls, 1)
        second, fm2 = self._budget(FREE_AT_CRASH)
        self.assertEqual(fm2.calls, 1, "the default must not reuse a reading")
        self.assertLessEqual(first, CAP)
        self.assertLessEqual(second, int(FREE_AT_CRASH * FRAC))

    def test_reading_is_reused_inside_an_explicit_window(self):
        self.envs.SGLANG_DSA_MQA_LOGITS_BUDGET_CACHE_S.set(CACHE_S)
        first, _ = self._budget(FREE_AT_SNAPSHOT)
        self.now += CACHE_S - 1.0
        second, fm = self._budget(FREE_AT_CRASH)
        self.assertEqual(first, second)
        self.assertEqual(fm.calls, 0, "mem_get_info syncs the host; do not re-query")

    def test_window_expires_and_the_budget_shrinks(self):
        self.envs.SGLANG_DSA_MQA_LOGITS_BUDGET_CACHE_S.set(CACHE_S)
        first, _ = self._budget(FREE_AT_SNAPSHOT)
        self.now += CACHE_S + 1.0
        second, fm = self._budget(int(0.5 * GiB))
        self.assertEqual(fm.calls, 1)
        self.assertLess(second, first)
        self.assertLessEqual(second, int(0.5 * GiB * FRAC))

    def test_capture_mode_neither_queries_nor_caches(self):
        with mock.patch.object(self.dsa_mod, "get_is_capture_mode", lambda: True):
            budget, fm = self._budget(FREE_AT_SNAPSHOT)
        self.assertEqual(fm.calls, 0)
        self.assertLessEqual(budget, CAP)
        _, fm2 = self._budget(FREE_AT_SNAPSHOT)
        self.assertEqual(fm2.calls, 1, "capture must not seed the cache")

    def test_both_knobs_off_restore_the_previous_behaviour(self):
        self.envs.SGLANG_DSA_MQA_LOGITS_MAX_CHUNK_BYTES.set(0)
        self.envs.SGLANG_DSA_MQA_LOGITS_BUDGET_CACHE_S.set(-1.0)
        first, _ = self._budget(FREE_AT_SNAPSHOT)
        self.now += 10_000.0
        second, fm = self._budget(FREE_AT_CRASH)
        self.assertEqual(first, int(FREE_AT_SNAPSHOT * FRAC))
        self.assertEqual(first, second)
        self.assertEqual(fm.calls, 0)

    def test_budget_floor_keeps_the_chunk_loop_progressing(self):
        self.envs.SGLANG_DSA_MQA_LOGITS_MAX_CHUNK_BYTES.set(1)
        budget, _ = self._budget(FREE_AT_SNAPSHOT)
        self.assertGreaterEqual(budget, 1)
        # One row of a 512k-token context is 1.95 MiB, far above the cap.
        self.assertGreaterEqual(max(1, budget // (512_000 * 4)), 1)

    def test_no_free_memory_still_yields_a_positive_budget(self):
        budget, _ = self._budget(0)
        self.assertGreaterEqual(budget, 1)

    def test_unset_mem_fraction_static_falls_back_to_the_total_mem_cap(self):
        self._set_mem_fraction_static(None)
        budget, _ = self._budget(FREE_AT_SNAPSHOT)
        self.assertLessEqual(budget, CAP)


if __name__ == "__main__":
    unittest.main()
