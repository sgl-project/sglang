"""Tests for sglang.srt.kv_canary.sweep_plan_builder: _swa_translate pure-tensor helper."""

# Import blocker on this machine (macOS + Python 3.11):
# sweep_plan_builder -> radix_cache_walker -> radix_cache -> memory_pool ->
# dsa/index_buf_accessor -> quantization -> awq -> moe -> moe_runner/runner.py
# uses @torch.compile which triggers torch._inductor which fails with:
#   TypeError: unsupported operand type(s) for |: 'module' and 'type'
# This is a torch/Python 3.11 version mismatch on macOS. Linux CI should be unaffected.
# We mock the problematic imports in sys.modules at module level before the import occurs.

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import sys
import types
import unittest

import torch

# ---- Patch sys.modules so that sweep_plan_builder's deep import chain succeeds ----
# Only install mocks for modules not already loaded (avoids clobbering a real import).
if "sglang.srt.kv_canary.radix_cache_walker" not in sys.modules:
    _rw_stub = types.ModuleType("sglang.srt.kv_canary.radix_cache_walker")
    _rw_stub.walk_radix_cache_for_canary = (
        None  # sentinel; sweep_plan_builder imports this name
    )
    sys.modules["sglang.srt.kv_canary.radix_cache_walker"] = _rw_stub

if "sglang.srt.mem_cache.base_prefix_cache" not in sys.modules:
    _bpc_stub = types.ModuleType("sglang.srt.mem_cache.base_prefix_cache")
    _bpc_stub.BasePrefixCache = None
    sys.modules["sglang.srt.mem_cache.base_prefix_cache"] = _bpc_stub

# Also ensure VerifyPlan exists on the jit_kernel verify module if not already loaded.
_JIT_VERIFY = "sglang.jit_kernel.kv_canary.verify"
if _JIT_VERIFY not in sys.modules:
    _jit_mod = types.ModuleType(_JIT_VERIFY)
    _jit_mod.VerifyPlan = None
    sys.modules[_JIT_VERIFY] = _jit_mod

from sglang.srt.kv_canary.sweep_plan_builder import _swa_translate  # noqa: E402
from sglang.test.test_utils import CustomTestCase


class TestSwaTranslateEmpty(CustomTestCase):
    def test_empty_indices_returns_empty(self):
        indices = torch.tensor([], dtype=torch.int64)
        lut = torch.tensor([10, 20, 30], dtype=torch.int64)
        result = _swa_translate(indices=indices, lut=lut)
        self.assertEqual(result.numel(), 0)

    def test_empty_indices_same_object_returned(self):
        indices = torch.tensor([], dtype=torch.int64)
        lut = torch.tensor([10, 20, 30], dtype=torch.int64)
        result = _swa_translate(indices=indices, lut=lut)
        self.assertIs(result, indices)


class TestSwaTranslateNormal(CustomTestCase):
    def test_single_element(self):
        indices = torch.tensor([2], dtype=torch.int64)
        lut = torch.tensor([10, 20, 30], dtype=torch.int64)
        result = _swa_translate(indices=indices, lut=lut)
        self.assertEqual(result.tolist(), [30])

    def test_sequential_lookup(self):
        indices = torch.tensor([0, 1, 2], dtype=torch.int64)
        lut = torch.tensor([10, 20, 30], dtype=torch.int64)
        result = _swa_translate(indices=indices, lut=lut)
        self.assertEqual(result.tolist(), [10, 20, 30])

    def test_repeated_indices(self):
        indices = torch.tensor([1, 1, 0], dtype=torch.int64)
        lut = torch.tensor([100, 200, 300], dtype=torch.int64)
        result = _swa_translate(indices=indices, lut=lut)
        self.assertEqual(result.tolist(), [200, 200, 100])

    def test_output_dtype_is_int64(self):
        indices = torch.tensor([0, 1], dtype=torch.int32)
        lut = torch.tensor([5, 6, 7], dtype=torch.int64)
        result = _swa_translate(indices=indices, lut=lut)
        self.assertEqual(result.dtype, torch.int64)


class TestSwaTranslateAnchorPassthrough(CustomTestCase):
    """Negative indices are anchors and must be returned unchanged (not looked up)."""

    def test_negative_minus_one_is_preserved(self):
        indices = torch.tensor([-1], dtype=torch.int64)
        lut = torch.tensor([99, 88, 77], dtype=torch.int64)
        result = _swa_translate(indices=indices, lut=lut)
        self.assertEqual(result.tolist(), [-1])

    def test_negative_minus_two_is_preserved(self):
        indices = torch.tensor([-2], dtype=torch.int64)
        lut = torch.tensor([99, 88, 77], dtype=torch.int64)
        result = _swa_translate(indices=indices, lut=lut)
        self.assertEqual(result.tolist(), [-2])

    def test_mixed_negative_and_positive(self):
        # [-1, 0, 2] with lut=[10,20,30]: -1 stays -1, 0->10, 2->30
        indices = torch.tensor([-1, 0, 2], dtype=torch.int64)
        lut = torch.tensor([10, 20, 30], dtype=torch.int64)
        result = _swa_translate(indices=indices, lut=lut)
        self.assertEqual(result.tolist(), [-1, 10, 30])

    def test_all_negative_all_preserved(self):
        indices = torch.tensor([-1, -3, -7], dtype=torch.int64)
        lut = torch.tensor([99, 88, 77, 66], dtype=torch.int64)
        result = _swa_translate(indices=indices, lut=lut)
        self.assertEqual(result.tolist(), [-1, -3, -7])

    def test_zero_positive_index_maps_correctly(self):
        # index 0 is NOT an anchor (only < 0 is anchor)
        indices = torch.tensor([0], dtype=torch.int64)
        lut = torch.tensor([42], dtype=torch.int64)
        result = _swa_translate(indices=indices, lut=lut)
        self.assertEqual(result.tolist(), [42])


class TestSwaTranslateLutDataTypes(CustomTestCase):
    def test_int32_lut_is_cast_to_int64(self):
        indices = torch.tensor([0, 1], dtype=torch.int64)
        lut = torch.tensor([5, 6], dtype=torch.int32)
        result = _swa_translate(indices=indices, lut=lut)
        self.assertEqual(result.tolist(), [5, 6])
        self.assertEqual(result.dtype, torch.int64)


if __name__ == "__main__":
    unittest.main(verbosity=3)
