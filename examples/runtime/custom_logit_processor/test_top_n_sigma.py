"""Unit tests for the Top-n-sigma custom logit processor example.

Loads ``top_n_sigma.py`` from this directory via importlib so the real example
code is tested (not a copy). No server, no model loading. Run with ``pytest``.
"""

import importlib.util
import unittest
from pathlib import Path

import torch

from sglang.test.test_utils import CustomTestCase

_EXAMPLE_PATH = Path(__file__).resolve().parent / "top_n_sigma.py"

NEG = float("-inf")


def _load_processor_cls():
    spec = importlib.util.spec_from_file_location("top_n_sigma_example", _EXAMPLE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.TopNSigmaLogitProcessor


class TestTopNSigmaExample(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.processor = _load_processor_cls()()

    def _run(self, logits, params):
        return self.processor(logits.clone(), params)

    def test_argmax_invariant_and_masks(self):
        torch.manual_seed(0)
        logits = torch.randn(2, 1000)
        top = logits.argmax(dim=-1)
        out = self._run(logits, [{"top_n_sigma": 1.0}, {"top_n_sigma": 1.0}])
        self.assertTrue(torch.equal(top, out.argmax(dim=-1)))
        self.assertTrue((out == NEG).any())
        for i in range(2):
            self.assertTrue(out[i, top[i]].isfinite())

    def test_threshold_is_max_minus_n_std(self):
        row = torch.tensor([[10.0, 9.0, 0.0, -5.0]])
        out = self._run(row, [{"top_n_sigma": 0.5}])
        thr = 10.0 - 0.5 * float(row.std(dim=-1))
        self.assertTrue(torch.equal(row[0] < thr, out[0] == NEG))

    def test_std_zero_is_skipped(self):
        row = torch.full((1, 5), 2.0)
        self.assertTrue(torch.equal(self._run(row, [{"top_n_sigma": 1.0}]), row))

    def test_non_finite_row_is_skipped(self):
        row = torch.tensor([[1.0, float("nan"), 3.0, 4.0]])
        out = self._run(row, [{"top_n_sigma": 1.0}])
        self.assertFalse((out == NEG).any())
        self.assertTrue(torch.equal(out.nan_to_num(), row.nan_to_num()))

    def test_invalid_params_are_skipped(self):
        base = torch.randn(1, 50)
        for bad in (
            {"top_n_sigma": 0},
            {"top_n_sigma": -1.0},
            {"top_n_sigma": None},
            {"top_n_sigma": "x"},
            {"top_n_sigma": True},  # bool must be rejected (it is an int subclass)
            {},
            None,
        ):
            self.assertTrue(torch.equal(self._run(base, [bad]), base), msg=str(bad))

    def test_empty_param_list_unchanged(self):
        base = torch.randn(1, 50)
        self.assertTrue(torch.equal(self._run(base, []), base))
        self.assertTrue(torch.equal(self._run(base, None), base))

    def test_mixed_n_per_row(self):
        torch.manual_seed(1)
        batch = torch.randn(2, 200)
        out = self._run(batch, [{"top_n_sigma": 1.0}, {"top_n_sigma": 0}])
        self.assertTrue((out[0] == NEG).any())
        self.assertTrue(torch.equal(out[1], batch[1]))

    def test_n_near_zero_collapses_to_argmax(self):
        torch.manual_seed(3)
        logits = torch.randn(1, 5000)
        top = logits.argmax(dim=-1)
        out = self._run(logits, [{"top_n_sigma": 1e-9}])
        finite = torch.isfinite(out[0])
        self.assertEqual(int(finite.sum()), 1)
        self.assertTrue(bool(finite[top[0]]))

    def test_low_precision_input_is_handled(self):
        # In production, logits are upcast to fp32 before custom logit processors
        # run (LogitsProcessor._copy_logits_to_buffer -> Sampler), so std/amax are
        # always computed in fp32. This guards the processor's behavior even when
        # it is handed low-precision logits directly: a dominant max is preserved,
        # lower logits are masked, and the dtype is not silently changed.
        # A clearly-separated max keeps argmax deterministic under fp16/bf16 rounding.
        for dtype in (torch.float16, torch.bfloat16):
            row = torch.tensor([[8.0, 2.0, 1.0, 0.0, -3.0]], dtype=dtype)
            out = self._run(row, [{"top_n_sigma": 1.0}])
            self.assertEqual(out.dtype, dtype, msg=str(dtype))
            self.assertEqual(int(out.argmax(dim=-1)), 0, msg=str(dtype))
            self.assertTrue(out[0, 0].isfinite(), msg=str(dtype))
            self.assertTrue((out == NEG).any(), msg=str(dtype))


if __name__ == "__main__":
    unittest.main()
