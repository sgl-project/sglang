"""Unit tests for the Top-nσ custom logit processor example.

Loads ``top_n_sigma.py`` from this directory via importlib so the real example
code is exercised (not a copy). No server, no model loading. Run with ``pytest``.
"""

import importlib.util
import unittest
from pathlib import Path

import torch

from sglang.test.test_utils import CustomTestCase

_EXAMPLE_PATH = Path(__file__).resolve().parent / "top_n_sigma.py"
NEG = float("-inf")


def _load_example():
    spec = importlib.util.spec_from_file_location("top_n_sigma_example", _EXAMPLE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestTopNSigmaExample(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        module = _load_example()
        cls.processor = module.TopNSigmaLogitProcessor()

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
        # All-finite row: threshold uses the unbiased (ddof=1) std of the row.
        row = torch.tensor([[10.0, 9.0, 0.0, -5.0]])
        out = self._run(row, [{"top_n_sigma": 0.5}])
        thr = 10.0 - 0.5 * float(row.std(dim=-1))
        self.assertTrue(torch.equal(row[0] < thr, out[0] == NEG))

    def test_std_zero_is_skipped(self):
        row = torch.full((1, 5), 2.0)
        self.assertTrue(torch.equal(self._run(row, [{"top_n_sigma": 1.0}]), row))

    def test_inf_masked_tokens_still_truncate(self):
        # Differentiator vs. an all-finite guard: a row carrying -inf (vocab
        # padding / grammar masks) is NOT skipped. max/std are taken over the
        # finite logits only; the row is still truncated and the pre-existing
        # -inf stays masked.
        row = torch.tensor([[10.0, 9.0, 1.0, NEG]])
        out = self._run(row, [{"top_n_sigma": 1.0}])
        self.assertFalse(torch.equal(out, row))  # active, not passed through
        self.assertTrue(out[0, 0].isfinite() and out[0, 1].isfinite())
        self.assertTrue((out[0, 2] == NEG) and (out[0, 3] == NEG))
        self.assertTrue(out[0].argmax() == row[0, :3].argmax())  # argmax kept

    def test_single_finite_logit_is_skipped(self):
        # Only one finite logit -> std undefined -> row left unchanged.
        row = torch.tensor([[5.0, NEG, NEG, NEG]])
        self.assertTrue(torch.equal(self._run(row, [{"top_n_sigma": 1.0}]), row))

    def test_invalid_params_are_skipped(self):
        base = torch.randn(1, 50)
        for bad in (
            {"top_n_sigma": 0},
            {"top_n_sigma": -1.0},
            {"top_n_sigma": None},
            {"top_n_sigma": "x"},
            {"top_n_sigma": True},  # bool is an int subclass -> must be rejected
            {},
            None,
        ):
            with self.subTest(bad=bad):
                self.assertTrue(torch.equal(self._run(base, [bad]), base))

    def test_empty_param_list_unchanged(self):
        base = torch.randn(1, 50)
        self.assertTrue(torch.equal(self._run(base, []), base))
        self.assertTrue(torch.equal(self._run(base, None), base))

    def test_mixed_n_per_row(self):
        torch.manual_seed(1)
        batch = torch.randn(2, 200)
        out = self._run(batch, [{"top_n_sigma": 1.0}, {"top_n_sigma": 0}])
        self.assertTrue((out[0] == NEG).any())  # row 0 truncated
        self.assertTrue(torch.equal(out[1], batch[1]))  # row 1 untouched

    def test_input_not_mutated(self):
        # Docstring promise: the input tensor is left unmodified; a new tensor
        # is returned. Pass ``base`` directly (not via _run's clone) to check.
        base = torch.randn(1, 100)
        original = base.clone()
        out = self.processor(base, [{"top_n_sigma": 1.0}])
        self.assertTrue(torch.equal(base, original))  # input untouched
        self.assertFalse(torch.equal(out, base))  # something was masked


if __name__ == "__main__":
    unittest.main()
