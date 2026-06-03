"""
Unit tests for TokenizerManager.detokenize_top_logprobs_tokens.

Regression coverage for a crash where a per-position logprob value is a
multi-element torch.Tensor instead of the annotated List[float]. The old
truthiness check ``if token_logprobs_val[i]:`` raises::

    RuntimeError: Boolean value of Tensor with more than one value is ambiguous.

which propagates out of the detokenization handler, gets caught by
print_exception_wrapper, and SIGKILLs the whole (prefill) process. The fix
uses an explicit ``is not None`` test that matches the actual sentinel for
skipped positions and works for both lists and tensors.
"""

import unittest

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.tokenizer_manager import TokenizerManager

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _make_tokenizer_manager() -> TokenizerManager:
    """Create a bare TokenizerManager, bypassing __init__.

    detokenize_top_logprobs_tokens / detokenize_logprob_tokens only touch
    self.tokenizer when decode_to_text=True, so an uninitialised instance is
    sufficient for the decode_to_text=False paths exercised here.
    """
    return TokenizerManager.__new__(TokenizerManager)


class TestDetokenizeTopLogprobsTensor(CustomTestCase):
    def test_multi_element_tensor_value_does_not_crash(self):
        """A multi-element tensor position must be detokenized, not raise."""
        tm = _make_tokenizer_manager()

        val = [torch.tensor([-0.1, -0.2, -0.3])]
        idx = [[10, 20, 30]]

        ret = tm.detokenize_top_logprobs_tokens(val, idx, decode_to_text=False)

        self.assertEqual(len(ret), 1)
        self.assertEqual(
            ret[0],
            [(-0.1, 10, None), (-0.2, 20, None), (-0.3, 30, None)],
        )

    def test_none_position_yields_none(self):
        """None is the sentinel for skipped positions and must stay None."""
        tm = _make_tokenizer_manager()

        val = [None, torch.tensor([-0.5, -0.6])]
        idx = [None, [1, 2]]

        ret = tm.detokenize_top_logprobs_tokens(val, idx, decode_to_text=False)

        self.assertEqual(len(ret), 2)
        self.assertIsNone(ret[0])
        self.assertEqual(ret[1], [(-0.5, 1, None), (-0.6, 2, None)])

    def test_plain_list_values_still_work(self):
        """The ordinary List[float] path is unaffected by the fix."""
        tm = _make_tokenizer_manager()

        val = [[-0.1, -0.2]]
        idx = [[7, 8]]

        ret = tm.detokenize_top_logprobs_tokens(val, idx, decode_to_text=False)

        self.assertEqual(ret, [[(-0.1, 7, None), (-0.2, 8, None)]])


if __name__ == "__main__":
    unittest.main(verbosity=2)
