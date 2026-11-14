import sys
import unittest
from types import ModuleType
from unittest import mock

import torch

from sglang.kernels.ops.speculative.dspark import dspark_verify_window
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestNpuBuildOutTokens(CustomTestCase):
    def test_rejected_drafts_from_native_kernel_are_zeroed(self):
        """The NPU kernel's legacy suffix must not reach downstream gathers."""
        for gamma, lengths in ((0, [0]), (3, []), (3, [0, 1, 3])):
            for dtype in (torch.int32, torch.int64):
                with self.subTest(gamma=gamma, lengths=lengths, dtype=dtype):
                    bs = len(lengths)
                    draft_tokens = torch.arange(bs * gamma).reshape(bs, gamma) + 2**40
                    bonus_tokens = torch.arange(bs) - 2**40
                    # Preserve the native API's support for strided length inputs.
                    correct_len = torch.tensor(
                        [n for length in lengths for n in (length, -1)], dtype=dtype
                    )[::2]
                    inputs = dict(
                        draft_tokens=draft_tokens,
                        correct_len=correct_len,
                        bonus=bonus_tokens,
                        verify_num_draft_tokens=gamma + 1,
                        gamma=gamma,
                    )
                    # sgl-kernel-npu 2026.11.0.post1 copies every draft, including
                    # rejected ones, then inserts the bonus at correct_len.
                    native_output = torch.zeros((bs, gamma + 1), dtype=torch.int64)
                    native_output[:, :gamma] = draft_tokens
                    native_output.scatter_(
                        1, correct_len.long()[:, None], bonus_tokens[:, None]
                    )
                    sample = ModuleType("sgl_kernel_npu.operators.sample")
                    sample.build_out_tokens_triton = mock.Mock(
                        return_value=native_output
                    )
                    with (
                        mock.patch.dict(sys.modules, {sample.__name__: sample}),
                        mock.patch.object(dspark_verify_window, "_is_npu", True),
                    ):
                        actual = dspark_verify_window.BuildOutTokens.triton(**inputs)

                    expected = dspark_verify_window.BuildOutTokens.torch(**inputs)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    sample.build_out_tokens_triton.assert_called_once_with(**inputs)


if __name__ == "__main__":
    unittest.main()
