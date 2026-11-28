"""Small CPU-side checks for the NPU enablement bits used by Kimi K3.1 DFLASH:

- the empty-input quant matmul shortcut (idle DP ranks produce zero-row
  linears during verify graph replay) returns a ``[0, N]`` tensor with the
  GEMM's output shape and dtype;
- the sampling presence penalty applies the bool mask after promoting it,
  since ``aclnnInplaceAddcmul`` rejects ``DT_BOOL`` tensors.
"""

import unittest

import torch

from sglang.srt.sampling.penalties import _compute_logit_penalties
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")

try:
    from sglang.srt.hardware_backend.npu.quantization.linear_method_npu import (
        _empty_quant_matmul_output,
    )

    _HAS_NPU_QUANT_IMPORT = True
except Exception:
    _HAS_NPU_QUANT_IMPORT = False


@unittest.skipUnless(_HAS_NPU_QUANT_IMPORT, "NPU quant method import unavailable")
class TestEmptyQuantMatmulOutput(CustomTestCase):
    def test_zero_row_input_returns_zero_row_output(self):
        class _Layer:
            weight = torch.nn.Parameter(torch.zeros(16, 7))

        x = torch.zeros((0, 16), dtype=torch.bfloat16)
        out = _empty_quant_matmul_output(x, _Layer(), torch.bfloat16)
        self.assertEqual(out.shape, (0, 7))
        self.assertEqual(out.dtype, torch.bfloat16)


class TestBoolPresencePenalty(CustomTestCase):
    def test_bool_presence_mask_matches_manual_penalty(self):
        vocab = 4
        counts = torch.tensor([[1.0, 2.0, 0.0, 0.5]])
        presence = torch.tensor([[0.3, 0.3, 0.3, 0.3]])

        additive, _ = _compute_logit_penalties(
            counts=counts,
            lens=None,
            frequency=None,
            presence=presence,
            repetition=None,
            min_new_tokens=None,
            stop_token_ids=None,
        )
        present = counts > 0
        expected = torch.zeros(1, vocab)
        expected[present] -= presence[present]
        self.assertTrue(torch.allclose(additive, expected))


if __name__ == "__main__":
    unittest.main()
