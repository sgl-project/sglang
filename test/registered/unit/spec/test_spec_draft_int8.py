"""Unit tests for the opt-in int8 weights of a speculative draft: no server, no GPU."""

import unittest

import torch

from sglang.srt.speculative.draft_int8 import (
    Int8WeightOnlyLinearMethod,
    apply_draft_int8_weights,
    quantize_int8_rowwise,
    wants_int8_weights,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _DenseMethod:
    def __init__(self) -> None:
        self.calls = 0

    def apply(self, layer, x, bias=None):
        self.calls += 1
        out = x.float() @ layer.weight.float().t()
        return (out if bias is None else out + bias).to(x.dtype)

    def process_weights_after_loading(self, layer):
        return "original"


class _Linear(torch.nn.Module):
    def __init__(self, n: int, k: int, dtype=torch.bfloat16) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(
            (torch.randn(n, k) * 0.02).to(dtype), requires_grad=False
        )
        self.quant_method = _DenseMethod()


class _Draft(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.big = _Linear(128, 512)
        self.odd_rows = _Linear(100, 512)
        self.small = _Linear(64, 256)
        self.fp32 = _Linear(128, 512, dtype=torch.float32)
        self.norm = torch.nn.LayerNorm(512)


class SpecDraftInt8Test(CustomTestCase):
    def test_quantizer_keeps_about_one_percent_of_error(self) -> None:
        torch.manual_seed(0)
        weight = (torch.randn(128, 512) * 0.02).to(torch.bfloat16)
        q, scale = quantize_int8_rowwise(weight)
        self.assertEqual(q.dtype, torch.int8)
        self.assertEqual(tuple(scale.shape), (128,))
        self.assertEqual(int(q.abs().max()), 127)
        back = q.float() * scale[:, None]
        rms = (back - weight.float()).pow(2).mean().sqrt()
        rms = (rms / weight.float().pow(2).mean().sqrt()).item()
        self.assertLess(rms, 0.015)

    def test_all_zero_rows_stay_finite(self) -> None:
        q, scale = quantize_int8_rowwise(torch.zeros(4, 256, dtype=torch.bfloat16))
        self.assertEqual(int(q.abs().max()), 0)
        self.assertTrue(torch.isfinite(scale).all().item())

    def test_only_large_tileable_half_precision_linears_are_wrapped(self) -> None:
        draft = _Draft()
        wrapped = apply_draft_int8_weights(
            draft, min_params=128 * 512, linear_base=_Linear
        )
        self.assertEqual(wrapped, 1)
        self.assertIsInstance(draft.big.quant_method, Int8WeightOnlyLinearMethod)
        for other in (draft.odd_rows, draft.small, draft.fp32):
            self.assertIsInstance(other.quant_method, _DenseMethod)
        self.assertFalse(wants_int8_weights(None))
        again = apply_draft_int8_weights(
            draft, min_params=128 * 512, linear_base=_Linear
        )
        self.assertEqual(again, 0)

    def test_unserved_steps_keep_the_dense_weight(self) -> None:
        draft = _Draft()
        apply_draft_int8_weights(draft, min_params=128 * 512, linear_base=_Linear)
        method = draft.big.quant_method
        dense = method.original
        self.assertEqual(method.nbytes, 128 * 512 + 128 * 4)
        self.assertEqual(method.process_weights_after_loading(draft.big), "original")

        x = torch.randn(8, 512).to(torch.bfloat16)
        reference = method.apply(draft.big, x)  # a CPU tensor: the dense path
        self.assertEqual(dense.calls, 1)

        # As on a device: the kernel serves up to 16 rows. Its arithmetic is
        # stood in for by the product on the de-quantized weights.
        method.serves = lambda rows: rows.shape[0] <= 16
        method.compute = lambda rows: (
            rows.float() @ (method.weight_int8.float() * method.scale[:, None]).t()
        ).to(rows.dtype)
        bias = torch.ones(128, dtype=torch.bfloat16)
        packed = method.apply(draft.big, x, bias)
        self.assertEqual(dense.calls, 1)
        self.assertEqual(packed.dtype, x.dtype)
        err = (packed.float() - 1 - reference.float()).abs().max()
        self.assertLess((err / reference.float().abs().max()).item(), 0.05)

        method.apply(draft.big, torch.randn(40, 512).to(torch.bfloat16))
        self.assertEqual(dense.calls, 2)


if __name__ == "__main__":
    unittest.main()
