"""DFlash grouped conv: a block's taps must never read another request's rows."""

import contextlib
import unittest

import torch
import torch.nn.functional as F

from sglang.srt.models.dflash import DFlashGroupedConv, _grouped_conv
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

HIDDEN, GROUP, BLOCK = 64, 16, 8
NUM_GROUPS = HIDDEN // GROUP


def _inputs(bs, taps, dtype, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(bs * BLOCK, HIDDEN, generator=g).to(dtype)
    delta = (0.5 * torch.randn(bs * BLOCK, taps, NUM_GROUPS, generator=g)).to(dtype)
    base = torch.randn(taps, HIDDEN, generator=g).to(dtype)
    return x, delta, base


def _stance(compiled):
    # The engine calls the torch.compile'd _grouped_conv (inductor here); the
    # eager stance runs its original Python instead.
    if compiled:
        return contextlib.nullcontext()
    return torch.compiler.set_stance("force_eager")


def _conv(x, delta, base, taps, compiled=False):
    with _stance(compiled):
        return _grouped_conv(x, delta, base, BLOCK, NUM_GROUPS, GROUP, taps)


def _multiplicative_mask_conv(x, delta, base, taps):
    # The previous formulation, kept to show finite outputs are unchanged.
    blocks = x.unflatten(-1, (NUM_GROUPS, GROUP))
    coefficients = base.view(1, taps, NUM_GROUPS, GROUP) + delta.unsqueeze(-1)
    out = coefficients[:, 0] * blocks
    position = torch.arange(x.shape[0]) % BLOCK
    for tap in range(1, taps):
        shifted = F.pad(blocks[:-tap], (0, 0, 0, 0, tap, 0))
        out = out + coefficients[:, tap] * shifted * (position >= tap).view(-1, 1, 1)
    return out.flatten(-2)


class TestDFlashGroupedConv(CustomTestCase):
    def _check_nonfinite_rows_stay_in_their_request(self, compiled):
        for taps in (2, 3):
            for bad in (float("nan"), float("inf"), float("-inf")):
                for row in range(BLOCK):
                    with self.subTest(taps=taps, bad=bad, row=row):
                        x, delta, base = _inputs(bs=3, taps=taps, dtype=torch.bfloat16)
                        clean = _conv(x, delta, base, taps, compiled)
                        clean = clean.view(3, BLOCK, HIDDEN)
                        x[row, 5] = bad  # request 0
                        out = _conv(x, delta, base, taps, compiled)
                        out = out.view(3, BLOCK, HIDDEN)
                        torch.testing.assert_close(out[1:], clean[1:], rtol=0, atol=0)
                        bad_rows = (~torch.isfinite(out[0])).any(-1).nonzero()
                        expected = torch.arange(row, min(row + taps, BLOCK))
                        self.assertEqual(bad_rows.flatten().tolist(), expected.tolist())

    def _check_module_prepare_and_finish_do_not_leak(self, compiled):
        torch.manual_seed(0)
        conv = DFlashGroupedConv(HIDDEN, BLOCK, 2, GROUP).to(torch.bfloat16)
        x = torch.randn(2 * BLOCK, HIDDEN).to(torch.bfloat16)
        y = torch.randn(2 * BLOCK, HIDDEN).to(torch.bfloat16)
        with torch.no_grad(), _stance(compiled):
            ref_x, kernel = conv.prepare(x)
            ref_y = conv.finish(y, kernel)
            x_bad, y_bad = x.clone(), y.clone()
            x_bad[BLOCK - 1] = float("nan")
            y_bad[BLOCK - 1] = float("inf")
            out_x, _ = conv.prepare(x_bad)
            out_y = conv.finish(y_bad, kernel)
        torch.testing.assert_close(out_x[BLOCK:], ref_x[BLOCK:], rtol=0, atol=0)
        torch.testing.assert_close(out_y[BLOCK:], ref_y[BLOCK:], rtol=0, atol=0)

    def test_nonfinite_rows_stay_in_their_request(self):
        self._check_nonfinite_rows_stay_in_their_request(compiled=False)

    def test_compiled_nonfinite_rows_stay_in_their_request(self):
        self._check_nonfinite_rows_stay_in_their_request(compiled=True)

    def test_module_prepare_and_finish_do_not_leak(self):
        self._check_module_prepare_and_finish_do_not_leak(compiled=False)

    def test_compiled_module_prepare_and_finish_do_not_leak(self):
        self._check_module_prepare_and_finish_do_not_leak(compiled=True)

    def test_finite_output_unchanged(self):
        for dtype in (torch.float32, torch.bfloat16):
            for taps in (2, 3):
                for seed in range(4):
                    with self.subTest(dtype=dtype, taps=taps, seed=seed):
                        x, delta, base = _inputs(5, taps, dtype, seed)
                        # torch.equal treats -0.0 == 0.0: the old mask could
                        # produce -0.0 where the new one produces +0.0.
                        self.assertTrue(
                            torch.equal(
                                _conv(x, delta, base, taps),
                                _multiplicative_mask_conv(x, delta, base, taps),
                            )
                        )


if __name__ == "__main__":
    unittest.main()
