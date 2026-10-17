"""Guard: ``fused_gdn_gating`` must store the gate's ``beta`` in fp32.

``beta_output`` is allocated as an fp32 buffer, but the kernel narrowed
``sigmoid(b)`` to the *input* dtype before storing it -- so the buffer held
bf16-precision values. That tensor is what ``kernel_dispatcher.extend`` (and the
non-packed decode path, and the NPU backend) feeds into the delta-rule update of
the persistent SSM state, where beta scales the state every step. Rounding it
there loses up to bf16 epsilon (~0.39%) per step on a state that is read,
scaled and written back each token.

The sibling decode kernels in ``fused_recurrent.py`` and
``fused_recurrent_linear_replayssm.py`` carried the same round-trip on the
sigmoid expression itself and are fixed separately (sgl-project/sglang#38977).

The kernel cannot run on CPU, so this is a source scan plus a numerical
demonstration of the error class. The scan is the regression guard; the
numerical part exists so a reader can see the magnitude rather than take it on
faith.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import re
import unittest
from pathlib import Path

import torch

from sglang.test.test_utils import CustomTestCase

_KERNEL = (
    Path(__file__).resolve().parents[5]
    / "python"
    / "sglang"
    / "kernels"
    / "ops"
    / "attention"
    / "fla"
    / "fused_gdn_gating.py"
)


class TestGdnGatingBetaStaysFp32(CustomTestCase):
    def test_beta_output_store_is_not_narrowed(self):
        src = _KERNEL.read_text()
        narrowed = re.search(
            r"tl\.store\(\s*beta_output[^)]*\.to\([^)]*element_ty\s*\)", src
        )
        self.assertIsNone(
            narrowed,
            "fused_gdn_gating narrows beta before storing it into the fp32 "
            f"beta_output buffer: {narrowed and narrowed.group(0)}",
        )

    def test_beta_output_buffer_is_fp32(self):
        # The store must not narrow, and the buffer it stores into must stay
        # fp32 -- narrowing on one side or the other is the same defect.
        src = _KERNEL.read_text()
        self.assertRegex(
            src,
            r"beta_output\s*=\s*torch\.empty\([^)]*dtype=torch\.float32",
            "beta_output must be allocated as fp32",
        )

    def test_bf16_round_trip_of_sigmoid_is_a_real_loss(self):
        # The magnitude the guard protects against, on CPU.
        b = torch.randn(4096, dtype=torch.bfloat16)
        exact = torch.sigmoid(b.float())
        rounded = torch.sigmoid(b.float()).to(torch.bfloat16).float()

        max_rel = ((rounded - exact).abs() / exact.abs()).max().item()
        # bf16 has 8 explicit mantissa bits, so the round-trip is ~2**-9
        # relative; assert a band rather than one exact value.
        self.assertGreater(max_rel, 1e-3)
        self.assertLess(max_rel, 1e-2)


if __name__ == "__main__":
    unittest.main()
