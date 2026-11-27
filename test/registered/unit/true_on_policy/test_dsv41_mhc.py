"""TOP DeepSeek-V4.1 mHC: what the single program refuses instead of running another kernel."""

import unittest
from types import SimpleNamespace

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.models import deepseek_v4_mhc as mhc
from sglang.srt.true_on_policy import dsv41_mhc

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _cfg():
    return mhc.HcConfig(
        mult=4,
        sinkhorn_iters=20,
        eps=1e-6,
        rms_eps=1e-20,
        hidden=8,
        pre_from_prev=True,
        cp_prefill=False,
        top=True,
    )


class TestDsv41Mhc(CustomTestCase):
    def test_stats_refuse_fp32_streams(self):
        x = torch.zeros(2, 4, 8)
        with self.assertRaisesRegex(NotImplementedError, "BF16 streams"):
            dsv41_mhc.mix_stats(
                x,
                torch.zeros(24, 32),
                torch.ones(3),
                torch.zeros(24),
                mult=4,
                sinkhorn_iters=20,
                rms_eps=1e-20,
                eps=1e-6,
            )

    def test_combine_refuses_a_bf16_pre(self):
        x = torch.zeros(2, 4, 8, dtype=torch.bfloat16)
        with self.assertRaisesRegex(NotImplementedError, "FP32 pre"):
            dsv41_mhc.collapse(x, torch.zeros(2, 4, dtype=torch.bfloat16))

    def test_combine_emits_no_mxfp8_input(self):
        """A quantized combine would hand the next projection a different program's input."""
        hc = SimpleNamespace(cfg=_cfg(), norm=None)
        state = mhc.HcState(
            torch.zeros(2, 4, 8, dtype=torch.bfloat16), torch.ones(2, 4)
        )
        with self.assertRaisesRegex(NotImplementedError, "MXFP8"):
            mhc._combine(hc, state, quantized=[])

    def test_no_stats_side_stream(self):
        batch = SimpleNamespace(
            forward_mode=SimpleNamespace(
                is_decode=lambda: True, is_target_verify=lambda: False
            )
        )
        self.assertFalse(mhc.use_stats_stream(_cfg(), batch, torch.zeros(1, 4, 8)))


if __name__ == "__main__":
    unittest.main()
