"""K3 attn-res aggregate_stream: pre-norm mixture for dspark aux capture.

Pins:
- triton mix pair (aggregate_stream) vs the eager reference
  (aggregate_stream_torch) across nvb 1..8, small/large T, and a bank with
  NB > nvb rows (runtime-stride addressing);
- nvb == 0 passthrough (same tensor object);
- _aggregate_fused == out_norm(mixture) -- guards the _mix_fused split that
  the serving "fused" mode now routes through;
- read-only inputs.
"""

import unittest

import torch

from sglang.srt.layers.attn_residual import (
    _aggregate_fused,
    _mix_fused,
    aggregate_stream,
    aggregate_stream_torch,
)
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import ReplicatedLinear
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, stage="base-b", runner_config="1-gpu-small")

_H = 7168
_EPS = 1e-6


def _make_modules(seed: int):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    norm = RMSNorm(_H, eps=_EPS).to(device="cuda", dtype=torch.bfloat16)
    proj = ReplicatedLinear(
        _H, 1, bias=False, params_dtype=torch.bfloat16, quant_config=None
    ).to(device="cuda")
    with torch.no_grad():
        norm.weight.copy_(1 + 0.1 * torch.randn(_H, generator=gen, device="cuda"))
        proj.weight.copy_(torch.randn(1, _H, generator=gen, device="cuda") * _H**-0.5)
    return proj, norm


def _make_inputs(T: int, seed: int, num_bank_slots: int = 8):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    prefix = torch.randn(T, _H, generator=gen, device="cuda").to(torch.bfloat16)
    bank = torch.randn(T, num_bank_slots, _H, generator=gen, device="cuda").to(
        torch.bfloat16
    )
    return prefix, bank


class TestAggregateStream(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is not available")
        cls.proj, cls.norm = _make_modules(seed=0)

    def test_kernel_matches_eager(self):
        for nvb in range(1, 9):
            for T in (1, 5, 257):
                with self.subTest(nvb=nvb, T=T):
                    prefix, bank = _make_inputs(T, seed=8 * T + nvb)
                    prefix_ref, bank_ref = prefix.clone(), bank.clone()

                    # _mix_fused directly: pins the triton pair regardless
                    # of the capability dispatch.
                    out = _mix_fused(prefix, bank, nvb, self.proj, self.norm)
                    ref = aggregate_stream_torch(
                        prefix, bank, nvb, self.proj, self.norm
                    )
                    torch.testing.assert_close(
                        out.float(), ref.float(), rtol=2e-2, atol=4e-2
                    )

                    self.assertTrue(torch.equal(prefix, prefix_ref))
                    self.assertTrue(torch.equal(bank, bank_ref))

    def test_wide_bank_row_addressing(self):
        """NB > nvb: rows must be addressed with the true bank stride."""
        prefix, bank = _make_inputs(7, seed=1, num_bank_slots=11)
        out = _mix_fused(prefix, bank, 5, self.proj, self.norm)
        ref = aggregate_stream_torch(prefix, bank, 5, self.proj, self.norm)
        torch.testing.assert_close(out.float(), ref.float(), rtol=2e-2, atol=4e-2)

    def test_nvb0_passthrough(self):
        prefix, bank = _make_inputs(3, seed=2)
        out = aggregate_stream(prefix, bank, 0, self.proj, self.norm)
        self.assertIs(out, prefix)

    def test_fused_wrapper_applies_out_norm(self):
        out_norm = RMSNorm(_H, eps=_EPS).to(device="cuda", dtype=torch.bfloat16)
        with torch.no_grad():
            gen = torch.Generator(device="cuda").manual_seed(3)
            out_norm.weight.copy_(
                1 + 0.1 * torch.randn(_H, generator=gen, device="cuda")
            )
        prefix, bank = _make_inputs(17, seed=4)
        out = _aggregate_fused(prefix, bank, 6, self.proj, self.norm, out_norm)
        ref = out_norm(aggregate_stream_torch(prefix, bank, 6, self.proj, self.norm))
        torch.testing.assert_close(out.float(), ref.float(), rtol=2e-2, atol=4e-2)


if __name__ == "__main__":
    unittest.main()
