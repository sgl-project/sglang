"""Bit-exactness of the Volundr fused mHC kernels against their torch references."""

import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-large")

DEV = "cuda"


def _mismatches(a, b):
    return int((a.reshape(-1) != b.reshape(-1)).sum())


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestVolundrMhc(CustomTestCase):
    def _inputs(self, tokens, hidden=512, streams=4):
        scale = torch.tensor([1.0, 0.7, 1.3, 0.4], device=DEV)[None, :, None]
        H = (torch.randn(tokens, streams, hidden, device=DEV) * scale).to(
            torch.bfloat16
        )
        d = (torch.randn(tokens, 2 * streams + streams * streams, device=DEV) * 2).to(
            torch.bfloat16
        )
        pre = torch.eye(streams, device=DEV)[0] + 0.1 * torch.randn(streams, device=DEV)
        post = torch.ones(streams, device=DEV) + 0.1 * torch.randn(streams, device=DEV)
        res = torch.randn(streams, streams, device=DEV)
        return H, d, pre, post, res

    def test_mix_and_combine(self):
        from sglang.kernels.ops.layernorm import volundr_mhc as F

        torch.manual_seed(0)
        for tokens in (1, 3, 8, 64, 513):
            H, d, pre, post, res = self._inputs(tokens)
            rx, rpost, rres = F.ref_mix_from_d(d, pre, post, res, H, 10)
            fx, fpost, fres = F.mhc_mix(d, pre, post, res, H, 10)
            self.assertEqual(_mismatches(rx, fx), 0)
            self.assertEqual(_mismatches(rpost.to(torch.bfloat16), fpost), 0)
            self.assertEqual(_mismatches(rres.to(torch.bfloat16), fres), 0)
            y = (rx.float() + torch.randn_like(rx.float()) * 0.3).to(torch.bfloat16)
            rH = F.ref_combine(H, rpost, rres, y - rx)
            mixed = torch.einsum("tij,tjd->tid", fres, H)
            fH, fmean = F.mhc_combine(mixed, fpost, y, fx, True)
            self.assertEqual(_mismatches(rH, fH), 0)
            self.assertEqual(_mismatches(rH.mean(1), fmean), 0)


if __name__ == "__main__":
    unittest.main()
