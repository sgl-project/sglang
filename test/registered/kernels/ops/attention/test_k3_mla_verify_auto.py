"""``k3_mla_verify_auto`` (FP8 MLA target verify, QLEN=8) must match a causal
torch reference in both regimes the on-device switch can pick: the Gluon
kernel (thresh = INT32_MAX) and the HipKittens kernel (thresh = 0)."""

import unittest

import torch

from sglang.kernels.ops.attention.k3_mla_verify_auto import k3_mla_verify_auto
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=60, stage="jit-kernel-unit", runner_config="amd")

H, QLEN, D, DV = 12, 8, 576, 512
SCALE = D**-0.5
KV_SCALE = 0.5
BATCHES = [[64], [3000], [40000], [9, 700, 5000, 33000], [2048] * 16]


def _reference(q, kv, lens, slots):
    kv = kv.view(-1, D).float() * KV_SCALE
    out, start = [], 0
    for b, L in enumerate(lens):
        k = kv[slots[start : start + L]]
        start += L
        qb = q[b * QLEN : (b + 1) * QLEN].float()  # [QLEN, H, D]
        s = torch.einsum("qhd,kd->hqk", qb, k) * SCALE
        visible = torch.arange(L, device=q.device) <= (
            L - QLEN + torch.arange(QLEN, device=q.device)[:, None]
        )
        p = s.masked_fill(~visible, float("-inf")).softmax(-1)
        out.append(torch.einsum("hqk,kd->qhd", p, k[:, :DV]))
    return torch.cat(out).to(q.dtype)


class TestK3MlaVerifyAuto(CustomTestCase):
    def test_both_regimes_match_reference(self):
        for lens in BATCHES:
            torch.manual_seed(len(lens) * lens[0])
            total = sum(lens)
            kv = (torch.randn(total + 128, 1, D, device="cuda") * 0.5).to(
                torch.float8_e4m3fn
            )
            slots = torch.randperm(total + 128, device="cuda")[:total].to(torch.int32)
            indptr = torch.tensor([0] + lens, device="cuda").cumsum(0).to(torch.int32)
            q = torch.randn(len(lens) * QLEN, H, D, device="cuda", dtype=torch.bfloat16)
            ref = _reference(q, kv, lens, slots.long())
            for name, thresh in (("gluon", 2**31 - 1), ("hk", 0)):
                with self.subTest(lens=lens[:4], regime=name):
                    out = k3_mla_verify_auto(
                        q,
                        kv,
                        indptr,
                        slots,
                        SCALE,
                        torch.tensor([KV_SCALE], device="cuda"),
                        KV_SCALE,
                        thresh=thresh,
                    )
                    torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-2)


if __name__ == "__main__":
    unittest.main()
