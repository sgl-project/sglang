"""Kimi-K3 DFlash draft sliding-window verify attention.

``draft_swa_verify_fwd`` reads the new draft K/V in place and splits long
windows across programs. It must match ``extend_attention_fwd`` with the same
sliding window over mixed batches: short, exactly-window, just-over-window and
long sequences, and any split count.
"""

import unittest

import torch

from sglang.kernels.ops.attention.extend_attention import extend_attention_fwd
from sglang.kernels.ops.attention.k3_draft_swa_verify import (
    can_handle,
    draft_swa_verify_fwd,
)
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=60, suite="stage-b-test-1-gpu-small-amd-mi35x")

HQ, HKV, D, L, WINDOW, PAGE = 4, 1, 128, 8, 4096, 64
SLOTS = 200000
TOL = 2e-2
BATCHES = [
    [4096],
    [90000],
    [100],
    [4100],
    [5000, 64, 4096],
    [9000] * 4,
    [3000, 9000, 4096, 4095, 1, 70000, 4097, 20000],
    [6000] * 16,
]


class TestK3DraftSwaVerify(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        cls.kbuf = torch.randn(SLOTS, HKV, D, device="cuda", dtype=torch.bfloat16)
        cls.vbuf = torch.randn(SLOTS, HKV, D, device="cuda", dtype=torch.bfloat16)

    def _case(self, seqs):
        bs, t = len(seqs), len(seqs) * L
        qkv = torch.randn(t, (HQ + 2 * HKV) * D, device="cuda", dtype=torch.bfloat16)
        q, k, v = qkv.split([HQ * D, HKV * D, HKV * D], dim=-1)
        prefix = [min(s, WINDOW) for s in seqs]
        kv_indptr = torch.zeros(bs + 1, dtype=torch.int32, device="cuda")
        kv_indptr[1:] = torch.cumsum(torch.tensor(prefix, device="cuda"), 0)
        kv_indices = torch.randperm(SLOTS, device="cuda")[: sum(prefix)].long()
        qo_indptr = torch.arange(0, (bs + 1) * L, L, dtype=torch.int32, device="cuda")
        return (
            q.view(t, HQ, D),
            k.view(t, HKV, D),
            v.view(t, HKV, D),
            qo_indptr,
            kv_indptr,
            kv_indices,
        )

    def _ref(self, q, k, v, qo, kvp, kvi):
        o = torch.empty_like(q)
        extend_attention_fwd(
            q, k.contiguous(), v.contiguous(), o, self.kbuf, self.vbuf, qo, kvp,
            kvi, None, True, None, L, 1.0, 1.0, D**-0.5,
            sliding_window_size=WINDOW, page_size=PAGE,
        )  # fmt: skip
        return o.float()

    def _fused(self, q, k, v, qo, kvp, kvi, n_splits=None):
        o = torch.empty_like(q)
        draft_swa_verify_fwd(
            q, k, v, o, self.kbuf, self.vbuf, qo, kvp, kvi, L, D**-0.5, WINDOW,
            PAGE, n_splits=n_splits,
        )  # fmt: skip
        return o.float()

    def test_matches_extend_attention(self):
        for seqs in BATCHES:
            q, k, v, qo, kvp, kvi = self._case(seqs)
            self.assertTrue(
                can_handle(
                    q,
                    k,
                    v,
                    self.kbuf,
                    self.vbuf,
                    qo,
                    L,
                    causal=True,
                    sinks=None,
                    logit_cap=0.0,
                    xai_temperature_len=-1,
                    score_mod=None,
                )  # fmt: skip
            )
            err = self._ref(q, k, v, qo, kvp, kvi) - self._fused(q, k, v, qo, kvp, kvi)
            self.assertLess(err.abs().max().item(), TOL, seqs[:4])

    def test_split_count_does_not_change_output(self):
        for bs in (1, 4, 8):
            q, k, v, qo, kvp, kvi = self._case([9000] * bs)
            ref = self._ref(q, k, v, qo, kvp, kvi)
            for n_splits in (8, 16, 32, 64):
                out = self._fused(q, k, v, qo, kvp, kvi, n_splits)
                self.assertLess((ref - out).abs().max().item(), TOL, (bs, n_splits))


if __name__ == "__main__":
    unittest.main()
