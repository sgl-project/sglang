"""``fill_draft_verify_metadata``: one-launch DFlash draft verify attention
metadata against the per-request reference."""

import unittest

import torch

from sglang.kernels.ops.attention.dflash_draft_verify_metadata import (
    fill_draft_verify_metadata,
)
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=30, suite="stage-b-test-1-gpu-small-amd-mi35x")

DEV = "cuda"
L = 8  # tokens per verify step


class TestDFlashDraftVerifyMetadata(CustomTestCase):
    def test_matches_reference(self):
        torch.manual_seed(0)
        r2t = torch.randint(0, 10**6, (70, 9000), device=DEV, dtype=torch.int32)

        def filled(n, dtype):
            return torch.full((n,), -7, dtype=dtype, device=DEV)

        def cumsum0(x):
            out = torch.zeros(x.numel() + 1, dtype=torch.int64, device=DEV)
            out[1:] = torch.cumsum(x, 0)
            return out

        for bs in (1, 2, 5, 8, 16, 64):
            for window in (4096, 0):
                seq = torch.randint(1, 8900, (bs,), device=DEV, dtype=torch.int32)
                seq[0] = 4160 if bs > 1 else 4100
                rpi = torch.randperm(70, device=DEV)[:bs]
                qo, mip = filled(bs + 2, torch.int64), filled(bs + 2, torch.int64)
                kvp, wp = filled(bs + 2, torch.int32), filled(bs + 2, torch.int32)
                kvi, wi = filled(bs * 9000, torch.int64), filled(bs * 9000, torch.int64)
                wo = filled(bs + 1, torch.int32)
                fill_draft_verify_metadata(
                    bs=bs, seq_lens=seq, req_pool_indices=rpi, req_to_token=r2t,
                    num_tokens_per_req=L, qo_indptr=qo, kv_indptr=kvp,
                    kv_indices=kvi, mask_indptr=mip, window_size=window or None,
                    window_kv_indptr=wp, window_kv_indices=wi, window_kv_offsets=wo,
                )  # fmt: skip
                s64 = seq.long()
                ref_kvi = torch.cat([r2t[rpi[b], : seq[b]].long() for b in range(bs)])
                self.assertTrue(
                    torch.equal(
                        qo[: bs + 1], torch.arange(0, (bs + 1) * L, L, device=DEV)
                    )
                )
                self.assertTrue(torch.equal(kvp[: bs + 1].long(), cumsum0(s64)))
                self.assertTrue(torch.equal(kvi[: ref_kvi.numel()], ref_kvi))
                self.assertEqual(int(kvi[ref_kvi.numel()]), -7)
                self.assertTrue(torch.equal(mip[: bs + 1], cumsum0(L * (s64 + L))))
                if window:
                    wl = torch.clamp(s64, max=window)
                    off = s64 - wl
                    ref_wi = torch.cat(
                        [r2t[rpi[b], off[b] : off[b] + wl[b]].long() for b in range(bs)]
                    )
                    self.assertTrue(torch.equal(wp[: bs + 1].long(), cumsum0(wl)))
                    self.assertTrue(torch.equal(wo[:bs].long(), off))
                    self.assertTrue(torch.equal(wi[: ref_wi.numel()], ref_wi))


if __name__ == "__main__":
    unittest.main()
