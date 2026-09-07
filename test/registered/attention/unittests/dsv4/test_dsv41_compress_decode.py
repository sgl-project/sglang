"""Ratio-2 compressor, decode pairing: the fixed-shape CUDA-graph path must keep
the same per-request pending state as the torch extend path. Tokens compress in
pairs (2k, 2k+1) and the even token waits for its odd partner; in decode this is
a per-request gather / scatter. Padded graph rows carry req_pool_idx 0 and must
not disturb request 0. GPU only by registration; the math is plain torch.
"""

import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-large")


def reference_pairing(kv, score, pos, req, state_kv, state_score):
    """Loop form of the extend-path grouping for a one-token-per-request batch."""
    odd = pos % 2 == 1
    partner_kv = torch.empty_like(kv)
    partner_score = torch.empty_like(score)
    for i in range(kv.shape[0]):
        r = int(req[i])
        if bool(odd[i]):
            partner_kv[i] = state_kv[r]
            partner_score[i] = state_score[r]
        else:
            partner_kv[i] = 0
            partner_score[i] = 0
    for i in range(kv.shape[0]):
        if not bool(odd[i]):
            state_kv[int(req[i])] = kv[i]
            state_score[int(req[i])] = score[i]
    return partner_kv, partner_score


class TestCompressDecodePairing(CustomTestCase):
    def test_matches_extend_grouping(self):
        from sglang.srt.layers.attention.dsv4.dsv41_sparse import (
            pair_partners_decode,
        )

        torch.manual_seed(0)
        num_req_slots, bs, d = 16, 8, 32
        req = torch.randperm(num_req_slots, device="cuda")[:bs]
        pos = torch.randint(1, 1000, (bs,), device="cuda")
        odd = pos % 2 == 1
        state_kv = torch.randn(num_req_slots, d, device="cuda")
        state_score = torch.randn(num_req_slots, d, device="cuda")
        kv = torch.randn(bs, d, device="cuda")
        score = torch.randn(bs, d, device="cuda")

        ref_kv, ref_score = state_kv.clone(), state_score.clone()
        ref_pk, ref_ps = reference_pairing(kv, score, pos, req, ref_kv, ref_score)

        pk, ps = pair_partners_decode(kv, score, odd, req, state_kv, state_score)

        # Odd rows read the same partner the extend grouping would have used.
        self.assertTrue(torch.equal(pk[odd], ref_pk[odd]))
        self.assertTrue(torch.equal(ps[odd], ref_ps[odd]))
        # The pending state ends up identical: even rows parked, odd rows kept.
        self.assertTrue(torch.equal(state_kv, ref_kv))
        self.assertTrue(torch.equal(state_score, ref_score))

    def test_padded_rows_routed_to_spare_row(self):
        from sglang.srt.layers.attention.dsv4.dsv41_sparse import (
            pair_partners_decode,
        )

        torch.manual_seed(1)
        num_req_slots, d = 8, 16
        pad_row = num_req_slots
        state_kv = torch.randn(num_req_slots + 1, d, device="cuda")
        state_score = torch.randn(num_req_slots + 1, d, device="cuda")
        before_kv, before_score = state_kv.clone(), state_score.clone()

        # Request 0 is live at an odd position; three padded rows sit at
        # position 0 (even) and are routed to the spare row by the caller.
        req = torch.tensor([0, pad_row, pad_row, pad_row], device="cuda")
        pos = torch.tensor([7, 0, 0, 0], device="cuda")
        odd = pos % 2 == 1
        kv = torch.randn(4, d, device="cuda")
        score = torch.randn(4, d, device="cuda")

        pk, ps = pair_partners_decode(kv, score, odd, req, state_kv, state_score)

        # Request 0 read its own pending partner and its row is untouched.
        self.assertTrue(torch.equal(pk[0], before_kv[0]))
        self.assertTrue(torch.equal(ps[0], before_score[0]))
        self.assertTrue(torch.equal(state_kv[:pad_row], before_kv[:pad_row]))
        self.assertTrue(torch.equal(state_score[:pad_row], before_score[:pad_row]))
        # Only the spare row absorbed the padded writes.
        self.assertTrue(bool((state_kv[pad_row] != before_kv[pad_row]).any()))


if __name__ == "__main__":
    unittest.main()
