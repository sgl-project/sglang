"""Packed per-rank argmax for the Kimi-K3 DFlash draft LM head.

Each TP rank packs its local (max, index) into one int32 key, the keys are
all-gathered, and ``select_from_gathered`` picks the global winner. This has to
match ``torch.argmax`` over the unsharded logits, including the tie rule
(lowest global index wins), signed zeros and all-negative rows.
"""

import unittest

import torch

from sglang.kernels.ops.speculative.k3_draft_argmax import (
    pack_row_argmax,
    select_from_gathered,
)
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase


def reference_argmax(logits_shards):
    """torch reference of the baseline sampler: list of [n, V] shards."""
    n, V = logits_shards[0].shape
    mx = []
    ids = []
    for r, lg in enumerate(logits_shards):
        a, b = torch.max(lg, dim=-1)
        mx.append(a)
        ids.append(b + r * V)
    gm = torch.stack(mx)
    gi = torch.stack(ids)
    best = torch.argmax(gm, dim=0)
    return torch.gather(gi, 0, best[None]).view(-1)


register_amd_ci(est_time=30, suite="stage-b-test-1-gpu-small-amd-mi35x")

TP, VOCAB_TP = 8, 20480
ROWS = [1, 7, 14, 28, 56, 63, 112, 448]


def _logits(trial, n):
    full = torch.randn(n, TP * VOCAB_TP, device="cuda").to(torch.bfloat16)
    if trial % 3 == 1:
        full = (full * 2).round().to(torch.bfloat16)
    if trial % 5 == 2:
        mx = full.max(dim=-1).values
        for r in range(n):
            full[r, torch.randint(0, TP * VOCAB_TP, (3,))] = mx[r]
    if trial % 7 == 3:
        full[: n // 2] = 0.0
        full[: n // 2, ::3] = -0.0
    if trial % 11 == 4:
        full = -full.abs() - 1
    return full


class TestK3DraftArgmaxPack(CustomTestCase):
    def test_matches_unsharded_argmax(self):
        torch.manual_seed(0)
        for trial in range(200):
            n = ROWS[trial % len(ROWS)]
            full = _logits(trial, n)
            shards = [
                full[:, r * VOCAB_TP : (r + 1) * VOCAB_TP].contiguous()
                for r in range(TP)
            ]
            n_pad = (n + 3) // 4 * 4
            gathered = torch.empty(TP, n_pad, dtype=torch.float32, device="cuda")
            for r in range(TP):
                pack_row_argmax(shards[r], gathered[r].view(torch.int32))
            out = torch.empty(n, dtype=torch.int64, device="cuda")
            select_from_gathered(
                gathered.view(torch.int32).view(-1), out, n, n_pad, VOCAB_TP, TP
            )
            self.assertTrue(torch.equal(out, reference_argmax(shards)), trial)
            self.assertTrue(torch.equal(out, torch.argmax(full.float(), dim=-1)), trial)


if __name__ == "__main__":
    unittest.main()
