"""``moe_topk_reduce_add`` (the ROCm top-k reduction with the shared-expert add folded in) must match an fp32 reference bit for bit and be batch-invariant."""

import unittest

import torch

from sglang.srt.utils import is_hip
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=40, suite="stage-b-test-1-gpu-small-amd-mi35x")

D, TOPK, E = 5120, 6, 384


def _reference(x, shared, ids, mask, alpha):
    m = shared.shape[0]
    xs = x.view(m, TOPK, D).float()
    acc = torch.zeros_like(shared, dtype=torch.float32)
    for k in range(TOPK):
        v = xs[:, k]
        if mask is not None:
            v = torch.where((mask[ids[:, k]] != 0)[:, None], v, 0.0)
        acc = acc + v
    return (acc * alpha + shared.float()).to(shared.dtype)


def _inputs(m, seed, local_fraction=0.25):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(m * TOPK, D, device="cuda", dtype=torch.bfloat16, generator=g)
    shared = torch.randn(m, D, device="cuda", dtype=torch.bfloat16, generator=g)
    ids = torch.randint(0, E, (m, TOPK), device="cuda", dtype=torch.int32, generator=g)
    mask = (torch.arange(E, device="cuda") < int(E * local_fraction)).to(torch.int32)
    return x, shared, ids, mask


@unittest.skipUnless(is_hip(), "the fused reduction is the ROCm path")
class TestMoeTopkReduceAdd(CustomTestCase):
    def setUp(self):
        from sglang.kernels.ops.moe.moe_reduce_add_hip import moe_topk_reduce_add

        self.reduce_add = moe_topk_reduce_add

    def test_matches_reference(self):
        for m in (1, 33):
            for alpha in (1.0, 2.5):
                x, shared, ids, mask = _inputs(m, m)
                for use_mask in (True, False):
                    out = torch.empty_like(shared)
                    self.reduce_add(
                        x,
                        shared,
                        out,
                        TOPK,
                        ids if use_mask else None,
                        mask if use_mask else None,
                        alpha=alpha,
                    )
                    ref = _reference(x, shared, ids, mask if use_mask else None, alpha)
                    self.assertTrue(torch.equal(out, ref), (m, alpha, use_mask))

    def test_repeatable_and_batch_invariant(self):
        x, shared, ids, mask = _inputs(300, 7)
        full = torch.empty_like(shared)
        self.reduce_add(x, shared, full, TOPK, ids, mask)
        for _ in range(3):
            again = torch.empty_like(shared)
            self.reduce_add(x, shared, again, TOPK, ids, mask)
            self.assertTrue(torch.equal(again, full))
        for rows in ([0], list(range(0, 300, 7))):
            idx = torch.tensor(rows, device="cuda")
            sub = torch.empty(len(rows), D, device="cuda", dtype=torch.bfloat16)
            self.reduce_add(
                x.view(300, TOPK, D)[idx].reshape(-1, D).contiguous(),
                shared[idx].contiguous(),
                sub,
                TOPK,
                ids[idx].contiguous(),
                mask,
            )
            self.assertTrue(torch.equal(sub, full[idx]), rows)


if __name__ == "__main__":
    unittest.main()
