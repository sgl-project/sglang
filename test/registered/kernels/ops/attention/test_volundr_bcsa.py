"""Bit-exactness of the Volundr BCSA decode kernels against their torch references."""

import types
import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-large")

DEV = "cuda"


def _mismatches(a, b):
    return int((a.reshape(-1) != b.reshape(-1)).sum())


def _fake_bcsa(heads, kv_heads, head_dim, hidden, rows, max_reqs, ctx):
    """The attributes of models/volundr.py VolundrAttention the far branch reads."""
    wq = (torch.randn(512, hidden, device=DEV) * 0.02).to(torch.bfloat16)
    nw = (torch.randn(128, device=DEV) * 0.1).to(torch.bfloat16)

    def idx_norm(x):
        xf = x.float()
        y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6)
        return (y * (1.0 + nw.float())).to(x.dtype)

    return types.SimpleNamespace(
        compress=4,
        window=4096,
        bcsa_topk=512,
        idx_heads=4,
        idx_dim=128,
        idx_scale=128**-0.5,
        num_heads=heads,
        num_kv_heads=kv_heads,
        head_dim=head_dim,
        scaling=head_dim**-0.5,
        idx_cache=torch.zeros(rows + 1, 128, dtype=torch.bfloat16, device=DEV),
        blk_cache=(torch.randn(rows + 1, 128, device=DEV) * 2).to(torch.bfloat16),
        r2t=torch.randint(0, rows, (max_reqs, ctx), device=DEV, dtype=torch.int32),
        max_blocks=ctx // 4,
        idx_q=lambda x: (torch.nn.functional.linear(x, wq),),
        idx_norm=idx_norm,
    )


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestVolundrBcsa(CustomTestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_block_keys_decode(self):
        from sglang.kernels.ops.attention import volundr_bcsa as F

        rows, max_reqs, ctx = 20001, 9, 8192
        idx = (torch.randn(rows + 1, 128, device=DEV) * 3).to(torch.bfloat16)
        r2t = torch.randint(0, rows, (max_reqs, ctx), device=DEV, dtype=torch.int32)
        for trial in range(40):
            bs = (1, 2, 4, 8)[trial % 4]
            pos = torch.randint(0, ctx - 1, (bs,), device=DEV)
            if trial % 7 == 0:
                pos[0] = 0  # a CUDA-graph padding row
            reqs = torch.randperm(max_reqs, device=DEV)[:bs]
            loc = torch.randperm(rows, device=DEV)[:bs]
            ref = torch.zeros(rows + 1, 128, device=DEV, dtype=torch.bfloat16)
            out = ref.clone()
            F.ref_block_keys_decode(pos, reqs, loc, r2t, idx, ref, 4)
            F.bcsa_block_keys_decode(pos, reqs, loc, r2t, idx, out, 4)
            ref[-1], out[-1] = 0, 0  # the trash row's content is irrelevant
            self.assertEqual(_mismatches(ref, out), 0)

    def test_far_sparse(self):
        from sglang.kernels.ops.attention import volundr_bcsa as F

        rows, max_reqs, ctx, hidden, head_dim = 40000, 9, 16384, 512, 128
        for heads, kv_heads in ((24, 4), (12, 2)):
            attn = _fake_bcsa(heads, kv_heads, head_dim, hidden, rows, max_reqs, ctx)
            kc = torch.randn(rows + 1, kv_heads, head_dim, device=DEV).to(
                torch.bfloat16
            )
            vc = torch.randn(rows + 1, kv_heads, head_dim, device=DEV).to(
                torch.bfloat16
            )
            for bs in (1, 2, 6):
                pos = torch.randint(4000, ctx - 1, (bs,), device=DEV)
                pos[0] = 4098 + bs  # at the window edge: zero or one far block
                if bs > 1:
                    pos[-1] = 3000  # a request below the window in a far batch
                reqs = torch.randperm(max_reqs, device=DEV)[:bs]
                hs = torch.randn(bs, hidden, device=DEV).to(torch.bfloat16)
                qh = torch.randn(bs, heads, head_dim, device=DEV).to(torch.bfloat16)
                o_loc = torch.randn(bs, heads, head_dim, device=DEV).to(torch.bfloat16)
                lse = torch.randn(bs, heads, device=DEV) * 3 + 5
                ref = F.ref_far_sparse(attn, hs, qh, o_loc, lse, pos, reqs, kc, vc)
                out = F.far_sparse_fused(attn, hs, qh, o_loc, lse, pos, reqs, kc, vc)
                self.assertEqual(_mismatches(ref, out), 0)


if __name__ == "__main__":
    unittest.main()
