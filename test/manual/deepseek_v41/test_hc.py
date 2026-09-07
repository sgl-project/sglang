import unittest

import dsv41_hc as hc
import torch
from ref_loader import (
    RefTestCase,
    assert_equal,
    randomize_,
    report,
    requires_ref,
    small_args,
)


@requires_ref
class TestHyperConnections(RefTestCase):
    def test_split_sinkhorn(self):
        mixes = torch.randn(2, 5, 24, dtype=torch.float32) * 3
        hc_scale = torch.randn(3, dtype=torch.float32)
        hc_base = torch.randn(24, dtype=torch.float32)
        expected = self.kernel.hc_split_sinkhorn(mixes, hc_scale, hc_base, 4, 20, 1e-6)
        actual = hc.hc_split_sinkhorn(mixes, hc_scale, hc_base, 4, 20, 1e-6)
        for name, a, e in zip(("pre", "post", "comb"), actual, expected):
            report(name, a, e)
            torch.testing.assert_close(a, e, rtol=1e-5, atol=1e-6)

    def test_block_mixing(self):
        model = self.model
        args = small_args(model)
        block = model.Block(0, args)
        randomize_(block)
        x = torch.randn(2, 5, args.hc_mult, args.dim)
        sub = torch.randn(2, 5, args.dim)

        expected = block.hc_mixes(
            x, block.hc_attn_fn, block.hc_attn_scale, block.hc_attn_base
        )
        actual = hc.hc_mixes(
            x,
            block.hc_attn_fn,
            block.hc_attn_scale,
            block.hc_attn_base,
            hc_mult=args.hc_mult,
            sinkhorn_iters=args.hc_sinkhorn_iters,
            hc_eps=args.hc_eps,
            norm_eps=args.norm_eps,
        )
        for name, a, e in zip(("pre", "post", "comb"), actual, expected):
            report(name, a, e)
            torch.testing.assert_close(a, e, rtol=1e-5, atol=1e-6)

        pre, post, comb = expected
        assert_equal(hc.hc_pre(x, pre), block.hc_pre(x, pre))
        assert_equal(hc.hc_post(sub, x, post, comb), block.hc_post(sub, x, post, comb))


if __name__ == "__main__":
    unittest.main()
