"""Boundary tests for the packed indices in the DSV4 prefill write plan."""

from __future__ import annotations

import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kernels.deepseek_v4.common import (
    make_legacy_context,
    make_paged_context,
    to_seq_extend,
)
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-large")


class TestCompressWritePlanBounds(CustomTestCase):
    def test_64k_prefill_preserves_last_token(self):
        """65536 tokens fit uint16 indices; the last token must not wrap or vanish."""
        for cr in (4, 128):
            paged = make_paged_context(
                bs=16, compress_ratio=cr, num_swa_pages_per_req=16
            )
            legacy = make_legacy_context(bs=16, compress_ratio=cr)
            seq_lens, extend_lens, num_q = to_seq_extend([(4096, 4096)] * 16)
            for ctx, on_gpu in ((paged, False), (paged, True), (legacy, False)):
                with self.subTest(cr=cr, paged=ctx is paged, on_gpu=on_gpu):
                    device = "cuda" if on_gpu else "cpu"
                    plan = ctx.make_prefill_plan(
                        seq_lens.to(device), extend_lens.to(device), num_q
                    )
                    c = plan.plan_c.cpu().view(torch.int32).view(-1, 4)
                    valid_c = c[:, 0] != -1
                    ids = c[valid_c, 1].bitwise_and(0xFFFF).sort().values
                    torch.testing.assert_close(
                        ids, torch.arange(cr - 1, num_q, cr, dtype=torch.int32)
                    )
                    w = plan.plan_w.cpu().view(torch.int32).view(-1, 2)
                    last = w[w[:, 0] == 65535]
                    if cr == 4:
                        self.assertEqual(len(last), 1)
                        self.assertEqual(int(last[0, 1]), ctx.state_loc(15, 4095))
                    else:
                        # Non-overlapping C128 consumed the complete final block;
                        # no raw tail remains to persist into the state ring.
                        self.assertEqual(len(w[w[:, 0] != -1]), 0)

    def test_prefill_rejects_uint16_index_overflow(self):
        for ctx in (
            make_paged_context(bs=16, compress_ratio=4, num_swa_pages_per_req=17),
            make_legacy_context(bs=16, compress_ratio=4),
        ):
            seq_lens, extend_lens, num_q = to_seq_extend(
                [(4096, 4096)] * 15 + [(4097, 4097)]
            )
            with self.assertRaisesRegex(RuntimeError, "plan_compress_prefill"):
                ctx.make_prefill_plan(seq_lens, extend_lens, num_q)

    def test_prefill_rejects_packed_invalid_sentinel(self):
        # A 65536-request, one-token-per-request batch makes the last packed
        # (batch_id, ragged_id) equal (65535, 65535), the invalid write sentinel.
        ctx = make_legacy_context(bs=65536, compress_ratio=4)
        seq_lens, extend_lens, num_q = to_seq_extend([(1, 1)] * 65536)
        with self.assertRaisesRegex(RuntimeError, "plan_compress_prefill"):
            ctx.make_prefill_plan(seq_lens, extend_lens, num_q)


if __name__ == "__main__":
    unittest.main()
