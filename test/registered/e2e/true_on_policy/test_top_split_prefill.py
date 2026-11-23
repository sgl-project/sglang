"""TOP chunked GDN attention: a prefilled row takes its decode's key split bit for bit.

Under SGLANG_TOP_GDN_MODE=chunked the trainer scores every row as a prefill, so serving prefill attention runs each
row through the fixed-position decode split (``split_prefill_paged``): its own 4096-key piece causal, earlier pieces in
full, FA's combine. Here each extend row of a batch (prefixes on and off the split grid, rows in several pieces) is
compared with ``split_decode_paged`` (what serving decode runs) of the same row over the same paged cache. Qwen3.5
attention shapes: 16 query heads, 2 KV heads, head dim 256, page 128, a 131072-token page table."""

import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=120, stage="base-b", runner_config="1-gpu-large")

HQ, HKV, D, PAGE, SPLIT, CTX = 16, 2, 256, 128, 4096, 131072
dev = "cuda"


def _bits(x):
    return x.contiguous().view(torch.int16)


@unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
class TestSplitPrefill(unittest.TestCase):
    def test_prefill_rows_equal_decode_rows(self):
        from sglang.srt.true_on_policy.attention import (
            SplitPrefillPlan,
            split_decode_paged,
            split_prefill_paged,
        )

        torch.manual_seed(0)
        # (prefix, extend): fresh short, fresh across pieces, prefix on the split grid, prefix off it, tail rows only.
        reqs = [(0, 700), (0, 9000), (4096, 5000), (6000, 3000), (8191, 3)]
        width = CTX // PAGE
        n_pages = sum(-(-(p + n) // PAGE) for p, n in reqs) + 1
        kc = torch.randn(n_pages, PAGE, HKV, D, device=dev).to(torch.bfloat16)
        vc = torch.randn(n_pages, PAGE, HKV, D, device=dev).to(torch.bfloat16)
        perm = torch.randperm(n_pages - 1, device=dev).to(torch.int32) + 1
        pt = torch.zeros(len(reqs), width, device=dev, dtype=torch.int32)
        used = 0
        for i, (p, n) in enumerate(reqs):
            k = -(-(p + n) // PAGE)
            pt[i, :k] = perm[used : used + k]
            used += k
        T = sum(n for _, n in reqs)
        q = torch.randn(T, HQ, D, device=dev).to(torch.bfloat16)
        plan = SplitPrefillPlan(
            prefix=[p for p, _ in reqs],
            lens=[n for _, n in reqs],
            split=SPLIT,
            page_size=PAGE,
            page_table=pt,
            device=dev,
        )
        out = split_prefill_paged(q, kc, vc, plan, D**-0.5)
        self.assertEqual(out.shape, (T, HQ, D))
        # Decode each row (its own query against keys [0, t]) through serving's split decode, in batches.
        req = torch.cat(
            [torch.full((n,), i, device=dev) for i, (_, n) in enumerate(reqs)]
        )
        pos = torch.cat([torch.arange(p, p + n, device=dev) for p, n in reqs])
        pieces = -(-width // (SPLIT // PAGE))
        bad = 0
        for s in range(0, T, 512):
            e = min(T, s + 512)
            dec = split_decode_paged(
                q[s:e],
                kc,
                vc,
                pt[req[s:e]],
                (pos[s:e] + 1).to(torch.int32),
                SPLIT,
                pieces,
                D**-0.5,
            )
            bad += int((_bits(dec) != _bits(out[s:e])).flatten(1).any(1).sum())
        self.assertEqual(
            bad, 0, f"{bad} of {T} prefilled rows differ from their split decode"
        )


if __name__ == "__main__":
    unittest.main()
