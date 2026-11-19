import unittest

import torch

from sglang.kernels.ops.attention.flash_attn.cute.flash_fwd_sm100_bias_decode import (
    paged_rel_bias_decode,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=90, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

HEADS_Q, HEADS_K, HEAD_DIM, PAGE_SIZE, REL_EXTENT = 12, 2, 128, 128, 1024


def reference(q, k_cache, v_cache, page_table, seqlens, rel_bias, scale, window):
    outs, lses = [], []
    group = HEADS_Q // HEADS_K
    for b, length in enumerate(seqlens.tolist()):
        pages = page_table[b, : (length + PAGE_SIZE - 1) // PAGE_SIZE]
        k = k_cache[pages].reshape(-1, HEADS_K, HEAD_DIM)[:length].float()
        v = v_cache[pages].reshape(-1, HEADS_K, HEAD_DIM)[:length].float()
        k, v = k.repeat_interleave(group, 1), v.repeat_interleave(group, 1)
        scores = torch.einsum("hd,khd->hk", q[b].float(), k) * scale
        distance = length - 1 - torch.arange(length, device=q.device)
        bias = rel_bias[b][:, distance.clamp(max=REL_EXTENT - 1)].float()
        scores = scores + torch.where(distance < REL_EXTENT, bias, 0.0)
        if window is not None:
            scores = scores.masked_fill(distance > window, -torch.inf)
        lses.append(torch.logsumexp(scores, -1))
        outs.append(torch.einsum("hk,khd->hd", torch.softmax(scores, -1), v))
    return torch.stack(outs), torch.stack(lses, 1)


class TestPagedRelBiasDecode(CustomTestCase):
    def setUp(self):
        if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
            self.skipTest("requires SM100")

    def test_matches_reference(self):
        torch.manual_seed(0)
        scale = HEAD_DIM**-0.5
        max_len = 2600
        pages = max_len // PAGE_SIZE + 1
        for batch in (1, 3, 16, 32):
            seqlens = torch.randint(
                1, max_len, (batch,), device="cuda", dtype=torch.int32
            )
            seqlens[0] = 2048
            total_pages = batch * pages + 3
            k_cache = torch.randn(
                total_pages, PAGE_SIZE, HEADS_K, HEAD_DIM, device="cuda"
            ).bfloat16()
            v_cache = torch.randn_like(k_cache)
            # Column slice of a wider buffer, like the CUDA-graph page-table metadata.
            table = torch.zeros(batch, pages + 37, device="cuda", dtype=torch.int32)
            table[:, :pages] = (
                torch.randperm(total_pages, device="cuda")[: batch * pages]
                .view(batch, pages)
                .int()
            )
            page_table = table[:, :pages]
            q = (torch.randn(batch, HEADS_Q, HEAD_DIM, device="cuda") * 1.5).bfloat16()
            rel_bias = (
                torch.randn(batch, HEADS_Q, REL_EXTENT, device="cuda") * 2
            ).bfloat16()
            for window in (1023, None):
                with self.subTest(batch=batch, window=window):
                    out = torch.empty(q.shape, device="cuda", dtype=torch.float32)
                    out, lse = paged_rel_bias_decode(
                        q,
                        k_cache,
                        v_cache,
                        page_table,
                        seqlens,
                        rel_bias,
                        scale,
                        out,
                        window_size_left=window,
                    )
                    expected, expected_lse = reference(
                        q,
                        k_cache,
                        v_cache,
                        page_table,
                        seqlens,
                        rel_bias,
                        scale,
                        window,
                    )
                    torch.testing.assert_close(out, expected, rtol=2e-2, atol=2e-2)
                    torch.testing.assert_close(
                        lse.reshape(expected_lse.shape),
                        expected_lse,
                        rtol=0,
                        atol=1e-3,
                    )


if __name__ == "__main__":
    unittest.main()
