"""intel_xpu topk > 1 target-verify expand pass.

Each tree-verify query row must attend exactly its tree branch, and the pass
must be recordable in an XPU graph; the page_size=1 path it replaced was not
(its boolean-mask pack failed capture with UR_RESULT_ERROR_UNSUPPORTED_FEATURE).
"""

import unittest

import torch

from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.test_utils import CustomTestCase

register_xpu_ci(est_time=60, suite="stage-b-test-1-gpu-xpu")

_HAS_XPU = hasattr(torch, "xpu") and torch.xpu.is_available()
if _HAS_XPU:
    from sglang.srt.layers.attention.xpu_backend import (
        _tree_expand_attention,
        _tree_expand_scratch_page,
        _tree_verify_expand_metadata,
    )

HQ, HK, D = 32, 8, 128
PAGE_SIZE = 64


def _tree_inputs(device, prefix_lens, branches):
    """Two requests sharing one tree shape; ``branches[i]`` lists the draft
    columns query row i attends. Slots are a fixed permutation so a wrong
    gather cannot accidentally read the right KV."""
    num_draft = len(branches)
    num_slots = 4 * PAGE_SIZE
    perm = torch.randperm(num_slots, generator=torch.Generator().manual_seed(0))
    width = max(prefix_lens) + num_draft
    req_to_token = torch.stack(
        [perm[r * width : (r + 1) * width] for r in range(len(prefix_lens))]
    ).to(device=device, dtype=torch.int32)
    rows = []
    for prefix_len in prefix_lens:
        for branch in branches:
            row = [True] * prefix_len + [j in branch for j in range(num_draft)]
            rows.append(torch.tensor(row))
    custom_mask = torch.cat(rows).to(device)
    seq_lens = torch.tensor(prefix_lens, device=device, dtype=torch.int64)
    req_pool_indices = torch.arange(len(prefix_lens), device=device)
    return req_to_token, custom_mask, seq_lens, req_pool_indices


def _reference(q, k_flat, v_flat, page_table, cache_seqlens, scale):
    rows, width = page_table.shape
    keys = k_flat[page_table.long()].float()
    values = v_flat[page_table.long()].float()
    scores = torch.einsum(
        "rhgd,rkhd->rhgk", q.float().view(rows, HK, HQ // HK, D), keys
    )
    scores = scores * scale
    valid = torch.arange(width, device=q.device) < cache_seqlens[:, None]
    scores = scores.masked_fill(~valid[:, None, None, :], float("-inf"))
    out = torch.einsum("rhgk,rkhd->rhgd", torch.softmax(scores, dim=-1), values)
    return out.reshape(rows, HQ, D), torch.logsumexp(scores, -1).reshape(rows, HQ)


@unittest.skipUnless(_HAS_XPU, "XPU device required")
class TestXPUTreeVerifyExpand(CustomTestCase):
    def test_metadata_selects_each_rows_tree_branch(self):
        """Rows list exactly their branch slots first, also for the second
        request's offset mask rows and for a padded row past the real mask."""
        device = "xpu"
        branches = [[0], [0, 1], [0, 2], [0, 1, 3]]
        prefix_lens = [5, 9]
        req_to_token, custom_mask, seq_lens, req_pool_indices = _tree_inputs(
            device, prefix_lens, branches
        )
        page_table, cache_seqlens = _tree_verify_expand_metadata(
            req_to_token=req_to_token,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            custom_mask=custom_mask,
            num_draft_tokens=len(branches),
        )
        for req, prefix_len in enumerate(prefix_lens):
            for i, branch in enumerate(branches):
                row = req * len(branches) + i
                want = [int(req_to_token[req, prefix_len + j]) for j in branch]
                self.assertEqual(int(cache_seqlens[row]), len(branch))
                self.assertEqual(page_table[row, : len(branch)].tolist(), want)

        # Graph replay pads the batch but keeps the real (smaller) custom_mask.
        padded_seq_lens = torch.cat([seq_lens, seq_lens.new_ones(1)])
        padded_indices = torch.cat([req_pool_indices, req_pool_indices.new_zeros(1)])
        page_table, cache_seqlens = _tree_verify_expand_metadata(
            req_to_token=req_to_token,
            req_pool_indices=padded_indices,
            seq_lens=padded_seq_lens,
            custom_mask=custom_mask,
            num_draft_tokens=len(branches),
        )
        real_rows = len(prefix_lens) * len(branches)
        self.assertEqual(
            cache_seqlens[:real_rows].tolist(),
            [len(b) for b in branches] * len(prefix_lens),
        )
        self.assertTrue(bool((cache_seqlens[real_rows:] >= 1).all()))

    def test_expand_attention_matches_reference_in_graph(self):
        """Eager and XPU-graph replay both match an fp32 reference (output
        and natural-log LSE, which merge_state_v2 consumes)."""
        device = torch.device("xpu")
        gen = torch.Generator(device=device).manual_seed(0)
        num_draft = 8
        rows = 3 * num_draft
        num_pages = 32
        k_cache = torch.randn(
            num_pages,
            PAGE_SIZE,
            HK,
            D,
            device=device,
            dtype=torch.bfloat16,
            generator=gen,
        )
        v_cache = torch.randn_like(k_cache)
        k_flat = k_cache.view(-1, HK, D)
        v_flat = v_cache.view(-1, HK, D)

        def random_tree():
            slots = torch.stack(
                [
                    torch.randperm(num_pages * PAGE_SIZE, device=device, generator=gen)[
                        :num_draft
                    ]
                    for _ in range(rows)
                ]
            ).to(torch.int32)
            lens = torch.randint(
                1,
                num_draft + 1,
                (rows,),
                device=device,
                dtype=torch.int32,
                generator=gen,
            )
            return slots, lens

        q = torch.randn(rows, HQ, D, device=device, dtype=torch.bfloat16, generator=gen)
        page_table, cache_seqlens = random_tree()
        cu_seqlens_q = torch.arange(rows + 1, device=device, dtype=torch.int32)
        page = _tree_expand_scratch_page(num_draft)
        scratch_k = torch.zeros(rows, page, HK, D, device=device, dtype=torch.bfloat16)
        scratch_v = torch.zeros_like(scratch_k)
        scratch_page_table = torch.arange(rows, device=device, dtype=torch.int32)[
            :, None
        ]
        scale = D**-0.5

        def run():
            return _tree_expand_attention(
                q=q,
                key_cache=k_cache,
                value_cache=v_cache,
                page_table=page_table,
                cache_seqlens=cache_seqlens,
                cu_seqlens_q=cu_seqlens_q,
                scratch_k=scratch_k,
                scratch_v=scratch_v,
                scratch_page_table=scratch_page_table,
                scale=scale,
                softcap=0.0,
                window_size=(-1, -1),
            )

        def check(out, lse):
            ref_out, ref_lse = _reference(
                q, k_flat, v_flat, page_table, cache_seqlens, scale
            )
            torch.testing.assert_close(out.float(), ref_out, atol=2e-2, rtol=2e-2)
            torch.testing.assert_close(lse.T.float(), ref_lse, atol=1e-3, rtol=1e-3)

        out, lse, *_ = run()
        check(out, lse)

        stream = torch.xpu.Stream()
        with torch.xpu.stream(stream):
            run()
        torch.xpu.synchronize()
        graph = torch.xpu.XPUGraph()
        with torch.xpu.graph(graph, stream=stream):
            graph_out, graph_lse, *_ = run()

        new_page_table, new_cache_seqlens = random_tree()
        page_table.copy_(new_page_table)
        cache_seqlens.copy_(new_cache_seqlens)
        q.copy_(torch.randn(q.shape, device=device, dtype=q.dtype, generator=gen))
        graph.replay()
        torch.xpu.synchronize()
        check(graph_out, graph_lse)


if __name__ == "__main__":
    unittest.main()
